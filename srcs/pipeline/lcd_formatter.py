"""
Convert a real LCD document (truck/load manifest) into clean JSON.

This REPLACES the earlier CSV-based version. The real LCD you provided is a
PDF, not a CSV — a fixed-layout report with a header block, one or more
delivery-note ("NO BL") groups, and an item line per physical pallet/label
in the load. Tested end-to-end against a real 3-page LCD PDF (57 lines
across 3 pages, totals cross-checked against the document's own
"NBRE UM TOTAL" field).

Usage:
    python -m pipeline.lcd_formatter <input.pdf> <output.json>

Or programmatically:
    from pipeline.lcd_formatter import parse_lcd_pdf
    lcd = parse_lcd_pdf("data/lcd/example_lcd.pdf")
    lcd["lines"]  # flat list of {etq_palette, produit, designation, ...}

--------------------------------------------------------------------------
Real document structure (confirmed against an actual file)
--------------------------------------------------------------------------
Page header (repeats on every page):
    "LCD : MP22738635   Nom transport : ACT GROUP   N° du transport : 273117   Page 1 / 3"
    "Immatriculation : TTNU8964790   DATE : 30/09/26 14:20"
    "DUPLICATA"                                            <- only present on reprints
    "Transporteur : GEFCO INTERNATIONAL LOGISTICS   NBRE UM TOTAL : 57"
    "------ ... column separator rows ... ------"
    "ETQ Palette  PRODUIT  DESIGNATION  TYPE COLIS  NBRE COLIS  QTE / COLIS  QTE TOTALE  UNITE"

Per delivery note, once (only repeats when a NEW "NO BL" starts, which can
happen mid-page or carry across several pages without repeating):
    "NO BL : 2002344001 (AVX)  VENDEUR : A00A31 01 (BYD Lithium Battery Co., Ltd.)
     EXPEDITEUR : A00HZZ 01 (Xi'an FinDreams Battery Co., L)"

Then one line per pallet/label:
    "260109783 9869910680 MODULE BATTERIE M0745 1 12 12 UN"
     etq_palette  produit   designation    type_colis nbre_colis qte_colis qte_totale unite

"Immatriculation" is the trailer/container ID for the WHOLE load (ISO 6346
format: 4 letters + 7 digits, e.g. TTNU8964790) — not a truck license
plate, despite the French label. #TODO: confirmed on this one sample; if
you ever see an LCD covering more than one trailer, this assumption (one
immatriculation per LCD) would need revisiting.

"TYPE COLIS" (e.g. "M0745") is almost certainly a MABEC-style packaging
code, which is the "Code MABEC" field in the containers reference.json —
but that field was blank for every container in STANDARD.pdf, so there is
currently NO working crosswalk from "M0745" to a Code Emballage (e.g.
"00080"). #TODO: get the MABEC code list (or ask your source system for the
Code Emballage directly instead of/alongside TYPE COLIS) — without it,
fusion.py cannot use TYPE COLIS to pick which physical container a pallet
should be in.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber

from utils.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Line patterns, derived from and tested against a real LCD PDF
# --------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"LCD\s*:\s*(?P<lcd_number>\S+)\s+"
    r"Nom transport\s*:\s*(?P<transport_name>.+?)\s+"
    r"N°\s*du transport\s*:\s*(?P<transport_number>\S+)\s+"
    r"Page\s+(?P<page_num>\d+)\s*/\s*(?P<page_total>\d+)"
)
_IMMAT_RE = re.compile(
    r"Immatriculation\s*:\s*(?P<immatriculation>\S+)\s+"
    r"DATE\s*:\s*(?P<date>\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2})"
)
_TRANSPORTEUR_RE = re.compile(
    r"Transporteur\s*:\s*(?P<transporteur>.+?)\s+"
    r"NBRE UM TOTAL\s*:\s*(?P<nbre_um_total>\d+)"
)
_BL_RE = re.compile(
    r"NO BL\s*:\s*(?P<no_bl>\S+)\s*\((?P<no_bl_suffix>[^)]*)\)\s*"
    r"VENDEUR\s*:\s*(?P<vendeur_code>\S+)\s+(?P<vendeur_subcode>\S+)\s*\((?P<vendeur_name>[^)]*)\)\s*"
    r"EXPEDITEUR\s*:\s*(?P<expediteur_code>\S+)\s+(?P<expediteur_subcode>\S+)\s*\((?P<expediteur_name>[^)]*)\)"
)
# #TODO: DESIGNATION is captured non-greedily between PRODUIT and TYPE_COLIS.
# This works for every line in the tested sample ("MODULE BATTERIE", a fixed
# two-word designation) because the four trailing columns are always
# strictly numeric/short-code. If a future LCD has a DESIGNATION that itself
# contains a short all-caps alnum token resembling a TYPE COLIS code right
# before the real trailing columns, this regex could misparse — re-test
# against a few more real samples with varied DESIGNATION text if possible.
_LINE_RE = re.compile(
    r"^(?P<etq_palette>\d{5,15})\s+"
    r"(?P<produit>\S+)\s+"
    r"(?P<designation>.+?)\s+"
    r"(?P<type_colis>\S+)\s+"
    r"(?P<nbre_colis>\d+)\s+"
    r"(?P<qte_colis>\d+)\s+"
    r"(?P<qte_totale>\d+)\s+"
    r"(?P<unite>\S+)$"
)
_SEPARATOR_RE = re.compile(r"^[\s\-]+$")  # dashed column-separator rows


def _parse_date(raw: str) -> str | None:
    """'30/09/26 14:20' -> '2026-09-30T14:20:00' (DD/MM/YY, as used in this document)."""
    try:
        dt = datetime.strptime(raw, "%d/%m/%y %H:%M")
        return dt.isoformat()
    except ValueError:
        log.warning("Could not parse LCD date %r, keeping raw string", raw)
        return raw


def parse_lcd_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """
    Parse a real LCD PDF into a structured dict:
        {
          lcd_number, transport_name, transport_number, immatriculation,
          date, is_duplicata, transporteur, nbre_um_total,
          bons_livraison: [ {no_bl, vendeur_*, expediteur_*, lines: [...]}, ... ],
          lines: [ flat list of every line, each tagged with its no_bl ]
        }

    `lines` (flat) is what pipeline/fusion.py should consume — one entry per
    physical pallet/label expected in the load, keyed by `etq_palette`
    (which is exactly the value barcode-read off the "NO ETIQUETTE (S)"
    barcode on the physical tag — see models/barcode_model.py).
    """
    pdf_path = Path(pdf_path)

    header: dict[str, Any] = {}
    is_duplicata = False
    bl_groups: list[dict[str, Any]] = []
    current_bl: dict[str, Any] | None = None
    flat_lines: list[dict[str, Any]] = []
    unmatched: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line or _SEPARATOR_RE.match(line):
                    continue
                if line == "DUPLICATA":
                    is_duplicata = True
                    continue
                if line.startswith("NBRE") or line.startswith("ETQ Palette") or line.startswith("COLIS TOTALE"):
                    continue  # wrapped table-header fragments

                m = _HEADER_RE.search(line)
                if m:
                    header.update({k: v for k, v in m.groupdict().items()
                                    if k not in ("page_num", "page_total")})
                    continue

                m = _IMMAT_RE.search(line)
                if m:
                    header["immatriculation"] = m.group("immatriculation")
                    header["date"] = _parse_date(m.group("date"))
                    continue

                m = _TRANSPORTEUR_RE.search(line)
                if m:
                    header["transporteur"] = m.group("transporteur")
                    header["nbre_um_total"] = int(m.group("nbre_um_total"))
                    continue

                m = _BL_RE.search(line)
                if m:
                    current_bl = m.groupdict()
                    current_bl["lines"] = []
                    bl_groups.append(current_bl)
                    continue

                m = _LINE_RE.match(line)
                if m:
                    row = m.groupdict()
                    row["nbre_colis"] = int(row["nbre_colis"])
                    row["qte_colis"] = int(row["qte_colis"])
                    row["qte_totale"] = int(row["qte_totale"])

                    if current_bl is None:
                        log.warning(
                            "Item line found before any 'NO BL' header (etq_palette=%s) "
                            "— LCD may be malformed, attaching with no_bl=None",
                            row["etq_palette"],
                        )
                        flat_lines.append({**row, "no_bl": None,
                                            "vendeur_name": None, "expediteur_name": None})
                    else:
                        current_bl["lines"].append(row)
                        flat_lines.append({
                            **row,
                            "no_bl": current_bl["no_bl"],
                            "vendeur_name": current_bl["vendeur_name"],
                            "expediteur_name": current_bl["expediteur_name"],
                        })
                    continue

                unmatched.append(line)

    if unmatched:
        log.warning("Found %d unrecognized line(s) while parsing %s — these were skipped, not silently merged. "
                     "Inspect them: they may be a new field this parser doesn't know about yet.",
                     len(unmatched), pdf_path)
        for u in unmatched:
            log.warning("  unrecognized: %r", u)

    result = {
        **header,
        "is_duplicata": is_duplicata,
        "bons_livraison": bl_groups,
        "lines": flat_lines,
    }

    expected_total = header.get("nbre_um_total")
    if expected_total is not None and len(flat_lines) != expected_total:
        log.warning(
            "Parsed %d line(s) but LCD header declares NBRE UM TOTAL=%s — "
            "mismatch may indicate a parsing gap (check 'unmatched' lines above) "
            "or a genuinely incomplete scan (missing page).",
            len(flat_lines), expected_total,
        )
    else:
        log.info("Parsed %d line(s) across %d delivery note(s) from %s (matches declared total)",
                  len(flat_lines), len(bl_groups), pdf_path)

    return result


def write_lcd_json(lcd: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lcd, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", output_path)


def load_lcd_json(json_path: str | Path) -> dict[str, Any]:
    with Path(json_path).open(encoding="utf-8") as f:
        return json.load(f)


def _main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m pipeline.lcd_formatter <input.pdf> <output.json>")
        sys.exit(1)

    input_pdf, output_json = sys.argv[1], sys.argv[2]
    lcd = parse_lcd_pdf(input_pdf)
    write_lcd_json(lcd, output_json)


if __name__ == "__main__":
    _main()
