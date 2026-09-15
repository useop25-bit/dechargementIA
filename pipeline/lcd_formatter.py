"""
Convert the raw LCD CSV (truck's declared shipment list, full of columns we
don't care about) into a clean JSON list keeping only what the fusion
algorithm needs: reference_number, container_number, and a few extra useful
fields.

Usage:
    python -m pipeline.lcd_formatter <input.csv> <output.json>

Or programmatically:
    from pipeline.lcd_formatter import format_lcd_csv
    records = format_lcd_csv("data/lcd/example_lcd.csv")
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# #TODO: confirm against the *real* CSV export from your source system —
# this example was built from a plausible-looking freight CSV. Update the
# left-hand keys to match your actual CSV header names exactly (case
# sensitive). The right-hand values are the clean output field names and
# must match settings.LCD_OUTPUT_FIELDS.
LCD_CSV_COLUMN_MAP: dict[str, str] = {
    "reference_number": "reference_number",
    "container_number": "container_number",
    "description": "description",
    "quantity": "quantity",
    "weight_kg": "weight_kg",
}

# Columns that should be parsed as numbers rather than left as strings.
_NUMERIC_FIELDS = {"quantity", "weight_kg"}


def _coerce(field_name: str, raw_value: str) -> Any:
    raw_value = (raw_value or "").strip()
    if field_name in _NUMERIC_FIELDS:
        if raw_value == "":
            return None
        try:
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            log.warning("Could not parse numeric field %s=%r, keeping as string", field_name, raw_value)
            return raw_value
    return raw_value


def format_lcd_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Read the raw LCD CSV and return a list of clean dict records, one per
    reference number, keeping only the fields defined in LCD_CSV_COLUMN_MAP.

    Rows missing a reference_number or container_number are skipped (with a
    warning) since they can't be used downstream by the fusion algorithm.
    """
    csv_path = Path(csv_path)
    records: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        missing_columns = set(LCD_CSV_COLUMN_MAP) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"CSV is missing expected columns {missing_columns}. "
                f"Found columns: {reader.fieldnames}. "
                f"Update LCD_CSV_COLUMN_MAP in pipeline/lcd_formatter.py "
                f"to match your real CSV headers (see #TODO)."
            )

        for row_num, row in enumerate(reader, start=2):  # header is line 1
            record = {
                out_field: _coerce(out_field, row.get(csv_field, ""))
                for csv_field, out_field in LCD_CSV_COLUMN_MAP.items()
            }

            if not record.get("reference_number") or not record.get("container_number"):
                log.warning(
                    "Skipping CSV row %d: missing reference_number or container_number (%r)",
                    row_num,
                    record,
                )
                continue

            records.append(record)

    log.info("Formatted %d LCD records from %s", len(records), csv_path)
    return records


def write_lcd_json(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", output_path)


def load_lcd_json(json_path: str | Path = settings.CURRENT_LCD_JSON_PATH) -> list[dict[str, Any]]:
    with Path(json_path).open(encoding="utf-8") as f:
        return json.load(f)


def _main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m pipeline.lcd_formatter <input.csv> <output.json>")
        sys.exit(1)

    input_csv, output_json = sys.argv[1], sys.argv[2]
    records = format_lcd_csv(input_csv)
    write_lcd_json(records, output_json)


if __name__ == "__main__":
    _main()
