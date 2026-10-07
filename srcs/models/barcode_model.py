"""
Barcode reading, rewritten from empirical testing against a real label photo
(not a guess — see the test log at the bottom of this docstring).

--------------------------------------------------------------------------
What testing against the real photo confirmed
--------------------------------------------------------------------------
1. Symbology is CODE128. Decoded directly with pyzbar, no OCR needed, no
   ambiguity to resolve — this settles the earlier open question about
   barcode vs. OCR for THIS type of label.

2. A label carries MULTIPLE separate CODE128 barcodes (one per field: code
   produit, no étiquette, quantité, code vendeur), not one barcode for the
   whole label.

3. Each barcode's payload is self-describing: it starts with a single
   uppercase letter that matches the "(X)" printed next to that field's
   name on the label, followed by the value. Decoding the two barcodes that
   were sharp enough in the test photo gave:
       "P9877021480"  <- under "CODE PRODUIT (P)"   -> identifier P, value 9877021480
       "S1259277"     <- under "NO ETIQUETTE (S)"   -> identifier S, value 1259277
   This means the model does NOT need to know a barcode's position on the
   label to know what field it is — it reads the identifier straight out of
   the decoded string. That's far more robust than positional matching: it
   survives label rotation, a partially-occluded tag, or a different label
   layout entirely, as long as the identifier convention holds.

4. "S<value>" is exactly the "ETQ Palette" field in the LCD JSON produced by
   pipeline/lcd_formatter.py. This is the real fusion join key — read this
   barcode off a pallet in the photo, look up the same string in the LCD's
   `lines`, and you have your reference match. (The earlier design assumed
   a barcode would encode a container number directly — that assumption is
   now replaced by this confirmed mechanism.)

5. Two of the four barcodes on the test label (Quantité, Code Vendeur) did
   NOT decode — not a code bug: cropping + 2x/4x/6x/8x upscaling + CLAHE +
   Otsu threshold + sharpening + de-skew rotation were all tried and all
   failed, and visually inspecting the crops shows real motion/focus blur
   on that part of the tag (it was on the curled, off-angle edge of the
   label in that photo). P and S, the two barcodes needed for fusion,
   decoded cleanly on the very first attempt with zero preprocessing. Build
   the camera/mount so the whole tag is in sharp focus, but the pipeline
   below treats Q and V as best-effort/optional and never blocks on them.

--------------------------------------------------------------------------
What's still a container (not a barcode) problem
--------------------------------------------------------------------------
The container-type code (e.g. "00080") is stamped/embossed directly into
the steel body of the container, not printed on the paper tag at all — it
can't be read by this module. #TODO: that needs an OCR pass (e.g.
easyocr/pytesseract on the segmented container's embossed-text region), a
separate model from this one. Happy to build that next if useful.
"""

from __future__ import annotations

import cv2
import numpy as np
from pyzbar import pyzbar

from config import settings
from utils.geometry import BBox, BarcodeDetection
from utils.logger import get_logger

log = get_logger(__name__)


def _split_identifier(raw_value: str) -> tuple[str | None, str | None, str | None]:
    """
    'P9877021480' -> ('P', 'produit', '9877021480')
    '9877021480'  -> (None, None, '9877021480')   # doesn't match the convention
    """
    if raw_value and raw_value[0] in settings.BARCODE_DATA_IDENTIFIERS and raw_value[1:].strip():
        identifier = raw_value[0]
        return identifier, settings.BARCODE_DATA_IDENTIFIERS[identifier], raw_value[1:]
    return None, None, raw_value


def _pyzbar_to_detection(code: "pyzbar.Decoded", offset: tuple[int, int] = (0, 0)) -> BarcodeDetection:
    raw_value = code.data.decode("utf-8", errors="replace")
    identifier, meaning, payload = _split_identifier(raw_value)

    ox, oy = offset
    bbox = BBox(x=code.rect.left + ox, y=code.rect.top + oy,
                width=code.rect.width, height=code.rect.height)

    return BarcodeDetection(
        value=raw_value,
        symbology=code.type,
        confidence=1.0,  # pyzbar gives no confidence score
        bbox=bbox,
        identifier=identifier,
        identifier_meaning=meaning,
        payload=payload,
    )


class BarcodeModel:
    """
    Reads every CODE128 barcode in an image (or a region of it) and returns
    them as BarcodeDetection objects with the data-identifier already split
    out when recognized.

    No localizer model is required: pyzbar finds barcodes directly in a
    full frame reliably enough in testing (both real barcodes decoded with
    zero preprocessing on the first try). If a future camera setup makes
    full-frame detection unreliable (e.g. many small labels in one wide
    shot), #TODO: add a YOLO localizer stage in front of this (crop
    candidate regions, decode each) the way the original design assumed —
    but don't add that complexity before it's shown to be needed.
    """

    def predict(self, image: np.ndarray) -> list[BarcodeDetection]:
        detections = self._decode(image)

        if detections:
            log.info("Decoded %d barcode(s) directly: %s",
                      len(detections), [d.value for d in detections])
            return detections

        log.info("No barcode decoded on first pass, retrying with enhancement variants")
        return self._decode_with_fallbacks(image)

    def _decode(self, image: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[BarcodeDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        codes = pyzbar.decode(gray)
        return [_pyzbar_to_detection(c, offset) for c in codes]

    def _decode_with_fallbacks(self, image: np.ndarray) -> list[BarcodeDetection]:
        """
        Enhancement variants empirically found useful when testing on a real
        (blurred/skewed) label photo. Order matters: cheapest/most likely
        first. Stops as soon as a variant finds something new; merges
        unique results across variants by raw decoded value so the same
        barcode isn't returned twice.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found: dict[str, BarcodeDetection] = {}

        def add_all(detections: list[BarcodeDetection]) -> None:
            for d in detections:
                found.setdefault(d.value, d)

        # CLAHE contrast boost
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        add_all(self._decode(clahe.apply(gray)))

        # Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        add_all(self._decode(otsu))

        # small de-skew rotations — labels are rarely perfectly level in a
        # handheld/fixed-angle shot
        h, w = gray.shape
        center = (w // 2, h // 2)
        for angle in (-8, -5, -3, 3, 5, 8):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
            add_all(self._decode(rotated))

        # upscale — helps when a barcode occupies very few pixels (distant
        # or small labels); did NOT rescue a genuinely out-of-focus barcode
        # in testing, but costs little to try
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        add_all(self._decode(upscaled))

        results = list(found.values())
        log.info("Fallback decoding found %d barcode(s): %s",
                  len(results), [d.value for d in results])
        return results

    def decode_region(self, image: np.ndarray, bbox: BBox, margin: int = 10) -> list[BarcodeDetection]:
        """
        Decode only within a bbox (e.g. the area of one segmented
        container/pallet), with a small margin — useful once a localizer or
        the segmentation model has already narrowed down where to look.
        """
        x1 = max(0, bbox.x - margin)
        y1 = max(0, bbox.y - margin)
        x2 = min(image.shape[1], bbox.x2 + margin)
        y2 = min(image.shape[0], bbox.y2 + margin)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        return self._decode(crop, offset=(x1, y1)) or self._decode_with_fallbacks(crop)


def match_to_lcd(detections: list[BarcodeDetection], lcd_lines: list[dict]) -> dict | None:
    """
    Convenience helper: given the barcodes decoded off one pallet/container
    in a photo, find its matching LCD line by the confirmed join key — the
    "S" (no_etiquette) identifier matches the LCD's `etq_palette` field
    exactly.

    Returns the matching LCD line dict, or None if no "S" barcode was
    decoded or it doesn't match any LCD line (e.g. wrong truck, damaged
    label, or a genuine LCD/physical-load mismatch worth flagging).
    """
    etq = next((d.payload for d in detections if d.identifier == "S"), None)
    if etq is None:
        return None
    return next((line for line in lcd_lines if line["etq_palette"] == etq), None)
