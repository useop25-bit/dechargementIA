"""
Wrapper around the barcode detection pipeline: localize barcode(s) in the
picture, then decode each one, so downstream code has both *where* the code
is (to cross-reference with a segmented container instance) and *what it
says* (to cross-reference with the LCD / containers_db).

# TODO(symbology): you weren't sure yet whether this is a standard scannable
# barcode/QR or the ISO 6346 container code printed as plain text on the
# container door (which would need OCR, not barcode decoding). This wrapper
# currently assumes a real scannable barcode (via pyzbar) localized by your
# trained "barcode_localizer" YOLO model. If it turns out to be printed text
# instead:
#   1. Replace the `_decode_region` method's pyzbar call with an OCR call
#      (e.g. easyocr, pytesseract, or a custom OCR model),
#   2. Set `symbology="OCR-ISO6346"` on the resulting BarcodeDetection,
#   3. Optionally validate the OCR result against the ISO 6346 check-digit
#      algorithm to reject obviously wrong reads (a #TODO worth adding once
#      you confirm this path — happy to write that validator on request).
"""

from __future__ import annotations

import cv2
import numpy as np
from pyzbar import pyzbar
from ultralytics import YOLO

from config import settings
from utils.geometry import BBox, BarcodeDetection
from utils.logger import get_logger

log = get_logger(__name__)


class BarcodeModel:
    def __init__(self, weights_path=settings.BARCODE_MODEL_WEIGHTS) -> None:
        log.info("Loading barcode localizer model from %s", weights_path)
        self._localizer = YOLO(str(weights_path))

    def predict(self, image: np.ndarray) -> list[BarcodeDetection]:
        """
        Localize candidate barcode regions with the trained detector, then
        decode each cropped region with pyzbar. Falls back to scanning the
        whole image with pyzbar if the localizer finds nothing (better a
        slow full-image scan than silently missing a readable code).
        """
        detections: list[BarcodeDetection] = []

        results = self._localizer.predict(
            image, conf=settings.BARCODE_LOCALIZER_CONF_THRESHOLD, verbose=False
        )
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            loc_conf = float(box.conf[0])
            bbox = BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

            decoded = self._decode_region(image, bbox)
            if decoded is None:
                log.debug("Barcode localized at %s but could not be decoded", bbox)
                continue

            value, symbology = decoded
            detections.append(
                BarcodeDetection(
                    value=value,
                    symbology=symbology,
                    confidence=loc_conf,
                    bbox=bbox,
                )
            )

        if not detections:
            log.info("No barcode decoded via localizer, falling back to full-image scan")
            detections.extend(self._full_image_fallback(image))

        log.info("Barcode model decoded %d code(s)", len(detections))
        return detections

    def _decode_region(self, image: np.ndarray, bbox: BBox) -> tuple[str, str] | None:
        # small margin around the box helps pyzbar with quiet-zone requirements
        margin = 10
        x1 = max(0, bbox.x - margin)
        y1 = max(0, bbox.y - margin)
        x2 = min(image.shape[1], bbox.x2 + margin)
        y2 = min(image.shape[0], bbox.y2 + margin)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)
        if not codes:
            return None

        code = codes[0]
        return code.data.decode("utf-8", errors="replace"), code.type

    def _full_image_fallback(self, image: np.ndarray) -> list[BarcodeDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)

        detections = []
        for code in codes:
            x, y, w, h = code.rect
            detections.append(
                BarcodeDetection(
                    value=code.data.decode("utf-8", errors="replace"),
                    symbology=code.type,
                    confidence=1.0,  # pyzbar doesn't give a confidence score
                    bbox=BBox(x=x, y=y, width=w, height=h),
                )
            )
        return detections
