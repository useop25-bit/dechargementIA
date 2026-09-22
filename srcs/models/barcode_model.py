"""
Wrapper around the barcode detection pipeline: localize barcode(s) in the
picture, then decode each one, so downstream code has both *where* the code
is (to cross-reference with a segmented container instance) and *what it
says* (to cross-reference with the LCD / containers_db).
"""

from __future__ import annotations

from typing import Any

from config import settings
from srcs.utils.geometry import BBox, BarcodeDetection
from srcs.utils.logger import get_logger

log = get_logger(__name__)


class BarcodeModel:
    def __init__(self, weights_path=settings.BARCODE_MODEL_WEIGHTS) -> None:
        self._localizer = None
        if not settings.ENABLE_BARCODE_MODEL:
            log.info("Barcode model disabled by ENABLE_BARCODE_MODEL")
            return
        import cv2
        from pyzbar import pyzbar
        from ultralytics import YOLO

        self._cv2 = cv2
        self._pyzbar = pyzbar
        log.info("Loading barcode localizer model from %s", weights_path)
        self._localizer = YOLO(str(weights_path))

    def predict(self, image: Any) -> list[BarcodeDetection]:
        """
        Localize candidate barcode regions with the trained detector, then
        decode each cropped region with pyzbar. Falls back to scanning the
        whole image with pyzbar if the localizer finds nothing (better a
        slow full-image scan than silently missing a readable code).
        """
        if self._localizer is None:
            return []

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

    def _decode_region(self, image: Any, bbox: BBox) -> tuple[str, str] | None:
        # small margin around the box helps pyzbar with quiet-zone requirements
        margin = 10
        x1 = max(0, bbox.x - margin)
        y1 = max(0, bbox.y - margin)
        x2 = min(image.shape[1], bbox.x2 + margin)
        y2 = min(image.shape[0], bbox.y2 + margin)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2GRAY)
        codes = self._pyzbar.decode(gray)
        if not codes:
            return None

        code = codes[0]
        return code.data.decode("utf-8", errors="replace"), code.type

    def _full_image_fallback(self, image: Any) -> list[BarcodeDetection]:
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        codes = self._pyzbar.decode(gray)

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
