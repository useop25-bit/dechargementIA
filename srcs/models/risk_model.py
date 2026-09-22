"""
Wrapper around the (already trained) global risk-detection model.

The model looks at the whole picture and flags anything relevant to
security/safety (hazmat symbols, visible damage, suspicious items,
unauthorized objects — the exact taxonomy is whatever it was trained on,
see config.settings.RISK_CLASS_NAMES).

Assumed to be an Ultralytics-compatible detector (YOLOv8/11), since that's
what's specified for the segmentation model too and it's the simplest fit
for "already trained .pt weights". If your risk model is actually a
classifier (whole-image label, no bbox) or something else entirely, adjust
`RiskModel.predict` accordingly — the important part to preserve is the
*output contract*: a list of `RiskDetection`.
"""

from __future__ import annotations

from typing import Any

from config import settings
from srcs.utils.geometry import BBox, RiskDetection
from srcs.utils.logger import get_logger

log = get_logger(__name__)

class RiskModel:
    def __init__(self, weights_path=settings.RISK_MODEL_WEIGHTS) -> None:
        self._model = None
        if not settings.ENABLE_RISK_MODEL:
            log.info("Risk model disabled by ENABLE_RISK_MODEL")
            return
        from ultralytics import YOLO

        log.info("Loading risk model from %s", weights_path)
        self._model = YOLO(str(weights_path))

    def predict(self, image: Any) -> list[RiskDetection]:
        if self._model is None:
            return []

        results = self._model.predict(
            image, conf=settings.RISK_CONF_THRESHOLD, verbose=False
        )
        result = results[0]

        detections: list[RiskDetection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])

            detections.append(
                RiskDetection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1)),
                )
            )

        log.info("Risk model found %d detection(s)", len(detections))
        return detections