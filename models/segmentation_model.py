"""
Wrapper around the (already trained) instance segmentation model that finds
every container instance in the picture.

Assumed Ultralytics YOLOv8/11-seg, matching the "ultralytics" tooling
choice. Each detected instance gets a stable `instance_id` (its index in
this call's output) — that id is what pipeline/fusion.py will use to refer
to "this specific blob in this specific picture" when building the
association table with the LCD references.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from config import settings
from utils.geometry import BBox, SegmentationInstance
from utils.logger import get_logger

log = get_logger(__name__)


class SegmentationModel:
    def __init__(self, weights_path=settings.SEGMENTATION_MODEL_WEIGHTS) -> None:
        log.info("Loading segmentation model from %s", weights_path)
        self._model = YOLO(str(weights_path))

    def predict(self, image: np.ndarray) -> list[SegmentationInstance]:
        """
        Run instance segmentation on a single BGR image.

        # TODO(classes): fill config.settings.SEGMENTATION_CLASS_NAMES with
        # your real classes (container types, or container vs. non-container
        # if the model also picks up other objects) — pipeline/fusion.py and
        # pipeline/decision.py may want to filter/weight by class_name.
        """
        results = self._model.predict(
            image, conf=settings.SEGMENTATION_CONF_THRESHOLD, verbose=False
        )
        result = results[0]

        instances: list[SegmentationInstance] = []

        if result.masks is None:
            log.warning("Segmentation model returned no masks for this image")
            return instances

        masks = result.masks.data.cpu().numpy()  # (N, H, W) binary-ish masks

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            mask = masks[i] if i < len(masks) else None

            instances.append(
                SegmentationInstance(
                    instance_id=i,
                    class_name=class_name,
                    confidence=conf,
                    bbox=BBox(x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1)),
                    mask=mask,
                )
            )

        log.info("Segmentation model found %d container instance(s)", len(instances))
        return instances
