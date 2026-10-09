from __future__ import annotations

import unittest

import cv2
import numpy as np

from srcs.models.barcode_model import _restore_affine_bbox, _restore_scaled_bbox
from srcs.utils.geometry import BBox, BarcodeDetection


def _detection(bbox: BBox) -> BarcodeDetection:
    return BarcodeDetection(
        value="S1234567",
        symbology="CODE128",
        confidence=1.0,
        bbox=bbox,
    )


class BarcodeCoordinateTests(unittest.TestCase):
    def test_scaled_bbox_is_restored_to_original_image_coordinates(self) -> None:
        detection = _detection(BBox(x=40, y=60, width=80, height=20))

        restored = _restore_scaled_bbox(detection, scale=2.0)

        self.assertEqual(restored.bbox.as_tuple(), (20, 30, 40, 10))

    def test_rotated_bbox_is_restored_to_original_image_coordinates(self) -> None:
        original = BBox(x=10, y=20, width=30, height=20)
        matrix = cv2.getRotationMatrix2D((50, 50), 90, 1.0)
        original_corners = np.array(
            [[
                [original.x, original.y],
                [original.x2, original.y],
                [original.x2, original.y2],
                [original.x, original.y2],
            ]],
            dtype=np.float32,
        )
        rotated_corners = cv2.transform(original_corners, matrix)[0]
        x1 = int(np.floor(rotated_corners[:, 0].min()))
        y1 = int(np.floor(rotated_corners[:, 1].min()))
        rotated = BBox(
            x=x1,
            y=y1,
            width=int(np.ceil(rotated_corners[:, 0].max())) - x1,
            height=int(np.ceil(rotated_corners[:, 1].max())) - y1,
        )

        restored = _restore_affine_bbox(
            _detection(rotated),
            cv2.invertAffineTransform(matrix),
            image_shape=(100, 100),
        )

        self.assertEqual(restored.bbox.as_tuple(), original.as_tuple())


if __name__ == "__main__":
    unittest.main()
