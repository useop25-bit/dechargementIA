"""
USB webcam capture, via OpenCV.

Kept deliberately simple: open once, grab a frame on demand. If you later
switch to the official Pi camera module, this is the only file to change
(swap the backend to `picamera2` behind the same `Camera` interface).
"""

from __future__ import annotations

import cv2
import numpy as np

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(
        self,
        index: int = settings.CAMERA_INDEX,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera at index {self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Let auto-exposure / auto-white-balance settle before we trust a frame.
        for _ in range(settings.CAMERA_WARMUP_FRAMES):
            self._cap.read()

        log.info("Camera opened (index=%s, %sx%s)", self.index, self.width, self.height)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera closed")

    def capture(self) -> np.ndarray:
        """Grab a single BGR frame. Raises CameraError on failure."""
        if self._cap is None:
            raise CameraError("Camera is not open — call open() first")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera")

        return frame

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
