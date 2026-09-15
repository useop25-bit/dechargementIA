"""
Fullscreen display abstraction, built on an OpenCV window.

Using a plain OpenCV window keeps this screen-agnostic: it works the same
way on an HDMI monitor, an official Pi touchscreen running a desktop
session, or a small HDMI TFT. If the target screen turns out to be
SPI-only (framebuffer, no X11/Wayland), this is the only file that needs a
different backend (e.g. pygame + fbcon, or luma.lcd).
"""

from __future__ import annotations

import cv2
import numpy as np

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class Display:
    def __init__(self, window_name: str = settings.DISPLAY_WINDOW_NAME) -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        if settings.DISPLAY_FULLSCREEN:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.window_name, image)

    def show_message(self, text: str, size: tuple[int, int] = (1280, 720)) -> None:
        """Show a plain text placeholder (e.g. 'Processing...')."""
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
        x = (size[0] - text_size[0]) // 2
        y = (size[1] + text_size[1]) // 2
        cv2.putText(canvas, text, (x, y), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        self.show(canvas)

    def poll_key(self, wait_ms: int = 1) -> str | None:
        """
        Pump the OpenCV event loop and return the pressed key as a
        lowercase single character, or None. Must be called regularly
        (e.g. in the main loop) or the window will appear frozen — this is
        also how the keyboard mock backend for buttons.py gets its input.
        """
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 255:  # no key pressed
            return None
        try:
            return chr(key).lower()
        except ValueError:
            return None

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
