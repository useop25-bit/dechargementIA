"""
Physical button handling.

Exposes a single `ButtonController` with two callbacks:
    on_capture()          -> fired on a short press of the CAPTURE button
    on_validate()          -> fired on a short press of the VALIDATE button
    on_validate_long()     -> fired on a long press of the VALIDATE button
                              (used as "reject" in the RESULT_SHOWN state)

Two backends:
  - GPIO backend (real Raspberry Pi, RPi.GPIO), used when
    settings.MOCK_HARDWARE is False.
  - Keyboard backend (dev machine, OpenCV window key capture), used when
    settings.MOCK_HARDWARE is True. In that mode, key polling happens inside
    hardware/display.py's main loop (see `Display.poll_key`) and is routed
    here via `ButtonController.handle_key`.

Debouncing: GPIO backend uses a simple time-based debounce
(settings.BUTTON_DEBOUNCE_MS). Long-press detection compares press/release
timestamps against settings.BUTTON_LONG_PRESS_MS.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class ButtonController:
    def __init__(
        self,
        on_capture: Callable[[], None],
        on_validate: Callable[[], None],
        on_validate_long: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_capture = on_capture
        self.on_validate = on_validate
        self.on_validate_long = on_validate_long or (lambda: None)

        self._last_capture_press = 0.0
        self._last_validate_press = 0.0
        self._validate_press_started_at: float | None = None

        self._gpio = None
        if not settings.MOCK_HARDWARE:
            self._setup_gpio()
        else:
            log.info("MOCK_HARDWARE=1 — using keyboard fallback for buttons "
                      "('%s'=capture, '%s'=validate, hold-then-release for long press)",
                      settings.MOCK_KEY_CAPTURE, settings.MOCK_KEY_VALIDATE)

    # ------------------------------------------------------------------
    # Real GPIO backend
    # ------------------------------------------------------------------

    def _setup_gpio(self) -> None:
        import RPi.GPIO as GPIO  # imported lazily: not installed off-Pi

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(settings.BUTTON_CAPTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(settings.BUTTON_VALIDATE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(
            settings.BUTTON_CAPTURE_PIN,
            GPIO.FALLING,
            callback=self._handle_capture_edge,
            bouncetime=settings.BUTTON_DEBOUNCE_MS,
        )
        # For VALIDATE we watch both edges ourselves to measure press duration
        # (short press = validate/accept, long press = reject).
        GPIO.add_event_detect(
            settings.BUTTON_VALIDATE_PIN,
            GPIO.BOTH,
            callback=self._handle_validate_edge,
            bouncetime=50,
        )
        log.info("GPIO buttons configured (capture=%s, validate=%s)",
                  settings.BUTTON_CAPTURE_PIN, settings.BUTTON_VALIDATE_PIN)

    def _handle_capture_edge(self, channel: int) -> None:
        now = time.time() * 1000
        if now - self._last_capture_press < settings.BUTTON_DEBOUNCE_MS:
            return
        self._last_capture_press = now
        log.debug("CAPTURE button pressed")
        self.on_capture()

    def _handle_validate_edge(self, channel: int) -> None:
        GPIO = self._gpio
        pressed = GPIO.input(settings.BUTTON_VALIDATE_PIN) == GPIO.LOW
        now = time.time() * 1000

        if pressed:
            self._validate_press_started_at = now
            return

        # released
        if self._validate_press_started_at is None:
            return
        duration = now - self._validate_press_started_at
        self._validate_press_started_at = None

        if duration >= settings.BUTTON_LONG_PRESS_MS:
            log.debug("VALIDATE long-press detected (%.0fms) -> reject", duration)
            self.on_validate_long()
        else:
            log.debug("VALIDATE short-press detected (%.0fms) -> accept", duration)
            self.on_validate()

    # ------------------------------------------------------------------
    # Keyboard mock backend — called from hardware/display.py's event loop
    # ------------------------------------------------------------------

    def handle_key(self, key: str) -> None:
        """Feed a keypress (as returned by cv2.waitKey) when MOCK_HARDWARE."""
        if key == settings.MOCK_KEY_CAPTURE:
            log.debug("[mock] CAPTURE key pressed")
            self.on_capture()
        elif key == settings.MOCK_KEY_VALIDATE:
            log.debug("[mock] VALIDATE key pressed")
            self.on_validate()
        elif key == "r":
            log.debug("[mock] 'r' pressed -> simulate reject (long-press)")
            self.on_validate_long()

    def cleanup(self) -> None:
        if self._gpio is not None:
            self._gpio.cleanup()
