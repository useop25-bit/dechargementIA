"""
Wires the ButtonController to the StateMachine, and runs the main event
loop that keeps the OpenCV display window responsive.

The loop needs to call display.poll_key() regularly regardless of whether
MOCK_HARDWARE is on, because OpenCV requires its event loop to be pumped
(cv2.waitKey) for the window to redraw/respond — this is *also* how the
keyboard mock backend gets key events routed to the buttons controller.
"""

from __future__ import annotations

from config import settings
from hardware.buttons import ButtonController
from hardware.display import Display
from app.state_machine import StateMachine
from utils.logger import get_logger

log = get_logger(__name__)


class App:
    def __init__(self, state_machine: StateMachine, display: Display) -> None:
        self.state_machine = state_machine
        self.display = display
        self.buttons = ButtonController(
            on_capture=self.state_machine.on_capture,
            on_validate=self.state_machine.on_validate,
            on_validate_long=self.state_machine.on_validate_long,
        )
        self._running = False

    def run(self) -> None:
        self._running = True
        log.info("App started. %s",
                  "Keyboard mock active." if settings.MOCK_HARDWARE else "Listening on GPIO buttons.")

        try:
            while self._running:
                key = self.display.poll_key(wait_ms=30)

                if key == "q":  # always available: quit
                    log.info("Quit key pressed, shutting down")
                    self._running = False
                    break

                if settings.MOCK_HARDWARE and key is not None:
                    self.buttons.handle_key(key)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.buttons.cleanup()
        self.display.close()
        log.info("App shut down cleanly")
