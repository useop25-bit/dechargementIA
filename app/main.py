"""
Entry point.

    MOCK_HARDWARE=1 python -m app.main      # dev machine, no GPIO/real camera needed for buttons
    python -m app.main                      # on the Raspberry Pi

# TODO(lcd-lifecycle): this currently loads a single, pre-generated
# settings.CURRENT_LCD_JSON_PATH at startup, assuming one truck/load per
# run of the app. If the real workflow is "one app session covers multiple
# trucks", add a way to reload the LCD json (and containers_db, if it can
# also change) between trucks — e.g. a third button, or a small on-screen
# menu, or a filesystem watch on the LCD json path.
"""

from __future__ import annotations

import json
import sys

from config import settings
from hardware.camera import Camera
from hardware.display import Display
from models.barcode_model import BarcodeModel
from models.risk_model import RiskModel
from models.segmentation_model import SegmentationModel
from pipeline.lcd_formatter import load_lcd_json
from app.state_machine import StateMachine
from app.gui import App
from utils.logger import get_logger

log = get_logger(__name__)


def _load_containers_db() -> dict:
    with settings.CONTAINERS_DB_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_lcd_records() -> list[dict]:
    if not settings.CURRENT_LCD_JSON_PATH.exists():
        log.error(
            "No LCD json found at %s. Generate one first with:\n"
            "    python -m pipeline.lcd_formatter <input.csv> %s",
            settings.CURRENT_LCD_JSON_PATH,
            settings.CURRENT_LCD_JSON_PATH,
        )
        sys.exit(1)
    return load_lcd_json(settings.CURRENT_LCD_JSON_PATH)


def main() -> None:
    log.info("Starting Container Inspection app (MOCK_HARDWARE=%s)", settings.MOCK_HARDWARE)

    containers_db = _load_containers_db()
    lcd_records = _load_lcd_records()

    camera = Camera()
    camera.open()

    display = Display()

    risk_model = RiskModel()
    segmentation_model = SegmentationModel()
    barcode_model = BarcodeModel()

    state_machine = StateMachine(
        camera=camera,
        display=display,
        risk_model=risk_model,
        segmentation_model=segmentation_model,
        barcode_model=barcode_model,
        lcd_records=lcd_records,
        containers_db=containers_db,
    )

    app = App(state_machine=state_machine, display=display)

    try:
        app.run()
    finally:
        camera.close()


if __name__ == "__main__":
    main()
