"""
Entry point.
    RUN_MODE=1 python -m srcs.app.main  # laptop camera + keyboard
    python -m srcs.app.main             # Raspberry Pi camera + GPIO
"""

from __future__ import annotations

import json
import sys

from config import settings
from config.hardware.camera import Camera
from config.hardware.display import Display
from srcs.models.barcode_model import BarcodeModel
from srcs.models.risk_model import RiskModel
from srcs.models.segmentation_model import SegmentationModel
from srcs.utils.lcd_formatter import lcd_records_for_fusion, load_lcd_document
from srcs.app.state_machine import StateMachine
from srcs.app.gui import App
from srcs.utils.logger import get_logger

log = get_logger(__name__)

def _load_containers_db() -> dict:
    with settings.CONTAINERS_DB_PATH.open(encoding="utf-8") as f:
        return json.load(f)

def _load_lcd_records() -> list[dict]:
    lcd_document, lcd_source = load_lcd_document(
        settings.CURRENT_LCD_PDF_PATH, settings.EXAMPLE_LCD_JSON_PATH
    )
    if lcd_source == settings.EXAMPLE_LCD_JSON_PATH:
        log.warning("No current LCD PDF found; using example data at %s", lcd_source)
    return lcd_records_for_fusion(lcd_document)

def main() -> None:
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (hardware) or 1 (developer)")
        settings.RUN_MODE = mode
        settings.MOCK_HARDWARE = mode == 1

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