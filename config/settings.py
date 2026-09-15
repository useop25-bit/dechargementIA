"""
Central configuration for the whole project.

Every other module reads its constants from here instead of hard-coding
values, so a change of hardware pin, model path, or threshold only needs to
happen in one place.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# When set (env var MOCK_HARDWARE=1), buttons.py uses keyboard keys instead
# of real GPIO, and camera.py still uses the real webcam (a webcam is
# available on most dev laptops). Useful for developing off-Pi.
MOCK_HARDWARE = os.environ.get("MOCK_HARDWARE", "0") == "1"

# --------------------------------------------------------------------------
# Hardware — camera
# --------------------------------------------------------------------------

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_WARMUP_FRAMES = 5  # discard the first N frames (auto-exposure settle)

# --------------------------------------------------------------------------
# Hardware — buttons (BCM numbering)
# --------------------------------------------------------------------------

BUTTON_CAPTURE_PIN = 17
BUTTON_VALIDATE_PIN = 27
BUTTON_DEBOUNCE_MS = 200
BUTTON_LONG_PRESS_MS = 900  # long-press on VALIDATE == reject

# Keyboard fallback when MOCK_HARDWARE is True
MOCK_KEY_CAPTURE = "c"
MOCK_KEY_VALIDATE = "v"

# --------------------------------------------------------------------------
# Hardware — display
# --------------------------------------------------------------------------

DISPLAY_WINDOW_NAME = "Container Inspection"
DISPLAY_FULLSCREEN = True

# --------------------------------------------------------------------------
# Paths — data
# --------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
CONTAINERS_DB_PATH = DATA_DIR / "containers_db.json"

# The LCD json produced by pipeline/lcd_formatter.py for the *current* truck.
# In production this should probably be regenerated once per truck/load, not
# once per picture — see app/main.py for where it's loaded.
CURRENT_LCD_JSON_PATH = DATA_DIR / "lcd" / "current_lcd.json"

# #TODO: confirm the exact CSV column names you receive from your source
# system, then adjust LCD_CSV_COLUMN_MAP in pipeline/lcd_formatter.py to
# match (this file only defines *which output fields* we keep).
LCD_OUTPUT_FIELDS = [
    "reference_number",
    "container_number",
    "description",
    "quantity",
    "weight_kg",
]

# --------------------------------------------------------------------------
# Paths — model weights
# --------------------------------------------------------------------------

WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

# #TODO: point these at your actual trained weight files.
RISK_MODEL_WEIGHTS = WEIGHTS_DIR / "risk_model.pt"
SEGMENTATION_MODEL_WEIGHTS = WEIGHTS_DIR / "segmentation_model.pt"
BARCODE_MODEL_WEIGHTS = WEIGHTS_DIR / "barcode_localizer.pt"

# #TODO: fill in the real class names your risk model was trained with.
# This is used by pipeline/decision.py to decide severity per class.
RISK_CLASS_NAMES = [
    # "hazmat_symbol",
    # "package_damage",
    # "unauthorized_item",
]

# #TODO: fill in the real class names your segmentation model was trained
# with (container types, or "container" vs other objects, etc).
SEGMENTATION_CLASS_NAMES = [
    # "container_20ft",
    # "container_40ft",
]

# --------------------------------------------------------------------------
# Inference thresholds
# --------------------------------------------------------------------------

RISK_CONF_THRESHOLD = 0.4
SEGMENTATION_CONF_THRESHOLD = 0.5
BARCODE_LOCALIZER_CONF_THRESHOLD = 0.4

# --------------------------------------------------------------------------
# Fusion / decision thresholds
# --------------------------------------------------------------------------

# #TODO: tune these once you have real data. See pipeline/fusion.py and
# pipeline/decision.py for how they are used.
FUSION_BARCODE_MATCH_WEIGHT = 0.7      # confidence boost when a barcode is
                                        # read inside a segmented instance
                                        # and matches a known container number
FUSION_DIMENSION_MATCH_WEIGHT = 0.2    # boost when apparent size roughly
                                        # matches containers_db dimensions
FUSION_COUNT_CONSISTENCY_WEIGHT = 0.1  # boost when total instances found
                                        # roughly matches expected LCD count

DECISION_REJECT_RISK_CONF = 0.6        # risk confidence above which the
                                        # container is auto-flagged/selected
DECISION_MIN_FUSION_CONF_FOR_AUTOMATCH = 0.5  # below this, fusion result is
                                        # "unknown / needs manual review"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"
