"""
Central configuration for the whole project.

Every other module reads its constants from here instead of hard-coding
values, so a change of hardware pin, model path, or threshold only needs to
happen in one place.
"""

import os
from pathlib import Path


DEFAULT_SEGMENTATION_MODEL = "yolo11n-seg.pt"
DEFAULT_OBJECT_DETECTION_MODEL = "yolov8n.pt"

# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Runtime switches are deliberately environment-driven so scripts can select
# a test profile without editing application code.
RUN_MODE = int(os.environ.get("RUN_MODE", "0"))
MOCK_HARDWARE = RUN_MODE == 1 or _env_bool("MOCK_HARDWARE", False)
ENABLE_YOLO = _env_bool("ENABLE_YOLO", True)
ENABLE_RISK_MODEL = _env_bool("ENABLE_RISK_MODEL", False)
ENABLE_BARCODE_MODEL = _env_bool("ENABLE_BARCODE_MODEL", False)
ENABLE_FUSION = _env_bool("ENABLE_FUSION", True)
ENABLE_DECISION = _env_bool("ENABLE_DECISION", True)
DECISION_STRATEGY = os.environ.get("DECISION_STRATEGY", "leftmost")

WEIGHTS_DIR = PROJECT_ROOT / "config" / "models_weights"

MODEL_EXAMPLE_SEGMENTATION = os.environ.get(
    "SEGMENTATION_MODEL_WEIGHTS",
    DEFAULT_SEGMENTATION_MODEL,
)
MODEL_EXAMPLE_OBJECT_DETECTION = os.environ.get(
    "RISK_MODEL_WEIGHTS",
    str(WEIGHTS_DIR / DEFAULT_OBJECT_DETECTION_MODEL),
)

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
CURRENT_LCD_JSON_PATH = DATA_DIR / "lcd" / "current_lcd.json"
EXAMPLE_LCD_JSON_PATH = DATA_DIR / "lcd" / "example_lcd.json"
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
RISK_MODEL_WEIGHTS = Path(
    os.environ.get("RISK_MODEL_WEIGHTS", str(WEIGHTS_DIR / DEFAULT_OBJECT_DETECTION_MODEL))
)
SEGMENTATION_MODEL_WEIGHTS = os.environ.get(
    "SEGMENTATION_MODEL_WEIGHTS",
    str(WEIGHTS_DIR / DEFAULT_SEGMENTATION_MODEL),
)
BARCODE_MODEL_WEIGHTS = WEIGHTS_DIR / "barcode_localizer.pt"
RISK_CLASS_NAMES = [
    # "hazmat_symbol",
    # "package_damage",
    # "unauthorized_item",
]
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
FUSION_BARCODE_MATCH_WEIGHT = 0.7      # confidence boost when a barcode is read inside a segmented instance and matches a known container number
FUSION_DIMENSION_MATCH_WEIGHT = 0.2    # boost when apparent size roughly matches containers_db dimensions
FUSION_COUNT_CONSISTENCY_WEIGHT = 0.1  # boost when total instances found roughly matches expected LCD count
DECISION_REJECT_RISK_CONF = 0.6        # risk confidence above which the container is auto-flagged/selected
DECISION_MIN_FUSION_CONF_FOR_AUTOMATCH = 0.5
DECISION_MIN_BOX_WIDTH = 20
DECISION_MIN_BOX_HEIGHT = 20

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_DIR = DATA_DIR / "logs"
RESULTS_LOG_PATH = LOG_DIR / "test_results.jsonl"
SESSION_LOG_PATH = LOG_DIR / "session.jsonl"
LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "30"))
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 5
LOG_LEVEL = "INFO"