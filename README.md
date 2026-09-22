# Container Inspection Station

A Raspberry Pi-oriented computer-vision prototype for inspecting a truck load of containers. The station captures an image, detects visible objects/instances, compares the result with declared shipment data, and presents an explainable result for operator validation.

The project is in active development. The software architecture and test pipeline exist, but the generic model, fusion rules, and decision policy are not yet a validated industrial solution.

## Project purpose

The intended workflow is:

1. An operator points the camera at a truck load.
2. The operator captures an image.
3. The software runs the configured computer-vision stages.
4. Detected instances are compared with the declared LCD list and container metadata.
5. The software produces an annotated image and a decision explanation.
6. The operator accepts or rejects the displayed result.
7. The raw image, annotated image, and decision summary are stored for later review.

The current prototype is designed around a Raspberry Pi, camera, display, and two physical buttons, but it also supports keyboard-controlled development mode.

## Current status and limitations

### Implemented

- Python package layout under `srcs/`
- configuration under `config/`
- OpenCV camera abstraction
- OpenCV display abstraction
- GPIO buttons with keyboard fallback
- four-state application workflow
- Ultralytics YOLO segmentation wrapper
- optional risk/object-detection wrapper
- optional barcode localization and decoding wrapper
- LCD CSV-to-JSON conversion
- container metadata loading
- prototype fusion algorithm
- prototype decision algorithm
- annotated image rendering
- detailed one-shot pipeline runner in `tests/run_test.py`
- deterministic unit tests for geometry, LCD parsing, fusion, and decision behavior
- application and session logs under `data/logs/`

### Not production-ready yet

- `yolo11n-seg.pt` is a generic development segmentation model, not a container-trained model
- the risk model is disabled by default and has no final project taxonomy
- barcode/OCR identity handling is not finalized
- fusion is an explainable heuristic and has not been evaluated on a representative image set
- decision behavior is provisional and still needs operational approval
- startup validation for every hardware and data dependency is incomplete
- Raspberry Pi performance, camera placement, lighting, and systemd behavior require hardware validation
- session logging still needs a final retention and audit policy

A successful smoke test proves that the code path runs. It does not prove that the model recognizes containers or that an automatic decision is safe.

## Repository layout

```text
.
├── README.md                    Project reference
├── TODO                         Detailed remaining implementation work
├── .gitignore                   Local/runtime exclusions
├── config/
│   ├── __init__.py
│   ├── settings.py              Central runtime configuration
│   ├── requirements.txt         Python dependencies
│   ├── hardware/
│   │   ├── buttons.py           GPIO and keyboard button controller
│   │   ├── camera.py            OpenCV camera wrapper
│   │   └── display.py           OpenCV display wrapper
│   └── models_weights/
│       └── yolo11n-seg.pt       Development segmentation weight
├── srcs/
│   ├── __init__.py
│   ├── app/
│   │   ├── main.py              Interactive application entry point
│   │   ├── gui.py               Event loop and input wiring
│   │   └── state_machine.py     Capture/analyze/validation workflow
│   ├── models/
│   │   ├── segmentation_model.py
│   │   ├── risk_model.py
│   │   └── barcode_model.py
│   ├── pipeline/
│   │   ├── lcd_formatter.py     CSV normalization and JSON loading
│   │   ├── fusion.py             Detection-to-declaration matching
│   │   ├── decision.py           Action/selection policy
│   │   └── draw.py               Annotated image generation
│   └── utils/
│       ├── geometry.py            Shared dataclasses and bounding boxes
│       └── logger.py              Console/file/session logging
├── data/
│   ├── containers_db.json         Example container metadata
│   ├── lcd/
│   │   ├── example_lcd.csv       Example declaration input
│   │   └── example_lcd.json      Example normalized declaration
│   ├── captures/                  Runtime accepted/rejected sessions
│   └── logs/                      Runtime application and test logs
├── tests/
│   ├── test_lcd_formatter.py     LCD and basic decision tests
│   ├── test_pipeline_logic.py    Geometry/fusion/decision tests
│   └── run_test.py                Detailed camera/file pipeline diagnostic
└── Collab/                       Training notebooks and research work
```

The local `setup.md` file is intentionally ignored by Git. It contains machine-specific installation notes and deployment values and should not be treated as portable project documentation.

## Software architecture

```text
Camera or image file
        |
        v
Capture/input frame
        |
        +--> segmentation_model.py  -> container/instance detections
        +--> risk_model.py           -> optional risk detections
        +--> barcode_model.py        -> optional barcode detections
        |
        v
fusion.py
  combines detections, LCD records, and container database
        |
        v
decision.py
  determines review/selection behavior
        |
        v
draw.py
  renders boxes, labels, and explanations
        |
        v
Display or saved annotated image
```

The interactive application injects the camera, display, model wrappers, LCD records, and container database into the state machine. This keeps the workflow testable without requiring hardware in every unit test.

## Interactive application lifecycle

`srcs/app/state_machine.py` implements four states.

### `IDLE`

The application is waiting for a capture action and displays a ready message.

### `PICTURE_TAKEN`

A frame has been captured and frozen on screen.

- CAPTURE takes a new frame and replaces the current one.
- VALIDATE sends the frame to analysis.

### `ANALYZING`

The application runs the configured stages:

1. risk prediction, if enabled
2. segmentation prediction
3. barcode prediction, if enabled
4. fusion
5. decision
6. drawing

The display shows an analysis message while this work runs.

### `RESULT_SHOWN`

The annotated image is displayed.

- short VALIDATE accepts and saves the result
- long VALIDATE rejects and saves the result as rejected
- keyboard `r` simulates rejection in mock mode

The `q` key quits the application in keyboard mode.

## Hardware and input behavior

The default GPIO configuration uses BCM numbering:

- capture button: GPIO 17
- validate button: GPIO 27
- wiring: button between GPIO pin and GND
- input mode: internal pull-up, therefore pressed means LOW

The validate button has two meanings:

- short press: confirm the current stage
- long press: reject a displayed result

On a development computer, `MOCK_HARDWARE=1` uses keyboard input instead of GPIO:

- `c`: capture
- `v`: validate
- `r`: reject
- `q`: quit

OpenCV must continue receiving `waitKey` calls for the display window and keyboard fallback to work.

Concrete operating-system installation, GPIO wiring, camera setup, display setup, systemd service, and troubleshooting instructions belong in the local ignored [setup.md](setup.md).

## Models

### Segmentation model

`srcs/models/segmentation_model.py` loads an Ultralytics segmentation model and converts its output into `SegmentationInstance` objects containing:

- stable instance id for the current frame
- model class name
- confidence
- pixel bounding box
- optional mask array

The default configured weight is:

```text
config/models_weights/yolo11n-seg.pt
```

This model is present for software smoke testing. Its generic classes must not be interpreted as a validated container detector. A future production model must be trained on the target container images and its class names must be recorded in `config/settings.py`.

Override the path without editing code:

```bash
SEGMENTATION_MODEL_WEIGHTS=/path/to/container-segmentation.pt \
python3 tests/run_test.py --mode 0 --image path/to/image.jpg
```

### Risk/object detection model

`srcs/models/risk_model.py` is an optional Ultralytics object detector wrapper. It returns `RiskDetection` objects with a label, confidence, and bounding box.

It is disabled by default:

```text
ENABLE_RISK_MODEL=False
```

This is intentional. Until a real risk model and severity policy exist, the application must not imply that it can detect safety or security risks. Enable it only after configuring and validating a compatible model.

### Barcode model

`srcs/models/barcode_model.py` optionally combines a YOLO localizer with `pyzbar` decoding. It returns the decoded value, symbology, confidence, and bounding box.

The final project must decide whether container identity will come from:

- a machine-readable barcode
- OCR of the printed ISO 6346 container code
- both, with a defined precedence and fallback

The current barcode wrapper is not a substitute for that decision.

## Data contracts

### LCD declaration data

The LCD is the declared shipment list. The raw CSV is normalized by `srcs/pipeline/lcd_formatter.py` into records containing:

```json
{
  "reference_number": "REF-00231",
  "container_number": "MSCU1234567",
  "description": "Example cargo",
  "quantity": 12,
  "weight_kg": 340.5
}
```

The formatter:

- reads the CSV header using `csv.DictReader`
- checks that the required columns exist
- converts quantity and weight when possible
- skips rows without a reference or container number
- writes normalized JSON when requested

The current declaration can be placed at `data/lcd/current_lcd.json`. If it is absent, the application uses `data/lcd/example_lcd.json`.

### Container database

`data/containers_db.json` maps a container number to physical metadata. Current example fields include:

- type
- length in millimeters
- width in millimeters
- height in millimeters
- tare weight
- maximum payload

The fusion algorithm uses this metadata as supporting evidence. It is not a replacement for camera calibration or a reliable identity reading.

### Shared geometry objects

`srcs/utils/geometry.py` defines the interfaces between stages:

- `BBox`: pixel coordinate rectangle with area, center, containment, and IoU
- `RiskDetection`: one risk/object detection
- `SegmentationInstance`: one segmented model instance
- `BarcodeDetection`: one decoded/localized barcode
- `ContainerMatch`: one instance-to-declaration association
- `Decision`: one output action/selection result

Keeping these contracts centralized allows pipeline tests to use synthetic objects without loading YOLO weights.

## Pipeline behavior

### LCD formatting

The LCD formatter runs once per declaration input, not necessarily once per image. It creates the normalized records used by fusion.

### Fusion

`srcs/pipeline/fusion.py` currently uses weighted evidence:

1. barcode overlap and matching container number are the strongest signal
2. apparent bounding-box ratio versus database dimensions is a weak signal
3. detected count versus declared container count is a weak global signal

The current behavior is deliberately conservative when no usable barcode matches an LCD container: the instance remains unresolved and receives no confident identity. This is an interim heuristic and is documented for replacement in [TODO](TODO).

Important unresolved cases include duplicate container claims, conflicting barcode values, barcode values absent from the LCD, several references in one container, and no-barcode images.

### Decision

`srcs/pipeline/decision.py` associates risks with overlapping instances and calculates a decision weight. The current prototype can:

- select a leftmost sufficiently large instance
- mark a high-confidence overlapping risk as a security risk
- mark low-confidence/unmatched fusion as manual-review-like behavior
- produce a human-readable explanation

This is not yet a final policy. In particular, the current leftmost strategy can stop after one instance, and the final system must evaluate all instances before selecting the primary UI result.

### Drawing

`srcs/pipeline/draw.py` does not make decisions. It draws:

- selection rectangles
- matched container labels
- risk rectangles and labels
- an explanation panel below the image

The original input image is copied before drawing.

## Running the project

All commands below are run from the repository root.

### Install dependencies

The dependency list is in `config/requirements.txt`. Use a virtual environment. The complete Raspberry Pi procedure is in the ignored `setup.md`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
```

### Run tests

```bash
python3 -m pytest -q
python3 tests/test_pipeline_logic.py
```

The tests are deterministic and do not require a camera, GPIO, or model weights.

### Run the interactive application

Keyboard development mode:

```bash
MOCK_HARDWARE=1 python3 -m srcs.app.main
```

Raspberry hardware mode:

```bash
python3 -m srcs.app.main
```

### Run the diagnostic pipeline test

`tests/run_test.py` is a non-interactive one-shot runner. It uses the same segmentation, LCD, fusion, decision, and drawing stages but does not require the GUI state machine.

Mode `1` captures an image from the camera:

```bash
python3 tests/run_test.py --mode 1
```

Mode `0` uses the newest supported image found recursively under `data/captures`:

```bash
python3 tests/run_test.py --mode 0
```

Mode `0` with a specific image:

```bash
python3 tests/run_test.py --mode 0 --image path/to/image.jpg
```

Useful options:

```bash
python3 tests/run_test.py --help
python3 tests/run_test.py --mode 1 --camera-index 1
python3 tests/run_test.py --mode 0 --weights path/to/model.pt
python3 tests/run_test.py --mode 0 --decision-strategy all
```

Each run is stored under:

```text
data/logs/run_test/<run-id>/
├── input.jpg
├── annotated.jpg
├── run_test.log
└── report.json
```

The report records configuration, data sources, image metadata, model loading, timings, detections, risk/barcode status, fusion matches, decisions, output paths, and full traceback information on failure.

## Runtime storage and logging

### Captures

The interactive state machine saves accepted or rejected sessions under:

```text
data/captures/<timestamp>/
├── raw.jpg
├── annotated.jpg
└── summary.json
```

### Application logs

The configured application logger writes rotating logs and session events under `data/logs/`. The diagnostic runner creates an isolated directory per run. Runtime files are ignored by Git.

The current logger is a debugging and prototype audit baseline. Before deployment, add or finalize:

- one session id per inspection
- UTC timestamps for every stage
- model filename and hash
- LCD source/version
- inference durations
- all candidate fusion evidence
- final decision and operator action
- retention and image privacy rules

## Configuration reference

Configuration is centralized in `config/settings.py`.

### Runtime switches

- `RUN_MODE`: `0` hardware mode, `1` development mode
- `MOCK_HARDWARE`: keyboard fallback when true
- `ENABLE_YOLO`: enable segmentation model loading
- `ENABLE_RISK_MODEL`: enable optional risk detector
- `ENABLE_BARCODE_MODEL`: enable barcode localizer/decoder
- `ENABLE_FUSION`: enable matching stage
- `ENABLE_DECISION`: enable decision stage
- `DECISION_STRATEGY`: current `leftmost` or `all`

Most switches can be overridden with environment variables. Model paths can be overridden with `SEGMENTATION_MODEL_WEIGHTS` and `RISK_MODEL_WEIGHTS`.

### Hardware values

- `CAMERA_INDEX`
- `CAMERA_WIDTH`
- `CAMERA_HEIGHT`
- `CAMERA_WARMUP_FRAMES`
- `BUTTON_CAPTURE_PIN`
- `BUTTON_VALIDATE_PIN`
- `BUTTON_DEBOUNCE_MS`
- `BUTTON_LONG_PRESS_MS`
- `DISPLAY_FULLSCREEN`

### Data paths

- `DATA_DIR`
- `CAPTURES_DIR`
- `CONTAINERS_DB_PATH`
- `CURRENT_LCD_JSON_PATH`
- `EXAMPLE_LCD_JSON_PATH`
- `LOG_DIR`
- `RESULTS_LOG_PATH`
- `SESSION_LOG_PATH`

The values are derived from the repository root; avoid hardcoding absolute paths into application modules.

## Development model and production model

The repository supports an incremental workflow:

1. Run the generic segmentation model to validate software integration.
2. Test with recorded images using `tests/run_test.py --mode 0`.
3. Collect and label representative container images.
4. Train a container-specific segmentation model.
5. Decide barcode versus OCR identity.
6. Implement and measure fusion behavior.
7. Define and test the decision policy.
8. Validate the complete workflow on the target Raspberry Pi.
9. Enable automatic startup only after manual hardware tests succeed.

The generic model should never be used as evidence that the final model is accurate.

## Documentation map

- [README.md](README.md): project architecture, behavior, data contracts, commands, and limitations
- [TODO](TODO): detailed remaining implementation work and acceptance criteria
- `setup.md`: local ignored machine-specific installation and Raspberry Pi deployment notes
- `Collab/`: model-training notebooks and experiments

## Project completion criteria

The system is ready for a real deployment only when:

- startup validates required files and hardware configuration
- a container-trained model has been evaluated on held-out scenes
- identity extraction has measured false-match and unresolved rates
- fusion handles conflicts and ambiguity explicitly
- decision policy is approved and covered by tests
- disabled risk mode cannot create a security rejection
- all detected instances are considered before primary selection
- session records are structured and retained according to policy
- Raspberry camera, display, buttons, performance, restart, and failure paths are validated
