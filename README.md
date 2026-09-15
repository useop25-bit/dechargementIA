# Container Inspection Station

An embedded inspection system built on a Raspberry Pi (4 or 5) that photographs
shipping containers inside a truck, cross-references that photo against
declared shipment data (the "LCD" list), runs three independent ML models on
the image, fuses everything into a single probabilistic read of "what
container is this, and what's in it", and produces an automatic **decision**
(what to select/remove from the load, and why) with a human validation step.

This document is the single source of truth for the project. Read it before
touching the code.

---

## 1. Problem statement

A truck arrives loaded with containers. An operator needs to:

1. Take a picture of the load (or of a container).
2. Have the system automatically:
   - detect security/safety **risks** visible in the picture,
   - detect and segment every **container instance** visible in the picture,
   - detect and decode every **barcode** visible in the picture and localize
     it in the image,
   - read the **LCD** (the shipment's declared reference list, given as a
     CSV) and the **container reference database** (a JSON mapping container
     numbers to their physical properties),
   - **fuse** the visual detections with the declared data to figure out,
     with a confidence score, which detected container instance corresponds
     to which declared reference,
   - **decide** whether a container should be pulled out of the load
     (security risk, mismatch, damage, etc.), producing a selection zone,
     and an explanation of the decision,
   - **draw** the decision on the picture (the zone to extract, risk
     annotations, text) and show it to the operator,
3. Let the operator **validate or reject** the automatic decision with a
   physical button.

Everything runs on a Raspberry Pi wired to a USB webcam, a screen, and two
push buttons.

---

## 2. Physical / hardware design

### 2.1 Bill of materials

| Component | Notes |
|---|---|
| Raspberry Pi 4 or 5 (4GB+ RAM recommended) | Pi 5 strongly preferred if segmentation model is heavy — see §6.1 |
| USB webcam | Accessed through OpenCV (`cv2.VideoCapture`), no `picamera2` dependency |
| Screen | Any HDMI monitor/touchscreen the Pi can drive as a normal X11/Wayland display |
| 2x momentary push buttons | Wired to GPIO with pull-down/pull-up + debouncing |
| Wires, breadboard or perfboard | For the buttons circuit |
| (later) 3D printed enclosure | Not addressed in this phase — see `hardware/` for pinout to design around |

### 2.2 Buttons

Two physical buttons drive the entire workflow as a simple state machine:

- **Button A — CAPTURE**: takes a picture from the webcam, freezes it on
  screen. Pressing it again while a picture is displayed (before validating)
  retakes the picture.
- **Button B — VALIDATE**: has a dual role depending on the current state of
  the state machine (see §4):
  - When a **raw picture** is on screen → confirms it should be sent to the
    analysis pipeline.
  - When the **annotated/decision picture** is on screen → confirms
    (accepts) the automatic decision. A **long press** (or a future 3rd
    button, configurable) rejects the decision and sends the operator back
    to CAPTURE.

GPIO pins are configurable in `config/settings.py`, default:
- `BUTTON_CAPTURE_PIN = 17`
- `BUTTON_VALIDATE_PIN = 27`

Wiring: each button between the GPIO pin and GND, using the Pi's internal
pull-up resistor (`GPIO.PUD_UP`), so a press reads `LOW`. Debounce is handled
in software (`hardware/buttons.py`).

### 2.3 Display

The display is driven as a normal desktop window (OpenCV `imshow` window,
full-screen). This keeps the code screen-agnostic — it works identically on
an HDMI monitor, an official touchscreen, or a small HDMI TFT. If the actual
screen ends up being SPI-only (no framebuffer/X11), `hardware/display.py` is
the only file that needs to change (e.g. swap to `luma.lcd` or `pygame` with
`fbcon`).

### 2.4 3D printed box

Not needed yet per current scope. Once the screen/case are chosen precisely,
add a parametric OpenSCAD file under `hardware/enclosure/` with cutouts for:
camera, screen, 2 buttons, cable routing, ventilation for the Pi.

---

## 3. Data model

### 3.1 The LCD (shipment reference list)

"LCD" = the truck's declared load list, given to us as a **raw CSV** that
contains a lot of irrelevant columns. `pipeline/lcd_formatter.py` converts it
into a clean JSON keeping only:

- `reference_number` — the shipment reference id,
- `container_number` — the container this reference is loaded in,
- a handful of other useful fields (quantity, description, weight — exact
  list is a `#TODO`, see the file).

Example output (`data/lcd/example_lcd.json`, generated from
`data/lcd/example_lcd.csv`):

```json
[
  {
    "reference_number": "REF-00231",
    "container_number": "MSCU1234567",
    "description": "Machine parts",
    "quantity": 12,
    "weight_kg": 340.5
  }
]
```

### 3.2 Container reference database

A separate, mostly-static JSON (`data/containers_db.json`) maps a
**container number** to its physical characteristics (dimensions, type,
tare weight, ISO code, etc.). This is looked up by the fusion algorithm once
it has a candidate container number (read from a barcode, or inferred).

```json
{
  "MSCU1234567": {
    "type": "20ft dry",
    "length_mm": 6058,
    "width_mm": 2438,
    "height_mm": 2591,
    "tare_weight_kg": 2300
  }
}
```

### 3.3 Model outputs (contracts between pipeline stages)

To keep every stage independently testable, all model wrappers return plain
Python dicts / dataclasses with a fixed shape — see `utils/geometry.py` and
the docstrings in each `models/*.py` file for the exact schema
(`RiskDetection`, `SegmentationInstance`, `BarcodeDetection`).

---

## 4. Software architecture

```
Webcam ──▶ hardware/camera.py
                     │
             app/state_machine.py  ◀── hardware/buttons.py (2 GPIO buttons)
                     │
             hardware/display.py  (shows raw pic / result pic)
                     │
      ┌──────────────┼───────────────────────┐
      ▼               ▼                        ▼
models/risk_model  models/segmentation_model  models/barcode_model
      │               │                        │
      │               └─────────┬──────────────┘
      │                         ▼
      │              pipeline/fusion.py  ◀── pipeline/lcd_formatter.py (CSV→JSON)
      │                         │          ◀── data/containers_db.json
      │                         ▼
      └───────────────▶ pipeline/decision.py
                                 │
                                 ▼
                         pipeline/draw.py
                                 │
                                 ▼
                        hardware/display.py (annotated result)
                                 │
                     operator presses VALIDATE (accept/reject)
```

### 4.1 State machine

`app/state_machine.py` implements the following states (also documented
inline in the file):

1. `IDLE` — waiting, showing a live-ish placeholder or the last result.
2. `PICTURE_TAKEN` — a raw frame is frozen on screen, waiting for VALIDATE
   (confirm → go to `ANALYZING`) or CAPTURE (retake → stay in
   `PICTURE_TAKEN`).
3. `ANALYZING` — pipeline is running (risk + segmentation + barcode + LCD +
   fusion + decision + draw). Screen shows a "processing" placeholder.
4. `RESULT_SHOWN` — annotated picture is on screen with the decision
   (selection zone + explanation text). Waiting for VALIDATE (accept → log
   + save + back to `IDLE`) or reject (back to `IDLE`/`PICTURE_TAKEN`,
   configurable).

### 4.2 Pipeline stages, in detail

1. **Risk model** (`models/risk_model.py`) — a YOLO(-style) detector,
   already trained, that looks at the whole picture and flags "risk" zones
   / labels (open flames, damaged packaging, hazmat symbols, etc. — the
   exact taxonomy is whatever the model was trained on). Output: a list of
   `RiskDetection(label, confidence, bbox)`.

2. **Segmentation model** (`models/segmentation_model.py`) — an Ultralytics
   instance-segmentation model (YOLOv8/11-seg) that finds every **container
   instance** in the picture and returns a class + a pixel mask + bbox per
   instance.

3. **Barcode model** (`models/barcode_model.py`) — localizes barcodes in the
   image (bbox) then decodes them (via `pyzbar`/`opencv` barcode detector).
   Returns `BarcodeDetection(value, symbology, bbox, confidence)`.

4. **LCD formatter** (`pipeline/lcd_formatter.py`) — CSV → JSON as described
   in §3.1. Runs once per truck/load (not necessarily once per picture —
   configurable, see file).

5. **Fusion algorithm** (`pipeline/fusion.py`) — this is the core
   "reasoning" piece and is intentionally left as a well-scaffolded
   `#TODO`: it must associate each segmented container **instance** (from
   step 2) with a **declared reference / container number** (from step 4),
   using the barcode reading (step 3) as the strongest signal when
   available (barcode bbox is contained in / overlaps a segmentation
   instance → very high confidence match) and falling back to weaker
   heuristics (dimensions from `containers_db.json` vs. apparent size in
   image, count of expected containers vs. detected instances, etc.) when
   no barcode is readable on a given instance. Output: a probabilistic
   association table — for each detected instance, a ranked list of
   candidate references with a confidence score.

6. **Decision algorithm** (`pipeline/decision.py`) — combines the fusion
   output with the risk output to decide, per picture:
   - a **selection zone** to extract: `(x, y, width, height)`,
   - a **confidence weight**,
   - a **human-readable explanation** (why this zone, why this decision —
     e.g. "container MSCU1234567 flagged: barcode confirms reference
     REF-00231 (declared: machine parts, 340.5kg) but risk model detected
     a hazmat symbol at 92% confidence inside this container's mask").
   This is the second big `#TODO` — the decision logic (thresholds, rules,
   how risk + fusion combine into one weight) needs your business rules.

7. **Draw** (`pipeline/draw.py`) — purely mechanical: draws the selection
   rectangle, risk boxes/labels, and the explanation text onto a copy of the
   original image, ready to display.

### 4.3 Why a "fusion" stage at all?

A barcode is not always readable (blur, angle, occlusion), and the
declared LCD doesn't tell you *where* in the picture a given reference is.
Fusion is what lets the system say "I'm 87% sure this segmented blob is
container MSCU1234567" even when the barcode is unreadable, by combining
partial evidence. This is standard **data fusion / evidence combination**
territory — naive Bayes weighted scoring, Dempster-Shafer belief
combination, or a simple rule-based scoring system are all reasonable
approaches; see the `#TODO` in `pipeline/fusion.py` for a proposed simple
implementation you can start from and refine.

---

## 5. Project layout

```
container_inspection/
├── README.md                     ← this file
├── requirements.txt
├── config/
│   └── settings.py                GPIO pins, camera index, paths, thresholds
├── hardware/
│   ├── camera.py                  webcam capture (OpenCV)
│   ├── buttons.py                 GPIO buttons with debounce + callbacks
│   └── display.py                 fullscreen window abstraction
├── data/
│   ├── lcd/
│   │   ├── example_lcd.csv        sample raw truck CSV
│   │   └── example_lcd.json       generated output (example)
│   ├── containers_db.json         container number → dimensions etc.
│   └── captures/                  runtime: saved pictures + results (gitignored)
├── models/
│   ├── risk_model.py              risk detector wrapper
│   ├── segmentation_model.py      container instance segmentation wrapper
│   ├── barcode_model.py           barcode localisation + decoding wrapper
│   └── weights/                   .pt weight files go here (not versioned)
├── pipeline/
│   ├── lcd_formatter.py           CSV → JSON
│   ├── fusion.py                  fuse segmentation + barcode + LCD
│   ├── decision.py                risk + fusion → decision
│   └── draw.py                    decision → annotated image
├── app/
│   ├── state_machine.py           the 4-state workflow
│   ├── gui.py                     glue between buttons/display/state machine
│   └── main.py                    entry point
├── utils/
│   ├── logger.py
│   └── geometry.py                shared dataclasses / bbox helpers
└── tests/
    └── test_lcd_formatter.py      example unit test to build on
```

---

## 6. Open points to settle before going further (tracked as `#TODO` in code)

1. **§6.1 Model weights** — you said the models already exist: drop the
   `.pt` (or `.onnx`/`.engine`) files into `models/weights/` and fill in the
   paths + class names in `config/settings.py`.
2. **§6.2 Risk taxonomy** — list the exact risk classes your model was
   trained on so `pipeline/decision.py` can map "risk label" → "how severe
   / does it force a rejection".
3. **§6.3 Barcode symbology** — confirm whether it's a standard barcode
   (EAN/Code128/QR) or the ISO 6346 container code printed as text/OCR
   rather than a scannable barcode; this changes `models/barcode_model.py`
   significantly (barcode decoding vs. OCR).
4. **§6.4 Fusion scoring** — the proposed default in `pipeline/fusion.py` is
   a simple weighted-evidence scorer; replace/tune once you know how
   reliable each signal (barcode, dimensions, count) is in practice.
5. **§6.5 Decision thresholds** — what confidence triggers an automatic
   "reject"/"select for extraction" vs. "needs manual review"?

---

## 7. Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate the LCD json from a CSV once per truck:
python -m pipeline.lcd_formatter data/lcd/example_lcd.csv data/lcd/example_lcd.json

# Run the app (needs a webcam + screen + buttons wired up; on a dev machine
# without GPIO, set MOCK_HARDWARE=1 to use keyboard keys instead of buttons):
MOCK_HARDWARE=1 python -m app.main
```

## 8. Dev without a Raspberry Pi

`config/settings.py` exposes `MOCK_HARDWARE`. When true, `hardware/buttons.py`
falls back to keyboard keys (`c` = capture, `v` = validate) instead of GPIO,
so the whole pipeline can be developed and tested on a laptop before
touching real hardware.
