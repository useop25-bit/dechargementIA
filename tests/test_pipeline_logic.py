"""Deterministic pipeline tests.

The repository root is added explicitly so this file can be run both with
``pytest`` and directly via ``python tests/test_pipeline_logic.py``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from srcs.pipeline.decision import decide
from srcs.pipeline.fusion import fuse
from srcs.utils.geometry import BarcodeDetection, BBox, ContainerMatch, RiskDetection, SegmentationInstance

def test_bbox_iou_calculation():
    left = BBox(0, 0, 10, 10)
    right = BBox(5, 5, 10, 10)

    assert round(left.iou(right), 2) == 0.14

def test_fusion_uses_barcode_match_when_available():
    instances = [SegmentationInstance(0, "container", 0.95, BBox(10, 10, 100, 80))]
    barcodes = [BarcodeDetection("MSCU1234567", "CODE128", 0.98, BBox(20, 20, 40, 20))]
    lcd_records = [{
        "reference_number": "REF-00231",
        "container_number": "MSCU1234567",
        "description": "Test",
        "quantity": 1,
        "weight_kg": 10.0,
    }]
    containers_db = {
        "MSCU1234567": {
            "type": "20ft dry",
            "length_mm": 6058,
            "width_mm": 2438,
            "height_mm": 2591,
            "tare_weight_kg": 2300,
        }
    }

    matches = fuse(instances, barcodes, lcd_records, containers_db)

    assert len(matches) == 1
    assert matches[0].container_number == "MSCU1234567"
    assert matches[0].reference_number == "REF-00231"
    assert matches[0].confidence > 0.0

def test_decide_returns_leftmost_risky_instance(monkeypatch):
    monkeypatch.setattr(settings, "DECISION_STRATEGY", "leftmost")
    monkeypatch.setattr(settings, "DECISION_MIN_BOX_WIDTH", 10)
    monkeypatch.setattr(settings, "DECISION_MIN_BOX_HEIGHT", 10)
    monkeypatch.setattr(settings, "DECISION_REJECT_RISK_CONF", 0.5)

    instances = [
        SegmentationInstance(0, "container", 0.95, BBox(30, 40, 80, 60)),
        SegmentationInstance(1, "container", 0.95, BBox(200, 40, 80, 60)),
    ]
    matches = [
        ContainerMatch(0, "REF-1", "CONT-1", 0.9, {"barcode_match": 0.7}),
        ContainerMatch(1, "REF-2", "CONT-2", 0.8, {"barcode_match": 0.7}),
    ]
    risks = [RiskDetection("hazmat", 0.9, BBox(35, 45, 30, 20))]

    decisions = decide(instances, matches, risks)

    assert len(decisions) == 1
    assert decisions[0].matched_container is not None
    assert decisions[0].matched_container.container_number == "CONT-1"
    assert decisions[0].is_security_risk is True