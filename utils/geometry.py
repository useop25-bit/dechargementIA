"""
Shared dataclasses used as the data "contract" between pipeline stages.

Keeping these in one place means models/*.py, pipeline/*.py and app/*.py
all agree on the exact shape of the data flowing between them, and every
stage can be unit-tested independently by constructing these objects by
hand (no need to run real inference in tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union with another bbox, in [0, 1]."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        union_area = self.area + other.area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class RiskDetection:
    """One detection from the risk model."""

    label: str
    confidence: float
    bbox: BBox


@dataclass
class SegmentationInstance:
    """One detected container instance from the segmentation model."""

    instance_id: int          # stable id within a single inference call
    class_name: str
    confidence: float
    bbox: BBox
    mask: "np.ndarray | None" = None  # binary mask, same size as image, optional in-memory only


@dataclass
class BarcodeDetection:
    """One barcode found in the image."""

    value: str
    symbology: str            # e.g. "CODE128", "QR", "EAN13", "OCR-ISO6346"
    confidence: float
    bbox: BBox


@dataclass
class ContainerMatch:
    """
    One candidate association between a detected SegmentationInstance and a
    declared LCD reference, produced by pipeline/fusion.py.
    """

    instance_id: int
    reference_number: Optional[str]
    container_number: Optional[str]
    confidence: float
    evidence: dict = field(default_factory=dict)  # human-readable breakdown,
                                                    # e.g. {"barcode_match": 0.7, "dimension_match": 0.1}


@dataclass
class Decision:
    """Final output of pipeline/decision.py."""

    selection_zone: BBox
    weight: float              # overall confidence of this decision, [0, 1]
    is_security_risk: bool
    explanation: str           # human-readable justification
    matched_container: Optional[ContainerMatch] = None
    risk_detections: list[RiskDetection] = field(default_factory=list)
