"""
Fusion algorithm.

Goal: for every SegmentationInstance detected in the picture (a "blob" the
segmentation model thinks is a container), figure out — with a confidence
score — which declared LCD reference / container_number it actually is.

# TODO(fusion-strategy): this is the most important open design point of the
# whole project and the implementation below is a deliberately simple
# starting point (weighted-evidence scoring), not a final answer. You said
# you were considering something "very own coded, just with some logic" —
# that is exactly what this is. Alternatives worth considering once you have
# real data to test against:
#   - Dempster-Shafer evidence combination if you want formal uncertainty
#     handling across independent, possibly-conflicting evidence sources.
#   - A small Bayesian network (barcode_read -> container_id,
#     apparent_dimensions -> container_id, expected_count -> container_id).
#   - A learned re-ranker (small classifier) once you have logged enough
#     real (instance, reference) match/no-match examples.
# The weighted scoring approach below is a reasonable, explainable default:
# it's easy to debug, easy to tune the weights in config/settings.py, and
# the `evidence` dict on each ContainerMatch keeps the reasoning visible —
# which the project needs anyway for the decision explanation text.

Evidence signals implemented below:
  1. Barcode match (strongest): a decoded barcode whose bbox falls inside
     (or heavily overlaps) a segmentation instance's bbox, and whose value
     matches a known container_number in the LCD/containers_db.
  2. Dimension match (weak, supporting signal): compare the apparent
     width/height ratio of the segmented instance's bbox against the real
     container's length/width/height ratio from containers_db.json — a very
     rough sanity check, not a precise measurement (no camera calibration /
     distance estimation is implemented here).
  3. Count consistency (weak, global signal): if the number of segmented
     instances roughly matches the number of *distinct* container_numbers
     expected from the LCD, that's a small extra confidence boost for all
     matches (the scene "looks complete").

# TODO(no-barcode-case): when no barcode overlaps a given instance at all,
# this implementation currently returns confidence 0.0 / reference=None for
# that instance ("needs manual review" territory — see decision.py). If you
# want a smarter fallback (e.g. "assume left-to-right order matches LCD
# order", or use relative size ranking against containers_db), that logic
# belongs here — flagged for you to develop once you've seen real pictures.
"""

from __future__ import annotations

from collections import defaultdict

from config import settings
from utils.geometry import BarcodeDetection, ContainerMatch, SegmentationInstance
from utils.logger import get_logger

log = get_logger(__name__)


def _find_overlapping_barcode(
    instance: SegmentationInstance, barcodes: list[BarcodeDetection]
) -> BarcodeDetection | None:
    """Return the barcode whose bbox overlaps this instance's bbox the most, if any."""
    best_barcode: BarcodeDetection | None = None
    best_iou = 0.0

    for barcode in barcodes:
        # A barcode is usually small relative to the container, so IoU alone
        # can be misleading — also accept "barcode center is inside instance
        # bbox" as a strong containment signal.
        center_x, center_y = barcode.bbox.center
        contained = instance.bbox.contains_point(center_x, center_y)
        iou = instance.bbox.iou(barcode.bbox)

        if contained and iou >= best_iou:
            best_iou = iou
            best_barcode = barcode

    return best_barcode


def _dimension_match_score(
    instance: SegmentationInstance, container_number: str, containers_db: dict
) -> float:
    """
    Very rough plausibility check: does the aspect ratio (width/height) of
    the segmented bbox roughly match the real container's length/height
    ratio? Returns a score in [0, 1].

    # TODO(calibration): without camera calibration / distance estimation
    # this can only ever be a weak sanity signal, not a real measurement.
    # If you add camera calibration later, this function is the place to
    # plug in a proper size-consistency estimate.
    """
    dims = containers_db.get(container_number)
    if not dims:
        return 0.0

    real_ratio = dims["length_mm"] / dims["height_mm"]
    if instance.bbox.height == 0:
        return 0.0
    observed_ratio = instance.bbox.width / instance.bbox.height

    # Score decays smoothly the further apart the ratios are.
    diff = abs(real_ratio - observed_ratio) / real_ratio
    return max(0.0, 1.0 - diff)


def fuse(
    instances: list[SegmentationInstance],
    barcodes: list[BarcodeDetection],
    lcd_records: list[dict],
    containers_db: dict,
) -> list[ContainerMatch]:
    """
    Main fusion entry point.

    Args:
        instances: output of models.segmentation_model.SegmentationModel.predict
        barcodes: output of models.barcode_model.BarcodeModel.predict
        lcd_records: output of pipeline.lcd_formatter.load_lcd_json
        containers_db: parsed content of data/containers_db.json

    Returns:
        One ContainerMatch per segmentation instance (best candidate found),
        in the same order as `instances`.
    """
    # Precompute: container_number -> list of LCD records (a container can
    # hold several references).
    records_by_container: dict[str, list[dict]] = defaultdict(list)
    for rec in lcd_records:
        records_by_container[rec["container_number"]].append(rec)

    expected_container_count = len(records_by_container)
    detected_count = len(instances)
    count_consistency = 1.0 - min(
        1.0, abs(expected_container_count - detected_count) / max(1, expected_container_count)
    )

    matches: list[ContainerMatch] = []

    for instance in instances:
        evidence: dict[str, float] = {}
        confidence = 0.0
        container_number: str | None = None
        reference_number: str | None = None

        barcode = _find_overlapping_barcode(instance, barcodes)
        if barcode is not None and barcode.value in records_by_container:
            container_number = barcode.value
            evidence["barcode_match"] = settings.FUSION_BARCODE_MATCH_WEIGHT
            confidence += settings.FUSION_BARCODE_MATCH_WEIGHT

            dim_score = _dimension_match_score(instance, container_number, containers_db)
            evidence["dimension_match"] = round(
                dim_score * settings.FUSION_DIMENSION_MATCH_WEIGHT, 3
            )
            confidence += evidence["dimension_match"]

            evidence["count_consistency"] = round(
                count_consistency * settings.FUSION_COUNT_CONSISTENCY_WEIGHT, 3
            )
            confidence += evidence["count_consistency"]

            # A container can carry several references — without further
            # disambiguation, pick the first declared one and surface the
            # rest in evidence for pipeline/decision.py to mention if useful.
            candidate_records = records_by_container[container_number]
            reference_number = candidate_records[0]["reference_number"]
            if len(candidate_records) > 1:
                evidence["other_references_in_container"] = [
                    r["reference_number"] for r in candidate_records[1:]
                ]
        else:
            log.info(
                "Instance %d (%s): no matching barcode found -> unresolved, "
                "needs manual review",
                instance.instance_id,
                instance.class_name,
            )

        matches.append(
            ContainerMatch(
                instance_id=instance.instance_id,
                reference_number=reference_number,
                container_number=container_number,
                confidence=round(min(1.0, confidence), 3),
                evidence=evidence,
            )
        )

    return matches
