"""
Decision algorithm.

Combines the fusion output (which container instance is probably which
declared reference) with the risk model output (what security/safety
issues were detected anywhere in the picture) to produce, per container
instance, a `Decision`:
    - selection_zone: the (x, y, width, height) zone to visually highlight /
      physically extract from the load,
    - weight: overall confidence in the decision,
    - is_security_risk: bool,
    - explanation: human-readable justification text,
    - matched_container / risk_detections: the evidence behind it.
"""

from __future__ import annotations

from config import settings
from srcs.utils.geometry import ContainerMatch, Decision, RiskDetection, SegmentationInstance
from srcs.utils.logger import get_logger

log = get_logger(__name__)

def _risks_overlapping(
    instance: SegmentationInstance, risks: list[RiskDetection]
) -> list[RiskDetection]:
    return [r for r in risks if instance.bbox.iou(r.bbox) > 0.0 or instance.bbox.contains_point(*r.bbox.center)]

def _build_explanation(
    instance: SegmentationInstance,
    match: ContainerMatch,
    overlapping_risks: list[RiskDetection],
) -> str:
    parts: list[str] = []

    if match.container_number:
        parts.append(
            f"Container identified as {match.container_number} "
            f"(reference {match.reference_number}, confidence {match.confidence:.0%})."
        )
        if "barcode_match" in match.evidence:
            parts.append("Identification confirmed by a barcode read inside this container's zone.")
        if match.evidence.get("other_references_in_container"):
            others = ", ".join(match.evidence["other_references_in_container"])
            parts.append(f"Note: this container also declares reference(s) {others}.")
    else:
        parts.append(
            "Could not confidently match this detected container to a declared "
            "LCD reference (no barcode read in this zone) — needs manual review."
        )

    if overlapping_risks:
        risk_desc = ", ".join(
            f"{r.label} ({r.confidence:.0%})" for r in overlapping_risks
        )
        parts.append(f"Security risk(s) detected in this zone: {risk_desc}.")
    else:
        parts.append("No security risk detected in this zone.")

    return " ".join(parts)

def decide(
    instances: list[SegmentationInstance],
    matches: list[ContainerMatch],
    risks: list[RiskDetection],
) -> list[Decision]:
    """
    Produce one Decision per detected container instance, ordered by the
    configured strategy. The initial test strategy selects the leftmost
    takeable instance only.

    Sorted by descending weight so the caller can easily pick the "most
    important" decision first (e.g. app/state_machine.py showing the
    single most urgent zone if only one can be validated at a time).
    """
    matches_by_instance = {m.instance_id: m for m in matches}
    decisions: list[Decision] = []

    ordered_instances = instances
    if settings.DECISION_STRATEGY == "leftmost":
        ordered_instances = sorted(instances, key=lambda item: item.bbox.x)

    for instance in ordered_instances:
        if settings.DECISION_STRATEGY == "leftmost" and (
            instance.bbox.width < settings.DECISION_MIN_BOX_WIDTH
            or instance.bbox.height < settings.DECISION_MIN_BOX_HEIGHT
        ):
            continue
        match = matches_by_instance.get(instance.instance_id)
        if match is None:
            log.warning("No fusion match found for instance %d, skipping", instance.instance_id)
            continue

        overlapping_risks = _risks_overlapping(instance, risks)
        max_risk_conf = max((r.confidence for r in overlapping_risks), default=0.0)
        is_security_risk = max_risk_conf >= settings.DECISION_REJECT_RISK_CONF

        # Weight combines "how sure are we this is a real risk / real
        # unmatched container" with "how sure are we about the container's
        # identity" — a security risk on a confidently-identified container
        # is the highest-priority, highest-weight decision.
        if is_security_risk:
            weight = max_risk_conf
        elif match.confidence < settings.DECISION_MIN_FUSION_CONF_FOR_AUTOMATCH:
            # Unmatched / low-confidence container: still worth flagging for
            # manual review, but with a capped, lower weight than a
            # confirmed security risk.
            weight = 0.3
        else:
            weight = 0.0  # confidently identified, no risk -> no action needed

        decision = Decision(
            selection_zone=instance.bbox,
            weight=round(weight, 3),
            is_security_risk=is_security_risk,
            explanation=_build_explanation(instance, match, overlapping_risks),
            matched_container=match,
            risk_detections=overlapping_risks,
        )
        decisions.append(decision)

        if settings.DECISION_STRATEGY == "leftmost":
            break

    if settings.DECISION_STRATEGY != "leftmost":
        decisions.sort(key=lambda d: d.weight, reverse=True)
    return decisions

def primary_decision(decisions: list[Decision]) -> Decision | None:
    """Convenience accessor: the single highest-weight decision, if any."""
    return decisions[0] if decisions else None