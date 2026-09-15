"""
Draws the decision(s) onto a copy of the original picture:
  - the selection zone (the container to extract), highlighted,
  - risk detections, boxed and labeled,
  - a text panel with the explanation.

Purely mechanical (no decision-making here) — safe to reuse as-is.
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from utils.geometry import Decision

# BGR colors
COLOR_RISK = (0, 0, 255)          # red
COLOR_SELECTION_RISK = (0, 0, 255)     # red — container selected because of a risk
COLOR_SELECTION_REVIEW = (0, 165, 255)  # orange — needs manual review
COLOR_SELECTION_OK = (0, 200, 0)        # green — identified, no risk
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT_FG = (255, 255, 255)


def _selection_color(decision: Decision) -> tuple[int, int, int]:
    if decision.is_security_risk:
        return COLOR_SELECTION_RISK
    if decision.matched_container is None or decision.weight >= 0.3:
        return COLOR_SELECTION_REVIEW
    return COLOR_SELECTION_OK


def draw_decisions(image: np.ndarray, decisions: list[Decision], max_text_width_chars: int = 60) -> np.ndarray:
    """
    Returns a new annotated image (original `image` is left untouched).

    Draws every decision's selection zone + risk boxes, and appends a text
    panel below the image summarizing each decision's explanation.
    """
    annotated = image.copy()

    for decision in decisions:
        color = _selection_color(decision)
        bbox = decision.selection_zone
        cv2.rectangle(annotated, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 3)

        label = decision.matched_container.container_number if decision.matched_container and decision.matched_container.container_number else "UNIDENTIFIED"
        cv2.putText(
            annotated, label, (bbox.x, max(0, bbox.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )

        for risk in decision.risk_detections:
            rb = risk.bbox
            cv2.rectangle(annotated, (rb.x, rb.y), (rb.x2, rb.y2), COLOR_RISK, 2)
            cv2.putText(
                annotated, f"{risk.label} {risk.confidence:.0%}", (rb.x, max(0, rb.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RISK, 1, cv2.LINE_AA,
            )

    text_panel = _build_text_panel(decisions, width=annotated.shape[1], max_chars=max_text_width_chars)
    return np.vstack([annotated, text_panel])


def _build_text_panel(decisions: list[Decision], width: int, max_chars: int) -> np.ndarray:
    lines: list[str] = []
    if not decisions:
        lines.append("No container detected in this picture.")
    for i, decision in enumerate(decisions, start=1):
        header = f"[{i}] weight={decision.weight:.2f} risk={'YES' if decision.is_security_risk else 'no'}"
        lines.append(header)
        lines.extend(textwrap.wrap(decision.explanation, width=max_chars))
        lines.append("")

    line_height = 24
    padding = 12
    panel_height = padding * 2 + line_height * max(1, len(lines))
    panel = np.full((panel_height, width, 3), COLOR_TEXT_BG, dtype=np.uint8)

    y = padding + line_height // 2
    for line in lines:
        cv2.putText(
            panel, line, (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_FG, 1, cv2.LINE_AA,
        )
        y += line_height

    return panel
