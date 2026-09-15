"""
The 4-state workflow described in README.md §4.1:

    IDLE -> (capture) -> PICTURE_TAKEN -> (validate) -> ANALYZING
      ^                        |  ^                         |
      |                   (capture, retake)                 |
      |                                                      v
      +---------------- (reject) ---------------------- RESULT_SHOWN
                                                              |
                                                         (validate/accept)
                                                              |
                                                              v
                                                            IDLE (+ saved)

This module holds no hardware/display/model instances itself — they're
injected (constructor params) so the state machine can be unit-tested with
fakes/mocks, independently of GPIO, a real camera, or real model weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from config import settings
from hardware.camera import Camera
from hardware.display import Display
from models.barcode_model import BarcodeModel
from models.risk_model import RiskModel
from models.segmentation_model import SegmentationModel
from pipeline import decision as decision_pipeline
from pipeline import draw as draw_pipeline
from pipeline import fusion as fusion_pipeline
from utils.geometry import Decision
from utils.logger import get_logger

log = get_logger(__name__)


class State(Enum):
    IDLE = auto()
    PICTURE_TAKEN = auto()
    ANALYZING = auto()
    RESULT_SHOWN = auto()


@dataclass
class SessionData:
    raw_frame: np.ndarray | None = None
    decisions: list[Decision] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None


class StateMachine:
    def __init__(
        self,
        camera: Camera,
        display: Display,
        risk_model: RiskModel,
        segmentation_model: SegmentationModel,
        barcode_model: BarcodeModel,
        lcd_records: list[dict],
        containers_db: dict,
    ) -> None:
        self.camera = camera
        self.display = display
        self.risk_model = risk_model
        self.segmentation_model = segmentation_model
        self.barcode_model = barcode_model
        self.lcd_records = lcd_records
        self.containers_db = containers_db

        self.state = State.IDLE
        self.session = SessionData()

        self.display.show_message("Ready — press CAPTURE")

    # ------------------------------------------------------------------
    # Button-triggered transitions
    # ------------------------------------------------------------------

    def on_capture(self) -> None:
        if self.state not in (State.IDLE, State.PICTURE_TAKEN):
            log.debug("CAPTURE ignored in state %s", self.state)
            return

        try:
            frame = self.camera.capture()
        except Exception:
            log.exception("Capture failed")
            self.display.show_message("Camera error — check connection")
            return

        self.session = SessionData(raw_frame=frame)
        self.state = State.PICTURE_TAKEN
        self.display.show(frame)
        log.info("Picture taken, waiting for VALIDATE to analyze (or CAPTURE to retake)")

    def on_validate(self) -> None:
        if self.state == State.PICTURE_TAKEN:
            self._run_analysis()
        elif self.state == State.RESULT_SHOWN:
            self._accept_result()
        else:
            log.debug("VALIDATE ignored in state %s", self.state)

    def on_validate_long(self) -> None:
        """Long-press on VALIDATE == reject, only meaningful in RESULT_SHOWN."""
        if self.state == State.RESULT_SHOWN:
            self._reject_result()
        else:
            log.debug("Long VALIDATE ignored in state %s", self.state)

    # ------------------------------------------------------------------
    # Internal transitions
    # ------------------------------------------------------------------

    def _run_analysis(self) -> None:
        assert self.session.raw_frame is not None
        self.state = State.ANALYZING
        self.display.show_message("Analyzing...")
        log.info("Running analysis pipeline")

        frame = self.session.raw_frame

        risks = self.risk_model.predict(frame)
        instances = self.segmentation_model.predict(frame)
        barcodes = self.barcode_model.predict(frame)

        matches = fusion_pipeline.fuse(
            instances=instances,
            barcodes=barcodes,
            lcd_records=self.lcd_records,
            containers_db=self.containers_db,
        )

        decisions = decision_pipeline.decide(
            instances=instances, matches=matches, risks=risks
        )

        annotated = draw_pipeline.draw_decisions(frame, decisions)

        self.session.decisions = decisions
        self.session.annotated_frame = annotated
        self.state = State.RESULT_SHOWN
        self.display.show(annotated)
        log.info("Analysis complete: %d decision(s)", len(decisions))

    def _accept_result(self) -> None:
        log.info("Result ACCEPTED by operator")
        self._save_session(accepted=True)
        self._reset_to_idle()

    def _reject_result(self) -> None:
        log.info("Result REJECTED by operator")
        self._save_session(accepted=False)
        self._reset_to_idle()

    def _reset_to_idle(self) -> None:
        self.session = SessionData()
        self.state = State.IDLE
        self.display.show_message("Ready — press CAPTURE")

    def _save_session(self, accepted: bool) -> None:
        """
        Persist the raw picture, annotated picture, and a JSON summary of
        the decisions + operator verdict, for audit/traceability.

        # TODO(storage): local disk storage under data/captures/ is a
        # reasonable default for a standalone device; if results need to be
        # centralized (e.g. uploaded to a server/DB for audit), add that
        # here — this is the single choke point where every finalized
        # session passes through.
        """
        settings.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        session_dir = settings.CAPTURES_DIR / ts
        session_dir.mkdir(parents=True, exist_ok=True)

        if self.session.raw_frame is not None:
            cv2.imwrite(str(session_dir / "raw.jpg"), self.session.raw_frame)
        if self.session.annotated_frame is not None:
            cv2.imwrite(str(session_dir / "annotated.jpg"), self.session.annotated_frame)

        summary = {
            "timestamp": ts,
            "accepted": accepted,
            "decisions": [
                {
                    "selection_zone": d.selection_zone.as_tuple(),
                    "weight": d.weight,
                    "is_security_risk": d.is_security_risk,
                    "explanation": d.explanation,
                    "container_number": d.matched_container.container_number if d.matched_container else None,
                    "reference_number": d.matched_container.reference_number if d.matched_container else None,
                }
                for d in self.session.decisions
            ],
        }
        with (session_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log.info("Session saved to %s", session_dir)
