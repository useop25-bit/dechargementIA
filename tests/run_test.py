"""Run one detailed, non-interactive inspection pipeline test.

Mode 1 captures a picture from the camera. Mode 0 reads a picture from an
image file or from an image folder. Both modes then execute the same pipeline
and write their complete output under ``data/logs/run_test/<run-id>/``.

Examples:
    python3 tests/run_test.py --mode 1
    python3 tests/run_test.py --mode 0 --image data/captures/example.jpg
    python3 tests/run_test.py --mode 0 --image-dir data/captures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRUE = "1"
FALSE = "0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _flag(value: bool) -> str:
    return TRUE if value else FALSE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one detailed container-inspection pipeline test"
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=(0, 1),
        default=1,
        help="1=camera capture, 0=read an image from --image or --image-dir",
    )
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for mode 0.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "captures",
        help="Folder searched recursively for an image in mode 0.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Override the segmentation model weight path.",
    )
    parser.add_argument("--enable-risk", action="store_true")
    parser.add_argument("--enable-barcode", action="store_true")
    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument("--disable-fusion", action="store_true")
    parser.add_argument("--disable-decision", action="store_true")
    parser.add_argument(
        "--decision-strategy",
        choices=("leftmost", "all"),
        default="all",
        help="Process all instances by default for diagnostics.",
    )
    return parser


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["RUN_MODE"] = str(args.mode)
    os.environ["MOCK_HARDWARE"] = _flag(args.mode == 1)
    os.environ["ENABLE_YOLO"] = _flag(not args.disable_yolo)
    os.environ["ENABLE_RISK_MODEL"] = _flag(args.enable_risk)
    os.environ["ENABLE_BARCODE_MODEL"] = _flag(args.enable_barcode)
    os.environ["ENABLE_FUSION"] = _flag(not args.disable_fusion)
    os.environ["ENABLE_DECISION"] = _flag(not args.disable_decision)
    os.environ["DECISION_STRATEGY"] = args.decision_strategy
    if args.camera_index is not None:
        os.environ["CAMERA_INDEX"] = str(args.camera_index)
    if args.weights is not None:
        os.environ["SEGMENTATION_MODEL_WEIGHTS"] = str(args.weights)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_tuple"):
        return value.as_tuple()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        data = {
            key: item
            for key, item in vars(value).items()
            if key != "mask"
        }
        return _json_safe(data)
    if hasattr(value, "item"):
        return value.item()
    return value


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def _image_candidates(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _create_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"run_test.{run_dir.name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "run_test.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _load_input_image(args: argparse.Namespace, run_dir: Path, logger: logging.Logger):
    import cv2

    start = time.perf_counter()
    if args.mode == 1:
        from config import settings
        from config.hardware.camera import Camera

        camera_index = args.camera_index if args.camera_index is not None else settings.CAMERA_INDEX
        logger.info("Opening camera index %s", camera_index)
        camera = Camera(index=camera_index)
        try:
            camera.open()
            frame = camera.capture()
        finally:
            camera.close()
        source = f"camera:{camera_index}"
    else:
        if args.image is not None:
            source_path = args.image
        else:
            candidates = _image_candidates(args.image_dir)
            if not candidates:
                raise FileNotFoundError(
                    f"No image found in {args.image_dir}. Use --image or add an image to the folder."
                )
            source_path = candidates[0]
        logger.info("Reading image from %s", source_path)
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise ValueError(f"OpenCV could not read image: {source_path}")
        source = str(source_path)

    if frame is None or frame.size == 0:
        raise RuntimeError(f"Input source returned an empty image: {source}")

    raw_path = run_dir / "input.jpg"
    if not cv2.imwrite(str(raw_path), frame):
        raise RuntimeError(f"Could not write input image to {raw_path}")
    logger.info(
        "Input ready: source=%s size=%sx%s channels=%s elapsed=%.4fs",
        source,
        frame.shape[1],
        frame.shape[0],
        frame.shape[2] if len(frame.shape) > 2 else 1,
        _elapsed(start),
    )
    return frame, {"source": source, "saved_path": str(raw_path), "elapsed_seconds": _elapsed(start)}


def _load_data(logger: logging.Logger):
    from config import settings
    from srcs.utils.lcd_formatter import lcd_records_for_fusion, load_lcd_document

    lcd_document, lcd_source = load_lcd_document(
        settings.CURRENT_LCD_PDF_PATH, settings.EXAMPLE_LCD_JSON_PATH
    )
    lcd_records = lcd_records_for_fusion(lcd_document)
    containers_db = json.loads(settings.CONTAINERS_DB_PATH.read_text(encoding="utf-8"))
    logger.info("LCD loaded from %s: %d records", lcd_source, len(lcd_records))
    logger.info("Container database loaded: %d entries", len(containers_db))
    return lcd_records, containers_db, {
        "lcd_source": str(lcd_source),
        "lcd_records": len(lcd_records),
        "containers_db_records": len(containers_db),
    }


def run(args: argparse.Namespace) -> int:
    configure_environment(args)

    from config import settings
    from srcs.models.risk_model import RiskModel
    from srcs.models.segmentation_model import SegmentationModel
    from srcs.pipeline import decision as decision_pipeline
    from srcs.utils import draw as draw_pipeline
    from srcs.pipeline import fusion as fusion_pipeline

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = settings.LOG_DIR / "run_test" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_run_logger(run_dir)
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "mode_description": "camera" if args.mode == 1 else "image_folder",
        "status": "started",
        "steps": [],
        "risk_detection": {"enabled": args.enable_risk, "executed": False},
    }

    def step(name: str, **details: Any) -> None:
        report["steps"].append({
            "name": name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "details": _json_safe(details),
        })
        logger.info("STEP %s | %s", name, details)

    try:
        step(
            "configuration",
            project_root=PROJECT_ROOT,
            mode=args.mode,
            camera_index=settings.CAMERA_INDEX,
            segmentation_enabled=settings.ENABLE_YOLO,
            segmentation_weights=settings.SEGMENTATION_MODEL_WEIGHTS,
            segmentation_weights_exists=Path(settings.SEGMENTATION_MODEL_WEIGHTS).exists(),
            risk_enabled=settings.ENABLE_RISK_MODEL,
            barcode_enabled=settings.ENABLE_BARCODE_MODEL,
            fusion_enabled=settings.ENABLE_FUSION,
            decision_enabled=settings.ENABLE_DECISION,
            decision_strategy=settings.DECISION_STRATEGY,
            python=sys.version.replace("\n", " "),
            platform=platform.platform(),
        )

        lcd_records, containers_db, data_details = _load_data(logger)
        step("data_loaded", **data_details)

        frame, input_details = _load_input_image(args, run_dir, logger)
        step("input_loaded", **input_details)

        model_start = time.perf_counter()
        segmentation_model = SegmentationModel()
        step(
            "segmentation_model_loaded",
            elapsed_seconds=_elapsed(model_start),
            weights=settings.SEGMENTATION_MODEL_WEIGHTS,
        )

        prediction_start = time.perf_counter()
        instances = segmentation_model.predict(frame)
        step(
            "segmentation_completed",
            elapsed_seconds=_elapsed(prediction_start),
            count=len(instances),
            instances=[
                {
                    "instance_id": item.instance_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "bbox": item.bbox.as_tuple(),
                    "mask_present": item.mask is not None,
                    "mask_shape": list(item.mask.shape) if item.mask is not None else None,
                }
                for item in instances
            ],
        )

        risks = []
        if settings.ENABLE_RISK_MODEL:
            risk_model = RiskModel()
            risks = risk_model.predict(frame)
            report["risk_detection"]["executed"] = True
            step("risk_detection_completed", count=len(risks), detections=_json_safe(risks))
        else:
            step("risk_detection_skipped", reason="disabled for this run", count=0)

        barcodes = []
        if settings.ENABLE_BARCODE_MODEL:
            barcode_model = BarcodeModel()
            barcodes = barcode_model.predict(frame)
            step("barcode_detection_completed", count=len(barcodes), detections=_json_safe(barcodes))
        else:
            step("barcode_detection_skipped", reason="disabled for this run", count=0)

        if settings.ENABLE_FUSION:
            fusion_start = time.perf_counter()
            matches = fusion_pipeline.fuse(instances, barcodes, lcd_records, containers_db)
            step("fusion_completed", elapsed_seconds=_elapsed(fusion_start), matches=_json_safe(matches))
        else:
            matches = []
            step("fusion_skipped", reason="disabled for this run")

        if settings.ENABLE_DECISION:
            decision_start = time.perf_counter()
            decisions = decision_pipeline.decide(instances, matches, risks)
            step(
                "decision_completed",
                elapsed_seconds=_elapsed(decision_start),
                count=len(decisions),
                decisions=_json_safe(decisions),
            )
        else:
            decisions = []
            step("decision_skipped", reason="disabled for this run")

        draw_start = time.perf_counter()
        annotated = draw_pipeline.draw_decisions(frame, decisions)
        annotated_path = run_dir / "annotated.jpg"
        import cv2
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Could not write annotated image to {annotated_path}")
        step(
            "output_written",
            elapsed_seconds=_elapsed(draw_start),
            annotated_path=annotated_path,
            annotated_size=[int(annotated.shape[1]), int(annotated.shape[0])],
        )

        report["status"] = "success"
        report["summary"] = {
            "instances": len(instances),
            "matches": len(matches),
            "risks": len(risks),
            "barcodes": len(barcodes),
            "decisions": len(decisions),
            "input_image": str(run_dir / "input.jpg"),
            "annotated_image": str(annotated_path),
        }
        logger.info("RUN COMPLETED SUCCESSFULLY")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        logger.exception("RUN FAILED")
        return 1
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["output_directory"] = str(run_dir)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Report written to %s", run_dir / "report.json")
        for handler in logger.handlers:
            handler.flush()


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
