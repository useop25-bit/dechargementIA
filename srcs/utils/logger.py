"""Project-wide logger with rotation, retention, and session trace support."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


def _prune_old_logs(log_dir: Path, retention_days: int) -> None:
    if not log_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=retention_days)
    for log_file in log_dir.glob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff.timestamp():
                log_file.unlink(missing_ok=True)
        except OSError:
            continue


def _ensure_log_dir() -> Path:
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(settings.LOG_DIR, settings.LOG_RETENTION_DAYS)
    return settings.LOG_DIR


def log_session_event(event: str, **payload: object) -> None:
    """Append a structured JSON event to the session audit log."""
    try:
        _ensure_log_dir()
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            "session_id": payload.pop("session_id", uuid.uuid4().hex),
            "payload": payload,
        }
        with settings.SESSION_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        return


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_dir = _ensure_log_dir()
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file, console-only logging.")

    return logger