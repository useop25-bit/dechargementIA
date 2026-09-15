"""Simple project-wide logger, writes to console + rotating file."""

import logging
from logging.handlers import RotatingFileHandler

from config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # already configured (avoid duplicate handlers on repeated calls)
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # e.g. read-only filesystem — fall back to console-only logging
        logger.warning("Could not create log file, console-only logging.")

    return logger
