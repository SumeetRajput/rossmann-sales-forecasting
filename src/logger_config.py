"""
Logger Configuration — Required by Task 1.2 of the brief

Provides a centralised logger that writes structured logs to both
file (timestamped) and console (clean output).

Usage:
    from src.logger_config import get_logger
    logger = get_logger("my_module")
    logger.info("Hello")
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name="rossmann", log_dir="outputs/logs", level=logging.INFO):
    """
    Create a logger that writes to both a timestamped file and the console.

    Args:
        name:    Logger name (typically the module name)
        log_dir: Directory to write log files to
        level:   Logging level (INFO, DEBUG, WARNING, ERROR)

    Returns:
        Configured logger instance ready to use
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when called repeatedly
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── File handler: detailed structured logs ──
    log_filename = datetime.now().strftime("rossmann_%Y-%m-%d_%H-%M-%S") + ".log"
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_filename), encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # ── Console handler: clean output ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger
