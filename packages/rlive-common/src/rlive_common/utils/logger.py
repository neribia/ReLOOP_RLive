import logging
from logging.config import dictConfig
from typing import Optional
from pathlib import Path

from rlive_common.config import config as cfg


CONFIGURED = False


# -------------------------------------------------------------
# Predefined format styles
# -------------------------------------------------------------
FORMAT_STYLES = {
    "simple": "[%(asctime)s] %(levelname)-8s - %(message)s",
    "detailed": "[%(asctime)s.%(msecs)03d] %(levelname)-8s - %(name)s.py:%(lineno)d - %(message)s",
    "color": "\033[1;36m%(asctime)s\033[0m \033[1;33m[%(levelname)-8s]\033[0m %(name)s: %(message)s",
    "thread": "%(asctime)s (%(threadName)-22s) %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    "json": (
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "msg": "%(message)s"}'
    ),
}

DATE_STYLES = {
    "short": "%H:%M:%S",
    "long": "%Y-%m-%d %H:%M:%S",
    "iso": "%Y-%m-%dT%H:%M:%S",
    "date_only": "%Y-%m-%d",
    "time_ms": "%H:%M:%S.%f",
}


# -------------------------------------------------------------
# Logger setup function
# -------------------------------------------------------------
def setup_logger(
    level: int = logging.DEBUG,
    formatter: str = "detailed",
    date_style: str = "long",
    stream: bool = True,
    logfile: Optional[str | Path] = None,
) -> None:

    if formatter not in FORMAT_STYLES:
        raise ValueError(f"Unknown formatter '{formatter}'. Choose from {list(FORMAT_STYLES.keys())}")
    if date_style not in DATE_STYLES:
        raise ValueError(f"Unknown date style '{date_style}'. Choose from {list(DATE_STYLES.keys())}")

    formatters = {
        # All available named formatters
        name: {"format": fmt, "datefmt": DATE_STYLES[date_style]}
        for name, fmt in FORMAT_STYLES.items()
    }

    # Default alias (points to selected style)
    formatters["default"] = {
        "format": FORMAT_STYLES[formatter],
        "datefmt": DATE_STYLES[date_style],
    }

    # --- Step 1: Build handlers dynamically ---
    handlers: dict[str, dict] = {}

    if stream:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": logging.getLevelName(level),
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }

    if logfile:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": logfile,
            "maxBytes": 5_000_000,  # 5 MB per log file
            "backupCount": 3,
            "encoding": "utf-8",
        }

    # --- Step 2: Compose the dictConfig ---
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "rlive_common": {
                "level": logging.getLevelName(level),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "rlive_world": {
                "level": logging.getLevelName(level),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "rlive_env": {
                "level": logging.getLevelName(level),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            # # Change the Unicorn logger
            # "uvicorn": {
            #     "level": "INFO",
            #     "handlers": list(handlers.keys()),
            # },
            # "uvicorn.error": {
            #     "level": "INFO",
            #     "handlers": list(handlers.keys()),
            #     "propagate": False,
            # },
            # "uvicorn.access": {
            #     "level": "INFO",
            #     "handlers": list(handlers.keys()),
            #     "propagate": False,
            # },
            # "httpx": {"level": "WARNING", "propagate": False},
            # "httpcore": {"level": "WARNING", "propagate": False},
            # "asyncio": {"level": "WARNING", "propagate": False},
            # "bleak": {"level": "WARNING", "propagate": False},
            # "uvicorn.error": {"level": "INFO", "propagate": False},
            # "uvicorn.access": {"level": "INFO", "propagate": False},
        },
        # Root logger
        "root": {
            "level": "WARNING",
            "handlers": list(handlers.keys()) or ["console"],
        },
    }

    # --- Step 3: Apply configuration ---
    dictConfig(config)
    logging.getLogger(__name__).debug(
        f"Logger initialized with formatter='{formatter}', date_style='{date_style}'"
    )
    global CONFIGURED
    CONFIGURED = True


# -------------------------------------------------------------
# Convenience accessor
# -------------------------------------------------------------
def get_logger(name: Optional[str] = None) -> logging.Logger:
    global CONFIGURED
    if not CONFIGURED:
        setup_logger(
            level=getattr(logging, cfg.LOGGING_LEVEL, logging.INFO),
            formatter = cfg.LOGGING_FORMAT,
            date_style = cfg.LOGGING_DATE,
            stream = cfg.LOGGING_STREAM,
            logfile = cfg.LOGGING_FILE,
        )
    return logging.getLogger(name or __name__)


# -------------------------------------------------------------
# Example usage / test run
# -------------------------------------------------------------
if __name__ == "__main__":
    #from rlive_common.utils import get_logger
    logger = get_logger(__name__)


    logger.debug("Debug message (only in file).")
    logger.info("Camera service initialized.")
    logger.warning("Low light detected.")
    logger.error("Camera disconnected.")
