"""Logging utility."""

import logging
import sys
from pathlib import Path

import colorlog


class ConsoleLogger:
    """Singleton logger that supports colored console output and optional file logging."""

    _instance = None  # Singleton

    def _init_singleton(self) -> None:
        self._logger = logging.getLogger("SpikeRL")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = colorlog.ColoredFormatter(
            "%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "green",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        self._file_path = None

    def __new__(cls) -> "ConsoleLogger":
        """Ensure that only one instance of ConsoleLogger exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_singleton()  # noqa: SLF001
        return cls._instance

    @property
    def logger(self) -> logging.Logger:
        """Returns the logging.Logger object for use in modules."""
        return self._logger

    def configure(self, level: int | str = logging.INFO, file_path: str | None = None) -> None:
        """Configure the logger.

        Args:
            level (int | str, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                Defaults to logging.INFO.
            file_path (str | None, optional): Path to a file to write logs.
                f None, logs only to console. Defaults to None.

        """
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

        if file_path and file_path != self._file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(file_path, encoding="utf-8")
            fh.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)
            self._file_path = file_path


# Global object for use in the project
console_logger = ConsoleLogger()
logger = console_logger.logger
