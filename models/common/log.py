"""
Logging utilities for AirGesture project.
"""

from datetime import datetime


def log_info(msg: str) -> None:
    """Print INFO message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] INFO - {msg}")


def log_warn(msg: str) -> None:
    """Print WARNING message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] WARNING - {msg}")


def log_err(msg: str) -> None:
    """Print ERROR message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ERROR - {msg}")
