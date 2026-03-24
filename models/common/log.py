from datetime import datetime


def log_info(msg):
    """Print INFO message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] INFO - {msg}")


def log_warn(msg):
    """Print WARNING message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] WARNING - {msg}")


def log_err(msg):
    """Print ERROR message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ERROR - {msg}")