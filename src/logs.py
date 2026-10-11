from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def log_path(name: str) -> Path:
    """Map a typed name or path to a file under ./logs (adds .csv if no extension)."""
    path = (LOG_DIR / name.strip().lstrip("/\\")).resolve()
    path.relative_to(LOG_DIR)  # raises ValueError (and exits) if it escapes ./logs
    return path.with_suffix(path.suffix or ".csv")


def ask_log_path(default: str) -> Path:
    """Prompt for a log file name; blank input falls back to `default`."""
    return log_path(input(f"Log file name [{default}]: ").strip() or default)
