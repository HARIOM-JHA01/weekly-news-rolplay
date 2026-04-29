import random
import time
from datetime import datetime


def log_line(level: str, message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<5} | {message}")


def with_retry(fn, *args, max_attempts: int = 5, base_delay: float = 1.0, label: str = "", **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on any exception."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            tag = f"[{label}] " if label else ""
            log_line(
                "WARN",
                f"{tag}attempt {attempt + 1}/{max_attempts} failed — retrying in {delay:.1f}s: {exc}",
            )
            time.sleep(delay)
    raise last_exc
