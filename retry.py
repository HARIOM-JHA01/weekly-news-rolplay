import random
import signal
import time
from datetime import datetime


def log_line(level: str, message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<5} | {message}", flush=True)


def _make_timeout_handler(label: str):
    def handler(signum, frame):
        raise TimeoutError(f"[{label}] per-attempt timeout exceeded")
    return handler


def with_retry(
    fn, *args,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    label: str = "",
    per_attempt_timeout: int = 0,
    **kwargs,
):
    """Call fn(*args, **kwargs) with exponential backoff on any exception.

    per_attempt_timeout: if > 0, each attempt is killed after this many seconds
    via SIGALRM (Unix only). Prevents hanging on a single blocked call.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if per_attempt_timeout > 0:
                signal.signal(signal.SIGALRM, _make_timeout_handler(label))
                signal.alarm(per_attempt_timeout)
            result = fn(*args, **kwargs)
            if per_attempt_timeout > 0:
                signal.alarm(0)
            return result
        except Exception as exc:
            if per_attempt_timeout > 0:
                signal.alarm(0)
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
