import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_whatsapp: str
    twilio_content_sid: str
    users_csv: str
    name_column: str
    phone_column: str
    delay_seconds: float
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send WhatsApp template messages in bulk using Twilio API."
    )
    parser.add_argument(
        "--users-csv",
        default="users.csv",
        help="Path to CSV file containing users (default: users.csv).",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Column name for user name (default: name).",
    )
    parser.add_argument(
        "--phone-column",
        default="phone",
        help="Column name for user phone (default: phone).",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.3,
        help="Delay between sends in seconds (default: 0.3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print requests without sending them.",
    )
    return parser.parse_args()


def require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(args: argparse.Namespace) -> Config:
    return Config(
        twilio_account_sid=require_env("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=require_env("TWILIO_AUTH_TOKEN"),
        twilio_from_whatsapp=require_env("TWILIO_WHATSAPP_FROM"),
        twilio_content_sid=require_env("TWILIO_CONTENT_SID"),
        users_csv=args.users_csv,
        name_column=args.name_column,
        phone_column=args.phone_column,
        delay_seconds=args.delay_seconds,
        dry_run=args.dry_run,
    )


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if not digits:
        return ""
    if phone.strip().startswith("+"):
        return f"+{digits}"
    return f"+{digits}"


def format_whatsapp_address(phone: str) -> str:
    return f"whatsapp:{phone}"


def send_template(cfg: Config, to_whatsapp: str, name: str) -> tuple[bool, str]:
    endpoint = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{cfg.twilio_account_sid}/Messages.json"
    )

    form_body = {
        "To": to_whatsapp,
        "From": cfg.twilio_from_whatsapp,
        "ContentSid": cfg.twilio_content_sid,
        # Twilio Content Templates use numeric keys for variables.
        # "1" maps to the first variable in the template body.
        "ContentVariables": json.dumps({"1": name}, ensure_ascii=True),
    }

    basic_auth = (
        f"{cfg.twilio_account_sid}:{cfg.twilio_auth_token}".encode("utf-8")
    )
    auth_header = "Basic " + base64.b64encode(basic_auth).decode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=urllib.parse.urlencode(form_body).encode("utf-8"),
        headers={
            "Authorization": auth_header,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
        return True, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return False, f"Network error: {exc.reason}"


def load_users(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def log_line(level: str, message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<5} | {message}")


def parse_json_safely(raw: str) -> dict:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return {"raw": raw}
    except json.JSONDecodeError:
        return {"raw": raw}


def summarize_twilio_success(raw: str) -> str:
    data = parse_json_safely(raw)
    sid = data.get("sid", "n/a")
    status = data.get("status", "n/a")
    to_value = data.get("to", "n/a")
    return f"sid={sid} status={status} to={to_value}"


def summarize_twilio_error(raw: str) -> str:
    data = parse_json_safely(raw)
    if "code" in data or "message" in data:
        code = data.get("code", "n/a")
        message = data.get("message", "Unknown error")
        more_info = data.get("more_info")
        if more_info:
            return f"code={code} message={message} more_info={more_info}"
        return f"code={code} message={message}"
    return raw


def print_summary(
    total: int,
    sent: int,
    failed: int,
    skipped: int,
    dry_run_count: int,
    dry_run: bool,
) -> None:
    print("\n" + "=" * 58)
    print("Bulk Send Summary")
    print("=" * 58)
    print(f"Total rows      : {total}")
    print(f"Sent            : {sent}")
    print(f"Failed          : {failed}")
    print(f"Skipped         : {skipped}")
    print(f"Dry-run previews: {dry_run_count}")
    print(f"Mode            : {'DRY-RUN' if dry_run else 'LIVE'}")
    print("=" * 58)


def run(cfg: Config) -> int:
    users = load_users(cfg.users_csv)
    if not users:
        log_line("ERROR", "No users found in CSV.")
        return 1

    sent = 0
    failed = 0
    skipped = 0
    dry_run_count = 0

    log_line(
        "INFO",
        (
            f"Starting bulk send: rows={len(users)} mode="
            f"{'DRY-RUN' if cfg.dry_run else 'LIVE'} "
            f"content_sid={cfg.twilio_content_sid}"
        ),
    )

    for index, row in enumerate(users, start=1):
        raw_name = (row.get(cfg.name_column) or "").strip()
        raw_phone = (row.get(cfg.phone_column) or "").strip()

        if not raw_name or not raw_phone:
            skipped += 1
            log_line(
                "WARN",
                (
                    f"[{index}/{len(users)}] SKIP missing required columns "
                    f"'{cfg.name_column}' or '{cfg.phone_column}'"
                ),
            )
            continue

        phone = normalize_phone(raw_phone)
        if not phone:
            skipped += 1
            log_line("WARN", f"[{index}/{len(users)}] SKIP invalid phone='{raw_phone}'")
            continue

        to_whatsapp = format_whatsapp_address(phone)

        if cfg.dry_run:
            dry_run_count += 1
            log_line(
                "INFO",
                (
                    f"[{index}/{len(users)}] PREVIEW to={to_whatsapp} "
                    f"nombre='{raw_name}' content_sid={cfg.twilio_content_sid}"
                ),
            )
            continue

        ok, result = send_template(cfg, to_whatsapp=to_whatsapp, name=raw_name)
        if ok:
            sent += 1
            log_line(
                "INFO",
                (
                    f"[{index}/{len(users)}] SENT nombre='{raw_name}' "
                    f"{summarize_twilio_success(result)}"
                ),
            )
        else:
            failed += 1
            log_line(
                "ERROR",
                (
                    f"[{index}/{len(users)}] FAIL nombre='{raw_name}' to={to_whatsapp} "
                    f"{summarize_twilio_error(result)}"
                ),
            )

        if cfg.delay_seconds > 0:
            time.sleep(cfg.delay_seconds)

    print_summary(
        total=len(users),
        sent=sent,
        failed=failed,
        skipped=skipped,
        dry_run_count=dry_run_count,
        dry_run=cfg.dry_run,
    )
    return 0 if failed == 0 else 2


def main() -> None:
    args = parse_args()
    load_dotenv_file(".env")
    try:
        cfg = load_config(args)
    except ValueError as exc:
        log_line("ERROR", str(exc))
        sys.exit(1)

    exit_code = run(cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
