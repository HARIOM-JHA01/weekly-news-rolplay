import argparse
import base64
import csv
import json
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from news_fetcher import NewsItem, fetch_ai_news
from blog_generator import generate_blog_post, generate_cover_image
from blog_api import upload_image, create_blog_post
from retry import log_line, with_retry


BLOG_BASE_URL = "https://blog.rolplay.ai"


@dataclass
class Config:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_whatsapp: str
    twilio_content_sid: str
    gemini_api_key: str
    blog_api_key: str
    users_csv: str
    name_column: str
    phone_column: str
    delay_seconds: float
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AI news, publish blog post, send WhatsApp summary to users."
    )
    parser.add_argument("--users-csv", default="users.csv")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--phone-column", default="phone")
    parser.add_argument(
        "--delay-seconds", type=float, default=0.3,
        help="Delay between WhatsApp sends in seconds (default: 0.3).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print requests without sending WhatsApp messages.",
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
        gemini_api_key=require_env("GEMINI_API_KEY"),
        blog_api_key=require_env("BLOG_API_KEY"),
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
    return f"+{digits}"


def format_whatsapp_address(phone: str) -> str:
    return f"whatsapp:{phone}"


def send_template(
    cfg: Config,
    to_whatsapp: str,
    variables: dict[str, str],
) -> tuple[bool, str]:
    endpoint = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{cfg.twilio_account_sid}/Messages.json"
    )
    form_body = {
        "To": to_whatsapp,
        "From": cfg.twilio_from_whatsapp,
        "ContentSid": cfg.twilio_content_sid,
        "ContentVariables": json.dumps(variables, ensure_ascii=True),
    }
    basic_auth = f"{cfg.twilio_account_sid}:{cfg.twilio_auth_token}".encode("utf-8")
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


def build_template_variables(name: str, news_items: list[NewsItem], slug: str) -> dict[str, str]:
    return {
        "1": name,
        "2": news_items[0].headline,
        "3": news_items[1].headline,
        "4": news_items[2].headline,
        "5": news_items[3].headline,
        "6": news_items[4].headline,
        "7": slug,
    }


def load_users(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_json_safely(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"raw": raw}
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


def print_summary(total: int, sent: int, failed: int, skipped: int,
                  dry_run_count: int, dry_run: bool) -> None:
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
    # --- Step 1: Fetch top 5 AI news ---
    log_line("INFO", "Fetching top 5 AI news stories via Gemini + Google Search...")
    try:
        news_items = fetch_ai_news(cfg.gemini_api_key)
    except Exception as exc:
        log_line("ERROR", f"Failed to fetch news: {exc}")
        return 1

    log_line("INFO", "Top 5 AI news this week:")
    for i, item in enumerate(news_items, start=1):
        log_line("INFO", f"  {i}. {item.headline}")

    # --- Step 2: Generate blog post content ---
    log_line("INFO", "Generating blog post content via Gemini...")
    try:
        blog = generate_blog_post(cfg.gemini_api_key, news_items)
    except Exception as exc:
        log_line("ERROR", f"Failed to generate blog post: {exc}")
        return 1
    log_line("INFO", f"Blog title: {blog.title}")

    # --- Step 3: Generate cover image ---
    log_line("INFO", "Generating cover image via Imagen...")
    try:
        image_bytes = generate_cover_image(cfg.gemini_api_key, blog.title)
    except Exception as exc:
        log_line("ERROR", f"Failed to generate cover image: {exc}")
        return 1
    log_line("INFO", f"Cover image generated ({len(image_bytes):,} bytes)")

    if cfg.dry_run:
        log_line("INFO", "[DRY-RUN] Skipping image upload and blog creation.")
        slug = "dry-run-slug"
        log_line("INFO", f"[DRY-RUN] Would publish: {BLOG_BASE_URL}/news/{slug}")
    else:
        # --- Step 4: Upload cover image ---
        log_line("INFO", "Uploading cover image...")
        try:
            cover_url = upload_image(BLOG_BASE_URL, cfg.blog_api_key, image_bytes)
        except Exception as exc:
            log_line("ERROR", f"Failed to upload image: {exc}")
            return 1
        log_line("INFO", f"Cover image URL: {cover_url}")

        # --- Step 5: Create blog post ---
        log_line("INFO", "Publishing blog post...")
        try:
            slug = create_blog_post(
                BLOG_BASE_URL,
                cfg.blog_api_key,
                title=blog.title,
                summary=blog.summary,
                content=blog.content,
                cover_image_url=cover_url,
                tags=blog.tags,
            )
        except Exception as exc:
            log_line("ERROR", f"Failed to create blog post: {exc}")
            return 1
        log_line("INFO", f"Blog published: {BLOG_BASE_URL}/news/{slug}")

    # --- Step 6: Send WhatsApp messages ---
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
        f"Sending WhatsApp messages: users={len(users)} mode={'DRY-RUN' if cfg.dry_run else 'LIVE'}",
    )

    for index, row in enumerate(users, start=1):
        raw_name = (row.get(cfg.name_column) or "").strip()
        raw_phone = (row.get(cfg.phone_column) or "").strip()

        if not raw_name or not raw_phone:
            skipped += 1
            log_line("WARN", f"[{index}/{len(users)}] SKIP missing name or phone")
            continue

        phone = normalize_phone(raw_phone)
        if not phone:
            skipped += 1
            log_line("WARN", f"[{index}/{len(users)}] SKIP invalid phone='{raw_phone}'")
            continue

        to_whatsapp = format_whatsapp_address(phone)
        variables = build_template_variables(raw_name, news_items, slug)

        if cfg.dry_run:
            dry_run_count += 1
            log_line(
                "INFO",
                f"[{index}/{len(users)}] PREVIEW to={to_whatsapp} name='{raw_name}' "
                f"variables={json.dumps(variables)}",
            )
            continue

        ok, result = with_retry(send_template, cfg, to_whatsapp=to_whatsapp, variables=variables, label=f"send_whatsapp/{raw_name}")
        if ok:
            sent += 1
            log_line(
                "INFO",
                f"[{index}/{len(users)}] SENT name='{raw_name}' {summarize_twilio_success(result)}",
            )
        else:
            failed += 1
            log_line(
                "ERROR",
                f"[{index}/{len(users)}] FAIL name='{raw_name}' to={to_whatsapp} "
                f"{summarize_twilio_error(result)}",
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


GLOBAL_TIMEOUT_SECONDS = 30 * 60  # 30-minute hard ceiling for the entire script


def _global_timeout_handler(signum, frame):
    log_line("ERROR", f"GLOBAL TIMEOUT: script exceeded {GLOBAL_TIMEOUT_SECONDS}s — forcing exit")
    sys.exit(3)


def main() -> None:
    signal.signal(signal.SIGALRM, _global_timeout_handler)
    signal.alarm(GLOBAL_TIMEOUT_SECONDS)

    args = parse_args()
    load_dotenv_file(".env")
    try:
        cfg = load_config(args)
    except ValueError as exc:
        log_line("ERROR", str(exc))
        sys.exit(1)

    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
