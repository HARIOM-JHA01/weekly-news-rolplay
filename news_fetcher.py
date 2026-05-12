import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from google import genai
from google.genai import types

from retry import with_retry


@dataclass
class NewsItem:
    headline: str  # ≤10 words, used in WhatsApp template
    summary: str   # 2-3 sentences, used in blog post


GEMINI_TIMEOUT = 120_000  # milliseconds (HttpOptions.timeout is in ms)


PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"


def _call_gemini(client, prompt: str, model: str = PRIMARY_MODEL):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3,
        ),
    )


def fetch_ai_news(api_key: str) -> list[NewsItem]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT),
    )

    today = datetime.now()
    week_ago = today - timedelta(days=7)
    date_range = f"{week_ago.strftime('%B %d, %Y')} to {today.strftime('%B %d, %Y')}"

    prompt = (
        f"Today is {today.strftime('%B %d, %Y')}. "
        f"Search the web for the top 5 most important AI news stories published between {date_range}. "
        "Focus on: major model releases, research breakthroughs, company announcements, "
        "funding rounds, policy/regulation, and industry shifts. "
        "Only include stories that actually happened in this date range — do not use older news. "
        "Return exactly 5 items as a JSON array with this structure:\n"
        '[{"headline": "max 10 word headline", "summary": "2-3 sentence factual summary with dates and numbers"}]\n'
        "Headlines and summaries must be written in Spanish. "
        "Headlines must be concise, specific, and newsworthy (max 10 words). "
        "Return ONLY the JSON array, no code blocks, no extra text."
    )

    from retry import log_line
    log_line("INFO", "→ Gemini: fetch_ai_news request sent (google_search enabled)...")
    try:
        response = with_retry(
            _call_gemini, client, prompt, PRIMARY_MODEL,
            label="fetch_ai_news",
            per_attempt_timeout=GEMINI_TIMEOUT // 1000 + 10,
        )
    except Exception:
        log_line("WARN", f"Primary model {PRIMARY_MODEL} failed — retrying with fallback {FALLBACK_MODEL}...")
        response = with_retry(
            _call_gemini, client, prompt, FALLBACK_MODEL,
            label="fetch_ai_news[fallback]",
            per_attempt_timeout=GEMINI_TIMEOUT // 1000 + 10,
        )
    log_line("INFO", f"← Gemini: fetch_ai_news response received ({len(response.text)} chars)")

    raw = response.text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(f"Expected list of 5 news items, got: {raw}")

    return [NewsItem(headline=item["headline"], summary=item["summary"]) for item in data]
