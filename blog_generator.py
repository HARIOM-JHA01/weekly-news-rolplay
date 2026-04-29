import json
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from news_fetcher import NewsItem
from retry import with_retry


@dataclass
class BlogPost:
    title: str
    summary: str
    content: str  # HTML
    tags: list[str] = field(default_factory=list)


def _call_blog_gemini(client, prompt: str):
    return client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )


def _call_image_gemini(client, prompt: str):
    return client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt],
    )


def generate_blog_post(api_key: str, news_items: list[NewsItem]) -> BlogPost:
    client = genai.Client(api_key=api_key)

    news_text = "\n".join(
        f"{i + 1}. {item.headline}\n   {item.summary}"
        for i, item in enumerate(news_items)
    )

    prompt = (
        f"Write a professional weekly AI news roundup blog post based on these 5 stories:\n\n"
        f"{news_text}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{"title": "...", "summary": "...", "content": "...", "tags": [...]}\n\n'
        "Requirements:\n"
        "- title: engaging title for the weekly roundup (max 80 chars)\n"
        "- summary: 1-2 sentence overview of the week (50-300 chars)\n"
        "- content: well-structured HTML using <h2>, <p>, <ul>, <li>, <strong> tags. "
        "  Include an intro paragraph, one <h2> section per news story with 2-3 paragraphs each, "
        "  and a concluding paragraph. Do not use <html>, <head>, or <body> tags.\n"
        "- tags: 4-6 relevant tags as strings (e.g. ['AI', 'Machine Learning', 'OpenAI'])\n"
        "Return ONLY the JSON object, no code blocks, no extra text."
    )

    response = with_retry(_call_blog_gemini, client, prompt, label="generate_blog_post")

    raw = response.text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

    data = json.loads(raw)
    return BlogPost(
        title=data["title"],
        summary=data["summary"],
        content=data["content"],
        tags=data.get("tags", ["AI", "Technology"]),
    )


def generate_cover_image(api_key: str, title: str) -> bytes:
    client = genai.Client(api_key=api_key)

    prompt = (
        f"Create a professional blog cover image for a weekly AI news roundup titled '{title}'. "
        "Clean digital art style, futuristic tech aesthetic. Blue and purple gradient background. "
        "Abstract neural network nodes, glowing circuit patterns, floating data elements. "
        "No text. High quality, suitable for a tech blog header."
    )

    response = with_retry(_call_image_gemini, client, prompt, label="generate_cover_image")

    for part in response.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("No image returned by Gemini image model")
