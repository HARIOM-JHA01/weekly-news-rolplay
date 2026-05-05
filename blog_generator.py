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


GEMINI_TEXT_TIMEOUT = 120   # seconds per attempt
GEMINI_IMAGE_TIMEOUT = 180  # image generation gets more time


def _call_blog_gemini(client, prompt: str):
    return client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )


def _call_image_gemini(client, prompt: str):
    return client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )


def generate_blog_post(api_key: str, news_items: list[NewsItem]) -> BlogPost:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TEXT_TIMEOUT),
    )

    news_text = "\n".join(
        f"{i + 1}. {item.headline}\n   {item.summary}"
        for i, item in enumerate(news_items)
    )

    prompt = (
        f"Escribe un artículo de blog profesional de resumen semanal de noticias de IA basado en estas 5 historias:\n\n"
        f"{news_text}\n\n"
        "El artículo debe estar escrito completamente en español mexicano (México), usando expresiones, "
        "vocabulario y tono propios del español mexicano.\n\n"
        "Devuelve un objeto JSON con exactamente estas claves:\n"
        '{"title": "...", "summary": "...", "content": "...", "tags": [...]}\n\n'
        "Requisitos:\n"
        "- title: título atractivo para el resumen semanal (máx. 80 caracteres), en español mexicano\n"
        "- summary: resumen de 1-2 oraciones de la semana (50-300 caracteres), en español mexicano\n"
        "- content: HTML bien estructurado usando etiquetas <h2>, <p>, <ul>, <li>, <strong>. "
        "  Incluye un párrafo introductorio, una sección <h2> por cada noticia con 2-3 párrafos cada una, "
        "  y un párrafo de conclusión. No uses etiquetas <html>, <head> ni <body>. Todo en español mexicano.\n"
        "- tags: 4-6 etiquetas relevantes como cadenas de texto (ej. ['IA', 'Aprendizaje Automático', 'OpenAI'])\n"
        "Devuelve ÚNICAMENTE el objeto JSON, sin bloques de código, sin texto adicional."
    )

    from retry import log_line
    log_line("INFO", "→ Gemini: generate_blog_post request sent...")
    response = with_retry(
        _call_blog_gemini, client, prompt,
        label="generate_blog_post",
        per_attempt_timeout=GEMINI_TEXT_TIMEOUT + 10,
    )
    log_line("INFO", f"← Gemini: generate_blog_post response received ({len(response.text)} chars)")

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
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_IMAGE_TIMEOUT),
    )

    prompt = (
        f"Create a professional blog cover image for a weekly AI news roundup titled '{title}'. "
        "Clean digital art style, futuristic tech aesthetic. Blue and purple gradient background. "
        "Abstract neural network nodes, glowing circuit patterns, floating data elements. "
        "No text. High quality, suitable for a tech blog header."
    )

    from retry import log_line
    log_line("INFO", "→ Gemini: generate_cover_image request sent (Imagen)...")
    response = with_retry(
        _call_image_gemini, client, prompt,
        label="generate_cover_image",
        per_attempt_timeout=GEMINI_IMAGE_TIMEOUT + 10,
    )
    images = response.generated_images if response.generated_images else []
    log_line("INFO", f"← Gemini: generate_cover_image response received ({len(images)} images)")

    if images and images[0].image and images[0].image.image_bytes:
        return images[0].image.image_bytes

    raise RuntimeError("No image returned by Imagen model")
