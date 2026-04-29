import json
import uuid
import urllib.request
import urllib.error
from datetime import datetime

from retry import with_retry


def _encode_multipart(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return boundary, body


def _do_upload(base_url: str, api_key: str, image_bytes: bytes, filename: str) -> str:
    boundary, body = _encode_multipart(image_bytes, filename)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/upload",
        data=body,
        headers={
            "x-api-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"]["url"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Image upload failed HTTP {exc.code}: {body_text}") from exc


def _do_create_blog(base_url: str, api_key: str, payload: bytes) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/blogs/create",
        data=payload,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"]["slug"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Blog creation failed HTTP {exc.code}: {body_text}") from exc


def upload_image(base_url: str, api_key: str, image_bytes: bytes) -> str:
    filename = f"ai-weekly-{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    return with_retry(_do_upload, base_url, api_key, image_bytes, filename, label="upload_image")


def create_blog_post(
    base_url: str,
    api_key: str,
    title: str,
    summary: str,
    content: str,
    cover_image_url: str,
    tags: list[str],
) -> str:
    payload = json.dumps({
        "title": title,
        "summary": summary,
        "content": content,
        "coverImage": cover_image_url,
        "tags": tags,
        "source": "Rolplay AI",
        "published": True,
    }).encode("utf-8")
    return with_retry(_do_create_blog, base_url, api_key, payload, label="create_blog_post")
