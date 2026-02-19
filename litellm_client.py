"""LiteLLM Local — Python SDK.

Routes all requests through the LiteLLM gateway (default http://localhost:8400).
Env vars: GATEWAY_URL, GATEWAY_KEY.
"""

__all__ = [
    "chat", "ocr", "ocr_layout_parse", "ocr_restructure_pages", "ocr_document",
    "embed", "transcribe", "health_check", "discover_services",
    "GATEWAY_URL", "GATEWAY_KEY", "MODELS",
]
__version__ = "1.0.0"

import base64
import io
import json
import logging
import mimetypes
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Union

from typing_extensions import TypedDict

try:
    from openai import OpenAI
    import httpx
except ImportError:
    raise ImportError("Install dependencies: pip install openai httpx")

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8400")
GATEWAY_KEY = os.getenv("GATEWAY_KEY", "not-needed")
MODELS: Dict[str, str] = {}  # populated at runtime via /models endpoints
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # >= gateway request_timeout

_EMBED_IMAGE_PRESETS = [  # (max_side, jpeg_quality) — progressively smaller
    (1024, 85), (896, 80), (768, 75), (640, 70), (512, 65), (448, 60), (384, 55),
]

class TextEmbedInput(TypedDict):
    text: str

class MultimodalEmbedInput(TypedDict):
    text: str
    image: str

EmbedInput = Union[
    str, List[str],
    TextEmbedInput, List[TextEmbedInput],
    MultimodalEmbedInput, List[MultimodalEmbedInput],
]

# Service endpoint URLs
_CHAT_MODELS_URL = f"{GATEWAY_URL}/v1/models"
_EMBED_URL = f"{GATEWAY_URL}/api/embeddings"
_EMBED_MODELS_URL = f"{GATEWAY_URL}/api/embeddings/models"
_ASR_TRANSCRIBE_URL = f"{GATEWAY_URL}/api/asr/transcriptions"
_ASR_MODELS_URL = f"{GATEWAY_URL}/api/asr/models"
_OCR_MODELS_URL = f"{GATEWAY_URL}/api/ocr/models"
_OCR_LAYOUT_URL = f"{GATEWAY_URL}/api/ocr/layout-parsing"
_OCR_RESTRUCTURE_URL = f"{GATEWAY_URL}/api/ocr/restructure-pages"

_MODEL_ENDPOINTS = {
    "chat": _CHAT_MODELS_URL,
    "embed": _EMBED_MODELS_URL,
    "asr": _ASR_MODELS_URL,
    "ocr": _OCR_MODELS_URL,
}
_MODEL_CACHE_TTL_SEC = float(os.getenv("MODEL_DISCOVERY_TTL_SEC", "60"))
_MODEL_CACHE_AT: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# Core internals — HTTP client, JSON fetcher, model discovery
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Cached OpenAI client pointed at the gateway."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=f"{GATEWAY_URL}/v1", api_key=GATEWAY_KEY,
            timeout=_TIMEOUT, max_retries=2,
        )
    return _client


def _reset_client() -> None:
    """Discard the cached client so the next call creates a fresh one."""
    global _client
    _client = None


def _fetch_json(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    context: str,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch a JSON response and normalize common network errors."""
    try:
        resp = http.request(method, url, json=json_body, data=data, files=files)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError as e:
        raise ConnectionError(f"{context} unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError(f"{context} timed out") from e
    except Exception as e:
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise
        raise RuntimeError(f"{context} error: {e}") from e


def _extract_model_ids(payload: Dict[str, Any]) -> List[str]:
    """Extract model IDs from an OpenAI-compatible /models payload."""
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


def _fetch_model_ids(url: str, http: Optional[httpx.Client] = None) -> List[str]:
    """Fetch model IDs from a model-list endpoint."""
    if http is not None:
        payload = _fetch_json(http, "GET", url, context="Model-list")
        return _extract_model_ids(payload)
    with httpx.Client(timeout=_TIMEOUT) as temp_http:
        payload = _fetch_json(temp_http, "GET", url, context="Model-list")
        return _extract_model_ids(payload)


def _discover_model(service: str, http: Optional[httpx.Client] = None, force_refresh: bool = False) -> str:
    """Cached model-ID lookup for a service from its /models endpoint."""
    if service not in _MODEL_ENDPOINTS:
        raise KeyError(f"Unknown service: {service}")
    now = time.time()
    cached = MODELS.get(service)
    if cached and not force_refresh and (now - _MODEL_CACHE_AT.get(service, 0.0)) < _MODEL_CACHE_TTL_SEC:
        return cached
    ids = _fetch_model_ids(_MODEL_ENDPOINTS[service], http=http)
    if not ids:
        raise RuntimeError(f"No models for service '{service}'")
    MODELS[service] = ids[0]
    _MODEL_CACHE_AT[service] = now
    return ids[0]

# ---------------------------------------------------------------------------
# File & image processing helpers
# ---------------------------------------------------------------------------

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
}


def _process_image(image: Union[str, Path]) -> str:
    """Convert local image path to base64 data URI; pass URLs through."""
    image_str = str(image)
    if image_str.startswith(("http://", "https://", "data:")):
        return image_str
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    suffix = path.suffix.lower()
    mime = _IMAGE_MIME.get(suffix) or mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _process_paddle_file(file_input: Union[str, Path]) -> str:
    """Convert local file path to base64 for Paddle OCR APIs; pass URLs through."""
    file_str = str(file_input)
    if file_str.startswith(("http://", "https://")):
        return file_str
    path = Path(file_input)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_input}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _embedding_image_candidates(image: Union[str, Path]) -> List[str]:
    """Progressively downscaled JPEG data-URIs for embedding; URLs pass through."""
    image_str = str(image)
    if image_str.startswith(("http://", "https://", "data:")):
        return [image_str]
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if Image is None or ImageOps is None:
        return [_process_image(path)]
    candidates: List[str] = []
    seen: set[str] = set()
    for max_side, jpeg_quality in _EMBED_IMAGE_PRESETS:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
            if uri not in seen:
                seen.add(uri)
                candidates.append(uri)
    return candidates or [_process_image(path)]

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def chat(
    message: str,
    image: Optional[Union[str, Path]] = None,
    system: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Union[str, Generator[str, None, None]]:
    """Chat with the VLM. Returns string or generator of chunks if streaming."""
    logger.info("Chat: %d chars%s", len(message), f", image={image}" if image else "")
    try:
        client = _get_client()
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if history:
            msgs.extend(history)
        content: List[Dict[str, Any]] = [{"type": "text", "text": message}]
        if image:
            content.insert(0, {"type": "image_url", "image_url": {"url": _process_image(image)}})
        msgs.append({"role": "user", "content": content})
        chat_model = _discover_model("chat")

        resp = client.chat.completions.create(
            model=chat_model, messages=msgs,
            max_tokens=max_tokens, temperature=temperature, stream=stream,
        )
        if stream:
            def _stream_chunks() -> Generator[str, None, None]:
                try:
                    for chunk in resp:
                        yield chunk.choices[0].delta.content or ""
                except httpx.ConnectError as exc:
                    raise ConnectionError(f"Chat stream interrupted: {exc}") from exc
                except httpx.TimeoutException as exc:
                    raise TimeoutError("Chat stream timed out") from exc
            return _stream_chunks()
        return resp.choices[0].message.content or ""

    except httpx.ConnectError as e:
        raise ConnectionError(f"Chat service unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError("Chat service timed out") from e
    except Exception as e:
        raise RuntimeError(f"Chat error: {e}") from e

# ---------------------------------------------------------------------------
# OCR (Paddle OCR pipeline)
# ---------------------------------------------------------------------------

def ocr_layout_parse(
    file_input: Union[str, Path],
    file_type: int = 1,
) -> Dict[str, Any]:
    """Call Paddle OCR /layout-parsing via gateway pass-through."""
    logger.info("OCR layout-parse: %s", file_input)
    payload = {
        "file": _process_paddle_file(file_input),
        "fileType": file_type,
    }
    with httpx.Client(timeout=_TIMEOUT) as http:
        return _fetch_json(
            http,
            "POST",
            _OCR_LAYOUT_URL,
            context="OCR layout-parsing",
            json_body=payload,
        )


def ocr_restructure_pages(
    pages: List[Dict[str, Any]],
    concatenate_pages: bool = True,
) -> Dict[str, Any]:
    """Call Paddle OCR /restructure-pages via gateway pass-through."""
    payload = {
        "pages": pages,
        "concatenatePages": concatenate_pages,
    }
    with httpx.Client(timeout=_TIMEOUT) as http:
        return _fetch_json(
            http,
            "POST",
            _OCR_RESTRUCTURE_URL,
            context="OCR restructure-pages",
            json_body=payload,
        )


def ocr_document(
    file_input: Union[str, Path],
    concatenate_pages: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Full OCR pipeline: layout-parse → restructure → optional save to output_dir."""
    layout_json = ocr_layout_parse(file_input=file_input, file_type=1)
    layout_results = (layout_json.get("result") or {}).get("layoutParsingResults") or []
    if not layout_results:
        raise RuntimeError("layout-parsing returned no layoutParsingResults")

    pages = [{"prunedResult": p.get("prunedResult", ""),
              "markdownImages": (p.get("markdown") or {}).get("images") or {}}
             for p in layout_results]

    restructure_json = ocr_restructure_pages(
        pages=pages,
        concatenate_pages=concatenate_pages,
    )
    merged = (restructure_json.get("result") or {}).get("layoutParsingResults") or []
    if not merged:
        raise RuntimeError("restructure-pages returned no layoutParsingResults")

    first = merged[0]
    md = first.get("markdown") or {}
    markdown_text = md.get("text", "")

    if output_dir:
        out_root = Path(output_dir)
        layout_images_dir = out_root / "layout_images"
        markdown_dir = out_root / "markdown"
        layout_images_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        for page_index, page in enumerate(layout_results):
            for image_name, image_b64 in (page.get("outputImages") or {}).items():
                image_path = layout_images_dir / f"{image_name}_{page_index}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(base64.b64decode(image_b64))

        md_file = markdown_dir / "doc.md"
        md_file.write_text(markdown_text, encoding="utf-8")
        for rel_path, image_b64 in (md.get("images") or {}).items():
            image_path = markdown_dir / rel_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(base64.b64decode(image_b64))

    return {
        "layout": layout_json,
        "restructure": restructure_json,
        "pruned_result": first.get("prunedResult", ""),
        "markdown_text": markdown_text,
    }


def ocr(
    image: Union[str, Path],
    prompt: str = "Extract all text from this image",
    max_tokens: int = 2048,
) -> str:
    """Extract text from an image via the Paddle OCR pipeline."""
    logger.info("OCR: %s", image)
    try:
        result = ocr_document(file_input=image, concatenate_pages=True, output_dir=None)
        pruned = result.get("pruned_result", "")
        if isinstance(pruned, str) and pruned.strip():
            return pruned
        if isinstance(pruned, dict):
            items = pruned.get("parsing_res_list") or []
            lines = [(item.get("block_content") or "").strip()
                     for item in items if isinstance(item, dict)]
            joined = "\n".join(line for line in lines if line)
            return joined if joined else json.dumps(pruned, ensure_ascii=False)
        return result.get("markdown_text") or str(pruned)
    except (ConnectionError, TimeoutError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"OCR error: {e}") from e

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _embed_text_batch(
    http: httpx.Client,
    inputs: List[Union[str, Dict[str, str]]],
    model_id: str,
) -> List[List[float]]:
    """Send a single batched text-embedding request."""
    texts = [item if isinstance(item, str) else item["text"] for item in inputs]
    payload = {
        "model": model_id,
        "input": texts,
        "encoding_format": "float",
        "truncate_prompt_tokens": 5119,
    }
    resp = http.post(_EMBED_URL, json=payload)
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def _build_embedding_messages(text: str, image_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build vLLM EmbeddingChatRequest messages payload."""
    user_content: List[Dict[str, Any]] = []
    if image_url:
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    user_content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def _is_embedding_context_error(resp: httpx.Response) -> bool:
    """Return True when backend rejects input for context length overflow."""
    if resp.status_code != 400:
        return False
    body_text = resp.text.lower()
    return "context length" in body_text or "input_tokens" in body_text


def _embed_multimodal_item(
    http: httpx.Client,
    item: Union[str, Dict[str, str]],
    model_id: str,
) -> List[float]:
    """Send one multimodal-compatible embedding request for a single item."""
    if isinstance(item, str):
        text, image_candidates = item, [None]
    elif isinstance(item, dict):
        text = item.get("text", "")
        image_candidates = _embedding_image_candidates(item["image"]) if "image" in item else [None]
    else:
        raise ValueError(f"Unsupported embed input type: {type(item)}")

    last_ctx_err: Optional[str] = None
    for idx, image_url in enumerate(image_candidates):
        payload = {
            "model": model_id,
            "messages": _build_embedding_messages(text=text, image_url=image_url),
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }
        resp = http.post(_EMBED_URL, json=payload)
        if resp.status_code < 400:
            return resp.json()["data"][0]["embedding"]
        if _is_embedding_context_error(resp) and idx < len(image_candidates) - 1:
            last_ctx_err = resp.text
            logger.warning("Embedding image too large; retrying smaller preset (%d/%d)",
                           idx + 2, len(image_candidates))
            continue
        resp.raise_for_status()

    if last_ctx_err is not None:
        raise RuntimeError(f"Embedding exceeds context after all presets: {last_ctx_err}")
    raise RuntimeError("Embedding request failed after image preprocessing retries")


def embed(input_data: EmbedInput) -> Union[List[float], List[List[float]]]:
    """Generate embeddings (text-only batches via ``input``, multimodal via ``messages``)."""
    logger.info("Embed: %s", type(input_data).__name__)
    if isinstance(input_data, (str, dict)):
        inputs = [input_data]
        single = True
    else:
        inputs = list(input_data)
        single = False

    has_multimodal = any(isinstance(item, dict) and "image" in item for item in inputs)

    try:
        with httpx.Client(timeout=_TIMEOUT) as http:
            embed_model = _discover_model("embed", http=http)
            if not has_multimodal:
                vecs = _embed_text_batch(http, inputs, embed_model)
            else:
                vecs = [_embed_multimodal_item(http, item, embed_model) for item in inputs]

        return vecs[0] if single else vecs

    except httpx.ConnectError as e:
        raise ConnectionError(f"Embedding service unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError("Embedding service timed out") from e
    except Exception as e:
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise
        raise RuntimeError(f"Embedding error: {e}") from e

# ---------------------------------------------------------------------------
# ASR (speech-to-text)
# ---------------------------------------------------------------------------

def transcribe(
    audio: Union[str, Path],
    language: Optional[str] = None,
    timestamps: Optional[Literal["word", "segment"]] = None,
) -> Dict[str, Any]:
    """Transcribe audio file to text. Returns dict with 'text' key."""
    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")
    logger.info("ASR: %s%s", path.name, f", lang={language}" if language else "")

    with httpx.Client(timeout=_TIMEOUT) as http:
        asr_model = _discover_model("asr", http=http)
        form_data: Dict[str, Any] = {"model": asr_model}
        if language:
            form_data["language"] = language
        if timestamps:
            form_data["timestamp_granularities[]"] = timestamps

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            payload = _fetch_json(
                http,
                "POST",
                _ASR_TRANSCRIBE_URL,
                context="ASR service",
                data=form_data,
                files=files,
            )

    if timestamps:
        return payload
    return {"text": payload.get("text", "")}

# ---------------------------------------------------------------------------
# Discovery & health
# ---------------------------------------------------------------------------

def _endpoint_exists(http: httpx.Client, path: str) -> Dict[str, Any]:
    """Probe whether a gateway path exists (OPTIONS then GET)."""
    url = f"{GATEWAY_URL}{path}"
    last_error: Optional[str] = None
    for method in ("OPTIONS", "GET"):
        try:
            resp = http.request(method, url)
            if resp.status_code != 404:
                return {"path": path, "exists": True, "status_code": resp.status_code, "method_used": method}
        except Exception as e:
            last_error = str(e)
    return {"path": path, "exists": False, "status_code": None, "method_used": None, "error": last_error}


def _services_from_openapi_paths(paths: List[str]) -> List[str]:
    """Extract service names from /api/<service>/models paths."""
    services: set[str] = set()
    pattern = re.compile(r"^/api/([^/]+)/models$")
    for path in paths:
        match = pattern.match(path)
        if match:
            services.add(match.group(1))
    return sorted(services)


def discover_services(timeout_sec: float = 8.0, service_hints: Optional[List[str]] = None) -> Dict[str, Any]:
    """Discover gateway capabilities by probing health, OpenAPI, and /api/<svc>/models."""
    timeout = httpx.Timeout(timeout_sec, connect=min(5.0, timeout_sec))
    report: Dict[str, Any] = {
        "gateway_url": GATEWAY_URL,
        "health": {},
        "openapi": {"available": False, "path_count": 0},
        "gateway_models": [],
        "services": {},
        "capabilities": {},
    }

    try:
        with httpx.Client(timeout=timeout) as http:
            for health_path in ("/health", "/health/liveliness"):
                try:
                    r = http.get(f"{GATEWAY_URL}{health_path}")
                    report["health"][health_path] = {
                        "ok": r.status_code < 400,
                        "status_code": r.status_code,
                    }
                except Exception as e:
                    report["health"][health_path] = {
                        "ok": False,
                        "status_code": None,
                        "error": str(e),
                    }

            try:
                openapi_resp = http.get(f"{GATEWAY_URL}/openapi.json")
                if openapi_resp.status_code < 400:
                    spec = openapi_resp.json()
                    paths = list((spec.get("paths") or {}).keys())
                    report["openapi"] = {
                        "available": True,
                        "path_count": len(paths),
                        "paths": paths,
                    }
            except Exception:
                paths = []

            try:
                report["gateway_models"] = _fetch_model_ids(f"{GATEWAY_URL}/v1/models", http=http)
            except Exception as e:
                report["gateway_models_error"] = str(e)

            discovered = _services_from_openapi_paths(paths if "paths" in locals() else [])
            hinted = sorted({s.strip() for s in (service_hints or []) if isinstance(s, str) and s.strip()})
            service_candidates = sorted(set(discovered + hinted))
            report["service_candidates"] = service_candidates

            for service in service_candidates:
                models_path = f"/api/{service}/models"
                service_entry: Dict[str, Any] = {
                    "models_path": models_path,
                    "models_endpoint": _endpoint_exists(http, models_path),
                    "models": [],
                }

                if service_entry["models_endpoint"].get("exists"):
                    try:
                        service_entry["models"] = _fetch_model_ids(f"{GATEWAY_URL}{models_path}", http=http)
                    except Exception as e:
                        service_entry["models_error"] = str(e)

                report["services"][service] = service_entry

    except Exception as e:
        report["fatal_error"] = str(e)
        return report

    report["capabilities"] = {
        service: bool(entry.get("models_endpoint", {}).get("exists"))
        for service, entry in report["services"].items()
    }
    return report


def health_check() -> Dict[str, Any]:
    """Check gateway and backend service health. Returns dict of service→bool."""
    status: Dict[str, Any] = {}
    try:
        urllib.request.urlopen(f"{GATEWAY_URL}/health", timeout=5)
        status["gateway"] = True
    except Exception as e:
        logger.warning("Gateway unhealthy: %s", e)
        return {"gateway": False, **{s: False for s in _MODEL_ENDPOINTS}}
    try:
        with httpx.Client(timeout=_TIMEOUT) as http:
            for svc in _MODEL_ENDPOINTS:
                try:
                    _discover_model(svc, http=http, force_refresh=True)
                    status[svc] = True
                except Exception as e:
                    status[svc] = False
                    logger.warning("%s check failed: %s", svc, e)
    except Exception as e:
        for svc in _MODEL_ENDPOINTS:
            status.setdefault(svc, False)
        logger.warning("Models check failed: %s", e)
    return status

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"LiteLLM Client v{__version__} — Gateway: {GATEWAY_URL}")
    for svc in _MODEL_ENDPOINTS:
        try:
            print(f"  {svc:6s} → model=\"{_discover_model(svc)}\"")
        except Exception as e:
            print(f"  {svc:6s} → failed ({e})")
    print("\nHealth:")
    for svc, ok in health_check().items():
        print(f"  {svc:8s} {'OK' if ok else 'DOWN'}")
