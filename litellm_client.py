"""
LiteLLM Local — Python SDK

All requests go through the LiteLLM gateway (default port 8400).
The gateway provides: unified endpoint, caching, retries, logging, model aliasing.

Usage:
    import litellm_client

    litellm_client.chat("Hello!")
    litellm_client.chat("Describe this", image="photo.png")
    litellm_client.ocr("invoice.png")
    litellm_client.embed("search query")
    litellm_client.embed(["batch", "of", "texts"])
    litellm_client.transcribe("meeting.wav")

Environment variables:
    GATEWAY_URL  — Gateway base URL (default: http://localhost:8400)
    GATEWAY_KEY  — API key if master_key is set (default: not-needed)
"""

__all__ = [
    "chat", "ocr", "embed", "transcribe", "health_check",
    "GATEWAY_URL", "GATEWAY_KEY", "MODELS",
]
__version__ = "1.0.0"

import base64
import logging
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Union

from typing_extensions import TypedDict

try:
    from openai import OpenAI
    import httpx
except ImportError:
    raise ImportError("Install dependencies: pip install openai httpx")

# ─── Logging ──────────────────────────────────────────────────────────────────
# Library convention: only define a logger; callers configure logging themselves.
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8400")
GATEWAY_KEY = os.getenv("GATEWAY_KEY", "not-needed")

# Model aliases — must match litellm_config.yaml model_name entries
MODELS = {
    "chat": "chat",           # → Qwen/Qwen3-VL-4B-Instruct-FP8 on :8070
    "ocr": "ocr",             # → zai-org/GLM-OCR on :8080
    "embed": "embedding",     # → Qwen3-VL-Embedding-2B-FP8 on :8090
    "asr": "asr",             # → Qwen/Qwen3-ASR-1.7B on :8000
}

# Timeouts — must be >= gateway's request_timeout (300s in litellm_config.yaml).
# Audio/image uploads are large; the previous 60s caused premature client-side timeouts.
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# ─── Type definitions ─────────────────────────────────────────────────────────

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


# ─── Internal helpers ─────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client pointed at the LiteLLM gateway.

    A single shared client is reused across all calls to benefit from httpx
    connection pooling.  Call ``_reset_client()`` after changing ``GATEWAY_URL``
    or ``GATEWAY_KEY`` at runtime.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=f"{GATEWAY_URL}/v1",
            api_key=GATEWAY_KEY,
            timeout=_TIMEOUT,
            max_retries=2,  # gateway does its own retries; keep client retries low
        )
        logger.debug("Client initialized → %s", GATEWAY_URL)
    return _client


def _reset_client() -> None:
    """Discard the cached client so the next call creates a fresh one."""
    global _client
    _client = None


def _model(service: str) -> str:
    """Return the gateway model alias for a service."""
    return MODELS[service]


# Common MIME types for images; falls back to mimetypes module for others.
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


# =============================================================================
# PUBLIC API
# =============================================================================

def chat(
    message: str,
    image: Optional[Union[str, Path]] = None,
    system: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Union[str, Generator[str, None, None]]:
    """
    Chat with the vision-language model.

    Args:
        message:     User text.
        image:       Optional image path/URL for vision queries.
        system:      System prompt.
        history:     Prior messages in OpenAI format.
        stream:      Stream token-by-token.
        max_tokens:  Max response length.
        temperature: Sampling temperature.

    Returns:
        Full response string, or a generator of chunks when streaming.

    Raises:
        ConnectionError: Gateway/service unreachable.
        TimeoutError:    Request exceeded timeout.
        RuntimeError:    Any other API error.
    """
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

        resp = client.chat.completions.create(
            model=_model("chat"), messages=msgs,
            max_tokens=max_tokens, temperature=temperature, stream=stream,
        )
        if stream:
            # Wrap in generator so exceptions during iteration are properly caught
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


def ocr(
    image: Union[str, Path],
    prompt: str = "Extract all text from this image",
    max_tokens: int = 2048,
) -> str:
    """
    Extract text from an image using the OCR model.

    Args:
        image:      Image path or URL.
        prompt:     Instruction for the OCR task.
        max_tokens: Max response length.

    Returns:
        Extracted text.

    Raises:
        ConnectionError: Gateway/service unreachable.
        TimeoutError:    Request exceeded timeout.
        RuntimeError:    Any other API error.
    """
    logger.info("OCR: %s", image)
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_model("ocr"),
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _process_image(image)}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    except httpx.ConnectError as e:
        raise ConnectionError(f"OCR service unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError("OCR service timed out") from e
    except Exception as e:
        raise RuntimeError(f"OCR error: {e}") from e


def embed(input_data: EmbedInput) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings (text, image, or multimodal).

    Args:
        input_data: A string, list of strings, or dict(s) with text/image keys.

    Returns:
        Single embedding vector, or list of vectors for batch input.

    Raises:
        ConnectionError: Gateway/service unreachable.
        TimeoutError:    Request exceeded timeout.
        RuntimeError:    Any other API error.
    """
    logger.info("Embed: %s", type(input_data).__name__)
    try:
        client = _get_client()

        # Normalize to list
        if isinstance(input_data, (str, dict)):
            inputs = [input_data]
            single = True
        else:
            inputs = list(input_data)
            single = False

        # Convert local image paths to data URIs
        processed = []
        for item in inputs:
            if isinstance(item, dict) and "image" in item:
                cp = item.copy()
                if os.path.exists(item["image"]):
                    cp["image"] = _process_image(item["image"])
                processed.append(cp)
            else:
                processed.append(item)

        resp = client.embeddings.create(model=_model("embed"), input=processed)
        vecs = [d.embedding for d in resp.data]
        return vecs[0] if single else vecs

    except httpx.ConnectError as e:
        raise ConnectionError(f"Embedding service unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError("Embedding service timed out") from e
    except Exception as e:
        raise RuntimeError(f"Embedding error: {e}") from e


def transcribe(
    audio: Union[str, Path],
    language: Optional[str] = None,
    timestamps: Optional[Literal["word", "segment"]] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio to text.

    Args:
        audio:      Path to audio file (.wav, .mp3, .flac).
        language:   Language code (e.g. "en", "zh").
        timestamps: Request word- or segment-level timestamps.

    Returns:
        Dict with 'text' key; includes timestamp data if requested.

    Raises:
        FileNotFoundError: Audio file does not exist.
        ConnectionError:   Gateway/service unreachable.
        TimeoutError:      Request exceeded timeout.
        RuntimeError:      Any other API error.
    """
    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")
    logger.info("ASR: %s%s", path.name, f", lang={language}" if language else "")

    try:
        client = _get_client()
        kwargs: Dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        if timestamps:
            kwargs["timestamp_granularities"] = [timestamps]

        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=_model("asr"), file=f,
                response_format="verbose_json" if timestamps else "json",
                **kwargs,
            )
        return resp.model_dump() if timestamps else {"text": resp.text}

    except httpx.ConnectError as e:
        raise ConnectionError(f"ASR service unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise TimeoutError("ASR service timed out") from e
    except Exception as e:
        raise RuntimeError(f"ASR error: {e}") from e


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check() -> Dict[str, Any]:
    """
    Check health of the gateway and all backend services.

    Returns:
        Dict with 'gateway' bool and per-service bools.
    """
    status: Dict[str, Any] = {}

    # 1. Check gateway is alive
    try:
        urllib.request.urlopen(f"{GATEWAY_URL}/health", timeout=5)
        status["gateway"] = True
    except Exception as e:
        status["gateway"] = False
        logger.warning("Gateway unhealthy: %s", e)
        for svc in MODELS:
            status[svc] = False
        return status

    # 2. Check each model alias is registered (single API call)
    try:
        models = _get_client().models.list()
        registered = {m.id for m in models.data}
        for svc, alias in MODELS.items():
            status[svc] = alias in registered
            if not status[svc]:
                logger.warning("%s model alias '%s' not registered", svc, alias)
    except Exception as e:
        for svc in MODELS:
            status[svc] = False
        logger.warning("Models check failed: %s", e)

    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Only configure logging when run directly
    print(f"LiteLLM Client v{__version__} — Gateway: {GATEWAY_URL}")
    for svc, alias in MODELS.items():
        print(f"  {svc:6s} → model=\"{alias}\"")

    print("\nHealth:")
    for svc, ok in health_check().items():
        print(f"  {svc:8s} {'OK' if ok else 'DOWN'}")
