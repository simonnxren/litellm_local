"""
LiteLLM Local — Python SDK

All requests go through the LiteLLM gateway (port 8400).
The gateway provides: unified endpoint, caching, retries, logging, model aliasing.

Usage:
    import litellm_client
    litellm_client.chat("Hello!")
    litellm_client.ocr("invoice.png")
    litellm_client.embed("search query")
    litellm_client.transcribe("meeting.wav")

Environment variables:
    GATEWAY_URL   — Gateway base URL (default: http://localhost:8400)
    GATEWAY_KEY   — API key if master_key is set (default: not-needed)
"""

import base64
import os
import logging
from pathlib import Path
from typing import Union, Optional, Iterator, Literal, List, Dict, Any
from typing_extensions import TypedDict

try:
    from openai import OpenAI
    import httpx
except ImportError:
    raise ImportError("Install dependencies: pip install openai httpx")

# ─── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── Configuration ────────────────────────────────────────────────────────────
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8400")
GATEWAY_KEY = os.getenv("GATEWAY_KEY", "not-needed")

# Model aliases — must match litellm_config.yaml
MODELS = {
    "chat": "chat",           # → Qwen/Qwen3-VL-4B-Instruct-FP8 on :8070
    "ocr": "ocr",             # → zai-org/GLM-OCR on :8080
    "embed": "embedding",     # → Qwen3-VL-Embedding-2B-FP8 on :8090
    "asr": "asr",             # → Qwen/Qwen3-ASR-1.7B on :8000
}

# Type definitions
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

def _get_client(service: str) -> OpenAI:
    """Return an OpenAI client pointed at the LiteLLM gateway."""
    logger.debug(f"[gateway] {service} → {GATEWAY_URL}")
    return OpenAI(base_url=f"{GATEWAY_URL}/v1", api_key=GATEWAY_KEY,
                  timeout=httpx.Timeout(60.0, connect=10.0), max_retries=3)


def _model(service: str) -> str:
    """Return the gateway model alias for a service."""
    return MODELS[service]


def _process_image(image: Union[str, Path]) -> str:
    """Convert local image path to base64 data URI; pass URLs through."""
    image_str = str(image)
    if image_str.startswith(("http://", "https://", "data:")):
        return image_str
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


# =============================================================================
# PUBLIC API
# =============================================================================

def chat(
    message: str,
    image: Optional[Union[str, Path]] = None,
    system: Optional[str] = None,
    history: Optional[List[Dict]] = None,
    stream: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Union[str, Iterator[str]]:
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
        Full response string, or an iterator of chunks when streaming.
    """
    logger.info(f"Chat: {len(message)} chars" + (f", image={image}" if image else ""))
    try:
        client = _get_client("chat")
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
            return (c.choices[0].delta.content or "" for c in resp)
        return resp.choices[0].message.content

    except httpx.ConnectError as e:
        raise ConnectionError(f"Chat service unreachable: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("Chat service timed out")
    except Exception as e:
        raise RuntimeError(f"Chat error: {e}")


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
    """
    logger.info(f"OCR: {image}")
    try:
        client = _get_client("ocr")
        resp = client.chat.completions.create(
            model=_model("ocr"),
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _process_image(image)}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    except httpx.ConnectError as e:
        raise ConnectionError(f"OCR service unreachable: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("OCR service timed out")
    except Exception as e:
        raise RuntimeError(f"OCR error: {e}")


def embed(input_data: EmbedInput) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings (text, image, or multimodal).

    Args:
        input_data: A string, list of strings, or dict(s) with text/image keys.

    Returns:
        Single embedding vector, or list of vectors for batch input.
    """
    logger.info(f"Embed: {type(input_data).__name__}")
    try:
        client = _get_client("embed")

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
        raise ConnectionError(f"Embedding service unreachable: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("Embedding service timed out")
    except Exception as e:
        raise RuntimeError(f"Embedding error: {e}")


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
    """
    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")
    logger.info(f"ASR: {path.name}" + (f", lang={language}" if language else ""))

    try:
        client = _get_client("asr")
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
        raise ConnectionError(f"ASR service unreachable: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("ASR service timed out")
    except Exception as e:
        raise RuntimeError(f"ASR error: {e}")


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check() -> Dict[str, Any]:
    """
    Check health of the gateway and all backend services.

    Returns:
        Dict mapping service name → bool (healthy or not).
    """
    status: Dict[str, Any] = {}

    try:
        import urllib.request
        urllib.request.urlopen(f"{GATEWAY_URL}/health", timeout=5)
        status["gateway"] = True
    except Exception as e:
        status["gateway"] = False
        logger.warning(f"Gateway unhealthy: {e}")
        return status  # No point checking backends

    for svc in MODELS:
        try:
            _get_client(svc).models.list()
            status[svc] = True
        except Exception as e:
            status[svc] = False
            logger.warning(f"{svc} unhealthy: {e}")

    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print(f"LiteLLM Client — Gateway: {GATEWAY_URL}")
    for svc, alias in MODELS.items():
        print(f"  {svc:6s} → model=\"{alias}\"")

    print("\nHealth:")
    for svc, ok in health_check().items():
        print(f"  {svc:8s} {'OK' if ok else 'DOWN'}")
