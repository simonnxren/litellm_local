"""
LiteLLM Local - Direct Connection Client

Connects directly to vLLM services without a central gateway:
- Chat/Vision: Port 8070 (Qwen3-VL-4B)
- OCR: Port 8080 (GLM-OCR)
- Embeddings: Port 8090 (Qwen3-VL-Embedding)
- ASR: Port 8000 (Qwen3-ASR)
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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration with environment variable support
SERVICES = {
    "chat": {
        "port": int(os.getenv("CHAT_PORT", 8070)),
        "model": os.getenv("CHAT_MODEL", "Qwen/Qwen3-VL-4B-Instruct-FP8")
    },
    "ocr": {
        "port": int(os.getenv("OCR_PORT", 8080)),
        "model": os.getenv("OCR_MODEL", "zai-org/GLM-OCR")
    },
    "embed": {
        "port": int(os.getenv("EMBED_PORT", 8090)),
        "model": os.getenv("EMBED_MODEL", "shigureui/Qwen3-VL-Embedding-2B-FP8")
    },
    "asr": {
        "port": int(os.getenv("ASR_PORT", 8000)),
        "model": os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    },
}

# Type definitions for better type safety
class TextEmbedInput(TypedDict):
    text: str

class MultimodalEmbedInput(TypedDict):
    text: str
    image: str

EmbedInput = Union[str, List[str], TextEmbedInput, List[TextEmbedInput], MultimodalEmbedInput, List[MultimodalEmbedInput]]

def _get_client(service: str) -> OpenAI:
    """Get OpenAI client for a specific service with timeout and retry configuration."""
    port = SERVICES[service]["port"]
    base_url = f"http://localhost:{port}/v1"
    
    logger.debug(f"Creating client for {service} service at {base_url}")
    
    return OpenAI(
        base_url=base_url,
        api_key="dummy",  # vLLM doesn't require real API keys
        timeout=httpx.Timeout(30.0, connect=5.0),
        max_retries=3
    )

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
    Chat with Qwen3-VL-4B model supporting text and image inputs.
    
    Args:
        message (str): The user's text message
        image (str|Path, optional): Path or URL to image file
        system (str, optional): System prompt to set model behavior
        history (List[Dict], optional): Previous conversation history
        stream (bool): Whether to stream responses (default: False)
        max_tokens (int): Maximum tokens in response (default: 1024)
        temperature (float): Sampling temperature 0.0-2.0 (default: 0.7)
        
    Returns:
        str|Iterator[str]: Complete response string or streaming iterator
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ConnectionError: If chat service is unreachable
        RuntimeError: If service returns an error
    """
    logger.info(f"Chat request: {len(message)} chars" + (f", image: {image}" if image else ""))
    
    try:
        client = _get_client("chat")
        model = SERVICES["chat"]["model"]
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
            
        content: List[Dict[str, Any]] = [{"type": "text", "text": message}]
        
        if image:
            image_url = _process_image(image)
            content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
            
        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )

        if stream:
            return (chunk.choices[0].delta.content or "" for chunk in response)
        
        result = response.choices[0].message.content
        logger.debug(f"Chat response: {len(result)} chars")
        return result
        
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to chat service: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("Chat service timed out")
    except Exception as e:
        raise RuntimeError(f"Chat service error: {e}")

def ocr(
    image: Union[str, Path],
    prompt: str = "Extract all text from this image",
    max_tokens: int = 2048
) -> str:
    """
    Extract text using GLM-OCR model.
    
    Args:
        image (str|Path): Path or URL to image file
        prompt (str): Instruction for OCR task (default: "Extract all text from this image")
        max_tokens (int): Maximum tokens in response (default: 2048)
        
    Returns:
        str: Extracted text from image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ConnectionError: If OCR service is unreachable
        RuntimeError: If service returns an error
    """
    logger.info(f"OCR request for image: {image}")
    
    try:
        client = _get_client("ocr")
        model = SERVICES["ocr"]["model"]
        
        image_url = _process_image(image)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        logger.debug(f"OCR response: {len(result)} chars")
        return result
        
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to OCR service: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("OCR service timed out")
    except Exception as e:
        raise RuntimeError(f"OCR service error: {e}")

def embed(
    input_data: EmbedInput
) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings using Qwen3-VL-Embedding (Text/Image/Multimodal).
    
    Input formats:
    - Text: "Hello" or ["Hello", "World"]
    - Image: {"image": "path/to/img.jpg"}
    - Multimodal: {"text": "A cat", "image": "cat.jpg"}
    
    Args:
        input_data: Input text, image, or multimodal data
        
    Returns:
        List[float]|List[List[float]]: Single embedding vector or list of vectors
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ConnectionError: If embedding service is unreachable
        RuntimeError: If service returns an error
    """
    logger.info(f"Embed request: {type(input_data).__name__}")
    
    try:
        client = _get_client("embed")
        model = SERVICES["embed"]["model"]
        
        # Normalize input to list
        if isinstance(input_data, (str, dict)):
            inputs = [input_data]
            is_single = True
        else:
            inputs = input_data
            is_single = False
            
        # Process images in inputs if present (assuming dict format for multimodal)
        processed_inputs = []
        for item in inputs:
            if isinstance(item, dict) and "image" in item:
                 # If local path, convert to data URI because server might be remote/container
                 # Ideally the server supports base64, usually it does.
                 # Note: For embedding server, we often pass the dict directly if using OpenAI format?
                 # Standard OpenAI embedding doesn't support dict input officially, 
                 # but vLLM VL embedding does. We assume it handles data URIs or paths.
                 # Let's ensure paths are converted to data URIs if they are local files.
                 item_copy = item.copy()
                 if os.path.exists(item["image"]):
                      item_copy["image"] = _process_image(item["image"])
                 processed_inputs.append(item_copy)
            else:
                processed_inputs.append(item)

        response = client.embeddings.create(model=model, input=processed_inputs)
        embeddings = [d.embedding for d in response.data]
        
        result = embeddings[0] if is_single else embeddings
        logger.debug(f"Embed response: {len(result) if is_single else f'{len(result)} vectors'}")
        return result
        
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to embedding service: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("Embedding service timed out")
    except Exception as e:
        raise RuntimeError(f"Embedding service error: {e}")

def transcribe(
    audio: Union[str, Path],
    language: Optional[str] = None,
    timestamps: Optional[Literal["word", "segment"]] = None
) -> Dict[str, Any]:
    """
    Transcribe audio using Qwen3-ASR.
    
    Args:
        audio (str|Path): Path to audio file
        language (str, optional): Language code (e.g., "en", "zh")
        timestamps (Literal["word", "segment"], optional): Request word or segment timestamps
        
    Returns:
        Dict[str, Any]: Dictionary with 'text' and optional 'chunks'/'words' if timestamps requested
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ConnectionError: If ASR service is unreachable
        RuntimeError: If service returns an error
    """
    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")
        
    logger.info(f"ASR request: {path.name}" + (f", language: {language}" if language else ""))
    
    try:
        # Qwen3-ASR via vLLM OpenAI API usually uses the standard /v1/audio/transcriptions
        # But for complex timestamp params, sometimes direct requests are safer if the client lib creates issues.
        # We will use standard OpenAI client first.
        client = _get_client("asr")
        model = SERVICES["asr"]["model"]
        
        with open(path, "rb") as f:
            # Note: openai-python doesn't fully support all extra params for all backends
            # but let's try standard create.
            kwargs = {}
            if language: kwargs["language"] = language
            if timestamps: kwargs["timestamp_granularities"] = [timestamps] # List format for standard API
            
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json" if timestamps else "json",
                **kwargs
            )
        
        # Unpack response object to dict
        if timestamps:
            result = response.model_dump()
        else:
            result = {"text": response.text}
            
        logger.debug(f"ASR response: {len(result.get('text', ''))} chars")
        return result
        
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to ASR service: {e}")
    except httpx.TimeoutException:
        raise TimeoutError("ASR service timed out")
    except Exception as e:
        raise RuntimeError(f"ASR service error: {e}")

# =============================================================================
# HELPERS
# =============================================================================

def _process_image(image: Union[str, Path]) -> str:
    """Convert image path/url to base64 data URI or return as-is."""
    image_str = str(image)
    if image_str.startswith(("http://", "https://", "data:")):
        return image_str
        
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
        
    mime_type = "image/jpeg"
    if path.suffix.lower() == ".png": mime_type = "image/png"
    elif path.suffix.lower() == ".webp": mime_type = "image/webp"
    
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"

def health_check() -> Dict[str, bool]:
    """
    Check health status of all services.
    
    Returns:
        Dict[str, bool]: Health status for each service
    """
    status = {}
    for service_name in SERVICES.keys():
        try:
            client = _get_client(service_name)
            # Simple health check - try to get model info
            response = client.models.list()
            status[service_name] = True
            logger.debug(f"{service_name} service is healthy")
        except Exception as e:
            status[service_name] = False
            logger.warning(f"{service_name} service is unhealthy: {e}")
    
    return status

if __name__ == "__main__":
    print("LiteLLM Client (Direct Mode)")
    print(f"Chat:  Port {SERVICES['chat']['port']} ({SERVICES['chat']['model']})")
    print(f"OCR:   Port {SERVICES['ocr']['port']} ({SERVICES['ocr']['model']})")
    print(f"Embed: Port {SERVICES['embed']['port']} ({SERVICES['embed']['model']})")
    print(f"ASR:   Port {SERVICES['asr']['port']} ({SERVICES['asr']['model']})")
    
    # Show health status
    print("\nHealth Check:")
    statuses = health_check()
    for service, healthy in statuses.items():
        status_str = "✅ UP" if healthy else "❌ DOWN"
        print(f"  {service}: {status_str}")
