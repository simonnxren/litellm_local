"""
LiteLLM Local - Python Client SDK

A unified client for multimodal embeddings, chat completions, and OCR
via the LiteLLM gateway with OpenAI-compatible APIs.

Quick Start:
    from litellm_client import LiteLLMClient

    client = LiteLLMClient()

    # Text embeddings (2048 dims)
    embedding = client.embed("Hello world")

    # Image embeddings
    embedding = client.embed_image("/path/to/photo.jpg")

    # Chat completion
    response = client.chat("What is 2+2?")

    # OCR / vision
    text = client.ocr("document.png")

    # Audio transcription (ASR)
    text = client.transcribe("speech.mp3")

Features:
    - Multimodal embeddings (text, image, video) via Qwen3-VL-Embedding-2B
    - Text embeddings (OpenAI-compatible)
    - Chat completions with streaming support
    - OCR with vision-language models
    - OpenAI-compatible client interface

Convenience Functions:
    from litellm_client import embed, embed_image, chat, ocr, transcribe

    # These use a singleton client instance
    embedding = embed("Hello world")
    text = ocr("image.png")
    text = transcribe("speech.mp3")

See README.md for detailed documentation and model information.
"""

from typing import Optional, Union, Iterator, cast, overload, Literal
from pathlib import Path
import base64
import time
from functools import wraps


__all__ = [
    "LiteLLMClient",
    "get_client",
    "embed",
    "embed_text",
    "embed_image",
    "chat",
    "ocr",
    "transcribe",
]


class LiteLLMClient:
    """Client SDK for LiteLLM Local services."""

    def __init__(
        self,
        base_url: str = "http://localhost:8200/v1",
        api_key: str = "dummy",
        embedding_model: str = "embedding",
        chat_model: str = "completions",
        ocr_model: str = "ocr",
    ):
        """
        Initialize the LiteLLM client.

        Args:
            base_url: LiteLLM gateway URL (default: http://localhost:8200/v1)
            api_key: API key for authentication (default: "dummy" for local use)
            embedding_model: Default model for embeddings (default: "embedding")
            chat_model: Default model for chat (default: "completions")
            ocr_model: Default model for OCR (default: "ocr")

        Example:
            >>> client = LiteLLMClient()
            >>> client = LiteLLMClient(base_url="http://gateway:8200/v1")
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.ocr_model = ocr_model

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def embed(
        self,
        input_data: Union[str, list[str], dict, list[dict]],
        model: Optional[str] = None,
    ) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings for text, images, or multimodal inputs.

        Unified method supporting:
        - Text embeddings: "Hello world" or ["text1", "text2"]
        - Image embeddings: {"image": "/path/to/img.jpg"}
        - Multimodal: {"text": "...", "image": "..."}
        - Video embeddings: {"video": "/path/to/video.mp4"}

        Args:
            input_data: Content to embed (string, list of strings, or dict)
            model: Optional model name override

        Returns:
            Single embedding (2048 dims) or list of embeddings

        Examples:
            >>> # Text
            >>> emb = client.embed("Hello world")
            >>>
            >>> # Image
            >>> emb = client.embed({"image": "photo.jpg"})
            >>>
            >>> # Text + Image
            >>> emb = client.embed({
            ...     "text": "A photo of",
            ...     "image": "cat.jpg"
            ... })
            >>>
            >>> # Batch
            >>> embs = client.embed([
            ...     {"text": "doc1"},
            ...     {"image": "img1.jpg"},
            ...     {"text": "query", "image": "img2.jpg"}
            ... ])

        Multimodal Format:
            - {"text": "..."} - Text only
            - {"text": "...", "instruction": "..."} - Text with instruction
            - {"image": "path_or_url"} - Image only
            - {"text": "...", "image": "..."} - Combined
            - {"video": "path_or_url", "fps": 1, "max_frames": 10}

        Note:
            Default model outputs 2048-dimensional vectors.
            Multimodal requires server support (Qwen3-VL-Embedding-2B).
        """
        # Validate input is not empty
        if isinstance(input_data, list) and len(input_data) == 0:
            raise ValueError("Input list cannot be empty")

        is_multimodal = isinstance(input_data, dict) or (
            isinstance(input_data, list)
            and len(input_data) > 0
            and isinstance(input_data[0], dict)
        )

        if is_multimodal:
            # Narrow the type for multimodal data
            multimodal_data: Union[dict, list[dict]] = input_data  # type: ignore
            return self._embed_multimodal(multimodal_data, model)
        else:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=input_data,
            )
            if isinstance(input_data, str):
                return response.data[0].embedding
            return [item.embedding for item in response.data]

    def _embed_multimodal(
        self,
        input_data: Union[dict, list[dict]],
        model: Optional[str] = None,
    ) -> Union[list[float], list[list[float]]]:
        """Internal: multimodal embeddings (dict format)."""
        if isinstance(input_data, dict):
            inputs = [input_data]
            single_input = True
        else:
            inputs = input_data
            single_input = False

        response = self.client.embeddings.create(
            model=model or self.embedding_model,
            input=inputs,
        )

        embeddings = [item.embedding for item in response.data]

        if single_input:
            return embeddings[0]
        return embeddings

    def embed_text(
        self,
        text: Union[str, list[str]],
        model: Optional[str] = None,
    ) -> Union[list[float], list[list[float]]]:
        """
        Text-only embeddings.

        Convenience method equivalent to embed() for text.

        Args:
            text: Text string or list of strings
            model: Optional model override

        Returns:
            Single embedding or list of embeddings

        Example:
            >>> client.embed_text("Hello world")
            >>> client.embed_text(["Hello", "World"])
        """
        return self.embed(text, model=model)

    def embed_image(
        self,
        image: Union[str, Path],
        instruction: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Image embeddings for similarity search and retrieval.

        Args:
            image: File path, URL, or base64 encoded image
            instruction: Optional instruction for the model
            model: Optional model override

        Returns:
            Single embedding vector (2048 dimensions)

        Example:
            >>> client.embed_image("/path/to/photo.jpg")
            >>> client.embed_image("https://example.com/img.png")
            >>> client.embed_image(
            ...     "document.jpg",
            ...     instruction="Extract visual features"
            ... )

        Supported: PNG, JPEG, WebP, GIF (path, URL, or base64)
        """
        # Validate file exists if it's a path (not URL or base64)
        image_str = str(image)
        if not image_str.startswith(("http://", "https://", "data:image")):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image}")

        input_dict: dict = {"image": image_str}
        if instruction:
            input_dict["instruction"] = instruction
        result = self.embed(input_dict, model=model)
        return cast(list[float], result)

    def embed_video(
        self,
        video: str,
        fps: Optional[int] = None,
        max_frames: Optional[int] = None,
        instruction: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Video embeddings for video search and analysis.

        Args:
            video: Video file path or URL
            fps: Frames per second sampling (default: 1)
            max_frames: Maximum frames to process
            instruction: Optional instruction for the model
            model: Optional model override

        Returns:
            Single embedding vector (2048 dimensions)

        Example:
            >>> client.embed_video("/path/to/video.mp4")
            >>> client.embed_video("video.mp4", fps=2, max_frames=32)
        """
        input_dict: dict = {"video": video}
        if fps is not None:
            input_dict["fps"] = fps
        if max_frames is not None:
            input_dict["max_frames"] = max_frames
        if instruction:
            input_dict["instruction"] = instruction
        result = self.embed(input_dict, model=model)
        return cast(list[float], result)

    # =========================================================================
    # CHAT
    # =========================================================================

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        """
        Chat completion with streaming support.

        Args:
            message: User message
            system: Optional system prompt
            history: Optional conversation history (OpenAI format)
            max_tokens: Maximum response tokens (default: 1024)
            temperature: Randomness 0.0-1.0 (default: 0.7)
            stream: Return token iterator if True
            model: Optional model override
            **kwargs: Additional API parameters

        Returns:
            Response text, or iterator if streaming

        Example:
            >>> client.chat("What is Python?")
            >>> client.chat("Hello", system="You are a pirate")
            >>>
            >>> # With history
            >>> history = [
            ...     {"role": "user", "content": "My name is Alice"},
            ...     {"role": "assistant", "content": "Hi Alice!"}
            ... ]
            >>> client.chat("What's my name?", history=history)
            >>>
            >>> # Streaming
            >>> for token in client.chat("Write a poem", stream=True):
            ...     print(token, end="")
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=model or self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

        if stream:
            return self._stream_response(response)
        return response.choices[0].message.content  # type: ignore

    def _stream_response(self, stream) -> Iterator[str]:
        """Yield tokens from a streaming response."""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # =========================================================================
    # OCR / VISION
    # =========================================================================

    def ocr(
        self,
        image: Union[str, Path, bytes],
        prompt: str = "Extract all text from this image",
        max_tokens: int = 2048,
        model: Optional[str] = None,
    ) -> str:
        """
        OCR text extraction from images.

        Uses vision-language model to extract text from images.

        Args:
            image: File path, Path object, or raw bytes
            prompt: Instruction for OCR (default: "Extract all text from this image")
            max_tokens: Maximum response tokens (default: 2048)
            model: Optional model override

        Returns:
            Extracted text from the image

        Example:
            >>> client.ocr("/path/to/document.png")
            >>> client.ocr("receipt.jpg", prompt="Extract total amount and date")
            >>>
            >>> # From bytes
            >>> with open("doc.png", "rb") as f:
            ...     text = client.ocr(f.read())

        Supported: PNG, JPEG, WebP
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            suffix = image_path.suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/png")
        else:
            image_bytes = image
            mime_type = "image/png"

        b64_image = base64.b64encode(image_bytes).decode()

        response = self.client.chat.completions.create(
            model=model or self.ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content  # type: ignore

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def list_models(self) -> list[str]:
        """
        List available models from the gateway.

        Returns:
            List of model ID strings

        Example:
            >>> models = client.list_models()
            >>> print(models)  # ['embedding', 'completions', 'ocr', 'asr']
        """
        models = self.client.models.list()
        return [m.id for m in models.data]

    def health(self) -> bool:
        """
        Check if gateway service is healthy.

        Returns:
            True if service is responding, False otherwise

        Example:
            >>> if client.health():
            ...     response = client.chat("Hello")
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    # =========================================================================
    # ASR / AUDIO TRANSCRIPTION
    # =========================================================================

    def transcribe(
        self,
        audio: Union[str, Path],
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Transcribe audio file to text using ASR (Automatic Speech Recognition).

        Supports various audio formats including MP3, WAV, M4A, FLAC.
        Uses the OpenAI-compatible audio transcription API.

        Args:
            audio: Path to audio file (MP3, WAV, M4A, FLAC, etc.)
            model: Optional model name (default: "asr" which routes to Qwen3-ASR-1.7B)
            language: Optional language (e.g., "English", "Chinese", "Japanese")
            prompt: Optional prompt/context to guide transcription
            response_format: Output format - "json", "text", "srt", "vtt", "verbose_json"
            temperature: Sampling temperature 0.0-1.0 (default: 0.0 for deterministic)
            **kwargs: Additional parameters

        Returns:
            Transcribed text from the audio

        Example:
            >>> client = LiteLLMClient()
            >>>
            >>> # Basic transcription
            >>> text = client.transcribe("speech.mp3")
            >>> print(text)
            >>>
            >>> # Specify language
            >>> text = client.transcribe("chinese.mp3", language="Chinese")
            >>>
            >>> # With context prompt
            >>> text = client.transcribe(
            ...     "interview.wav",
            ...     prompt="Technical interview about machine learning"
            ... )

        Supported Formats:
            - MP3 (.mp3)
            - WAV (.wav)
            - M4A (.m4a)
            - FLAC (.flac)
            - OGG (.ogg)
            - And other common audio formats

        Note:
            The ASR service (Qwen3-ASR-1.7B) supports multilingual transcription.
            Specify language for better accuracy on non-English content.
        """
        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        with open(audio_path, "rb") as f:
            # Build parameters dynamically to avoid type issues with None
            params = {
                "file": f,
                "model": model or "asr",  # Uses the "asr" alias from config
                "temperature": temperature,
            }

            # Only add optional parameters if they are provided
            if language is not None:
                params["language"] = language
            if prompt is not None:
                params["prompt"] = prompt
            if response_format != "json":  # Only add if non-default
                params["response_format"] = response_format  # type: ignore
            if kwargs:
                params.update(kwargs)

            response = self.client.audio.transcriptions.create(**params)

        # Return the transcribed text
        return response.text


# =============================================================================
# CONVENIENCE FUNCTIONS (use singleton client)
# =============================================================================

_default_client: Optional[LiteLLMClient] = None


def _with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying network operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2**attempt))
                        continue
                    raise
            return None

        return wrapper

    return decorator


def get_client(base_url: str = "http://localhost:8200/v1") -> LiteLLMClient:
    """
    Get or create singleton client instance.

    Args:
        base_url: Gateway URL (default: http://localhost:8200/v1)

    Returns:
        Singleton LiteLLMClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = LiteLLMClient(base_url=base_url)
    return _default_client


def embed(
    input_data: Union[str, list[str], dict, list[dict]],
) -> Union[list[float], list[list[float]]]:
    """
    Generate embeddings (text, image, or multimodal).

    Uses singleton client. Supports:
    - Text: "Hello world" or ["text1", "text2"]
    - Image: {"image": "path/to/img.jpg"}
    - Multimodal: {"text": "...", "image": "..."}

    Example:
        >>> from litellm_client import embed
        >>> embed("Hello world")
        >>> embed({"image": "photo.jpg"})
        >>> embed({"text": "A photo of", "image": "cat.jpg"})
    """
    return get_client().embed(input_data)


def embed_text(text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
    """Text-only embeddings using singleton client."""
    return get_client().embed_text(text)


def embed_image(
    image: Union[str, Path], instruction: Optional[str] = None
) -> list[float]:
    """
    Image embeddings using singleton client.

    Example:
        >>> from litellm_client import embed_image
        >>> embed_image("/path/to/photo.jpg")
        >>> embed_image("https://example.com/img.png")
    """
    return get_client().embed_image(image, instruction=instruction)


@overload
def chat(message: str, *, stream: Literal[False] = False, **kwargs) -> str: ...


@overload
def chat(message: str, *, stream: Literal[True], **kwargs) -> Iterator[str]: ...


def chat(message: str, **kwargs) -> Union[str, Iterator[str]]:
    """
    Chat completion using singleton client.

    Example:
        >>> from litellm_client import chat
        >>> chat("What is Python?")
        >>> chat("Hello", system="You are a pirate")
    """
    return get_client().chat(message, **kwargs)


def ocr(image: Union[str, Path, bytes], **kwargs) -> str:
    """
    OCR text extraction using singleton client.

    Example:
        >>> from litellm_client import ocr
        >>> ocr("document.png")
        >>> ocr("receipt.jpg", prompt="Extract total amount")
    """
    return get_client().ocr(image, **kwargs)


def transcribe(audio: Union[str, Path], **kwargs) -> str:
    """
    Audio transcription (ASR) using singleton client.

    Transcribe audio files to text using Qwen3-ASR model.
    Supports MP3, WAV, M4A, FLAC and other formats.

    Args:
        audio: Path to audio file
        **kwargs: Optional parameters:
            - language: Language ("English", "Chinese", "Japanese", etc.)
            - prompt: Context prompt for transcription
            - response_format: "json", "text", "srt", "vtt"
            - temperature: 0.0-1.0 (default 0.0)

    Returns:
        Transcribed text from the audio

    Example:
        >>> from litellm_client import transcribe
        >>>
        >>> # Basic transcription
        >>> text = transcribe("speech.mp3")
        >>>
        >>> # With language specification
        >>> text = transcribe("chinese.mp3", language="Chinese")
        >>>
        >>> # With context prompt
        >>> text = transcribe(
        ...     "interview.wav",
        ...     prompt="Technical interview"
        ... )
    """
    return get_client().transcribe(audio, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="LiteLLM Local Client")
    parser.add_argument("--url", default="http://localhost:8200/v1", help="Gateway URL")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with the model")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument("--stream", action="store_true", help="Stream response")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("text", help="Text to embed")

    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="Extract text from image")
    ocr_parser.add_argument("image", help="Path to image file")
    ocr_parser.add_argument("--prompt", default="Extract all text from this image")

    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe audio to text"
    )
    transcribe_parser.add_argument("audio", help="Path to audio file")
    transcribe_parser.add_argument(
        "--language", help="Language (e.g., English, Chinese)"
    )

    # Models command
    subparsers.add_parser("models", help="List available models")

    # Health command
    subparsers.add_parser("health", help="Check service health")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = LiteLLMClient(base_url=args.url)

    if args.command == "chat":
        if args.stream:
            for token in client.chat(args.message, system=args.system, stream=True):
                print(token, end="", flush=True)
            print()
        else:
            print(client.chat(args.message, system=args.system))

    elif args.command == "embed":
        embedding = client.embed(args.text)
        print(f"Embedding ({len(embedding)} dims): {embedding[:5]}...")

    elif args.command == "ocr":
        print(client.ocr(args.image, prompt=args.prompt))

    elif args.command == "transcribe":
        text = client.transcribe(args.audio, language=args.language)
        print(text)

    elif args.command == "models":
        for model in client.list_models():
            print(f"  - {model}")

    elif args.command == "health":
        if client.health():
            print("✅ Service is healthy")
        else:
            print("❌ Service is not responding")
            sys.exit(1)
