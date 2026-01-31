"""
LiteLLM Local - Python Client SDK

A simple wrapper around the OpenAI client for easy interaction with
local vLLM services via the LiteLLM gateway.

Usage:
    from litellm_client import LiteLLMClient

    client = LiteLLMClient()

    # Embeddings
    embeddings = client.embed("Hello world")

    # Chat
    response = client.chat("What is 2+2?")

    # OCR
    text = client.ocr("image.png")
"""

from typing import Optional, Union, Iterator
from pathlib import Path
import base64


class LiteLLMClient:
    """Client SDK for LiteLLM Local services."""

    def __init__(
        self,
        base_url: str = "http://localhost:8200/v1",
        api_key: str = "dummy",
        embedding_model: str = "qwen3-embedding-0.6b",
        chat_model: str = "qwen3-8b-fp8",
        ocr_model: str = "hunyuan-ocr",
    ):
        """
        Initialize the LiteLLM client.

        Args:
            base_url: LiteLLM gateway URL (default: http://localhost:8200/v1)
            api_key: API key (default: "dummy" for local deployment)
            embedding_model: Model for embeddings
            chat_model: Model for chat completions
            ocr_model: Model for OCR/vision tasks
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
        text: Union[str, list[str]],
        model: Optional[str] = None,
    ) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings for text.

        Args:
            text: Single string or list of strings to embed
            model: Override default embedding model

        Returns:
            Single embedding vector or list of vectors (1024 dimensions each)
        """
        response = self.client.embeddings.create(
            model=model or self.embedding_model,
            input=text,
        )

        if isinstance(text, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]

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
        Send a chat message and get a response.

        Args:
            message: User message
            system: Optional system prompt
            history: Optional conversation history
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            stream: If True, returns a generator of tokens
            model: Override default chat model
            **kwargs: Additional parameters passed to the API

        Returns:
            Response text, or iterator of tokens if streaming
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
        return response.choices[0].message.content

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
        Extract text from an image using OCR.

        Args:
            image: Path to image file, or raw bytes
            prompt: Instruction for the OCR model
            max_tokens: Maximum tokens in response
            model: Override default OCR model

        Returns:
            Extracted text from the image
        """
        # Load and encode image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            # Detect MIME type
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

        return response.choices[0].message.content

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def list_models(self) -> list[str]:
        """List available models."""
        models = self.client.models.list()
        return [m.id for m in models.data]

    def health(self) -> bool:
        """Check if the service is healthy."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_client: Optional[LiteLLMClient] = None


def get_client(base_url: str = "http://localhost:8200/v1") -> LiteLLMClient:
    """Get or create default client instance."""
    global _default_client
    if _default_client is None:
        _default_client = LiteLLMClient(base_url=base_url)
    return _default_client


def embed(text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
    """Quick embedding function using default client."""
    return get_client().embed(text)


def chat(message: str, **kwargs) -> str:
    """Quick chat function using default client."""
    return get_client().chat(message, **kwargs)


def ocr(image: Union[str, Path, bytes], **kwargs) -> str:
    """Quick OCR function using default client."""
    return get_client().ocr(image, **kwargs)


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

    elif args.command == "models":
        for model in client.list_models():
            print(f"  - {model}")

    elif args.command == "health":
        if client.health():
            print("✅ Service is healthy")
        else:
            print("❌ Service is not responding")
            sys.exit(1)
