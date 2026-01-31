"""Comprehensive tests for vLLM models via LiteLLM Gateway."""

import pytest
import requests
from openai import OpenAI

BASE_URL = "http://localhost:8200"


@pytest.fixture
def client():
    """OpenAI client fixture."""
    return OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")


class TestEmbeddings:
    """Test embedding model."""

    def test_single_embedding(self, client):
        """Test single text embedding."""
        response = client.embeddings.create(
            model="qwen3-embedding-0.6b",
            input="Hello world"
        )
        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 1024

    def test_batch_embeddings(self, client):
        """Test batch embeddings."""
        response = client.embeddings.create(
            model="qwen3-embedding-0.6b",
            input=["Text 1", "Text 2", "Text 3"]
        )
        assert len(response.data) == 3
        assert all(len(d.embedding) == 1024 for d in response.data)


class TestCompletions:
    """Test completion model."""

    def test_simple_completion(self, client):
        """Test basic text completion."""
        response = client.chat.completions.create(
            model="qwen3-8b-fp8",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=20
        )
        assert len(response.choices) == 1
        assert len(response.choices[0].message.content) > 0
        assert response.usage.total_tokens > 0

    def test_completion_with_parameters(self, client):
        """Test completion with temperature and stop."""
        response = client.chat.completions.create(
            model="qwen3-8b-fp8",
            messages=[{"role": "user", "content": "Count to 5"}],
            max_tokens=30,
            temperature=0.7,
            stop=["\n"]
        )
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason in ["stop", "length"]

    def test_streaming_completion(self, client):
        """Test streaming response."""
        stream = client.chat.completions.create(
            model="qwen3-8b-fp8",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20,
            stream=True
        )
        chunks = list(stream)
        assert len(chunks) > 0
        assert any(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)


class TestOCR:
    """Test OCR model."""

    @pytest.mark.skipif(
        not __import__('os').path.exists('assets/Screenshot From 2025-11-02 16-41-52.png'),
        reason="Test image not found - place test image in assets/ directory"
    )
    def test_ocr_image(self, client):
        """Test OCR on image.
        
        Requires a test image at: assets/Screenshot From 2025-11-02 16-41-52.png
        """
        import base64
        with open('assets/Screenshot From 2025-11-02 16-41-52.png', 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        
        response = client.chat.completions.create(
            model="hunyuan-ocr",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Extract text"}
                ]
            }],
            max_tokens=100
        )
        assert len(response.choices) == 1
        assert len(response.choices[0].message.content) > 0


class TestHealth:
    """Test service health."""

    def test_litellm_health(self):
        """Test LiteLLM gateway is healthy."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200

    def test_models_available(self, client):
        """Test models are listed."""
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        assert "qwen3-embedding-0.6b" in model_ids
        assert "qwen3-8b-fp8" in model_ids
        assert "hunyuan-ocr" in model_ids
