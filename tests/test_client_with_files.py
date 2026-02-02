#!/usr/bin/env python3
"""
Test script for litellm_client with image and audio files.

This script demonstrates how to use the client with the test files
in the tests/ directory.

Test files available:
- tests/Screenshot 2026-01-06 at 19.15.45.png (image for OCR/embeddings)
- tests/000981_jfk-space-race-speech-59951.mp3 (audio for transcription)

Usage:
    python test_client_with_files.py
    pytest test_client_with_files.py -v
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litellm_client import LiteLLMClient, embed, embed_image, ocr, chat
from pathlib import Path
import pytest


@pytest.fixture
def client():
    """Create a LiteLLMClient instance."""
    return LiteLLMClient()


@pytest.fixture
def image_path():
    """Get the test image path."""
    path = Path(__file__).parent / "Screenshot 2026-01-06 at 19.15.45.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture
def audio_path():
    """Get the test audio path."""
    path = Path(__file__).parent / "000981_jfk-space-race-speech-59951.mp3"
    if not path.exists():
        pytest.skip(f"Test audio not found: {path}")
    return path


def test_with_image(client, image_path):
    """Test OCR and image embeddings with the screenshot."""
    # Check if service is running
    assert client.health(), (
        "Gateway not running. Start services with: docker-compose up -d"
    )

    print("✅ Gateway is healthy\n")

    print(f"Testing with image: {image_path}")
    print("-" * 60)

    # Test 1: OCR - Extract text from image
    print("\n1. Testing OCR (text extraction)...")
    text = client.ocr(image_path, prompt="Extract all text visible in this image")
    assert text, "OCR returned empty result"
    assert len(text) > 0, "OCR returned empty text"
    print(f"✅ OCR Result:\n{text[:500]}...")  # Show first 500 chars

    # Test 2: Image embedding
    print("\n2. Testing image embedding...")
    embedding = client.embed_image(str(image_path))
    assert embedding is not None, "Image embedding returned None"
    assert len(embedding) > 0, "Image embedding returned empty list"
    print(f"✅ Image embedding shape: {len(embedding)} dimensions")
    print(f"   Sample values: {embedding[:5]}")

    # Test 3: Image with instruction
    print("\n3. Testing image embedding with instruction...")
    embedding = client.embed_image(
        str(image_path),
        instruction="Extract visual features for document classification",
    )
    assert embedding is not None, "Image embedding with instruction returned None"
    assert len(embedding) > 0, "Image embedding with instruction returned empty list"
    print(f"✅ Image embedding with instruction: {len(embedding)} dimensions")


def test_with_audio(client, audio_path):
    """Test audio transcription with the JFK speech."""
    # Check if service is running
    assert client.health(), "Gateway not running"

    print(f"\nTesting with audio: {audio_path}")
    print("-" * 60)

    # Note: The current client doesn't have a transcribe method
    # but we can show how it would be used with the underlying client
    print("\nAudio transcription (via whisper model)...")
    print(f"File: {audio_path}")
    print(f"Size: {audio_path.stat().st_size / 1024:.1f} KB")

    # This would require adding a transcribe method to the client
    print("\n💡 To transcribe audio, use the 'asr' model directly:")
    print("   client.client.audio.transcriptions.create(")
    print("       model='asr',")
    print("       file=open('audio.mp3', 'rb')")
    print("   )")


def test_text_embeddings(client):
    """Test text embeddings."""
    assert client.health(), "Gateway not running"

    print("\nTesting text embeddings:")
    print("-" * 60)

    # Test single text
    print("\n1. Single text embedding...")
    embedding = client.embed_text("Hello world")
    assert embedding is not None, "Text embedding returned None"
    assert len(embedding) > 0, "Text embedding returned empty list"
    print(f"✅ Embedding: {len(embedding)} dimensions")

    # Test batch
    print("\n2. Batch text embeddings...")
    texts = ["First document", "Second document", "Third document"]
    embeddings = client.embed_text(texts)
    assert embeddings is not None, "Batch embedding returned None"
    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
    assert len(embeddings[0]) > 0, "First embedding is empty"
    print(f"✅ Batch: {len(embeddings)} embeddings, each {len(embeddings[0])} dims")

    # Test multimodal (text + image) - only if image exists
    image_path = Path(__file__).parent / "Screenshot 2026-01-06 at 19.15.45.png"
    if image_path.exists():
        print("\n3. Multimodal (text + image) embedding...")
        embedding = client.embed(
            {"text": "This is a screenshot showing", "image": str(image_path)}
        )
        assert embedding is not None, "Multimodal embedding returned None"
        assert len(embedding) > 0, "Multimodal embedding is empty"
        print(f"✅ Multimodal embedding: {len(embedding)} dimensions")
    else:
        print("\n3. Skipping multimodal test (image not found)")


def test_chat(client):
    """Test chat completion."""
    assert client.health(), "Gateway not running"

    print("\nTesting chat completion:")
    print("-" * 60)

    print("\n1. Simple chat...")
    response = client.chat("What is 2+2?")
    assert response is not None, "Chat returned None"
    assert len(response) > 0, "Chat returned empty response"
    print(f"✅ Response: {response[:200]}...")

    print("\n2. Chat with system prompt...")
    response = client.chat("Tell me a joke", system="You are a helpful assistant")
    assert response is not None, "Chat with system prompt returned None"
    assert len(response) > 0, "Chat with system prompt returned empty response"
    print(f"✅ Response: {response[:200]}...")

    print("\n3. Streaming chat...")
    print("Response: ", end="")
    tokens = []
    for token in client.chat("Say hello", stream=True):
        print(token, end="")
        tokens.append(token)
    print()
    assert len(tokens) > 0, "Streaming returned no tokens"
    print("✅ Streaming completed")


def test_convenience_functions():
    """Test convenience functions (singleton client)."""
    print("\nTesting convenience functions:")
    print("-" * 60)

    # Test embed
    print("\n1. embed() - unified embedding function...")
    result = embed("Hello world")
    assert result is not None, "embed() returned None"
    assert len(result) > 0, "embed() returned empty result"
    print(f"✅ Text embedding: {len(result)} dimensions")

    # Test embed_image - only if image exists
    image_path = Path(__file__).parent / "Screenshot 2026-01-06 at 19.15.45.png"
    if image_path.exists():
        print("\n2. embed_image() - image embedding...")
        result = embed_image(str(image_path))
        assert result is not None, "embed_image() returned None"
        assert len(result) > 0, "embed_image() returned empty result"
        print(f"✅ Image embedding: {len(result)} dimensions")
    else:
        print("\n2. Skipping embed_image() test (image not found)")

    # Test chat
    print("\n3. chat() - chat function...")
    result = chat("What is Python?")
    assert result is not None, "chat() returned None"
    assert len(result) > 0, "chat() returned empty result"
    print(f"✅ Chat response: {result[:100]}...")

    # Test ocr - only if image exists
    if image_path.exists():
        print("\n4. ocr() - OCR function...")
        result = ocr(str(image_path))
        assert result is not None, "ocr() returned None"
        assert len(result) > 0, "ocr() returned empty result"
        print(f"✅ OCR result: {result[:200]}...")


def main():
    """Run all tests."""
    print("=" * 70)
    print("LiteLLM Client Test Suite")
    print("=" * 70)

    # Check if client can be imported
    print("\n✅ Client imported successfully")

    # Test health
    client = LiteLLMClient()
    if not client.health():
        print("\n" + "=" * 70)
        print("⚠️  SERVICES NOT RUNNING")
        print("=" * 70)
        print("\nTo run these tests, start the services:")
        print("  cd /home/simon/Desktop/projects/litellm_local")
        print("  docker-compose up -d")
        print("\nAvailable test files:")
        print("  - tests/Screenshot 2026-01-06 at 19.15.45.png")
        print("  - tests/000981_jfk-space-race-speech-59951.mp3")
        print("\nTest categories:")
        print("  1. Image: OCR, image embeddings, multimodal embeddings")
        print("  2. Audio: Transcription (requires ASR service)")
        print("  3. Text: Text embeddings, chat completion")
        return 1

    # Get test file paths
    test_dir = Path(__file__).parent
    image_path = test_dir / "Screenshot 2026-01-06 at 19.15.45.png"
    audio_path = test_dir / "000981_jfk-space-race-speech-59951.mp3"

    # Run tests
    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)

    if image_path.exists():
        test_with_image(client, image_path)
    else:
        print("\n⚠️ Skipping test_with_image (image not found)")

    if audio_path.exists():
        test_with_audio(client, audio_path)
    else:
        print("\n⚠️ Skipping test_with_audio (audio not found)")

    test_text_embeddings(client)
    test_chat(client)
    test_convenience_functions()

    print("\n" + "=" * 70)
    print("Tests Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
