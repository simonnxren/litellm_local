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
"""

import sys
sys.path.insert(0, '/Users/simonren/Desktop/projects/litellm_local')

from litellm_client import LiteLLMClient, embed, embed_image, ocr, chat
from pathlib import Path


def test_with_image():
    """Test OCR and image embeddings with the screenshot."""
    client = LiteLLMClient()
    
    # Check if service is running
    if not client.health():
        print("❌ Gateway not running. Start services with: docker-compose up -d")
        return False
    
    print("✅ Gateway is healthy\n")
    
    # Test file path
    image_path = Path("tests/Screenshot 2026-01-06 at 19.15.45.png")
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"Testing with image: {image_path}")
    print("-" * 60)
    
    # Test 1: OCR - Extract text from image
    print("\n1. Testing OCR (text extraction)...")
    try:
        text = client.ocr(
            image_path,
            prompt="Extract all text visible in this image"
        )
        print(f"✅ OCR Result:\n{text[:500]}...")  # Show first 500 chars
    except Exception as e:
        print(f"❌ OCR failed: {e}")
    
    # Test 2: Image embedding
    print("\n2. Testing image embedding...")
    try:
        embedding = client.embed_image(str(image_path))
        print(f"✅ Image embedding shape: {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Image embedding failed: {e}")
    
    # Test 3: Image with instruction
    print("\n3. Testing image embedding with instruction...")
    try:
        embedding = client.embed_image(
            str(image_path),
            instruction="Extract visual features for document classification"
        )
        print(f"✅ Image embedding with instruction: {len(embedding)} dimensions")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    return True


def test_with_audio():
    """Test audio transcription with the JFK speech."""
    client = LiteLLMClient()
    
    # Check if service is running
    if not client.health():
        print("❌ Gateway not running")
        return False
    
    audio_path = Path("tests/000981_jfk-space-race-speech-59951.mp3")
    
    if not audio_path.exists():
        print(f"❌ Audio not found: {audio_path}")
        return False
    
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
    
    return True


def test_text_embeddings():
    """Test text embeddings."""
    client = LiteLLMClient()
    
    if not client.health():
        print("❌ Gateway not running")
        return False
    
    print("\nTesting text embeddings:")
    print("-" * 60)
    
    # Test single text
    print("\n1. Single text embedding...")
    try:
        embedding = client.embed_text("Hello world")
        print(f"✅ Embedding: {len(embedding)} dimensions")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test batch
    print("\n2. Batch text embeddings...")
    try:
        texts = ["First document", "Second document", "Third document"]
        embeddings = client.embed_text(texts)
        print(f"✅ Batch: {len(embeddings)} embeddings, each {len(embeddings[0])} dims")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test multimodal (text + image)
    image_path = Path("tests/Screenshot 2026-01-06 at 19.15.45.png")
    if image_path.exists():
        print("\n3. Multimodal (text + image) embedding...")
        try:
            embedding = client.embed({
                "text": "This is a screenshot showing",
                "image": str(image_path)
            })
            print(f"✅ Multimodal embedding: {len(embedding)} dimensions")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return True


def test_chat():
    """Test chat completion."""
    client = LiteLLMClient()
    
    if not client.health():
        print("❌ Gateway not running")
        return False
    
    print("\nTesting chat completion:")
    print("-" * 60)
    
    print("\n1. Simple chat...")
    try:
        response = client.chat("What is 2+2?")
        print(f"✅ Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n2. Chat with system prompt...")
    try:
        response = client.chat(
            "Tell me a joke",
            system="You are a helpful assistant"
        )
        print(f"✅ Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n3. Streaming chat...")
    try:
        print("Response: ", end="")
        for token in client.chat("Say hello", stream=True):
            print(token, end="")
        print()
        print("✅ Streaming completed")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    return True


def test_convenience_functions():
    """Test convenience functions (singleton client)."""
    print("\nTesting convenience functions:")
    print("-" * 60)
    
    # Test embed
    print("\n1. embed() - unified embedding function...")
    try:
        result = embed("Hello world")
        print(f"✅ Text embedding: {len(result)} dimensions")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test embed_image
    image_path = Path("tests/Screenshot 2026-01-06 at 19.15.45.png")
    if image_path.exists():
        print("\n2. embed_image() - image embedding...")
        try:
            result = embed_image(str(image_path))
            print(f"✅ Image embedding: {len(result)} dimensions")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Test chat
    print("\n3. chat() - chat function...")
    try:
        result = chat("What is Python?")
        print(f"✅ Chat response: {result[:100]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test ocr
    if image_path.exists():
        print("\n4. ocr() - OCR function...")
        try:
            result = ocr(str(image_path))
            print(f"✅ OCR result: {result[:200]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return True


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
        print("  cd /Users/simonren/Desktop/projects/litellm_local")
        print("  docker-compose up -d")
        print("\nAvailable test files:")
        print("  - tests/Screenshot 2026-01-06 at 19.15.45.png")
        print("  - tests/000981_jfk-space-race-speech-59951.mp3")
        print("\nTest categories:")
        print("  1. Image: OCR, image embeddings, multimodal embeddings")
        print("  2. Audio: Transcription (requires ASR service)")
        print("  3. Text: Text embeddings, chat completion")
        return 1
    
    # Run tests
    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)
    
    test_with_image()
    test_with_audio()
    test_text_embeddings()
    test_chat()
    test_convenience_functions()
    
    print("\n" + "=" * 70)
    print("Tests Complete")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
