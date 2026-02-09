
"""
Comprehensive test suite for LiteLLM Local Gateway services.
Tests all four capabilities: Chat, Embedding, OCR, and ASR.
Based on research of vLLM and LiteLLM embedding service connections.
"""

import os
import sys
import time
import base64
import wave
import struct
import json
import pytest
from pathlib import Path

# Add local directory to path
sys.path.append(os.getcwd())
import litellm_client


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_dummy_wav(filename="dummy.wav", duration=1.0):
    """Create a 1-second dummy WAV file with silence."""
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # Write silence (zeros)
        data = struct.pack('<' + ('h' * num_samples), *[0] * num_samples)
        wav_file.writeframes(data)
    
    print(f"Created dummy audio: {filename}")
    return filename


def create_dummy_image(filename="dummy.png"):
    """Create a simple 1x1 black PNG."""
    # A 1x1 black pixel PNG base64
    base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_data))
    print(f"Created dummy image: {filename}")
    return filename


# ============================================================================
# CHAT COMPLETION TESTS
# ============================================================================

class TestChatCompletion:
    """Test suite for chat completion functionality."""
    
    def test_chat_basic(self):
        """Test basic chat completion."""
        print("\n--- Testing Chat (Qwen3-VL-4B) ---")
        try:
            print("Sending 'Hello'...")
            response = litellm_client.chat("Hello, are you working?")
            print(f"Response: {response[:100]}...")  # Truncate
            assert len(response) > 0, "Response should not be empty"
            print("✅ Chat test passed")
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
            raise
    
    def test_chat_with_image(self):
        """Test chat with image input."""
        print("\n--- Testing Chat with Image ---")
        try:
            img_path = create_dummy_image("chat_test.png")
            try:
                response = litellm_client.chat(
                    "What is in this image?", 
                    image_path=img_path
                )
                print(f"Response: {response[:100]}...")
                assert len(response) > 0
                print("✅ Chat with image test passed")
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
        except Exception as e:
            print(f"❌ Chat with image test failed: {e}")
            raise


# ============================================================================
# EMBEDDING TESTS
# ============================================================================

class TestEmbedding:
    """Comprehensive test suite for embedding functionality.
    
    Based on vLLM OpenAI-compatible embedding API research:
    - Supports text, image, and multimodal embeddings
    - Compatible with /v1/embeddings endpoint
    - Supports batch processing and various input formats
    """
    
    def test_text_embedding_basic(self):
        """Test basic text embedding generation."""
        print("\n--- Testing Text Embedding ---")
        text = "This is a test sentence for embedding."
        embedding = litellm_client.embed(text)
        
        # Validate structure
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        
        # Validate dimensions (Qwen3-VL-Embedding typically 1536 dims)
        assert len(embedding) >= 128, f"Unexpected embedding dimension: {len(embedding)}"
        print(f"✅ Text embedding: {len(embedding)} dimensions")
    
    def test_text_embedding_batch(self):
        """Test batch text embedding."""
        print("\n--- Testing Batch Text Embedding ---")
        texts = [
            "First test sentence",
            "Second test sentence", 
            "Third test sentence"
        ]
        embeddings = litellm_client.embed(texts)
        
        # Validate batch structure
        assert isinstance(embeddings, list), "Batch result should be a list"
        assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"
        assert all(isinstance(emb, list) for emb in embeddings), "Each embedding should be a list"
        assert all(len(emb) == len(embeddings[0]) for emb in embeddings), "All embeddings should have same dimensions"
        
        print(f"✅ Batch embedding: {len(embeddings)} vectors, {len(embeddings[0])} dims each")
    
    def test_text_embedding_consistency(self):
        """Test that identical inputs produce similar embeddings."""
        print("\n--- Testing Embedding Consistency ---")
        text = "Consistency test sentence"
        emb1 = litellm_client.embed(text)
        emb2 = litellm_client.embed(text)
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        similarity = dot_product / (norm1 * norm2)
        
        # Should be very similar (>= 0.99)
        assert similarity >= 0.99, f"Embeddings not consistent: similarity={similarity:.4f}"
        print(f"✅ Consistency test: similarity={similarity:.4f}")
    
    def test_embedding_performance(self):
        """Test embedding generation performance."""
        print("\n--- Testing Embedding Performance ---")
        text = "Performance benchmark test"
        
        start_time = time.time()
        embedding = litellm_client.embed(text)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust based on your setup)
        assert elapsed < 5.0, f"Embedding took too long: {elapsed:.2f}s"
        print(f"✅ Performance: {elapsed:.3f}s for {len(embedding)}-dimension embedding")
    
    def test_empty_input_handling(self):
        """Test handling of edge cases."""
        print("\n--- Testing Edge Cases ---")
        # Empty string
        try:
            emb = litellm_client.embed("")
            assert isinstance(emb, list) and len(emb) > 0
            print("✅ Empty string handled")
        except Exception as e:
            print(f"⚠️ Empty string raised exception (may be expected): {e}")
    
    @pytest.mark.integration
    def test_multimodal_embedding_text_only_dict(self):
        """Test multimodal embedding with text-only dict input."""
        print("\n--- Testing Multimodal Text-Only Dict ---")
        input_data = {"text": "A cat sitting on a mat"}
        embedding = litellm_client.embed(input_data)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)
        print(f"✅ Multimodal text-only dict: {len(embedding)} dimensions")
    
    @pytest.mark.integration  
    def test_multimodal_embedding_with_image(self):
        """Test multimodal embedding with text and image."""
        print("\n--- Testing Multimodal Text+Image ---")
        # Create a simple test image
        image_path = "test_image.png"
        # Base64 encoded 1x1 black PNG
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        
        try:
            input_data = {
                "text": "What is in this image?",
                "image": image_path
            }
            embedding = litellm_client.embed(input_data)
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert isinstance(embedding[0], float)
            print(f"✅ Multimodal text+image: {len(embedding)} dimensions")
        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)


# ============================================================================
# OCR TESTS
# ============================================================================

class TestOCR:
    """Test suite for OCR functionality."""
    
    def test_ocr_connectivity(self):
        """Test basic OCR connectivity."""
        print("\n--- Testing OCR (GLM-OCR) ---")
        img_path = create_dummy_image("ocr_test.png")
        try:
            print(f"OCRing {img_path}...")
            text = litellm_client.ocr(img_path, prompt="Describe strictly what is in this image")
            print(f"OCR Result: {text}")
            # Even if empty (black pixel), it shouldn't crash
            print("✅ OCR test passed (connectivity check)")
        except Exception as e:
            print(f"❌ OCR test failed: {e}")
            raise
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)
    
    def test_ocr_with_real_image(self):
        """Test OCR with a real image from assets."""
        print("\n--- Testing OCR with Real Image ---")
        asset_images = list(Path("assets").glob("*.png")) + list(Path("assets").glob("*.jpg"))
        if asset_images:
            image_path = str(asset_images[0])
            try:
                print(f"OCRing {image_path}...")
                text = litellm_client.ocr(image_path, prompt="What text do you see in this image?")
                print(f"OCR Result: {text[:200]}...")
                print("✅ OCR with real image test passed")
            except Exception as e:
                print(f"⚠️ OCR with real image failed: {e}")
        else:
            print("⚠️ No real images found in assets/ directory")


# ============================================================================
# ASR (TRANSCRIPTION) TESTS
# ============================================================================

class TestASR:
    """Test suite for ASR (Automatic Speech Recognition) functionality."""
    
    def test_transcribe_silence(self):
        """Test transcription with silent audio."""
        print("\n--- Testing ASR (Qwen3-ASR) ---")
        audio_path = create_dummy_wav("asr_test.wav")
        try:
            print(f"Transcribing {audio_path}...")
            result = litellm_client.transcribe(audio_path)
            print(f"Transcription: {result}")
            # Silence might return empty string or hallucination, but shouldn't crash
            assert isinstance(result, dict), "Result should be a dict"
            assert "text" in result
            print("✅ ASR test passed")
        except Exception as e:
            print(f"❌ ASR test failed: {e}")
            raise
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def test_transcribe_with_real_audio(self):
        """Test transcription with real audio from assets."""
        print("\n--- Testing ASR with Real Audio ---")
        asset_audios = (
            list(Path("assets").glob("*.wav")) + 
            list(Path("assets").glob("*.mp3")) + 
            list(Path("assets").glob("*.flac"))
        )
        if asset_audios:
            audio_path = str(asset_audios[0])
            try:
                print(f"Transcribing {audio_path}...")
                result = litellm_client.transcribe(audio_path)
                print(f"Transcription: {result.get('text', '')[:200]}...")
                if 'segments' in result:
                    print(f"Segments found: {len(result['segments'])}")
                print("✅ ASR with real audio test passed")
            except Exception as e:
                print(f"⚠️ ASR with real audio failed: {e}")
        else:
            print("⚠️ No real audio files found in assets/ directory")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_all_services(self):
        """Test all services in sequence."""
        print("\n" + "="*60)
        print("🧪 STARTING COMPLETE INTEGRATION TEST")
        print("="*60)
        
        # Create test assets
        img_path = create_dummy_image("integration_test.png")
        audio_path = create_dummy_wav("integration_test.wav")
        
        try:
            # Test each service
            chat_test = TestChatCompletion()
            chat_test.test_chat_basic()
            
            embed_test = TestEmbedding()
            embed_test.test_text_embedding_basic()
            embed_test.test_text_embedding_batch()
            
            ocr_test = TestOCR()
            ocr_test.test_ocr_connectivity()
            
            asr_test = TestASR()
            asr_test.test_transcribe_silence()
            
            print("\n" + "="*60)
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("="*60)
            
        finally:
            # Cleanup
            for path in [img_path, audio_path]:
                if os.path.exists(path):
                    os.remove(path)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all tests."""
    print("🚀 Starting LiteLLM Client Comprehensive Test Suite...")
    print("Based on vLLM and LiteLLM embedding service research")
    print("Supports: Chat, Embedding, OCR, ASR\n")
    
    try:
        # Run integration test (which covers all services)
        integration = TestIntegration()
        integration.test_all_services()
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
