"""
Gateway smoke tests — validates all services through the LiteLLM gateway.

Uses real media files from assets/ for meaningful test validation:
  - Screenshots (PNG) for vision chat and OCR tests
  - Audio files (FLAC/MP3) for ASR transcription tests

Requires:
  1. vLLM services running (docker-compose.vllm_cu130_nightly.yml)
  2. Gateway running        (docker-compose.gateway.yml)
  3. Real media files in    assets/

Run:
    pytest test_gateway.py -v
    python test_gateway.py
"""

import os
import sys
import json
import pytest
from pathlib import Path

sys.path.append(os.getcwd())
import litellm_client

GATEWAY_URL = litellm_client.GATEWAY_URL

# ============================================================================
# ASSET PATHS
# ============================================================================

ASSETS_DIR = Path(__file__).parent / "assets"

# Pick one screenshot for vision/OCR tests
SAMPLE_IMAGE = str(ASSETS_DIR / "Screenshot From 2025-11-28 00-38-10.png")
# Additional image for multi-image tests
SAMPLE_IMAGE_2 = str(ASSETS_DIR / "Screenshot From 2025-11-04 23-09-53.png")
# Short audio clip for ASR tests
SAMPLE_AUDIO_FLAC = str(ASSETS_DIR / "sample1.flac")
# Longer speech for ASR content validation
SAMPLE_AUDIO_JFK = str(ASSETS_DIR / "000981_jfk-space-race-speech-59951.mp3")


def _asset_exists(path: str) -> bool:
    return Path(path).is_file()


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_image():
    """Real screenshot PNG from assets/."""
    assert _asset_exists(SAMPLE_IMAGE), f"Missing asset: {SAMPLE_IMAGE}"
    return SAMPLE_IMAGE


@pytest.fixture
def sample_image_2():
    """Second screenshot PNG from assets/."""
    assert _asset_exists(SAMPLE_IMAGE_2), f"Missing asset: {SAMPLE_IMAGE_2}"
    return SAMPLE_IMAGE_2


@pytest.fixture
def sample_audio():
    """Real FLAC audio from assets/."""
    assert _asset_exists(SAMPLE_AUDIO_FLAC), f"Missing asset: {SAMPLE_AUDIO_FLAC}"
    return SAMPLE_AUDIO_FLAC


@pytest.fixture
def jfk_audio():
    """JFK speech MP3 from assets/."""
    assert _asset_exists(SAMPLE_AUDIO_JFK), f"Missing asset: {SAMPLE_AUDIO_JFK}"
    return SAMPLE_AUDIO_JFK


# ============================================================================
# GATEWAY HEALTH
# ============================================================================

class TestGatewayHealth:
    """Validate the gateway itself is reachable."""

    def test_gateway_liveliness(self):
        """GET /health/liveliness should return 200."""
        import urllib.request
        resp = urllib.request.urlopen(f"{GATEWAY_URL}/health/liveliness", timeout=5)
        assert resp.status == 200

    def test_gateway_models_list(self):
        """GET /v1/models should list all registered models."""
        import urllib.request
        resp = urllib.request.urlopen(f"{GATEWAY_URL}/v1/models", timeout=5)
        body = json.loads(resp.read().decode())
        ids = {m["id"] for m in body["data"]}
        # Friendly aliases must be present
        for alias in ("chat", "ocr", "embedding", "asr"):
            assert alias in ids, f"Model alias '{alias}' not found in gateway models"

    def test_health_check_function(self):
        """litellm_client.health_check() in gateway mode."""
        status = litellm_client.health_check()
        assert "gateway" in status
        assert status["gateway"] is True


# ============================================================================
# CHAT THROUGH GATEWAY
# ============================================================================

class TestGatewayChat:

    def test_basic_chat(self):
        """Simple text chat through gateway."""
        resp = litellm_client.chat("Say hello in one word.")
        assert isinstance(resp, str) and len(resp) > 0

    def test_chat_streaming(self):
        """Streaming chat through gateway."""
        chunks = list(litellm_client.chat("Count to 3.", stream=True))
        full = "".join(chunks)
        assert len(full) > 0
        # Should contain at least some numbers
        assert any(c.isdigit() for c in full), f"Expected digits in: {full[:100]}"

    def test_chat_with_image(self, sample_image):
        """Vision chat with a real screenshot through gateway."""
        resp = litellm_client.chat(
            "Describe what you see in this screenshot in one sentence.",
            image=sample_image,
        )
        assert isinstance(resp, str) and len(resp) > 10


# ============================================================================
# OCR THROUGH GATEWAY
# ============================================================================

class TestGatewayOCR:

    def test_ocr(self, sample_image):
        """OCR extraction on a real screenshot through gateway."""
        text = litellm_client.ocr(sample_image, prompt="Extract all visible text from this screenshot.")
        assert isinstance(text, str) and len(text) > 0

    def test_ocr_second_image(self, sample_image_2):
        """OCR on a different screenshot to validate consistency."""
        text = litellm_client.ocr(sample_image_2)
        assert isinstance(text, str) and len(text) > 0


# ============================================================================
# EMBEDDINGS THROUGH GATEWAY
# ============================================================================

class TestGatewayEmbedding:

    def test_single_text(self):
        """Single text embedding through gateway."""
        emb = litellm_client.embed("Test sentence for embedding.")
        assert isinstance(emb, list) and len(emb) >= 128
        assert all(isinstance(x, float) for x in emb)

    def test_batch_text(self):
        """Batch text embedding through gateway."""
        embs = litellm_client.embed(["Hello", "World"])
        assert len(embs) == 2
        assert len(embs[0]) == len(embs[1])

    def test_consistency(self):
        """Same input should produce near-identical embeddings."""
        a = litellm_client.embed("Deterministic test")
        b = litellm_client.embed("Deterministic test")
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        sim = dot / (na * nb)
        assert sim >= 0.99, f"Cosine similarity = {sim:.4f}"


# ============================================================================
# ASR THROUGH GATEWAY
# ============================================================================

class TestGatewayASR:

    def test_transcribe_audio(self, sample_audio):
        """Transcribe real FLAC audio through gateway."""
        result = litellm_client.transcribe(sample_audio)
        assert isinstance(result, dict) and "text" in result
        assert len(result["text"]) > 0, "Transcription should produce non-empty text"

    def test_transcribe_jfk_speech(self, jfk_audio):
        """Transcribe JFK speech — should contain recognizable words."""
        result = litellm_client.transcribe(jfk_audio)
        assert isinstance(result, dict) and "text" in result
        text = result["text"].lower()
        assert len(text) > 20, f"JFK speech transcription too short: {text}"


# ============================================================================
# INTEGRATION
# ============================================================================

class TestGatewayIntegration:
    """End-to-end: hit every service in one test with real assets."""

    def test_all_services(self, sample_image, sample_audio):
        results = {}

        # Chat
        results["chat"] = litellm_client.chat("Ping")
        assert len(results["chat"]) > 0

        # OCR
        results["ocr"] = litellm_client.ocr(sample_image)
        assert isinstance(results["ocr"], str) and len(results["ocr"]) > 0

        # Embedding
        results["embed"] = litellm_client.embed("test")
        assert len(results["embed"]) >= 128

        # ASR
        results["asr"] = litellm_client.transcribe(sample_audio)
        assert "text" in results["asr"]
        assert len(results["asr"]["text"]) > 0

        print("\nGateway integration — all services OK")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def main():
    print(f"Gateway: {GATEWAY_URL}")
    print(f"Mode:    gateway\n")

    # Verify assets exist
    missing = []
    for label, path in [("image", SAMPLE_IMAGE), ("image2", SAMPLE_IMAGE_2),
                         ("audio", SAMPLE_AUDIO_FLAC), ("jfk", SAMPLE_AUDIO_JFK)]:
        if not _asset_exists(path):
            missing.append(f"{label}: {path}")
    if missing:
        print("ERROR — missing assets:")
        for m in missing:
            print(f"  {m}")
        sys.exit(2)

    passed = 0
    failed = 0

    def run(name, fn, *args):
        nonlocal passed, failed
        try:
            fn(*args)
            print(f"  OK  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1

    # Health
    print("--- Health ---")
    run("gateway liveliness", TestGatewayHealth().test_gateway_liveliness)
    run("models list", TestGatewayHealth().test_gateway_models_list)
    run("health_check()", TestGatewayHealth().test_health_check_function)

    # Chat
    print("--- Chat ---")
    run("basic chat", TestGatewayChat().test_basic_chat)
    run("streaming", TestGatewayChat().test_chat_streaming)
    run("vision chat", TestGatewayChat().test_chat_with_image, SAMPLE_IMAGE)

    # Embedding
    print("--- Embedding ---")
    run("single text", TestGatewayEmbedding().test_single_text)
    run("batch text", TestGatewayEmbedding().test_batch_text)
    run("consistency", TestGatewayEmbedding().test_consistency)

    # OCR
    print("--- OCR ---")
    run("ocr screenshot", TestGatewayOCR().test_ocr, SAMPLE_IMAGE)
    run("ocr second image", TestGatewayOCR().test_ocr_second_image, SAMPLE_IMAGE_2)

    # ASR
    print("--- ASR ---")
    run("transcribe flac", TestGatewayASR().test_transcribe_audio, SAMPLE_AUDIO_FLAC)
    run("transcribe jfk speech", TestGatewayASR().test_transcribe_jfk_speech, SAMPLE_AUDIO_JFK)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
