# LiteLLM Local — Self-Hosted AI Inference Stack

GPU-accelerated inference on RTX 5090 (32GB VRAM) with a unified OpenAI-compatible API via LiteLLM gateway.

## Services

| Service | Port | Model | GPU Memory | Purpose |
|---------|------|-------|------------|---------|
| LiteLLM Gateway | 8400 | — (proxy) | — | Unified API, caching, retries |
| Chat / Vision | 8070 | Qwen/Qwen3-VL-4B-Instruct-FP8 | 38% (~12 GB) | Chat and image understanding |
| OCR API | 8080 | PaddleOCR-VL-1.5 pipeline | ~5 GB | Layout parsing + page restructure |
| OCR VLM Server | 8118 | PaddleOCR-VL-1.5-0.9B | ~5 GB | OpenAI-compatible OCR backend |
| Embeddings | 8090 | alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC | 15% (~5 GB) | Multimodal embeddings (2048d) |
| ASR | 8060 | Qwen/Qwen3-ASR-1.7B | 20% (~6 GB) | Speech-to-text |

**Total:** ~88% GPU utilization on 32GB RTX 5090

## Quick Start

```bash
# 1. Start vLLM services (sequential startup with health checks)
docker compose -f docker-compose.vllm_cu130_nightly.yml up -d

# 2. Start LiteLLM gateway (after vLLM services are healthy)
docker compose -f docker-compose.gateway.yml up -d

# 3. Verify
curl http://localhost:8400/health
curl http://localhost:8400/v1/models
```

## Usage

### Python SDK

```python
import litellm_client

# Chat
litellm_client.chat("Hello!")
litellm_client.chat("What's in this image?", image="photo.png")

# OCR
text = litellm_client.ocr("invoice.png")
layout = litellm_client.ocr_layout_parse("invoice.png")
doc = litellm_client.ocr_document("invoice.png", output_dir="outputs/invoice")

# Embeddings
vec = litellm_client.embed("semantic search query")
vecs = litellm_client.embed(["batch", "of", "texts"])

# Speech-to-text
result = litellm_client.transcribe("meeting.wav")
print(result["text"])
```

### curl

```bash
# Chat
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local_vlm", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl http://localhost:8400/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC", "input": "Hello world"}'

# ASR (unified endpoint, with or without timestamps)
curl http://localhost:8400/api/asr/transcriptions \
  -F "file=@assets/sample1.flac" \
  -F "model=Qwen/Qwen3-ASR-1.7B"

# OCR VLM (OpenAI-compatible pass-through)
curl http://localhost:8400/api/ocr/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "PaddleOCR-VL-1.5-0.9B", "messages": [{"role":"user","content":[{"type":"text","text":"Extract text"},{"type":"image_url","image_url":{"url":"https://example.com/doc.png"}}]}]}'

# Paddle OCR pass-through endpoints via gateway
curl http://localhost:8400/api/ocr/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{"file":"<base64-or-url>","fileType":1}'

curl http://localhost:8400/api/ocr/restructure-pages \
  -H "Content-Type: application/json" \
  -d '{"pages":[],"concatenatePages":true}'
```

**Gateway model IDs:** built-in `/v1/models` includes `local_vlm` and external providers. OCR, ASR, and embeddings are exposed via `/api/*` pass-through endpoints.

## Files

| File | Purpose |
|------|---------|
| `docker-compose.vllm_cu130_nightly.yml` | vLLM services — 4 models on single GPU |
| `docker-compose.gateway.yml` | LiteLLM gateway on port 8400 |
| `litellm_config.yaml` | Gateway routing, caching, retry config |
| `pipeline_config_vllm.yaml` | PaddleOCR pipeline config for the OCR API service |
| `vllm_config.yaml` | Backend tuning for Paddle genai vLLM server |
| `litellm_client.py` | Python SDK (gateway-only) |
| `test_gateway.py` | Test suite using real media assets |
| `pytest.ini` | Pytest markers and config |

## Testing

```bash
# All tests
pytest test_gateway.py -v

# Standalone runner (no pytest needed)
python test_gateway.py

# Selective
pytest test_gateway.py -m embedding
pytest test_gateway.py -m chat
pytest test_gateway.py -m ocr
pytest test_gateway.py -m asr
pytest test_gateway.py -m integration
```

Tests use real screenshots and audio files from `assets/`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_URL` | `http://localhost:8400` | LiteLLM gateway URL |
| `GATEWAY_KEY` | `not-needed` | API key (if `master_key` is set) |

## System Requirements

- **GPU:** NVIDIA RTX 5090 (32GB) or similar Blackwell GPU
- **Driver:** 580.x+
- **Image:** `vllm/vllm-openai:cu130-nightly`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA driver mismatch (error 803) | Use `vllm/vllm-openai:cu130-nightly` image |
| libcuda.so compat conflict | `rm -f /usr/local/cuda/compat/libcuda.so*` (already in compose entrypoint) |
| FlashAttention segfault on sm_100 | Use `--attention-backend FLASHINFER` (handled by nightly image) |
| Embedding 404 errors | Add `--runner pooling` to embedding service (already configured) |

## Stopping

```bash
docker compose -f docker-compose.gateway.yml down
docker compose -f docker-compose.vllm_cu130_nightly.yml down
```
