# LiteLLM Local — Self-Hosted AI Inference Stack

GPU-accelerated inference on RTX 5090 (32GB VRAM) with a unified OpenAI-compatible API via LiteLLM gateway.

## Services

| Service | Port | Model | GPU Memory | Purpose |
|---------|------|-------|------------|---------|
| LiteLLM Gateway | 8400 | — (proxy) | — | Unified API, caching, retries |
| Chat / Vision | 8070 | Qwen/Qwen3-VL-4B-Instruct-FP8 | 38% (~12 GB) | Chat and image understanding |
| OCR | 8080 | zai-org/GLM-OCR | 15% (~5 GB) | Image text extraction |
| Embeddings | 8090 | shigureui/Qwen3-VL-Embedding-2B-FP8 | 15% (~5 GB) | Multimodal embeddings (2048d) |
| ASR | 8000 | Qwen/Qwen3-ASR-1.7B | 20% (~6 GB) | Speech-to-text |

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
  -d '{"model": "chat", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl http://localhost:8400/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embedding", "input": "Hello world"}'
```

**Model aliases:** `chat`, `ocr`, `embedding`, `asr` (full HuggingFace names also work).

## Files

| File | Purpose |
|------|---------|
| `docker-compose.vllm_cu130_nightly.yml` | vLLM services — 4 models on single GPU |
| `docker-compose.gateway.yml` | LiteLLM gateway on port 8400 |
| `litellm_config.yaml` | Gateway routing, caching, retry config |
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
