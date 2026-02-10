# AGENTS.md — Project Context for AI Agents

## What This Project Is

**litellm_local** is a self-hosted, GPU-accelerated AI inference stack that exposes OpenAI-compatible APIs for four capabilities: **chat completions**, **OCR**, **audio transcription (ASR)**, and **multimodal embeddings**. It runs on a single-machine setup (RTX 5090 with 32GB VRAM) with all services orchestrated via Docker Compose.

The goal: run powerful open-source models locally with a single unified API endpoint. All requests flow through a LiteLLM gateway on port 8400 which provides caching, retries, logging, and model aliasing.

## Architecture

```
Clients (Python SDK, curl, any OpenAI client)
    │
    ▼
LiteLLM Gateway (:8400)  — Unified API, caching, retries, logging
    │
    ├── model="chat"       → Chat Service       (:8070)  Qwen3-VL-4B-Instruct-FP8   12 GB
    ├── model="ocr"        → OCR Service        (:8080)  zai-org/GLM-OCR             4 GB
    ├── model="embedding"  → Embedding Service  (:8090)  Qwen3-VL-Embedding-2B-FP8   4 GB
    └── model="asr"        → ASR Service        (:8000)  Qwen3-ASR-1.7B              6 GB
```

All services run on a single RTX 5090 GPU with dynamic memory allocation via vLLM's `--gpu-memory-utilization` parameter. Services start sequentially with health checks to prevent VRAM contention.

## Key Models

| Capability | Model | Size | Port | Backend | VRAM |
|------------|-------|------|------|---------|------|
| Chat / Vision | `Qwen/Qwen3-VL-4B-Instruct-FP8` | 4B | 8070 | vLLM nightly (cu130) | 12 GB |
| OCR | `zai-org/GLM-OCR` | ~4B | 8080 | vLLM nightly (cu130) | 4 GB |
| ASR (speech→text) | `Qwen/Qwen3-ASR-1.7B` | 1.7B | 8000 | vLLM nightly (cu130) | 6 GB |
| Embeddings (multimodal) | `shigureui/Qwen3-VL-Embedding-2B-FP8` | 2B | 8090 | vLLM nightly (cu130) + pooling runner | 4 GB |

**Total VRAM Usage**: ~28 GB (88% of 32GB RTX 5090)

## File Map

### Docker / Infrastructure

| File | Purpose |
|------|---------|
| `docker-compose.vllm_cu130_nightly.yml` | **Main compose file** - Single machine orchestration with all 4 services. Sequential startup with health checks. |
| `docker-compose.gateway.yml` | **Gateway compose file** - LiteLLM proxy on port 8400, routes to all vLLM services via host networking. |

### Client SDK

| File | Purpose |
|------|---------|
| `litellm_client.py` | **Gateway-only Python SDK**. Routes all requests through the LiteLLM gateway. Supports chat, OCR, embeddings, ASR with streaming. |
| `test_gateway.py` | **Test suite**. Pytest-based tests for all 4 services through the gateway, using real media assets. |

### Configuration & Testing

| File | Purpose |
|------|---------|
| `pytest.ini` | Pytest configuration with custom markers (`integration`, `embedding`, `chat`, `ocr`, `asr`) |
| `litellm_config.yaml` | LiteLLM gateway configuration — model routing, caching, retry settings |
| `.gitignore` | Git ignore rules including assets, logs, and environment files |
| `.dockerignore` | Docker ignore rules for build context optimization |

## Port Map

| Port | Service | Model | Status |
|------|---------|-------|--------|
| 8400 | LiteLLM Gateway | — (proxy) | ✓ Operational |
| 8070 | Chat | Qwen3-VL-4B-Instruct-FP8 | ✅ Operational |
| 8080 | OCR | zai-org/GLM-OCR | ✅ Operational |
| 8090 | Embedding | Qwen3-VL-Embedding-2B-FP8 | ✅ Operational (2048 dims) |
| 8000 | ASR | Qwen3-ASR-1.7B | ✅ Operational |

## GPU Memory Allocation (RTX 5090 - 32GB)

Services share GPU via vLLM's dynamic memory management:

- **Chat Service**: `--gpu-memory-utilization 0.38` (~12.16 GB)
- **OCR Service**: `--gpu-memory-utilization 0.15` (~4.8 GB)
- **Embedding Service**: `--gpu-memory-utilization 0.15` (~4.8 GB)
- **ASR Service**: `--gpu-memory-utilization 0.20` (~6.4 GB)

Services start sequentially with health checks to prevent VRAM contention during model loading.

## Build & Deploy

```bash
# 1. Start vLLM services
docker compose -f docker-compose.vllm_cu130_nightly.yml up -d

# 2. Start LiteLLM gateway (after vLLM services are healthy)
docker compose -f docker-compose.gateway.yml up -d

# 3. Check all services
docker compose -f docker-compose.vllm_cu130_nightly.yml ps
docker compose -f docker-compose.gateway.yml ps

# 4. Test gateway
curl http://localhost:8400/health
curl http://localhost:8400/v1/models

# 5. Use via gateway (single endpoint)
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "chat", "messages": [{"role": "user", "content": "Hello"}]}'

# 6. Stop everything
docker compose -f docker-compose.gateway.yml down
docker compose -f docker-compose.vllm_cu130_nightly.yml down
```

## Key Features & Enhancements

### Client SDK (`litellm_client.py`)
- **Gateway-Only**: All requests (chat, OCR, embeddings, ASR) route through the LiteLLM gateway
- **Unified Embedding Endpoint**: Text and multimodal embeddings use a single gateway pass-through endpoint (`/embeddings/v1/embeddings`) that forwards requests directly to vLLM
- **Environment Variable Configuration**: Gateway URL and API key configurable via env vars
- **Robust Error Handling**: Specific exceptions for different failure modes
- **Comprehensive Logging**: Structured logging with INFO/DEBUG levels
- **Type Safety**: TypedDict definitions and proper type hints
- **Health Checks**: Built-in service status monitoring
- **Streaming Support**: Async streaming for chat completions
- **Batch Operations**: Efficient batch processing for text embeddings

### Testing Framework
- **Unified Test Suite**: Single file (`test_gateway.py`) with real media assets from `assets/`
- **Pytest Integration**: Markers for selective test execution
- **Comprehensive Coverage**: Tests for all 4 services plus integration tests
- **Standalone Runner**: `python test_gateway.py` for quick validation without pytest

### Recent Improvements
- ✅ **Gateway-Only Architecture**: All SDK requests route through LiteLLM gateway (no direct mode)
- ✅ **LiteLLM Gateway**: Unified API on port 8400 with model aliasing, caching, retries, and logging
- ✅ **Unified Embedding Endpoint**: Pass-through endpoint routes both text and multimodal embeddings through gateway to vLLM
- ✅ **Full Embedding Functionality Restored**: Added `--runner pooling` flag for proper 2048-dimensional embeddings
- ✅ **Real Asset Testing**: Tests use actual screenshots and audio files from `assets/`
- ✅ **Production Ready**: Proper error handling, logging, and configuration

## Known Issues & Solutions

These were discovered and resolved during development:

1. **cu128 vs cu130 torch mismatch**: `uv pip install vllm` without `--prerelease=allow` picks stable PyPI vLLM (cu128 torch) instead of nightly cu130. 
   - **Fix**: Use official `vllm/vllm-openai:cu130-nightly` Docker image

2. **compat libcuda.so shadowing host driver**: The CUDA 13.0 container base image ships `/usr/local/cuda/compat/libcuda.so.580.x` which is too old for Blackwell. ldconfig picks it over the host's 590.x driver.
   - **Fix**: `rm -f /usr/local/cuda/compat/libcuda.so*` in Dockerfile before `ldconfig` (already in compose entrypoint)

3. **FlashAttention segfault on sm_100**: FlashAttention's precompiled CUDA kernels don't support Blackwell compute capability 10.0. Crashes in `flash::mha_varlen_fwd` during encoder cache profiling.
   - **Fix**: Use `--attention-backend FLASHINFER` flag on vLLM serve commands (handled automatically in nightly image)

4. **Embedding Service Not Working**: Missing `--runner pooling` flag caused 404 errors on `/v1/embeddings` endpoint.
   - **Fix**: Added `--runner pooling` to embedding service configuration

## Coding Conventions

- **Docker images**: Use `vllm/vllm-openai:cu130-nightly` for Blackwell compatibility
- **Gateway-only**: All client requests go through LiteLLM gateway on :8400 (embeddings use a pass-through endpoint)
- **Environment variables**: `GATEWAY_URL`, `GATEWAY_KEY` for SDK config
- **GPU allocation**: Dynamic via `--gpu-memory-utilization` (0.0-1.0 fraction)
- **Service dependencies**: Sequential startup with health checks via `depends_on`
- **Testing**: Pytest with custom markers for selective execution
- **Logging**: Python logging module with structured INFO/DEBUG levels

## Agent Guidelines

When working on this project:

- **Architecture**: Gateway-only — all requests go through LiteLLM on :8400
- **Configuration**: Environment variables preferred over hardcoded values
- **Error Handling**: Specific exceptions with meaningful messages
- **Testing**: `pytest test_gateway.py -v` or `python test_gateway.py`
- **GPU Management**: Monitor VRAM usage — current allocation is ~90% of 32GB
- **Model Updates**: FP8 quantized models for optimal performance on RTX 5090
- **Documentation**: Keep AGENTS.md and README.md synchronized with current implementation
