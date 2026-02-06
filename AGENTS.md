# AGENTS.md — Project Context for AI Agents

## What This Project Is

**litellm_local** is a self-hosted, GPU-accelerated AI inference stack that exposes OpenAI-compatible APIs for four capabilities: **chat completions**, **OCR**, **audio transcription (ASR)**, and **multimodal embeddings**. It runs on a two-machine LAN setup (Machine A with an RTX 5090, Machine B with a smaller GPU) behind a unified LiteLLM gateway proxy.

The goal: run powerful open-source models locally with a single OpenAI-compatible endpoint, no cloud APIs needed.

## Architecture

```
Clients (Python SDK, curl, any OpenAI client)
    │
    ▼
LiteLLM Gateway (:8200)              ← docker-compose.litellm.yml
    │   Routes by model name, load-balances, retries
    │
    ├── Machine A  (RTX 5090, 32 GB VRAM)    ← docker-compose.vllma.yml
    │   ├── vllm-completions (:8101)  — Qwen3-VL-4B-Instruct  (15 GB)
    │   ├── vllm-ocr         (:8102)  — zai-org/GLM-OCR        (4 GB)
    │   └── vllm-asr         (:8103)  — Qwen3-ASR-0.6B + ForcedAligner (custom image)
    │
    └── Machine B  (8 GB GPU)                ← docker-compose.vllmb.yml
        └── vllm-embedding   (:8100)  — Qwen3-VL-Embedding-2B  (2 GB)
```

There is also a fallback compose for Machine B using a GTX 1070 Ti (`docker-compose.1070b.yml`, `qwen3vl-embedding-server/`) which runs a custom FastAPI embedding server instead of vLLM (Pascal GPUs lack sm_100/flash-attention support).

## Key Models

| Capability | Model | Size | Port | Backend |
|------------|-------|------|------|---------|
| Chat / Vision | `Qwen/Qwen3-VL-4B-Instruct` | 4B | 8101 | vLLM nightly (cu130) |
| OCR | `zai-org/GLM-OCR` | ~4B | 8102 | vLLM nightly (cu130) |
| ASR (speech→text) | `Qwen/Qwen3-ASR-0.6B` + `Qwen3-ForcedAligner-0.6B` | 0.6B each | 8103 | Transformers + Gunicorn |
| Embeddings (multimodal) | `Qwen/Qwen3-VL-Embedding-2B` | 2B | 8100 | vLLM or custom FastAPI |

## File Map

### Docker / Infrastructure

| File | Purpose |
|------|---------|
| `Dockerfile.vllm-nightly` | Builds vLLM image from nightly wheels on CUDA 13.0 for Blackwell GPUs. Handles compat-lib removal for host driver passthrough. |
| `Dockerfile.vllm` | Builds vLLM from source (slower, used as fallback). |
| `docker-compose.vllma.yml` | Machine A services: completions, OCR, ASR. Shared GPU with dynamic `gpu-memory-utilization`. |
| `docker-compose.vllmb.yml` | Machine B services: embedding via vLLM. |
| `docker-compose.1070b.yml` | Machine B fallback: custom FastAPI embedding server for Pascal GPUs. |
| `docker-compose.litellm.yml` | LiteLLM gateway proxy. Inline config routes model names to backends. |
| `qwen3-asr/Dockerfile` | ASR image extending vllm-blackwell-nightly with qwen-asr, flask, gunicorn. |
| `qwen3-asr/serve.py` | Flask app: `/v1/audio/transcriptions` endpoint with word/segment timestamps. |
| `qwen3vl-embedding-server/Dockerfile` | Custom embedding server image for Pascal GPUs (CUDA 12.4, float16). |
| `qwen3vl-embedding-server/server.py` | FastAPI server: `/v1/embeddings` with multimodal support (text, image, video). |

### Management Scripts

| File | Purpose |
|------|---------|
| `vllma.sh` | Start/stop/status/logs for Machine A services (completions + OCR + ASR). |
| `vllmb.sh` | Start/stop/status/logs for Machine B services (embedding). |
| `litellm.sh` | Start/stop/status/logs for the LiteLLM gateway. |
| `1070b.sh` | Start/stop/build/test for the GTX 1070 Ti fallback embedding server. |

### Client SDK

| File | Purpose |
|------|---------|
| `litellm_client.py` | Full Python SDK (946 lines). Unified client for embed, chat, OCR, transcribe. Includes CLI, streaming, retries, convenience functions. |
| `CLIENT_REVIEW.md` | Production review of the client SDK. Status: approved (grade A-). |

### Tests

| File | Purpose |
|------|---------|
| `tests/test_models.py` | OpenAI API compliance tests (embeddings, completions, OCR, health). |
| `tests/test_client_with_files.py` | Integration tests using real image/audio files against live gateway. |
| `pytest.ini` | Pytest config with `integration` and `compliance` markers. |

## Port Map

| Port | Service | Location |
|------|---------|----------|
| 8100 | Embedding | Machine B |
| 8101 | Completions | Machine A |
| 8102 | OCR | Machine A |
| 8103 | ASR | Machine A |
| 8200 | LiteLLM Gateway | Either machine |

## GPU Memory Budget (Machine A — 32 GB)

Services share one GPU via vLLM's `--gpu-memory-utilization` (fraction of total VRAM):

- **Completions**: 15 GB → `gpu_util = 15/32 ≈ 0.47`
- **OCR**: 4 GB → `gpu_util = 4/32 = 0.125`
- **ASR**: Uses Transformers (not vLLM), loads into remaining VRAM

Services start sequentially (depends_on with healthcheck) to avoid VRAM contention during model loading.

## Build & Deploy

```bash
# 1. Build the vLLM nightly image (Blackwell / CUDA 13.0)
docker build -f Dockerfile.vllm-nightly -t vllm-blackwell-nightly:latest .

# 2. Create shared network
docker network create vllm-shared-network 2>/dev/null || true

# 3. Start Machine A services
./vllma.sh start        # or: docker compose -f docker-compose.vllma.yml up -d

# 4. Start Machine B services (on machine B)
./vllmb.sh start        # or: docker compose -f docker-compose.vllmb.yml up -d

# 5. Start the gateway (on whichever machine)
./litellm.sh start      # or: docker compose -f docker-compose.litellm.yml up -d
```

## Known Blackwell (RTX 5090) Issues

These were discovered and resolved during development:

1. **cu128 vs cu130 torch mismatch**: `uv pip install vllm` without `--prerelease=allow` picks stable PyPI vLLM (cu128 torch) instead of nightly cu130. Fix: use `--prerelease=allow` with nightly index as primary.

2. **compat libcuda.so shadowing host driver**: The CUDA 13.0 container base image ships `/usr/local/cuda/compat/libcuda.so.580.x` which is too old for Blackwell. ldconfig picks it over the host's 590.x driver. Fix: `rm -f /usr/local/cuda/compat/libcuda.so*` in Dockerfile before `ldconfig`.

3. **FlashAttention segfault on sm_100**: FlashAttention's precompiled CUDA kernels don't support Blackwell compute capability 10.0. Crashes in `flash::mha_varlen_fwd` during encoder cache profiling. Fix: use `--attention-backend FLASHINFER` flag on vLLM serve commands.

## Coding Conventions

- **Docker images** are tagged `*-nightly:latest` for nightly builds, plain `:latest` for source builds.
- **Environment variables** in compose files use `$$` escaping (compose syntax) and are evaluated at container startup via inline bash.
- **GPU allocation** is computed dynamically: `GPU_UTIL = SERVICE_GPU_GB / GPU_TOTAL_GB`.
- **Model aliases** in LiteLLM config: short names (`completions`, `ocr`, `embedding`, `asr`) map to full HuggingFace model IDs.
- **Client SDK** uses `openai` library under the hood; ASR uses raw `httpx` for multipart file upload.
- **Tests** require live services. Run with `pytest -m integration` or `pytest -m compliance`.

## Agent Guidelines

When working on this project:

- **Before editing compose files**: Read the current file first — it uses multi-line bash `command` blocks with `$$` variable escaping that are easy to break.
- **Before editing Dockerfiles**: The nightly Dockerfile (`Dockerfile.vllm-nightly`) is the active one. `Dockerfile.vllm` (from-source) is the fallback.
- **GPU constraints**: All Machine A services share 32 GB. Any new service needs a VRAM budget that fits within the remaining headroom.
- **Testing changes**: After compose changes, `docker compose -f <file> up -d` then check logs with `docker logs <container> --tail 50`.
- **Network**: All services communicate over `vllm-shared-network` (external Docker network). The gateway resolves service hostnames within this network.
- **The ASR service** is special — it does NOT use vLLM for inference. It uses HuggingFace Transformers with a Flask/Gunicorn server, but builds on top of the vllm-blackwell-nightly base image for CUDA compatibility.
