# MLX Inference Stack — Apple Silicon

Run **chat completions**, **OCR**, **ASR (speech-to-text)**, and **embeddings** locally on Apple Silicon through a single OpenAI-compatible LiteLLM gateway.

```
Clients  (OpenAI SDK / curl / any HTTP client)
    │
    ▼
LiteLLM Gateway (:8200)                ← docker-compose.litellm-mlx.yml
    │   routes by model name, retries, caches
    │
    ├── mlx_lm.server      (:8101)  — Chat / Completions
    ├── mlx_vlm.server      (:8102)  — OCR / Vision
    ├── mlx_audio.server    (:8103)  — ASR (Whisper-compatible)
    └── embedding_server.py (:8100)  — Text Embeddings (custom FastAPI)
```

## Quick Start

```bash
# 1. Activate the conda environment
conda activate mlx

# 2. Start all MLX services
cd mlx && python manager.py start

# 3. Verify health
python manager.py health
```

That's it. All four capabilities are running at ports 8100-8103.

## Port Map

| Port | Service   | Backend            | Endpoint                        |
|------|-----------|--------------------|----------------------------------|
| 8100 | Embedding | `embedding_server.py` (FastAPI)  | `/v1/embeddings`           |
| 8101 | Chat      | `mlx_lm.server`      | `/v1/chat/completions`          |
| 8102 | OCR       | `mlx_vlm.server`     | `/chat/completions` (no `/v1/`) |
| 8103 | ASR       | `mlx_audio.server`   | `/v1/audio/transcriptions`      |
| 8200 | Gateway   | LiteLLM (Docker)     | unified OpenAI-compatible API   |

## Default Models

| Capability | Model | Package |
|------------|-------|---------|
| Chat / Vision | `mlx-community/Qwen3-VL-8B-Instruct-8bit` | `mlx_lm` |
| OCR | `mlx-community/GLM-OCR-8bit` | `mlx_vlm` |
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` | `mlx_audio` |
| Embedding | `jedisct1/Qwen3-VL-Embedding-8B-mlx` | `mlx_lm` (hidden-state extraction) |

Override any model via environment variables (see below).

## File Map

```
mlx/
├── manager.py                     # Unified CLI + FastAPI orchestrator
├── docker-compose.litellm-mlx.yml # LiteLLM gateway Docker config
├── servers/
│   ├── __init__.py
│   └── embedding_server.py        # Custom OpenAI-compatible embedding server
├── logs/                          # Runtime logs (auto-created)
└── README.md                      # This file
```

## Management Commands

### CLI Commands (Python)
```bash
python manager.py start              # Start all services
python manager.py start chat         # Start only chat
python manager.py stop               # Stop all services
python manager.py stop asr           # Stop only ASR
python manager.py restart chat       # Restart chat
python manager.py status             # JSON status of all services
python manager.py health             # Health check with symbols
python manager.py logs chat 50       # Show last 50 lines
```

### API Server Mode
```bash
python manager.py server             # Start FastAPI server on port 9000
python manager.py server --port 8500 # Custom port

# Then access via HTTP
curl http://localhost:9000/health
curl http://localhost:9000/services
curl -X POST http://localhost:9000/services/start-all

# Interactive docs
open http://localhost:9000/docs
```

## Client Usage

### Via LiteLLM Gateway (recommended)

All requests go through `http://localhost:8200` using standard OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="not-needed")

# ── Chat ─────────────────────────────────────────────
response = client.chat.completions.create(
    model="completions",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence"}],
)
print(response.choices[0].message.content)

# ── Chat with streaming ──────────────────────────────
stream = client.chat.completions.create(
    model="completions",
    messages=[{"role": "user", "content": "Write a haiku about MLX"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# ── Embeddings ───────────────────────────────────────
result = client.embeddings.create(
    model="embedding",
    input=["Hello world", "Goodbye world"],
)
print(f"Dimension: {len(result.data[0].embedding)}")

# ── OCR (vision) ─────────────────────────────────────
import base64
with open("document.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Extract all text from this image"},
        ],
    }],
)
print(response.choices[0].message.content)

# ── ASR (audio transcription) ────────────────────────
transcript = client.audio.transcriptions.create(
    model="asr",
    file=open("speech.mp3", "rb"),
)
print(transcript.text)
```

### Direct to Services (bypass gateway)

```python
from openai import OpenAI

# Chat — direct to mlx_lm.server
chat_client = OpenAI(base_url="http://localhost:8101/v1", api_key="not-needed")

# Embedding — direct to custom server
embed_client = OpenAI(base_url="http://localhost:8100/v1", api_key="not-needed")

# ASR — direct to mlx_audio.server
asr_client = OpenAI(base_url="http://localhost:8103/v1", api_key="not-needed")
```

### curl Examples

```bash
# Chat
curl http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{"model":"completions","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'

# Embedding
curl http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{"model":"embedding","input":"Hello world"}'

# ASR
curl http://localhost:8200/v1/audio/transcriptions \
  -H "Authorization: Bearer not-needed" \
  -F model=asr \
  -F file=@speech.mp3
```

## Environment Variables

All models and ports can be overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_CONDA_ENV` | `mlx` | Conda environment name |
| `MLX_CHAT_MODEL` | `mlx-community/Qwen3-VL-8B-Instruct-8bit` | Chat model |
| `MLX_OCR_MODEL` | `mlx-community/GLM-OCR-8bit` | OCR model |
| `MLX_ASR_MODEL` | `mlx-community/Qwen3-ASR-1.7B-8bit` | ASR model |
| `MLX_EMBED_MODEL` | `jedisct1/Qwen3-VL-Embedding-8B-mlx` | Embedding model |
| `MLX_CHAT_PORT` | `8101` | Chat server port |
| `MLX_OCR_PORT` | `8102` | OCR server port |
| `MLX_ASR_PORT` | `8103` | ASR server port |
| `MLX_EMBED_PORT` | `8100` | Embedding server port |
| `MLX_GATEWAY_PORT` | `8200` | LiteLLM gateway port |

Example — swap to a 4B chat model:

```bash
MLX_CHAT_MODEL=mlx-community/Qwen3-VL-4B-Instruct-8bit ./mlx.sh start chat
```

## Alternative Models

### Chat
- `mlx-community/Qwen3-VL-4B-Instruct-8bit` — lighter, faster
- `mlx-community/Qwen3-VL-8B-Instruct-8bit` — more capable (default)

### ASR
- `mlx-community/Qwen3-ASR-1.7B-8bit` — Qwen ASR, more accurate, 52 languages (default)
- `mlx-community/Qwen3-ASR-0.6B-8bit` — Qwen ASR, smaller
- `mlx-community/whisper-large-v3-turbo` — fastest, good accuracy

### Embedding
- `jedisct1/Qwen3-VL-Embedding-8B-mlx` — multimodal, 4096-dim, state-of-the-art (default)
- `mlx-community/bge-small-en-v1.5` — small, fast, English (via mlx_embeddings fallback)

## Architecture Notes

### Why Four Separate Servers?

MLX models share the unified memory on Apple Silicon. Unlike GPU VRAM partitioning, there's no need to calculate memory fractions — the OS manages unified memory automatically. Running separate processes ensures:

1. **Isolation** — one model crashing doesn't affect others
2. **Independent scaling** — restart/swap models individually
3. **Different backends** — each package (`mlx_lm`, `mlx_vlm`, `mlx_audio`) has its own optimised server

### LiteLLM Routing

The gateway uses different LiteLLM provider prefixes depending on each backend's API compatibility:

| Service | LiteLLM Prefix | Reason |
|---------|---------------|--------|
| Chat | `hosted_vllm/` | `mlx_lm.server` exposes `/v1/chat/completions` |
| OCR | `hosted_vllm/` | `mlx_vlm.server` exposes `/chat/completions` (no `/v1/`) |
| ASR | `openai/` | `mlx_audio.server` exposes `/v1/audio/transcriptions` |
| Embedding | `openai/` | Custom server exposes `/v1/embeddings` |

### Memory Estimation

All models load into unified memory (RAM = VRAM on Apple Silicon):

| Model | Approx. Memory |
|-------|----------------|
| Qwen3-VL-8B-Instruct-8bit | ~8 GB |
| GLM-OCR-8bit | ~4 GB |
| whisper-large-v3-turbo | ~1.5 GB |
| Qwen3-VL-Embedding-8B-mlx | ~8.7 GB |
| **Total** | **~22 GB** |

A Mac with 32 GB+ unified memory can run all four comfortably. With 16 GB, consider using smaller model variants.

## Prerequisites

```bash
# Create and activate conda environment
conda create -n mlx python=3.12
conda activate mlx

# Install packages
pip install mlx-lm mlx-vlm mlx-audio mlx-embeddings
pip install fastapi uvicorn

# Docker (for LiteLLM gateway only)
# Install Docker Desktop for Mac from https://docker.com
```

## Troubleshooting

**Service won't start** — Check logs: `./mlx.sh logs <service>`

**Port already in use** — Stop the conflicting service: `lsof -i :<port>` then `kill <pid>`

**Model download slow** — Models are cached in `~/.cache/huggingface/`. First run downloads from HuggingFace Hub.

**OCR returns errors** — `mlx_vlm` loads models dynamically on first request. The first OCR request will be slow (~30s) as it downloads and loads the model. Subsequent requests use the cached model.

**Gateway can't reach services** — The LiteLLM Docker container uses `host.docker.internal` to reach host services. Ensure Docker Desktop's "Allow host networking" is enabled.

**Out of memory** — Use smaller model variants or stop services you don't need: `./mlx.sh stop ocr`