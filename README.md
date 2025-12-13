# memoria_vllm

**Production-ready vLLM inference stack** with LiteLLM gateway providing an OpenAI-compatible API for text embeddings, chat completions with reasoning, document OCR, and audio transcription — all on a single GPU.

## Overview

LiteLLM gateway routes traffic to dedicated vLLM instances, each running a specialized model with reserved GPU memory. Get enterprise-grade features like authentication, rate limiting, cost tracking, and observability out of the box.

### Models Available
- 📝 **Embeddings** – `Qwen/Qwen3-Embedding-0.6B` (1024-dim vectors)
- 💬 **Chat/Completions** – `Qwen/Qwen3-8B-FP8` (32K context, 8B params, optional thinking mode)
- 🖼️ **OCR** – `tencent/HunyuanOCR` (vision-language model for document understanding)
- 🎧 **Audio** – `openai/whisper-large-v3-turbo` (transcription)

### Service Architecture
```
Client → LiteLLM Gateway (port 8200)
            │
            ├─ vLLM Embedding (8100) - 15% GPU
            ├─ vLLM Completions (8101) - 70% GPU (minimal) or 40% (full)
            ├─ vLLM OCR (8102) - 25% GPU
            └─ vLLM Whisper (8103) - 11% GPU

LiteLLM Features:
• OpenAI-compatible API
• Model routing and load balancing
• Request/response logging
• Cost tracking per model
• Admin UI at /ui

Deployment Options:
• Minimal: Embeddings + Completions only (2 services, 32K context)
• Full: All 4 services (embeddings, chat, OCR, audio)
```

## Quick Start

### Prerequisites
- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with 24GB+ VRAM recommended
- CUDA 12.1+ drivers

### Installation

```bash
# 1. Clone repository
git clone https://github.com/simonnxren/memoria_vllm
cd memoria_vllm

# 2. Configure environment (uses defaults from .env.example)
cp .env.example .env

# 3. Choose deployment option:
#    Option A - Minimal (recommended): Embeddings + Chat only (32K context)
#    Option B - Full: All services (OCR + Whisper included)

# 4a. Start minimal setup (embeddings + chat with 32K context)
docker compose -f docker-compose.minimal.yml up -d

# OR 4b. Start full setup (all services)
docker compose up -d

# 5. Wait for services to be ready (first start downloads models ~15-20 mins)
docker compose logs -f

# 6. Verify all services are healthy
curl http://localhost:8200/health/liveliness
```

### First API Call

```bash
# List available models
curl http://localhost:8200/v1/models | jq '.data[] | .id'

# Generate embeddings
curl -X POST http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "input": "Hello world"
  }' | jq '.data[0].embedding | length'

# Chat completion (with thinking disabled for clean output)
curl -X POST http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-fp8",
    "messages": [{"role": "user", "content": "What is 2+2? /no_think"}],
    "max_tokens": 50
  }' | jq '.choices[0].message.content'
```

## Usage Examples

### Python Client (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="not-needed"  # No auth by default
)

# 1. Generate embeddings
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input=["Hello world", "How are you?"]
)
embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings of {len(embeddings[0])} dimensions")

# 2. Chat completion (standard mode - add /no_think for clean output)
chat = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "What is 15% of 200? /no_think"}
    ],
    max_tokens=100
)
print("Answer:", chat.choices[0].message.content)

# 3. Chat with thinking enabled (for complex reasoning)
chat_thinking = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[
        {"role": "user", "content": "Solve this step by step: If a train travels 120km in 2 hours, what's its speed?"}
    ],
    max_tokens=300
)
print("Full response:", chat_thinking.choices[0].message.content)

# 4. Streaming chat
stream = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Count from 1 to 5 /no_think"}],
    max_tokens=50,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 5. OCR (using HunyuanOCR via chat completions)
import base64

with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

ocr_response = client.chat.completions.create(
    model="hunyuan-ocr",  # or "tencent/HunyuanOCR"
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            },
            {
                "type": "text",
                "text": "Extract all text from this image."
            }
        ]
    }],
    max_tokens=2000
)
print("OCR Result:", ocr_response.choices[0].message.content)

# 6. Audio transcription
audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-large-v3-turbo",  # or "openai/whisper-large-v3-turbo"
    file=audio_file,
    language="en"
)
print("Transcript:", transcript.text)
```

### Direct API Calls (curl)

```bash
# Embeddings
curl -X POST http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "input": "Your text here"
  }'

# Chat completions (clean output with /no_think)
curl -X POST http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-fp8",
    "messages": [{"role": "user", "content": "Hello! /no_think"}],
    "max_tokens": 50
  }'

# Streaming
curl -X POST http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-fp8",
    "messages": [{"role": "user", "content": "Count to 5 /no_think"}],
    "max_tokens": 50,
    "stream": true
  }'

# Audio transcription
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo"
```

## Available Models

The gateway exposes 7 model names (4 primary + 3 HuggingFace ID aliases):

| Model Name | Type | Context | Description |
|------------|------|---------|-------------|
| `qwen3-embedding-0.6b` | Embeddings | N/A | 1024-dimensional vectors |
| `Qwen/Qwen3-Embedding-0.6B` | Embeddings | N/A | HuggingFace ID alias |
| `qwen3-8b-fp8` | Chat | 32K | 8B params, thinking mode (use `/no_think` for clean output) |
| `Qwen/Qwen3-8B-FP8` | Chat | 32K | HuggingFace ID alias |
| `hunyuan-ocr` | Vision | 8K | OCR and document understanding |
| `tencent/HunyuanOCR` | Vision | 8K | HuggingFace ID alias |
| `whisper-large-v3-turbo` | Audio | N/A | Audio transcription |
| `openai/whisper-large-v3-turbo` | Audio | N/A | OpenAI-compatible alias |

## Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# Model Selection (HuggingFace IDs)
MODEL_EMBED_NAME=Qwen/Qwen3-Embedding-0.6B
MODEL_COMPLETIONS_NAME=Qwen/Qwen3-8B-FP8
MODEL_OCR_NAME=tencent/HunyuanOCR
MODEL_WHISPER_NAME=openai/whisper-large-v3-turbo

# Service Ports
ROUTER_PORT=8200
VLLM_EMBED_PORT=8100
VLLM_COMPLETIONS_PORT=8101
VLLM_OCR_PORT=8102
VLLM_WHISPER_PORT=8103

# GPU Memory Allocation
# Minimal setup (embeddings + completions only):
VLLM_EMBED_GPU_MEMORY=0.15       # 15%
VLLM_COMPLETIONS_GPU_MEMORY=0.70 # 70% (for 32K context)

# Full setup (all services):
# VLLM_EMBED_GPU_MEMORY=0.10       # 10%
# VLLM_COMPLETIONS_GPU_MEMORY=0.40 # 40%
# VLLM_OCR_GPU_MEMORY=0.25         # 25%
# VLLM_WHISPER_GPU_MEMORY=0.11     # 11%

# Performance Tuning
VLLM_EMBED_MAX_NUM_SEQS=256
VLLM_COMPLETIONS_MAX_NUM_SEQS=128
VLLM_COMPLETIONS_MAX_MODEL_LEN=32768      # 32K context
VLLM_COMPLETIONS_MAX_BATCHED_TOKENS=65536
VLLM_OCR_MAX_NUM_SEQS=64
VLLM_WHISPER_MAX_NUM_SEQS=64
```

### Model Routing Configuration

Models are configured in `litellm_config.yaml`:

```yaml
model_list:
  - model_name: qwen3-8b-fp8
    litellm_params:
      model: hosted_vllm/Qwen/Qwen3-8B-FP8
      api_base: http://vllm-completions:8101/v1
      supports_response_schema: true
    model_info:
      supports_function_calling: true
```

## Service Management

```bash
# View all containers
docker compose ps

# View logs
docker compose logs -f                 # All services
docker compose logs -f litellm         # Gateway only
docker compose logs -f vllm-ocr        # OCR service

# Restart services
docker compose restart litellm         # Restart gateway
docker compose restart vllm-ocr        # Restart OCR

# Stop all services
docker compose down

# Rebuild and restart
docker compose up -d --build
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/completions` | POST | Text completions |
| `/v1/chat/completions` | POST | Chat completions (with vision support for OCR) |
| `/v1/audio/transcriptions` | POST | Whisper audio transcription |
| `/health/liveliness` | GET | LiteLLM health check |
| `/health/readiness` | GET | LiteLLM readiness check |

### Health Checks

```bash
# LiteLLM gateway health
curl http://localhost:8200/health/liveliness

# Individual vLLM services
curl http://localhost:8100/health  # Embeddings
curl http://localhost:8101/health  # Completions
curl http://localhost:8102/health  # OCR
curl http://localhost:8103/health  # Whisper
```

## Testing

Run the provided test scripts:

```bash
# Quick test all services
/tmp/quick_test.sh  # Created during setup

# Or manual tests
curl -s http://localhost:8200/v1/models | jq '.data[] | .id'
curl -s -X POST http://localhost:8200/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-0.6b","input":"test"}' | jq '.data[0].embedding | length'
```

## Troubleshooting

### Services won't start
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# View container logs
docker compose logs vllm-embedding
docker compose logs vllm-ocr
```

### 404 Errors on API calls
- Ensure `api_base` URLs in `litellm_config.yaml` include `/v1` suffix
- Check models are loaded: `curl http://localhost:8200/v1/models`
- Verify vLLM containers are healthy: `docker compose ps`

### Out of Memory
- Reduce GPU allocations in `.env` (must sum < 1.0)
- Decrease `MAX_NUM_SEQS` values for services
- Use smaller models

### Model Download Issues
- First startup downloads models (~15-20 mins)
- Check logs: `docker compose logs -f vllm-ocr`
- Ensure internet connectivity
- HuggingFace models cached in Docker volumes

## License
MIT

## Resources
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
