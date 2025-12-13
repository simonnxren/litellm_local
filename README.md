# memoria_vllm

vLLM inference stack with LiteLLM gateway - OpenAI-compatible API for embeddings, chat, OCR, and audio transcription.

## Models

- **Embeddings** – Qwen3-Embedding-0.6B (1024 dims)
- **Chat** – Qwen3-8B-FP8 (20K context, 8B params)
- **OCR** – HunyuanOCR (vision-to-text)
- **Audio** – Whisper-Large-v3-Turbo

## Architecture

```
Client → LiteLLM Gateway :8200
            ├─ vLLM Embedding :8100 (11% GPU)
            ├─ vLLM Completions :8101 (60% GPU)
            ├─ vLLM OCR :8102 (20% GPU)
            └─ vLLM Whisper :8103 (11% GPU)
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
**Requirements**: Docker + GPU support, NVIDIA GPU (24GB+ VRAM), CUDA 12.1+

```bash
# Start services
./start_vllm.sh

# Verify
curl http://localhost:8200/health

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
    stre

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="dummy"
)

# Embeddings
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input="Hello world"
)

# Chat
chat = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=100
)

# OCR
import base64
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

ocr = client.chat.completions.create(
    model="hunyuan-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract text"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=2048
)
```

**cURL examples**: See [API_USAGE_MINIMAL.md](API_USAGE_MINIMAL.md)LLM_WHISPER_GPU_MEMORY=0.11     # 11%

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

## API Endpoints

- `GET /v1/models` - List models
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/chat/completions` - Chat (supports vision for OCR)
- `POST /v1/audio/transcriptions` - Audio transcription
- `GET /health` - Health check

## Testing

```bash
# Run tests
pytest tests/test_models.py -v

# Manual test
curl http://localhost:8200/v1/models**GPU Issues**: Check `nvidia-smi` and `docker logs vllm-completions`

**Out of Memory**: Reduce GPU allocations in `.env` (must sum ≤ 1.0)

**Model Downloads**: First start takes 15-20 mins to download models

## License

MIT