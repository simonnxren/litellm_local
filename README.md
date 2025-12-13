# LiteLLM Local

vLLM inference stack with LiteLLM gateway - OpenAI-compatible API for embeddings, chat, and OCR.

## Models

- **Embeddings** – Qwen3-Embedding-0.6B (1024 dims)
- **Chat** – Qwen3-8B-FP8 (32K context, 8B params)
- **OCR** – HunyuanOCR (vision-to-text)

## Architecture

```
Client → LiteLLM Gateway :8200
            ├─ vLLM Embedding :8100 (~15% GPU)
            ├─ vLLM Completions :8101 (~70% GPU)
            └─ vLLM OCR :8102 (~20% GPU)
```

## Quick Start

### Prerequisites
- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with 24GB+ VRAM recommended
- CUDA 12.1+ drivers

### Installation

```bash
# 1. Clone repository
git clone https://github.com/simonnxren/litellm_local.git
cd litellm_local

# 2. Configure environment (optional, uses defaults)
cp .env.example .env

# 3. Start services
./start_vllm.sh

# 4. Verify
curl http://localhost:8200/health

## Usage

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="dummy"
)

# 1. Embeddings
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input=["Hello world", "How are you?"]
)
embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings of {len(embeddings[0])} dimensions")

# 2. Chat completion
chat = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "What is 15% of 200?"}],
    max_tokens=100
)
print("Answer:", chat.choices[0].message.content)

# 3. Streaming chat
stream = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    max_tokens=50,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# 4. OCR (vision)
import base64
with open("image.png", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="hunyuan-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": "Extract all text from this image"}
        ]
    }],
    max_tokens=500
)
print("Extracted text:", response.choices[0].message.content
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

### cURL Examples

See [API_USAGE_MINIMAL.md](API_USAGE_MINIMAL.md) for cURL examples.

## Configuration

### Environment Variables

Configure services in `.env` (copy from `.env.example`):

```bash
# Model configuration
MODEL_EMBED_NAME=Qwen/Qwen3-Embedding-0.6B
MODEL_COMPLETIONS_NAME=Qwen/Qwen3-8B-FP8
MODEL_OCR_NAME=tencent/HunyuanOCR

# GPU memory allocation (must sum ≤ 1.0)
VLLM_EMBED_GPU_MEMORY=0.15
VLLM_COMPLETIONS_GPU_MEMORY=0.70
VLLM_OCR_GPU_MEMORY=0.20

# Performance tuning
VLLM_COMPLETIONS_MAX_MODEL_LEN=32768  # 32K context
VLLM_COMPLETIONS_MAX_NUM_SEQS=128
```

### Model Routing

Models are configured in [litellm_config.yaml](litellm_config.yaml):

```yaml
model_list:
  - model_name: qwen3-8b-fp8
    litellm_params:
      model: hosted_vllm/Qwen/Qwen3-8B-FP8
      api_base: http://vllm-completions:8101/v1
      supports_response_schema: true
```

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/embeddings` - Generate text embeddings
- `POST /v1/chat/completions` - Chat completions (supports vision for OCR)
- `GET /health` - Health check

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py::TestEmbeddings -v

# Manual verification
curl http://localhost:8200/v1/models
```

## Troubleshooting

**Services not starting**: Check `docker logs litellm-gateway` or `docker logs vllm-completions`

**GPU errors**: Verify NVIDIA drivers with `nvidia-smi`

**Out of memory**: Reduce GPU allocations in `.env` (total must be ≤ 1.0)

**Slow first start**: Initial startup downloads models (15-20 minutes)

## License

MIT
