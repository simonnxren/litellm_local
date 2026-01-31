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
./llm.sh start

# 4. Verify
curl http://localhost:8200/health
```

### Management Commands

```bash
./llm.sh start          # Start vLLM services
./llm.sh start ollama   # Use Ollama backend instead
./llm.sh stop           # Stop all services
./llm.sh restart        # Restart services
./llm.sh status         # Show service status
./llm.sh logs gateway   # Tail gateway logs
./llm.sh info           # Show access info
./llm.sh health         # Check service health
```

## LAN Access

Services are accessible from any device on your local network:

```bash
# Check network configuration and get your IP
./llm.sh info

# Test from another device (replace with your IP)
curl http://192.168.1.100:8200/health
```

**Firewall Configuration** (if needed):
```bash
# Ubuntu/Debian
sudo ufw allow 8200/tcp  # LiteLLM Gateway
sudo ufw allow 8100/tcp  # vLLM Embedding
sudo ufw allow 8101/tcp  # vLLM Completions
sudo ufw allow 8102/tcp  # vLLM OCR

# Or allow from specific subnet only
sudo ufw allow from 192.168.1.0/24 to any port 8200
```

**Usage from LAN devices**:
```python
# Replace localhost with server IP
client = OpenAI(
    base_url="http://192.168.1.100:8200/v1",
    api_key="dummy"
)
```

## Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8200/v1", api_key="dummy")

# Embeddings
response = client.embeddings.create(model="qwen3-embedding-0.6b", input="Hello")

# Chat
response = client.chat.completions.create(
    model="qwen3-8b-fp8",
    messages=[{"role": "user", "content": "Hello"}]
)
```

See [API_USAGE_MINIMAL.md](API_USAGE_MINIMAL.md) for more examples.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List models |
| `POST /v1/embeddings` | Generate embeddings |
| `POST /v1/chat/completions` | Chat (supports vision) |

## Configuration

See [.env.example](.env.example) for environment variables.

## Troubleshooting

- **Services not starting**: `docker logs litellm-gateway`
- **GPU errors**: `nvidia-smi`
- **Out of memory**: Reduce GPU allocations in `.env`

## License

MIT

