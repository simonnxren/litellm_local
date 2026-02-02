# Qwen3-ASR Docker with vLLM Backend

This directory contains Docker configuration for running Qwen3-ASR with vLLM as the inference backend.

## Features

- 🚀 **vLLM Backend**: Fast inference with vLLM's optimized attention kernels
- 🎯 **Multiple Models**: Supports Qwen3-ASR-1.7B and Qwen3-ASR-0.6B
- ⏱️ **Timestamps**: Includes Qwen3-ForcedAligner for word-level timestamps
- 📡 **Multiple Interfaces**: OpenAI-compatible API, Gradio demo, and streaming demo
- 🐳 **Docker Compose**: Easy deployment with docker-compose

## Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB recommended for 1.7B model)
- **CUDA**: CUDA 12.8+ compatible driver
- **Docker**: Docker with NVIDIA Container Toolkit
- **Storage**: ~20GB for model weights

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the vLLM API server
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Option 2: Docker Build & Run

```bash
# Build the image
docker build -t qwen3-asr-vllm .

# Run vLLM API server
docker run --gpus all -d --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm serve

# Run Gradio demo
docker run --gpus all -d --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm demo

# Run streaming demo
docker run --gpus all -d --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm demo-streaming

# Run inference example
docker run --gpus all --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm inference

# Interactive shell
docker run --gpus all -it --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm bash
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR model to use |
| `ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner model |
| `GPU_MEMORY_UTILIZATION` | `0.8` | Fraction of GPU memory to use |
| `MAX_BATCH_SIZE` | `32` | Maximum inference batch size |
| `MAX_NEW_TOKENS` | `2048` | Maximum tokens to generate |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | Server port |

### Example: Use smaller model

```bash
docker run --gpus all -d --rm \
  -p 8000:8000 \
  -e ASR_MODEL=Qwen/Qwen3-ASR-0.6B \
  -e GPU_MEMORY_UTILIZATION=0.6 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=8gb \
  qwen3-asr-vllm serve
```

## API Usage

### vLLM OpenAI-Compatible API

The default `serve` mode uses `qwen-asr-serve` which wraps vLLM's OpenAI-compatible server.

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data)
content = response.json()['choices'][0]['message']['content']
print(content)

# Parse the ASR output
from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
print(f"Language: {language}")
print(f"Text: {text}")
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
        {"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
        ]}
    ]
}'
```

## Deployment Modes

### 1. vLLM API Server (Default)

```bash
docker run --gpus all -d -p 8000:8000 qwen3-asr-vllm serve
```

Uses `qwen-asr-serve` which is a wrapper around `vllm serve`.

### 2. Gradio Demo

```bash
docker run --gpus all -d -p 8000:8000 qwen3-asr-vllm demo
```

Launch a web UI at http://localhost:8000 with file upload and microphone recording.

### 3. Streaming Demo

```bash
docker run --gpus all -d -p 8000:8000 qwen3-asr-vllm demo-streaming
```

Real-time streaming transcription from microphone.

### 4. Custom Python Script

```bash
docker run --gpus all -it qwen3-asr-vllm python /path/to/your/script.py
```

### 5. Interactive Shell

```bash
docker run --gpus all -it qwen3-asr-vllm bash
```

## Pre-downloading Models

To avoid downloading models at runtime:

```bash
# Create a local directory for models
mkdir -p ~/models/qwen3-asr

# Download using huggingface-cli (outside Docker)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ~/models/qwen3-asr/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ~/models/qwen3-asr/Qwen3-ForcedAligner-0.6B

# Run with local models
docker run --gpus all -d -p 8000:8000 \
  -v ~/models/qwen3-asr:/models \
  -e ASR_MODEL=/models/Qwen3-ASR-1.7B \
  -e ALIGNER_MODEL=/models/Qwen3-ForcedAligner-0.6B \
  qwen3-asr-vllm serve
```

## GPU Memory Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Qwen3-ASR-0.6B | ~4GB | 6GB |
| Qwen3-ASR-1.7B | ~8GB | 12GB |
| + ForcedAligner-0.6B | +2GB | +4GB |

Adjust `GPU_MEMORY_UTILIZATION` if you encounter OOM errors:

```bash
docker run --gpus all -d -p 8000:8000 \
  -e GPU_MEMORY_UTILIZATION=0.6 \
  qwen3-asr-vllm serve
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `GPU_MEMORY_UTILIZATION` (e.g., 0.6)
- Use smaller model (`Qwen/Qwen3-ASR-0.6B`)
- Reduce `MAX_BATCH_SIZE`

### Slow Startup

First startup downloads models (~10GB). Mount the HuggingFace cache to persist:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```

### Container Exits Immediately

Check logs:

```bash
docker logs qwen3-asr
```

### GPU Not Detected

Verify NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Files

- `Dockerfile` - Main Docker image definition
- `docker-compose.yml` - Docker Compose configuration
- `entrypoint.sh` - Container entrypoint script
- `serve.py` - Custom Flask-based API server
- `example_inference.py` - Python inference examples

## License

Apache 2.0 (same as Qwen3-ASR)
