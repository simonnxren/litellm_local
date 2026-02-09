# vLLM Blackwell Nightly

vLLM deployment for RTX 5090 (Blackwell / sm_100) using CUDA 13.0.

## Quick Start

```bash
# Start all services
docker compose -f docker-compose.glm-ocr-qwen3-asr.yml up -d

# Check status
docker compose -f docker-compose.glm-ocr-qwen3-asr.yml ps
```

## Services

| Service | Port | Model | GPU Memory | Purpose |
|---------|------|-------|------------|---------|
| glm-ocr | 8080 | zai-org/GLM-OCR | 15% | OCR for images |
| qwen3-vl-embedding | 8090 | shigureui/Qwen3-VL-Embedding-2B-FP8 | 15% | Multimodal embeddings |
| qwen3-asr | 8000 | Qwen/Qwen3-ASR-1.7B | 20% | Speech recognition |
| qwen3-vl-4b | 8070 | Qwen/Qwen3-VL-4B-Instruct-FP8 | 38% | Vision-language chat |

**Total:** ~88% GPU memory utilization on 32GB RTX 5090

## System Requirements

- **GPU:** NVIDIA RTX 5090 (32GB)
- **Driver:** 580.x+
- **CUDA:** 13.0
- **Image:** `vllm/vllm-openai:cu130-nightly`

## Files

| File | Purpose |
|------|---------|
| `docker-compose.glm-ocr-qwen3-asr.yml` | Main stack with all 4 services |
| `Dockerfile.vllm-nightly` | Custom build (for reference) |
| `quantize_glm_ocr_fp8.py` | FP8 quantization script |

## API Usage

### OCR (GLM-OCR)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-OCR",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      {"type": "text", "text": "OCR this image"}
    ]}],
    "max_tokens": 500
  }'
```

### ASR (Qwen3-ASR)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{"role": "user", "content": [
      {"type": "input_audio", "input_audio": {"data": "BASE64_WAV", "format": "wav"}},
      {"type": "text", "text": "<|audio_bos|><|AUDIO|><|audio_eos|>"}
    ]}],
    "max_tokens": 500
  }'
```

## Troubleshooting

### Error 803: CUDA Driver Mismatch
Use `vllm/vllm-openai:cu130-nightly` (not `nightly`)

### libcuda.so Conflict
Container startup removes compat libcuda:
```bash
rm -f /usr/local/cuda/compat/libcuda.so* && ldconfig
```
(Already in docker-compose entrypoint)

### New Model Architectures
Install latest transformers:
```bash
pip install https://github.com/huggingface/transformers/archive/refs/heads/main.zip
```
(Already in docker-compose entrypoint)
