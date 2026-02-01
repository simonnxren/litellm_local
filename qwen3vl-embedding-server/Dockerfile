# Qwen3-VL-Embedding Server with OpenAI-compatible API
# Supports text, image, and video embeddings
# Optimized for GTX 1070 Ti (8GB VRAM) with the 2B model

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support (compatible with Pascal/GTX 1070 Ti)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone the official Qwen3-VL-Embedding repository
RUN git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git /app/qwen3vl

# Install dependencies from the repo's requirements
# Based on the setup_environment.sh requirements
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=0.26.0 \
    qwen-vl-utils>=0.0.8 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    numpy>=1.24.0 \
    decord>=0.6.0 \
    av>=10.0.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.0.0

# Copy the API server
COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV MODEL_NAME=Qwen/Qwen3-VL-Embedding-2B
ENV MAX_LENGTH=8192
ENV DEVICE=cuda
# Ensure HuggingFace uses the cache volume
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

CMD ["python", "server.py"]
