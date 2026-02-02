# Dockerfile for Qwen3-ASR with Transformers inference backend
# Lighter weight alternative to vLLM, using Hugging Face Transformers
#
# Supported ASR Models:
#   - Qwen/Qwen3-ASR-1.7B (default, higher accuracy)
#   - Qwen/Qwen3-ASR-0.6B (smaller, faster, less VRAM)
#
# Build:
#   docker build -t qwen3-asr-transformers .
#
# Run (interactive):
#   docker run --gpus all -it --rm -p 8000:8000 qwen3-asr-transformers
#
# Run (API server):
#   docker run --gpus all -d --rm -p 8000:8000 qwen3-asr-transformers serve
#
# Run with smaller model (less VRAM):
#   docker run --gpus all -d --rm -p 8000:8000 \
#     -e ASR_MODEL=Qwen/Qwen3-ASR-0.6B \
#     qwen3-asr-transformers serve
#
# Run with model cache mounted:
#   docker run --gpus all -it --rm -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     qwen3-asr-transformers

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

# Model configuration (can be overridden at runtime)
# ASR_MODEL options: Qwen/Qwen3-ASR-1.7B (default) or Qwen/Qwen3-ASR-0.6B
ENV ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
ENV ALIGNER_MODEL="Qwen/Qwen3-ForcedAligner-0.6B"
ENV MAX_BATCH_SIZE=32
ENV MAX_NEW_TOKENS=256
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEVICE_MAP=cuda:0
ENV DTYPE=bfloat16

# Install system dependencies
RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    curl \
    vim \
    libsndfile1 \
    libsox-dev \
    sox \
    ccache \
    software-properties-common \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# Install CMake (required for flash-attn)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

# Symlink python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Initialize git lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip3 install --no-cache-dir -U pip setuptools wheel

# Remove conflicting package (if present)
RUN apt remove python3-blinker -y 2>/dev/null || true

# Install qwen-asr with transformers backend (minimal installation)
RUN pip3 install --no-cache-dir -U qwen-asr

# Install additional dependencies for the API server
RUN pip3 install --no-cache-dir flask gunicorn

# Flash-attn is optional but recommended for better performance
# Uncomment the following lines to install flash-attn (adds ~30min to build time):
# RUN MAX_JOBS=8 pip3 install --no-cache-dir -U flash-attn --no-build-isolation || \
#     echo "Warning: flash-attn installation failed, continuing without it"

# Copy application files
COPY entrypoint.sh /app/entrypoint.sh
COPY serve.py /app/serve.py
COPY example_inference.py /app/example_inference.py

RUN chmod +x /app/entrypoint.sh

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
