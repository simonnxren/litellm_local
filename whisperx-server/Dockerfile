# WhisperX ASR Server with OpenAI-compatible API
# Supports speaker diarization and word-level timestamps
# Optimized for Pascal GPUs (GTX 1070 Ti)

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Disable PyTorch 2.6+ weights_only=True default for pyannote compatibility
ENV TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install WhisperX and dependencies
RUN pip install --no-cache-dir \
    whisperx \
    faster-whisper \
    pyannote.audio \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    aiofiles

# Copy the API server
COPY server.py /app/server.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV ASR_MODEL=large-v3-turbo
ENV COMPUTE_TYPE=int8
ENV DEVICE=cuda
ENV BATCH_SIZE=16

# Run the server
CMD ["python", "server.py"]
