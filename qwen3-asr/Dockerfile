# Qwen3-ASR with Timestamps - based on vLLM Blackwell nightly for CUDA compatibility
FROM vllm-blackwell-nightly:latest

# Default model config
ENV ASR_MODEL="Qwen/Qwen3-ASR-0.6B"
ENV ALIGNER_MODEL="Qwen/Qwen3-ForcedAligner-0.6B"
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEVICE_MAP=cuda:0
ENV DTYPE=bfloat16

# Install additional deps for ASR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install qwen-asr and flask
RUN uv pip install qwen-asr flask gunicorn

COPY serve.py /app/serve.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Reset entrypoint from vllm base image
ENTRYPOINT []
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "300", "serve:create_app()"]
