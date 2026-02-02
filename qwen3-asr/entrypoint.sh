#!/bin/bash
set -e

# Qwen3-ASR Docker Entrypoint Script (Transformers Backend)
# Supports multiple modes: serve, serve-dev, demo, inference, bash

# Default values (can be overridden by environment variables)
ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
ALIGNER_MODEL="${ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

# Gunicorn settings
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"

echo "=================================================="
echo "  Qwen3-ASR with Transformers Backend"
echo "=================================================="
echo "ASR Model: ${ASR_MODEL}"
echo "Aligner Model: ${ALIGNER_MODEL}"
echo "Device Map: ${DEVICE_MAP}"
echo "Dtype: ${DTYPE}"
echo "Max Batch Size: ${MAX_BATCH_SIZE}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Server: ${SERVER_HOST}:${SERVER_PORT}"
echo "=================================================="

case "${1:-serve}" in
    serve)
        echo "Starting Gunicorn production server..."
        echo "Workers: ${GUNICORN_WORKERS}, Threads: ${GUNICORN_THREADS}, Timeout: ${GUNICORN_TIMEOUT}s"
        # For CUDA models, we can't use preload (CUDA doesn't support fork)
        # Use single worker with multiple threads to handle concurrent requests
        # The model is loaded lazily on first request in the worker process
        exec gunicorn \
            --workers "${GUNICORN_WORKERS}" \
            --threads "${GUNICORN_THREADS}" \
            --bind "${SERVER_HOST}:${SERVER_PORT}" \
            --timeout "${GUNICORN_TIMEOUT}" \
            --worker-class gthread \
            --access-logfile - \
            --error-logfile - \
            --capture-output \
            'serve:app'
        ;;
    
    serve-dev)
        echo "Starting Flask development server..."
        exec python3 /app/serve.py
        ;;
    
    demo)
        echo "Starting Gradio demo with Transformers backend..."
        exec qwen-asr-demo \
            --asr-checkpoint "${ASR_MODEL}" \
            --aligner-checkpoint "${ALIGNER_MODEL}" \
            --backend transformers \
            --cuda-visible-devices 0 \
            --backend-kwargs "{\"device_map\":\"${DEVICE_MAP}\",\"dtype\":\"${DTYPE}\",\"max_inference_batch_size\":${MAX_BATCH_SIZE},\"max_new_tokens\":${MAX_NEW_TOKENS}}" \
            --aligner-kwargs "{\"device_map\":\"${DEVICE_MAP}\",\"dtype\":\"${DTYPE}\"}" \
            --ip "${SERVER_HOST}" \
            --port "${SERVER_PORT}"
        ;;
    
    inference)
        echo "Running inference example..."
        exec python3 /app/example_inference.py
        ;;
    
    python)
        shift
        exec python3 "$@"
        ;;
    
    bash|shell)
        exec /bin/bash
        ;;
    
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
