#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "🚀 Starting vLLM services..."
docker compose -f docker-compose.vllmMin.yml up -d

echo "⏳ Waiting for vLLM services to be healthy..."
# Wait for embedding service
until docker inspect --format='{{.State.Health.Status}}' vllm-embedding 2>/dev/null | grep -q "healthy"; do
    echo "  Waiting for vllm-embedding..."
    sleep 10
done
echo "  ✅ vllm-embedding is healthy"

# Wait for completions service
until docker inspect --format='{{.State.Health.Status}}' vllm-completions 2>/dev/null | grep -q "healthy"; do
    echo "  Waiting for vllm-completions..."
    sleep 10
done
echo "  ✅ vllm-completions is healthy"

# Wait for OCR service
until docker inspect --format='{{.State.Health.Status}}' vllm-ocr 2>/dev/null | grep -q "healthy"; do
    echo "  Waiting for vllm-ocr..."
    sleep 10
done
echo "  ✅ vllm-ocr is healthy"

echo "🌐 Starting LiteLLM gateway..."
docker compose -f docker-compose.litellm.yml up -d

echo "⏳ Waiting for LiteLLM to be ready..."
sleep 5

echo ""
echo "✅ All services started!"
echo "📊 vLLM Embedding:   http://localhost:${VLLM_EMBED_PORT:-8100}"
echo "📊 vLLM Completions: http://localhost:${VLLM_COMPLETIONS_PORT:-8101}"
echo "� vLLM OCR:         http://localhost:${VLLM_OCR_PORT:-8102}"
echo "�🚪 LiteLLM Gateway:  http://localhost:${ROUTER_PORT:-8200}"
echo ""
echo "📋 Check logs: docker logs -f litellm-gateway"
