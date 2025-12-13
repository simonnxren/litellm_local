#!/bin/bash
set -e

# Check if Ollama is running
curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || { echo "❌ Ollama not running on localhost:11434"; exit 1; }

# Pull required models
echo "📦 Pulling Ollama models..."
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:8b
ollama pull deepseek-ocr:3b

# Start LiteLLM Gateway
echo "🚀 Starting LiteLLM Gateway..."
docker compose -f docker-compose.litellm.yml up -d

echo "✅ LiteLLM Gateway started successfully!"
echo "📋 Check logs: docker logs -f litellm-gateway"

