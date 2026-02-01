#!/bin/bash
# LiteLLM Gateway
cd "$(dirname "$0")"

G='\033[32m' R='\033[31m' N='\033[0m'

case "${1:-status}" in
    start)
        docker network inspect vllm-shared-network &>/dev/null || docker network create vllm-shared-network
        docker compose -f docker-compose.litellm.yml up -d
        ;;
    stop)  docker compose -f docker-compose.litellm.yml down ;;
    status)
        curl -s "http://localhost:8200/health" &>/dev/null && echo -e "${G}✓ litellm${N}" || echo -e "${R}✗ litellm${N}"
        ;;
    logs) docker logs -f litellm-gateway ;;
    *) echo "Usage: $0 {start|stop|status|logs}" ;;
esac
