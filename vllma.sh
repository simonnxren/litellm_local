#!/bin/bash
# Machine A (32GB) - completions, ocr
cd "$(dirname "$0")"
[ -f .env.example ] && export $(grep -v '^#' .env.example | xargs)

G='\033[32m' R='\033[31m' N='\033[0m'

case "${1:-status}" in
    start)
        docker network inspect vllm-shared-network &>/dev/null || docker network create vllm-shared-network
        docker compose -f docker-compose.vllma.yml up -d
        ;;
    stop)  docker compose -f docker-compose.vllma.yml down ;;
    status)
        for p in completions:8101 ocr:8102; do
            n=${p%:*} port=${p#*:}
            curl -s "http://localhost:$port/health" &>/dev/null && echo -e "${G}✓ $n${N}" || echo -e "${R}✗ $n${N}"
        done ;;
    logs) docker logs -f "${2:-vllm-completions}" ;;
    *) echo "Usage: $0 {start|stop|status|logs [service]}" ;;
esac
