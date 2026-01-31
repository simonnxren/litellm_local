#!/bin/bash
# vLLM Deploy Script
cd "$(dirname "$0")"
[ -f .env.example ] && export $(grep -v '^#' .env.example | xargs)

G='\033[32m' Y='\033[33m' R='\033[31m' N='\033[0m'

services() {
    local s=""
    [ "${VLLM_EMBED_ENABLE:-false}" = "true" ] && s+=" vllm-embedding"
    [ "${VLLM_COMPLETIONS_ENABLE:-false}" = "true" ] && s+=" vllm-completions"
    [ "${VLLM_OCR_ENABLE:-false}" = "true" ] && s+=" vllm-ocr"
    [ "${VLLM_ASR_ENABLE:-false}" = "true" ] && s+=" vllm-asr"
    echo $s
}

case "${1:-status}" in
    start)
        s=$(services)
        [ -z "$s" ] && echo -e "${R}No services enabled${N}" && exit 1
        echo -e "${G}Starting:${N}$s"
        docker compose -f docker-compose.vllm.yml up -d $s
        ;;
    stop)
        docker compose -f docker-compose.vllm.yml down
        ;;
    status)
        for p in embed:8100 completions:8101 ocr:8102 asr:8103; do
            n=${p%:*} port=${p#*:} en="VLLM_${n^^}_ENABLE"
            [ "${!en:-false}" != "true" ] && echo -e "${Y}○ $n${N}" && continue
            curl -s "http://localhost:$port/health" >/dev/null 2>&1 \
                && echo -e "${G}✓ $n${N}" || echo -e "${R}✗ $n${N}"
        done
        ;;
    logs) docker logs -f "vllm-${2:-completions}" ;;
    *) echo "Usage: $0 {start|stop|status|logs [service]}" ;;
esac
