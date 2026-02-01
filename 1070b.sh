#!/bin/bash
# Machine B (GTX 1070 Ti) Service Manager
# Services: Qwen3-VL-Embedding (8100), WhisperX (8103)
cd "$(dirname "$0")"

G='\033[32m' R='\033[31m' Y='\033[33m' B='\033[34m' N='\033[0m'

check_service() {
    curl -s "http://localhost:$1/health" &>/dev/null && echo -e "${G}✓${N}" || echo -e "${R}✗${N}"
}

case "${1:-status}" in
    build)
        echo -e "${Y}Building all services...${N}"
        docker compose -f docker-compose.1070b.yml build
        ;;
    start)
        docker network inspect 1070b-network &>/dev/null || docker network create 1070b-network
        echo -e "${Y}Starting Machine B services...${N}"
        docker compose -f docker-compose.1070b.yml up -d
        echo -e "${B}Services starting. Use '$0 status' to check health.${N}"
        ;;
    stop)
        docker compose -f docker-compose.1070b.yml down
        ;;
    status)
        echo -e "${B}Machine B (GTX 1070 Ti) Services:${N}"
        echo -e "  qwen3vl-embedding (8100): $(check_service 8100)"
        echo -e "  whisperx          (8103): $(check_service 8103)"
        ;;
    logs)
        if [ -z "$2" ]; then
            docker compose -f docker-compose.1070b.yml logs -f
        else
            docker logs -f "$2"
        fi
        ;;
    test-embedding)
        echo "Testing embedding service..."
        curl -s -X POST "http://localhost:8100/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d '{"input": "Hello world", "model": "Qwen/Qwen3-VL-Embedding-2B"}' \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'✓ Embedding dim: {len(d[\"data\"][0][\"embedding\"])}')" 2>/dev/null \
            || echo -e "${R}✗ Failed${N}"
        ;;
    test-asr)
        if [ -z "$2" ]; then
            echo "Usage: $0 test-asr <audio_file>"
            exit 1
        fi
        echo "Testing ASR service..."
        curl -s -X POST "http://localhost:8103/v1/audio/transcriptions" \
            -F "file=@$2" -F "model=large-v3-turbo" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'✓ Transcription: {d[\"text\"][:100]}...')" 2>/dev/null \
            || echo -e "${R}✗ Failed${N}"
        ;;
    *)
        echo "Machine B (GTX 1070 Ti) Service Manager"
        echo ""
        echo "Usage: $0 {build|start|stop|status|logs|test-embedding|test-asr}"
        echo ""
        echo "Commands:"
        echo "  build          - Build all Docker images"
        echo "  start          - Start all services"
        echo "  stop           - Stop all services"
        echo "  status         - Check service health"
        echo "  logs [name]    - View logs (all or specific container)"
        echo "  test-embedding - Test embedding service"
        echo "  test-asr FILE  - Test ASR with audio file"
        echo ""
        echo "Services:"
        echo "  qwen3vl-embedding - Port 8100 (embeddings)"
        echo "  whisperx          - Port 8103 (ASR)"
        ;;
esac
