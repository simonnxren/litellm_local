#!/bin/bash
# Unified LLM Management Script
cd "$(dirname "$0")"

G='\033[32m' Y='\033[33m' R='\033[31m' B='\033[34m' N='\033[0m'

case "${1:-status}" in
    start)
        echo -e "${B}Starting vLLM services...${N}"
        ./vllm.sh start
        echo -e "${B}Starting LiteLLM gateway...${N}"
        python3 litellm_server.py start
        ;;
    stop)
        echo -e "${B}Stopping LiteLLM gateway...${N}"
        python3 litellm_server.py stop
        echo -e "${B}Stopping vLLM services...${N}"
        ./vllm.sh stop
        ;;
    status)
        echo -e "${B}vLLM Services:${N}"
        ./vllm.sh status
        echo -e "${B}LiteLLM Gateway:${N}"
        python3 litellm_server.py status
        ;;
    logs)
        if [ "$2" == "gateway" ]; then
            docker logs -f litellm-gateway
        else
            ./vllm.sh logs "$2"
        fi
        ;;
    info)
        IP=$(hostname -I | awk '{print $1}')
        PORT=$(grep ROUTER_PORT .env.example | cut -d'=' -f2)
        [ -z "$PORT" ] && PORT=8200
        echo -e "${B}Access Info:${N}"
        echo -e "  Local:   http://localhost:$PORT/v1"
        echo -e "  Network: http://$IP:$PORT/v1"
        ;;
    health)
        echo -e "${B}Checking Health...${N}"
        ./vllm.sh status
        python3 litellm_server.py status
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs [service]|info|health}"
        ;;
esac
