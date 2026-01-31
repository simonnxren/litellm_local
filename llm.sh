#!/bin/bash
# LiteLLM Local - Unified Management Script
# Usage: ./llm.sh <command> [options]

set -e
cd "$(dirname "$0")"

# Load environment if exists
[ -f .env ] && source .env

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default ports
ROUTER_PORT=${ROUTER_PORT:-8200}
VLLM_EMBED_PORT=${VLLM_EMBED_PORT:-8100}
VLLM_COMPLETIONS_PORT=${VLLM_COMPLETIONS_PORT:-8101}
VLLM_OCR_PORT=${VLLM_OCR_PORT:-8102}

show_help() {
    echo "LiteLLM Local - Management Script"
    echo ""
    echo "Usage: ./llm.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [mode]    Start services (modes: vllm, ollama)"
    echo "  stop            Stop all services"
    echo "  restart         Restart all services"
    echo "  status          Show service status"
    echo "  logs [service]  Tail logs (services: gateway, embedding, completions, ocr)"
    echo "  info            Show network info and access examples"
    echo "  health          Check service health"
    echo ""
    echo "Examples:"
    echo "  ./llm.sh start          # Start vLLM services"
    echo "  ./llm.sh start ollama   # Use Ollama backend"
    echo "  ./llm.sh logs gateway   # Tail LiteLLM gateway logs"
    echo "  ./llm.sh status         # Show all service status"
}

wait_for_health() {
    local container=$1
    local max_wait=${2:-300}
    local elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        if docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null | grep -q "healthy"; then
            return 0
        fi
        echo -e "  ${YELLOW}Waiting for $container...${NC}"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo -e "  ${RED}Timeout waiting for $container${NC}"
    return 1
}

cmd_start() {
    local mode=${1:-vllm}
    
    case $mode in
        vllm)
            echo -e "${BLUE}🚀 Starting vLLM services...${NC}"
            docker compose -f docker-compose.vllm.yml up -d
            
            echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
            wait_for_health vllm-embedding && echo -e "  ${GREEN}✅ vllm-embedding${NC}"
            wait_for_health vllm-completions && echo -e "  ${GREEN}✅ vllm-completions${NC}"
            wait_for_health vllm-ocr && echo -e "  ${GREEN}✅ vllm-ocr${NC}"
            wait_for_health vllm-whisper && echo -e "  ${GREEN}✅ vllm-whisper${NC}" || true
            ;;
        ollama)
            echo -e "${BLUE}🚀 Starting with Ollama backend...${NC}"
            # Check Ollama is running
            if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo -e "${RED}❌ Ollama not running on localhost:11434${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Ollama detected${NC}"
            
            # Pull models
            echo -e "${YELLOW}📦 Pulling Ollama models...${NC}"
            ollama pull qwen3-embedding:0.6b || true
            ollama pull qwen3:8b || true
            ;;
        *)
            echo -e "${RED}Unknown mode: $mode${NC}"
            echo "Valid modes: vllm, full, ollama"
            exit 1
            ;;
    esac
    
    # Start LiteLLM Gateway
    echo -e "${BLUE}🌐 Starting LiteLLM gateway...${NC}"
    docker compose -f docker-compose.litellm.yml up -d
    sleep 3
    
    echo ""
    echo -e "${GREEN}✅ All services started!${NC}"
    cmd_info
}

cmd_stop() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker compose -f docker-compose.litellm.yml down 2>/dev/null || true
    docker compose -f docker-compose.vllm.yml down 2>/dev/null || true
    echo -e "${GREEN}✅ All services stopped${NC}"
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start "${1:-vllm}"
}

cmd_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo ""
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "litellm|vllm|NAMES" || echo "No services running"
}

cmd_logs() {
    local service=${1:-gateway}
    
    case $service in
        gateway|litellm)
            docker logs -f litellm-gateway
            ;;
        embedding|embed)
            docker logs -f vllm-embedding
            ;;
        completions|chat)
            docker logs -f vllm-completions
            ;;
        ocr|vision)
            docker logs -f vllm-ocr
            ;;
        whisper|audio)
            docker logs -f vllm-whisper
            ;;
        *)
            echo "Unknown service: $service"
            echo "Valid services: gateway, embedding, completions, ocr, whisper"
            exit 1
            ;;
    esac
}

cmd_info() {
    # Get server IP
    local server_ip
    if command -v hostname &>/dev/null; then
        server_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
    else
        server_ip="localhost"
    fi
    [ -z "$server_ip" ] && server_ip="localhost"
    
    echo ""
    echo -e "${BLUE}🌐 Access Information${NC}"
    echo ""
    echo -e "  ${GREEN}Local:${NC}  http://localhost:$ROUTER_PORT"
    echo -e "  ${GREEN}LAN:${NC}    http://$server_ip:$ROUTER_PORT"
    echo ""
    echo -e "${BLUE}📝 Python Example${NC}"
    echo ""
    echo "  from openai import OpenAI"
    echo "  client = OpenAI("
    echo "      base_url=\"http://$server_ip:$ROUTER_PORT/v1\","
    echo "      api_key=\"dummy\""
    echo "  )"
    echo ""
    echo -e "${BLUE}🔗 cURL Test${NC}"
    echo ""
    echo "  curl http://$server_ip:$ROUTER_PORT/v1/models"
}

cmd_health() {
    echo -e "${BLUE}🏥 Health Check${NC}"
    echo ""
    
    # Check LiteLLM
    if curl -sf http://localhost:$ROUTER_PORT/health >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅${NC} LiteLLM Gateway (http://localhost:$ROUTER_PORT)"
    else
        echo -e "  ${RED}❌${NC} LiteLLM Gateway (http://localhost:$ROUTER_PORT)"
    fi
    
    # Check vLLM services
    for port_var in VLLM_EMBED_PORT VLLM_COMPLETIONS_PORT VLLM_OCR_PORT; do
        port=${!port_var}
        name=${port_var/VLLM_/}
        name=${name/_PORT/}
        if curl -sf http://localhost:$port/health >/dev/null 2>&1; then
            echo -e "  ${GREEN}✅${NC} vLLM $name (http://localhost:$port)"
        else
            echo -e "  ${YELLOW}⚠️${NC}  vLLM $name (http://localhost:$port) - not running or unhealthy"
        fi
    done
}

# Main
case "${1:-}" in
    start)
        cmd_start "$2"
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart "$2"
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$2"
        ;;
    info)
        cmd_info
        ;;
    health)
        cmd_health
        ;;
    -h|--help|help|"")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
