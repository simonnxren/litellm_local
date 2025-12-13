#!/bin/bash
# Network configuration helper for LiteLLM Local

echo "=== LiteLLM Local - Network Information ==="
echo ""

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "🌐 Server IP: $SERVER_IP"
echo ""

# Check if services are running
echo "📊 Service Status:"
echo "  LiteLLM Gateway: http://$SERVER_IP:8200"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "litellm|vllm" || echo "  ⚠️  No services running"
echo ""

# Test local connectivity
echo "🔍 Testing local connectivity:"
if curl -s http://localhost:8200/health > /dev/null 2>&1; then
    echo "  ✅ Local access: http://localhost:8200"
else
    echo "  ❌ Local access failed"
fi

# Test LAN connectivity
if curl -s http://$SERVER_IP:8200/health > /dev/null 2>&1; then
    echo "  ✅ LAN access: http://$SERVER_IP:8200"
else
    echo "  ❌ LAN access failed (check firewall)"
fi
echo ""

# Check firewall status
echo "🔥 Firewall Status:"
if command -v ufw &> /dev/null; then
    if sudo ufw status | grep -q "Status: active"; then
        echo "  UFW is active"
        echo "  Port 8200 status:"
        sudo ufw status | grep 8200 || echo "    ⚠️  Port 8200 not allowed"
    else
        echo "  UFW is inactive"
    fi
elif command -v firewall-cmd &> /dev/null; then
    if sudo firewall-cmd --state 2>/dev/null | grep -q "running"; then
        echo "  Firewalld is running"
        sudo firewall-cmd --list-ports | grep -q 8200 || echo "    ⚠️  Port 8200 not open"
    else
        echo "  Firewalld is not running"
    fi
else
    echo "  No firewall detected (ufw/firewalld)"
fi
echo ""

# Usage examples
echo "📝 LAN Access Examples:"
echo ""
echo "Python:"
echo "  from openai import OpenAI"
echo "  client = OpenAI("
echo "      base_url=\"http://$SERVER_IP:8200/v1\","
echo "      api_key=\"dummy\""
echo "  )"
echo ""
echo "cURL:"
echo "  curl http://$SERVER_IP:8200/v1/models"
echo ""
echo "JavaScript/Node.js:"
echo "  const openai = new OpenAI({"
echo "    baseURL: \"http://$SERVER_IP:8200/v1\","
echo "    apiKey: \"dummy\""
echo "  });"
echo ""

# Firewall setup recommendation
echo "🛡️  To allow LAN access (if needed):"
echo "  sudo ufw allow 8200/tcp"
echo "  # Or for specific subnet:"
echo "  sudo ufw allow from 192.168.1.0/24 to any port 8200"
