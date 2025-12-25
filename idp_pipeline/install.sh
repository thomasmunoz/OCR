#!/bin/bash
# IDP Pipeline - One-Command Installer
# Usage: curl -sSL <url>/install.sh | bash
# Or: ./install.sh

set -e

echo "🚀 IDP Pipeline Installer"
echo "========================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect OS
OS=$(uname -s)

# Step 1: Check Docker
echo -e "\n${BLUE}[1/4]${NC} Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    if [[ "$OS" == "Linux" ]]; then
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker $USER
        echo "⚠️  Log out and back in for Docker permissions to take effect"
    else
        echo "Please install Docker Desktop from https://docker.com"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} Docker $(docker --version | cut -d' ' -f3)"

# Step 2: Create data directories
echo -e "\n${BLUE}[2/4]${NC} Creating data directories..."
DATA_DIR="${IDP_DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"/{models,uploads,output}
echo -e "${GREEN}✓${NC} Data directory: $DATA_DIR"

# Step 3: Create production docker-compose
echo -e "\n${BLUE}[3/4]${NC} Generating docker-compose.prod.yml..."
cat > docker-compose.prod.yml << 'EOF'
# IDP Pipeline - Production Ready
# Scale: docker compose -f docker-compose.prod.yml up -d --scale worker=4

services:
  redis:
    image: redis:7-alpine
    container_name: idp-redis
    restart: always
    expose:
      - "6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      target: production
    container_name: idp-api
    restart: always
    ports:
      - "${IDP_PORT:-8080}:8080"
    environment:
      - IDP_QUEUE_TYPE=redis
      - IDP_REDIS_URL=redis://redis:6379/0
      - IDP_UPLOAD_DIR=/data/uploads
      - IDP_OUTPUT_DIR=/data/output
      - IDP_MODELS_DIR=/data/models
    volumes:
      - ./data/uploads:/data/uploads
      - ./data/output:/data/output
      - ./data/models:/data/models
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  worker:
    build:
      context: .
      target: worker
    restart: always
    environment:
      - IDP_QUEUE_TYPE=redis
      - IDP_REDIS_URL=redis://redis:6379/0
      - IDP_UPLOAD_DIR=/data/uploads
      - IDP_OUTPUT_DIR=/data/output
      - IDP_MODELS_DIR=/data/models
      - IDP_VRAM_GB=${IDP_VRAM_GB:-4.0}
      - HF_HOME=/data/models
    volumes:
      - ./data/uploads:/data/uploads:ro
      - ./data/output:/data/output
      - ./data/models:/data/models
    depends_on:
      redis:
        condition: service_healthy
      api:
        condition: service_healthy
    deploy:
      replicas: ${IDP_WORKERS:-1}

networks:
  default:
    name: idp-network
EOF
echo -e "${GREEN}✓${NC} Created docker-compose.prod.yml"

# Step 4: Build and start
echo -e "\n${BLUE}[4/4]${NC} Building and starting services..."
docker compose -f docker-compose.prod.yml up -d --build

# Wait for health
echo -e "\n⏳ Waiting for services to be ready..."
sleep 10

# Check status
echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓ IDP Pipeline Installed Successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo "📍 Web UI:    http://localhost:${IDP_PORT:-8080}"
echo "📂 Data:      $DATA_DIR/"
echo ""
echo "📋 Useful commands:"
echo "   docker compose -f docker-compose.prod.yml logs -f        # View logs"
echo "   docker compose -f docker-compose.prod.yml up -d --scale worker=4  # Scale to 4 workers"
echo "   docker compose -f docker-compose.prod.yml down           # Stop"
echo ""
echo "🔍 Check status:"
docker compose -f docker-compose.prod.yml ps
