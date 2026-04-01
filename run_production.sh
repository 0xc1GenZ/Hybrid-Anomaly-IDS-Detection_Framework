#!/bin/bash
echo "=================================================="
echo "   Hybrid-IDS Framework - Production Deployment"
echo "=================================================="
echo ""

# Create logs directory
mkdir -p logs
LOGFILE="logs/production.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting production deployment..." >> "$LOGFILE"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop / Engine."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Docker not running" >> "$LOGFILE"
    exit 1
fi

echo "[1/3] Stopping any old containers..."
docker compose down >> "$LOGFILE" 2>&1

echo "[2/3] Building and starting services..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: docker compose up --build -d" >> "$LOGFILE"
docker compose up --build -d

echo ""
echo "=================================================="
echo "✅ Deployment Successful!"
echo "=================================================="
echo ""
echo "📊 Streamlit Dashboard → http://localhost:8501"
echo "🔌 Flask REST API     → http://localhost:5000"
echo "📝 Log file saved at: logs/production.log"
echo ""
echo "To stop the services, run: ./stop_production.sh"
echo ""