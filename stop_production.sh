#!/bin/bash
echo "=================================================="
echo "   Hybrid-IDS Framework - Stopping Services"
echo "=================================================="
echo ""

LOGFILE="logs/production.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping all services..." >> "$LOGFILE"

docker compose down

echo ""
echo "✅ All services stopped successfully."
echo "Log saved to logs/production.log"
echo ""