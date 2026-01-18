#!/bin/bash
# Local Health Check Script
# Run every 5 minutes via cron:
# */5 * * * * /path/to/local-health-check.sh >> /var/log/omp-health.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:3001}"
ALERT_THRESHOLD_CPU=90
ALERT_THRESHOLD_MEM=85
ALERT_THRESHOLD_DISK=80

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Timestamp
echo "=== Health Check: $(date -Iseconds) ==="

# Track failures
FAILURES=0

# Check 1: API Health
echo -n "API Health: "
if curl -s -f "${DASHBOARD_URL}/api/health" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    FAILURES=$((FAILURES + 1))
fi

# Check 2: OMP Status
echo -n "OMP Status: "
OMP_STATUS=$(curl -s "${DASHBOARD_URL}/api/omp/status" 2>/dev/null || echo '{"status":"error"}')
if echo "$OMP_STATUS" | grep -q '"status":"running"'; then
    echo -e "${GREEN}running${NC}"
elif echo "$OMP_STATUS" | grep -q '"status":"paused"'; then
    echo -e "${YELLOW}paused${NC}"
elif echo "$OMP_STATUS" | grep -q '"status":"offline"'; then
    echo -e "${YELLOW}offline${NC}"
else
    echo -e "${RED}error${NC}"
    FAILURES=$((FAILURES + 1))
fi

# Check 3: CPU Usage
echo -n "CPU Usage: "
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)
if [ "$CPU_USAGE" -lt "$ALERT_THRESHOLD_CPU" ]; then
    echo -e "${GREEN}${CPU_USAGE}%${NC}"
else
    echo -e "${RED}${CPU_USAGE}% (threshold: ${ALERT_THRESHOLD_CPU}%)${NC}"
    FAILURES=$((FAILURES + 1))
fi

# Check 4: Memory Usage
echo -n "Memory Usage: "
MEM_USAGE=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
if [ "$MEM_USAGE" -lt "$ALERT_THRESHOLD_MEM" ]; then
    echo -e "${GREEN}${MEM_USAGE}%${NC}"
else
    echo -e "${RED}${MEM_USAGE}% (threshold: ${ALERT_THRESHOLD_MEM}%)${NC}"
    FAILURES=$((FAILURES + 1))
fi

# Check 5: Disk Usage
echo -n "Disk Usage: "
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {gsub("%",""); print $5}')
if [ "$DISK_USAGE" -lt "$ALERT_THRESHOLD_DISK" ]; then
    echo -e "${GREEN}${DISK_USAGE}%${NC}"
else
    echo -e "${YELLOW}${DISK_USAGE}% (threshold: ${ALERT_THRESHOLD_DISK}%)${NC}"
    # Trigger cleanup if disk usage is high
    if [ "$DISK_USAGE" -ge 85 ]; then
        echo "  -> Triggering auto-cleanup..."
        "$SCRIPT_DIR/auto_cleanup.sh" --runs --days 3 2>/dev/null || true
    fi
fi

# Check 6: Output Directory Size
echo -n "Output Dir Size: "
OUTPUT_SIZE=$(du -sh "$PROJECT_ROOT/output/scg" 2>/dev/null | cut -f1 || echo "0")
echo "${OUTPUT_SIZE}"

# Check 7: Hall of Fame Count
echo -n "Hall of Fame: "
HOF_COUNT=$(ls -1 "$PROJECT_ROOT/artifacts/hall_of_fame" 2>/dev/null | wc -l || echo "0")
echo "${HOF_COUNT} strategies"

# Check 8: Recent Campaign Activity
echo -n "Last Campaign: "
LAST_RUN=$(ls -t "$PROJECT_ROOT/output/scg" 2>/dev/null | head -1 || echo "none")
echo "${LAST_RUN}"

# Summary
echo "---"
if [ "$FAILURES" -eq 0 ]; then
    echo -e "Status: ${GREEN}HEALTHY${NC} (0 failures)"
    exit 0
else
    echo -e "Status: ${RED}DEGRADED${NC} (${FAILURES} failure(s))"
    exit 1
fi
