#!/bin/bash
# =============================================================================
# OMP Health Check Script
# =============================================================================
#
# Usage:
#   ./health-check.sh           # Check all services
#   ./health-check.sh --json    # Output JSON format
#   ./health-check.sh --quiet   # Exit code only (0=healthy, 1=unhealthy)
#
# Cron example (check every 5 minutes):
#   */5 * * * * /opt/alpha-forge/quant_b3_backtest/scripts/vps/health-check.sh --quiet || /opt/alpha-forge/quant_b3_backtest/scripts/vps/restart-services.sh

set -e

# Configuration
API_URL="${API_URL:-http://localhost:3001}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:5173}"
TIMEOUT=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Parse arguments
JSON_OUTPUT=false
QUIET=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --json) JSON_OUTPUT=true ;;
        --quiet) QUIET=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check functions
check_api_health() {
    local status
    if status=$(curl -s --max-time $TIMEOUT "$API_URL/api/health" 2>/dev/null); then
        echo "ok"
    else
        echo "fail"
    fi
}

check_omp_status() {
    local status
    if status=$(curl -s --max-time $TIMEOUT "$API_URL/api/omp/status" 2>/dev/null); then
        echo "$status" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown"
    else
        echo "unreachable"
    fi
}

check_dashboard() {
    if curl -s --max-time $TIMEOUT -o /dev/null -w "%{http_code}" "$DASHBOARD_URL" | grep -q "200\|304"; then
        echo "ok"
    else
        echo "fail"
    fi
}

check_pm2() {
    if command -v pm2 &> /dev/null && pm2 list 2>/dev/null | grep -q "online"; then
        echo "ok"
    else
        echo "fail"
    fi
}

get_pm2_processes() {
    if command -v pm2 &> /dev/null; then
        pm2 jlist 2>/dev/null | jq '[.[] | {name: .name, status: .pm2_env.status, memory: .monit.memory, cpu: .monit.cpu, restarts: .pm2_env.restart_time}]' 2>/dev/null || echo "[]"
    else
        echo "[]"
    fi
}

get_system_resources() {
    local cpu_usage mem_usage disk_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 2>/dev/null || echo "0")
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100}' 2>/dev/null || echo "0")
    disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%' 2>/dev/null || echo "0")
    echo "{\"cpu\": $cpu_usage, \"memory\": $mem_usage, \"disk\": $disk_usage}"
}

# Run checks
api_health=$(check_api_health)
omp_status=$(check_omp_status)
dashboard_health=$(check_dashboard)
pm2_health=$(check_pm2)
resources=$(get_system_resources)

# Determine overall health
all_healthy=true
[[ "$api_health" != "ok" ]] && all_healthy=false
[[ "$dashboard_health" != "ok" ]] && all_healthy=false
[[ "$pm2_health" != "ok" ]] && all_healthy=false

# Output
if $JSON_OUTPUT; then
    cat << EOF
{
  "timestamp": "$(date -Iseconds)",
  "healthy": $all_healthy,
  "services": {
    "api": "$api_health",
    "omp": "$omp_status",
    "dashboard": "$dashboard_health",
    "pm2": "$pm2_health"
  },
  "resources": $resources,
  "pm2_processes": $(get_pm2_processes)
}
EOF
elif $QUIET; then
    if $all_healthy; then
        exit 0
    else
        exit 1
    fi
else
    echo "=========================================="
    echo "  Alpha Forge Health Check"
    echo "  $(date)"
    echo "=========================================="
    echo ""
    
    # Services
    echo "Services:"
    if [[ "$api_health" == "ok" ]]; then
        echo -e "  ${GREEN}✓${NC} API Server: healthy"
    else
        echo -e "  ${RED}✗${NC} API Server: $api_health"
    fi
    
    if [[ "$omp_status" == "running" ]]; then
        echo -e "  ${GREEN}✓${NC} OMP Mining: $omp_status"
    elif [[ "$omp_status" == "offline" ]]; then
        echo -e "  ${YELLOW}○${NC} OMP Mining: $omp_status"
    else
        echo -e "  ${RED}✗${NC} OMP Mining: $omp_status"
    fi
    
    if [[ "$dashboard_health" == "ok" ]]; then
        echo -e "  ${GREEN}✓${NC} Dashboard: healthy"
    else
        echo -e "  ${RED}✗${NC} Dashboard: $dashboard_health"
    fi
    
    if [[ "$pm2_health" == "ok" ]]; then
        echo -e "  ${GREEN}✓${NC} PM2: running"
    else
        echo -e "  ${RED}✗${NC} PM2: $pm2_health"
    fi
    
    # Resources
    echo ""
    echo "Resources:"
    cpu=$(echo $resources | jq -r '.cpu')
    mem=$(echo $resources | jq -r '.memory')
    disk=$(echo $resources | jq -r '.disk')
    
    echo "  CPU Usage:  ${cpu}%"
    echo "  Memory:     ${mem}%"
    echo "  Disk:       ${disk}%"
    
    # Overall status
    echo ""
    if $all_healthy; then
        echo -e "${GREEN}Overall: HEALTHY${NC}"
    else
        echo -e "${RED}Overall: UNHEALTHY${NC}"
    fi
fi




















