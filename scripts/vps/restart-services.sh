#!/bin/bash
# =============================================================================
# OMP Service Restart Script
# =============================================================================
#
# Usage:
#   ./restart-services.sh           # Restart all services
#   ./restart-services.sh api       # Restart API only
#   ./restart-services.sh dashboard # Restart dashboard only
#   ./restart-services.sh omp       # Restart OMP (via API)

set -e

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/opt/alpha-forge/quant_b3_backtest}"
API_URL="${API_URL:-http://localhost:3001}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

restart_api() {
    log_info "Restarting API Server..."
    cd "$PROJECT_ROOT/dashboard"
    pm2 restart api-server || {
        log_warn "API not running, starting fresh..."
        pm2 start ecosystem.config.cjs --only api-server
    }
    sleep 2
    
    # Verify
    if curl -s --max-time 5 "$API_URL/api/health" &>/dev/null; then
        log_info "API Server restarted successfully"
    else
        log_error "API Server failed to start"
        return 1
    fi
}

restart_dashboard() {
    log_info "Restarting Dashboard..."
    cd "$PROJECT_ROOT/dashboard"
    pm2 restart alpha-dashboard || {
        log_warn "Dashboard not running, starting fresh..."
        pm2 start ecosystem.config.cjs --only alpha-dashboard
    }
    sleep 2
    log_info "Dashboard restarted"
}

restart_omp() {
    log_info "Restarting OMP Mining..."
    
    # Stop OMP via API
    curl -s -X POST "$API_URL/api/omp/stop" &>/dev/null || true
    sleep 1
    
    # Start OMP via API
    if curl -s -X POST "$API_URL/api/omp/start" &>/dev/null; then
        log_info "OMP Mining restarted"
    else
        log_error "Failed to restart OMP"
        return 1
    fi
}

restart_all() {
    log_info "Restarting all services..."
    restart_api
    restart_dashboard
    
    # Wait for API to be ready
    sleep 3
    
    # Optionally restart OMP if it was running
    status=$(curl -s "$API_URL/api/omp/status" 2>/dev/null | jq -r '.status // "offline"' 2>/dev/null || echo "offline")
    if [[ "$status" == "offline" ]]; then
        log_info "OMP is offline, not auto-starting"
    fi
    
    log_info "All services restarted"
}

# Main
case "${1:-all}" in
    api)
        restart_api
        ;;
    dashboard)
        restart_dashboard
        ;;
    omp)
        restart_omp
        ;;
    all|"")
        restart_all
        ;;
    *)
        echo "Usage: $0 [api|dashboard|omp|all]"
        exit 1
        ;;
esac













