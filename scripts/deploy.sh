#!/bin/bash
###############################################################################
# ALPHA FORGE - Professional Binary Deploy System
# Compiles static binaries locally, deploys only binaries to VPS
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/vps/.env.vps"

# Load VPS credentials
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
else
    echo "Error: $ENV_FILE not found" >&2
    exit 1
fi

# Configuration
VPS_PATH="/opt/alpha-forge"
MUSL_TARGET="x86_64-unknown-linux-musl"
BINARY_DIR="$PROJECT_ROOT/target/$MUSL_TARGET/release"
BINARIES=("combiner" "backtest")

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; GRAY='\033[0;90m'
NC='\033[0m'; BOLD='\033[1m'

# Timing
START_TIME=$(date +%s)
elapsed() { printf "%02d:%02d" $((($(date +%s) - START_TIME) / 60)) $((($(date +%s) - START_TIME) % 60)); }

# Logging
log()   { echo -e "${GREEN}[✓]${NC} $1"; }
info()  { echo -e "${BLUE}[i]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1" >&2; exit 1; }
step()  { echo -e "${CYAN}[→]${NC} $1"; }

# SSH wrapper
ssh_cmd() { sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$VPS_USER@$VPS_IP" "$@"; }

# Rsync wrapper (quiet, fast)
rsync_vps() {
    sshpass -p "$VPS_PASS" rsync -az --delete \
        -e "ssh -o StrictHostKeyChecking=no" "$@"
}

###############################################################################
# COMMANDS
###############################################################################

cmd_build() {
    echo -e "${BOLD}${CYAN}BUILD${NC} - Compiling static binaries"
    echo ""
    
    step "Checking Rust toolchain..."
    rustup target list --installed | grep -q "$MUSL_TARGET" || error "Target $MUSL_TARGET not installed. Run: rustup target add $MUSL_TARGET"
    command -v musl-gcc >/dev/null || error "musl-gcc not found. Run: sudo apt install musl-tools"
    
    step "Building release binaries (musl static)..."
    cd "$PROJECT_ROOT"
    cargo build --release --target "$MUSL_TARGET" -p combiner_cli -p backtester_cli 2>&1 | \
        grep -E "(Compiling|Finished|warning:.*generated|error)" | tail -10
    
    step "Verifying binaries..."
    for bin in "${BINARIES[@]}"; do
        if [[ -f "$BINARY_DIR/$bin" ]]; then
            size=$(ls -lh "$BINARY_DIR/$bin" | awk '{print $5}')
            linkage=$(ldd "$BINARY_DIR/$bin" 2>&1 | head -1)
            echo -e "  ${GREEN}✓${NC} $bin ($size) - $linkage"
        else
            error "Binary not found: $BINARY_DIR/$bin"
        fi
    done
    
    log "Build complete ($(elapsed))"
}

cmd_deploy() {
    echo -e "${BOLD}${YELLOW}DEPLOY${NC} - Sending binaries to VPS"
    echo ""
    
    # Pre-flight checks
    step "Pre-flight checks..."
    for bin in "${BINARIES[@]}"; do
        [[ -f "$BINARY_DIR/$bin" ]] || error "Binary not found: $bin. Run './deploy.sh build' first"
    done
    ssh_cmd "echo 'VPS OK'" >/dev/null || error "Cannot connect to VPS"
    echo -e "  ${GREEN}✓${NC} Binaries exist, VPS reachable"
    
    # Setup VPS directories
    step "Setting up VPS directories..."
    ssh_cmd "mkdir -p $VPS_PATH/{bin,bin/versions,configs,logs}"
    
    # Keep only last backup (minimize disk usage)
    step "Rotating backups (keep only previous)..."
    ssh_cmd "cd $VPS_PATH/bin && for b in combiner backtest; do [[ -f \$b ]] && mv \$b versions/\$b.prev 2>/dev/null || true; done"
    
    # Deploy binaries
    step "Deploying binaries..."
    for bin in "${BINARIES[@]}"; do
        rsync_vps "$BINARY_DIR/$bin" "$VPS_USER@$VPS_IP:$VPS_PATH/bin/"
        ssh_cmd "chmod +x $VPS_PATH/bin/$bin"
        echo -e "  ${GREEN}✓${NC} $bin"
    done
    
    # Deploy configs
    step "Deploying configs..."
    rsync_vps "$PROJECT_ROOT/configs/" "$VPS_USER@$VPS_IP:$VPS_PATH/configs/"
    
    # Deploy dashboard
    step "Deploying dashboard..."
    rsync_vps --exclude 'node_modules' --exclude 'dist' \
        "$PROJECT_ROOT/dashboard/" "$VPS_USER@$VPS_IP:$VPS_PATH/dashboard/"
    
    # Build dashboard on VPS
    step "Building dashboard on VPS..."
    ssh_cmd "cd $VPS_PATH/dashboard && npm ci --silent 2>/dev/null && npm run build 2>&1 | tail -2"
    
    # Restart services
    step "Restarting services..."
    ssh_cmd "pm2 restart all --update-env 2>/dev/null || pm2 start $VPS_PATH/dashboard/ecosystem.config.cjs 2>/dev/null" || true
    
    # Verify
    step "Verifying deployment..."
    sleep 2
    local health=$(curl -s -o /dev/null -w "%{http_code}" "http://$VPS_IP:3001/api/health" 2>/dev/null || echo "ERR")
    if [[ "$health" == "200" ]]; then
        echo -e "  ${GREEN}✓${NC} API health check passed"
    else
        warn "API health check failed (HTTP $health)"
    fi
    
    # Test binaries on VPS
    local ver=$(ssh_cmd "$VPS_PATH/bin/combiner --version 2>/dev/null" || echo "ERR")
    if [[ "$ver" != "ERR" ]]; then
        echo -e "  ${GREEN}✓${NC} combiner: $ver"
    else
        warn "combiner binary test failed"
    fi
    
    log "Deploy complete ($(elapsed))"
    echo ""
    echo -e "${CYAN}VPS:${NC} http://$VPS_IP"
    echo -e "${CYAN}API:${NC} http://$VPS_IP:3001/api/health"
}

cmd_quick() {
    echo -e "${BOLD}${CYAN}QUICK${NC} - Dashboard only (no binaries)"
    echo ""
    
    step "Syncing dashboard source..."
    rsync_vps --exclude 'node_modules' --exclude 'dist' \
        "$PROJECT_ROOT/dashboard/src/" "$VPS_USER@$VPS_IP:$VPS_PATH/dashboard/src/"
    rsync_vps "$PROJECT_ROOT/dashboard/server/" "$VPS_USER@$VPS_IP:$VPS_PATH/dashboard/server/"
    rsync_vps "$PROJECT_ROOT/dashboard/server.js" "$VPS_USER@$VPS_IP:$VPS_PATH/dashboard/"
    
    step "Syncing configs..."
    rsync_vps "$PROJECT_ROOT/configs/" "$VPS_USER@$VPS_IP:$VPS_PATH/configs/"
    
    step "Building dashboard..."
    ssh_cmd "cd $VPS_PATH/dashboard && npm run build 2>&1 | tail -2"
    
    step "Restarting services..."
    ssh_cmd "pm2 restart all --update-env 2>/dev/null" || true
    
    log "Quick deploy complete ($(elapsed))"
}

cmd_verify() {
    echo -e "${BOLD}${GREEN}VERIFY${NC} - Testing VPS"
    echo ""
    
    step "Testing API endpoints..."
    local ok=0 fail=0
    
    for ep in health overview campaigns omp/status strategies audits; do
        local status=$(curl -s -o /dev/null -w "%{http_code}" "http://$VPS_IP:3001/api/$ep" 2>/dev/null || echo "ERR")
        if [[ "$status" == "200" ]]; then
            echo -e "  ${GREEN}✓${NC} /api/$ep"
            ((ok++)) || true
        else
            echo -e "  ${RED}✗${NC} /api/$ep [$status]"
            ((fail++)) || true
        fi
    done
    
    step "Testing binaries..."
    local combiner_test=$(ssh_cmd "$VPS_PATH/bin/combiner --version 2>&1" || echo "ERR")
    local backtest_test=$(ssh_cmd "$VPS_PATH/bin/backtest --help 2>&1 | head -1" || echo "ERR")
    
    if [[ "$combiner_test" == *"combiner"* ]]; then
        echo -e "  ${GREEN}✓${NC} combiner: $combiner_test"
        ((ok++)) || true
    else
        echo -e "  ${RED}✗${NC} combiner: not working"
        ((fail++)) || true
    fi
    
    if [[ "$backtest_test" != "ERR" ]]; then
        echo -e "  ${GREEN}✓${NC} backtest: working"
        ((ok++)) || true
    else
        echo -e "  ${RED}✗${NC} backtest: not working"
        ((fail++)) || true
    fi
    
    echo ""
    log "Results: $ok passed, $fail failed"
}

cmd_rollback() {
    echo -e "${BOLD}${RED}ROLLBACK${NC} - Restoring previous binaries"
    echo ""
    
    step "Restoring from .prev backups..."
    ssh_cmd "cd $VPS_PATH/bin && \
        [[ -f versions/combiner.prev ]] && cp versions/combiner.prev combiner && echo '  ✓ combiner restored' || echo '  ✗ No combiner backup' && \
        [[ -f versions/backtest.prev ]] && cp versions/backtest.prev backtest && echo '  ✓ backtest restored' || echo '  ✗ No backtest backup'"
    
    step "Restarting services..."
    ssh_cmd "pm2 restart all --update-env 2>/dev/null" || true
    
    log "Rollback complete ($(elapsed))"
}

cmd_status() {
    echo -e "${BOLD}VPS STATUS${NC}"
    echo ""
    
    step "PM2 processes..."
    ssh_cmd "pm2 list 2>/dev/null" || warn "PM2 not running"
    
    step "Binary versions..."
    local combiner_ver=$(ssh_cmd "$VPS_PATH/bin/combiner --version 2>&1" || echo "not found")
    local backtest_ver=$(ssh_cmd "$VPS_PATH/bin/backtest --help 2>&1 | head -1" || echo "not found")
    echo -e "  combiner: $combiner_ver"
    echo -e "  backtest: $backtest_ver"
    
    step "Disk usage..."
    ssh_cmd "du -sh $VPS_PATH/{bin,dashboard,configs} 2>/dev/null" || true
    
    echo ""
    echo -e "${CYAN}URL:${NC} http://$VPS_IP"
}

cmd_full() {
    echo -e "${BOLD}${YELLOW}FULL DEPLOY${NC} - Build + Deploy"
    echo ""
    cmd_build
    echo ""
    cmd_deploy
}

cmd_ssh() {
    info "Connecting to VPS..."
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP"
}

cmd_help() {
    echo -e "${BOLD}ALPHA FORGE${NC} - Professional Deploy System"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build     Compile static binaries locally (~5 min)"
    echo "  deploy    Send binaries + dashboard to VPS (~90 sec)"
    echo "  quick     Dashboard only, no binaries (~45 sec)"
    echo "  full      Build + deploy (~6 min)"
    echo "  verify    Test VPS endpoints and binaries"
    echo "  rollback  Restore previous binary versions"
    echo "  status    Show VPS status"
    echo "  ssh       Open SSH session to VPS"
    echo ""
    echo "Typical workflow:"
    echo "  1. Make code changes"
    echo "  2. ./deploy.sh build    # If Rust code changed"
    echo "  3. ./deploy.sh deploy   # Send to VPS"
    echo "  4. ./deploy.sh verify   # Confirm working"
}

###############################################################################
# MAIN
###############################################################################

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}ALPHA FORGE${NC} Deploy System │ VPS: ${GRAY}$VPS_IP${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

case "${1:-help}" in
    build)    cmd_build ;;
    deploy)   cmd_deploy ;;
    quick)    cmd_quick ;;
    full)     cmd_full ;;
    verify)   cmd_verify ;;
    rollback) cmd_rollback ;;
    status)   cmd_status ;;
    ssh)      cmd_ssh ;;
    help|*)   cmd_help ;;
esac

echo ""

