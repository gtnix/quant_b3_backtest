#!/bin/bash
#############################################################################
# ALPHA FORGE - VPS Deploy Automation
# Professional deployment with sshpass + rsync
#############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/vps/.env.vps"

# Load credentials from .env file
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "Error: $ENV_FILE not found!"
    echo "Create it with VPS_IP, VPS_USER, VPS_PASS, VULTR_API_KEY"
    exit 1
fi

VPS_PATH="/opt/alpha-forge"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check dependencies
check_deps() {
    command -v sshpass >/dev/null 2>&1 || error "sshpass not installed. Run: sudo apt install sshpass"
    command -v rsync >/dev/null 2>&1 || error "rsync not installed. Run: sudo apt install rsync"
}

# SSH command wrapper
ssh_cmd() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP" "$@"
}

# Rsync wrapper
rsync_to_vps() {
    sshpass -p "$VPS_PASS" rsync -avz --progress \
        --exclude 'target/' \
        --exclude 'node_modules/' \
        --exclude '.git/' \
        --exclude 'cache/' \
        --exclude '*.log' \
        --exclude 'artifacts/cockpit_runs/' \
        --exclude 'artifacts/data_integrity/' \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$1" "$VPS_USER@$VPS_IP:$2"
}

# Initial setup (first time only)
initial_setup() {
    log "Starting INITIAL VPS setup..."
    
    # Copy setup scripts
    log "Copying setup scripts to VPS..."
    ssh_cmd "mkdir -p $VPS_PATH/scripts/vps"
    rsync_to_vps "$LOCAL_PATH/scripts/setup-vps.sh" "$VPS_PATH/scripts/"
    rsync_to_vps "$LOCAL_PATH/scripts/vps/" "$VPS_PATH/scripts/vps/"
    
    # Run setup
    log "Running VPS setup (this may take 10-15 minutes)..."
    ssh_cmd "chmod +x $VPS_PATH/scripts/setup-vps.sh && $VPS_PATH/scripts/setup-vps.sh"
    
    log "Initial setup complete!"
}

# Sync code to VPS
sync_code() {
    log "Syncing codebase to VPS..."
    
    # Sync main directories
    rsync_to_vps "$LOCAL_PATH/crates/" "$VPS_PATH/quant_b3_backtest/crates/"
    rsync_to_vps "$LOCAL_PATH/dashboard/" "$VPS_PATH/quant_b3_backtest/dashboard/"
    rsync_to_vps "$LOCAL_PATH/configs/" "$VPS_PATH/quant_b3_backtest/configs/"
    rsync_to_vps "$LOCAL_PATH/data/" "$VPS_PATH/quant_b3_backtest/data/"
    rsync_to_vps "$LOCAL_PATH/Cargo.toml" "$VPS_PATH/quant_b3_backtest/"
    rsync_to_vps "$LOCAL_PATH/Cargo.lock" "$VPS_PATH/quant_b3_backtest/"
    
    log "Code sync complete!"
}

# Build on VPS
build_rust() {
    log "Building Rust binaries on VPS..."
    ssh_cmd "cd $VPS_PATH/quant_b3_backtest && source ~/.cargo/env && cargo build --release -p combiner_cli -p backtester_cli"
    log "Rust build complete!"
}

# Build dashboard
build_dashboard() {
    log "Building dashboard on VPS..."
    ssh_cmd "cd $VPS_PATH/quant_b3_backtest/dashboard && npm ci && npm run build"
    log "Dashboard build complete!"
}

# Restart services
restart_services() {
    log "Restarting services..."
    ssh_cmd "cd $VPS_PATH/quant_b3_backtest/dashboard && pm2 restart ecosystem.config.cjs --update-env || pm2 start ecosystem.config.cjs"
    ssh_cmd "systemctl reload nginx"
    log "Services restarted!"
}

# Show status
show_status() {
    log "VPS Status:"
    ssh_cmd "pm2 status"
    echo ""
    log "Access URL: http://$VPS_IP"
}

# Deploy update (sync + build + restart)
deploy_update() {
    sync_code
    build_rust
    build_dashboard
    restart_services
    show_status
}

# Quick deploy (sync dashboard only, no Rust rebuild)
quick_deploy() {
    log "Quick deploy (dashboard only)..."
    rsync_to_vps "$LOCAL_PATH/dashboard/src/" "$VPS_PATH/quant_b3_backtest/dashboard/src/"
    rsync_to_vps "$LOCAL_PATH/dashboard/server.js" "$VPS_PATH/quant_b3_backtest/dashboard/"
    ssh_cmd "cd $VPS_PATH/quant_b3_backtest/dashboard && npm run build"
    restart_services
    show_status
}

# Main
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           ALPHA FORGE - VPS Deploy Automation                     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

check_deps

case "${1:-update}" in
    --initial|-i)
        initial_setup
        sync_code
        build_rust
        build_dashboard
        restart_services
        show_status
        ;;
    --quick|-q)
        quick_deploy
        ;;
    --sync|-s)
        sync_code
        ;;
    --build|-b)
        build_rust
        build_dashboard
        ;;
    --restart|-r)
        restart_services
        show_status
        ;;
    --status)
        show_status
        ;;
    --ssh)
        log "Opening SSH session..."
        sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP"
        ;;
    update|*)
        deploy_update
        ;;
esac

echo ""
log "Done!"

