#!/bin/bash
# VPS Sync & Campaign Runner - Professional Script
# Usage: ./vps-sync.sh [command]
# Commands: sync, build, run-5min, run-30min, status, logs, clean, full-test

set -euo pipefail

# Configuration
VPS_HOST="149.28.39.194"
VPS_USER="root"
VPS_PASS='Z]p2qwTJBqAwpubs'
VPS_PATH="/opt/alpha-forge/quant_b3_backtest"
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Database URL
export NEON_DATABASE_URL="postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?channel_binding=require&sslmode=require"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# SSH wrapper with timeout
vps_ssh() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$VPS_USER@$VPS_HOST" "$@"
}

vps_ssh_bg() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f "$VPS_USER@$VPS_HOST" "$@"
}

# Check VPS connectivity
check_vps() {
    log_info "Checking VPS connectivity..."
    if vps_ssh "echo 'VPS OK'" > /dev/null 2>&1; then
        log_ok "VPS reachable"
        return 0
    else
        log_error "VPS unreachable"
        return 1
    fi
}

# Sync code to VPS
cmd_sync() {
    log_info "Syncing code to VPS..."
    
    # Push local changes first
    cd "$LOCAL_PATH"
    if [[ -n $(git status --porcelain) ]]; then
        log_info "Committing local changes..."
        git add -A
        git commit -m "sync: auto-commit before VPS sync" || true
    fi
    
    log_info "Pushing to GitHub..."
    git push || { log_error "Push failed"; return 1; }
    
    # Pull on VPS
    log_info "Pulling on VPS..."
    vps_ssh "cd $VPS_PATH && git fetch origin && git reset --hard origin/main" || { log_error "VPS pull failed"; return 1; }
    
    log_ok "Sync complete"
}

# Build on VPS
cmd_build() {
    log_info "Building on VPS (this may take a few minutes)..."
    
    vps_ssh "source ~/.cargo/env 2>/dev/null || true; cd $VPS_PATH && cargo build --release 2>&1" | tail -20
    
    # Verify binary exists
    if vps_ssh "test -f $VPS_PATH/target/release/combiner"; then
        log_ok "Build successful"
    else
        log_error "Build failed - binary not found"
        return 1
    fi
}

# Clean artifacts
cmd_clean() {
    log_info "Cleaning local artifacts..."
    cd "$LOCAL_PATH"
    rm -rf output/scg/* cache/* artifacts/candidates/* 2>/dev/null || true
    log_ok "Local clean"
    
    log_info "Cleaning VPS artifacts..."
    vps_ssh "cd $VPS_PATH && rm -rf output/scg/* cache/* artifacts/candidates/* 2>/dev/null || true"
    log_ok "VPS clean"
    
    log_info "Cleaning database..."
    psql "$NEON_DATABASE_URL" -c "DELETE FROM scg_promotions; DELETE FROM scg_candidates; DELETE FROM scg_runs; DELETE FROM scg_campaigns;" 2>/dev/null || {
        log_warn "psql not available, skipping DB clean"
    }
    log_ok "Clean complete"
}

# Run 5 minute test locally
cmd_run_local_5min() {
    log_info "Running 5-min test LOCALLY..."
    cd "$LOCAL_PATH"
    
    export MACHINE_ORIGIN="local"
    ./target/release/combiner factory run \
        --campaign configs/campaigns/scg_5min_maxpower.toml \
        2>&1 | tee output/local_5min.log
    
    log_ok "Local 5-min test complete"
}

# Run 5 minute test on VPS
cmd_run_vps_5min() {
    log_info "Running 5-min test on VPS..."
    
    vps_ssh "cd $VPS_PATH && \
        source ~/.cargo/env 2>/dev/null || true && \
        export NEON_DATABASE_URL='$NEON_DATABASE_URL' && \
        export MACHINE_ORIGIN='vps' && \
        ./target/release/combiner factory run \
            --campaign configs/campaigns/scg_5min_maxpower.toml \
            2>&1" | tee "$LOCAL_PATH/output/vps_5min.log"
    
    log_ok "VPS 5-min test complete"
}

# Run 30 minute audit
cmd_run_30min() {
    local target="${1:-local}"
    log_info "Running 30-min audit on $target..."
    
    if [[ "$target" == "vps" ]]; then
        vps_ssh_bg "cd $VPS_PATH && \
            source ~/.cargo/env 2>/dev/null || true && \
            export NEON_DATABASE_URL='$NEON_DATABASE_URL' && \
            export MACHINE_ORIGIN='vps' && \
            nohup ./target/release/combiner factory run \
                --campaign configs/campaigns/scg_30min_audit.toml \
                > output/vps_30min.log 2>&1 &"
        log_ok "VPS 30-min started in background"
    else
        cd "$LOCAL_PATH"
        export MACHINE_ORIGIN="local"
        nohup ./target/release/combiner factory run \
            --campaign configs/campaigns/scg_30min_audit.toml \
            > output/local_30min.log 2>&1 &
        log_ok "Local 30-min started in background (PID: $!)"
    fi
}

# Check status
cmd_status() {
    log_info "=== LOCAL STATUS ==="
    echo "Hostname: $(hostname)"
    if pgrep -f "combiner" > /dev/null; then
        log_ok "Combiner running locally"
        ps aux | grep combiner | grep -v grep | head -3
    else
        log_warn "No combiner running locally"
    fi
    
    echo ""
    log_info "=== VPS STATUS ==="
    vps_ssh "hostname && uptime"
    if vps_ssh "pgrep -f combiner" > /dev/null 2>&1; then
        log_ok "Combiner running on VPS"
        vps_ssh "ps aux | grep combiner | grep -v grep | head -3"
    else
        log_warn "No combiner running on VPS"
    fi
    
    echo ""
    log_info "=== DATABASE STATUS ==="
    psql "$NEON_DATABASE_URL" -c "SELECT machine_origin, COUNT(*) as runs, MAX(started_at) as last_run FROM scg_runs GROUP BY machine_origin;" 2>/dev/null || {
        log_warn "psql not available"
    }
}

# Show logs
cmd_logs() {
    local target="${1:-local}"
    log_info "Showing $target logs..."
    
    if [[ "$target" == "vps" ]]; then
        vps_ssh "tail -50 $VPS_PATH/output/vps_*.log 2>/dev/null || echo 'No VPS logs'"
    else
        tail -50 "$LOCAL_PATH"/output/local_*.log 2>/dev/null || echo "No local logs"
    fi
}

# Full test: sync, build, clean, run 5min on both
cmd_full_test() {
    log_info "=== FULL TEST SEQUENCE ==="
    
    check_vps || exit 1
    
    cmd_sync
    cmd_build
    cmd_clean
    
    log_info "Running 5-min test on LOCAL..."
    cmd_run_local_5min
    
    log_info "Running 5-min test on VPS..."
    cmd_run_vps_5min
    
    log_info "=== COMPARING RESULTS ==="
    cmd_status
    
    log_ok "Full test complete!"
}

# Main
case "${1:-help}" in
    sync)       cmd_sync ;;
    build)      cmd_build ;;
    clean)      cmd_clean ;;
    run-local)  cmd_run_local_5min ;;
    run-vps)    cmd_run_vps_5min ;;
    run-30min)  cmd_run_30min "${2:-local}" ;;
    status)     cmd_status ;;
    logs)       cmd_logs "${2:-local}" ;;
    full-test)  cmd_full_test ;;
    *)
        echo "VPS Sync & Campaign Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  sync        Push local changes and pull on VPS"
        echo "  build       Build release binary on VPS"
        echo "  clean       Clean all artifacts (local, VPS, database)"
        echo "  run-local   Run 5-min test locally"
        echo "  run-vps     Run 5-min test on VPS"
        echo "  run-30min   Run 30-min audit [local|vps]"
        echo "  status      Check status of both environments"
        echo "  logs        Show logs [local|vps]"
        echo "  full-test   Complete: sync → build → clean → run both"
        ;;
esac

