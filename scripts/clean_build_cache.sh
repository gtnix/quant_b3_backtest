#!/bin/bash
# =============================================================================
# Script de Limpeza de Cache de Build Rust
# =============================================================================
# Limpa artefatos de compilação para liberar espaço em disco.
# 
# Uso:
#   ./scripts/clean_build_cache.sh [--full | --incremental | --deps | --all]
#
# Opções:
#   --incremental   Remove apenas cache incremental (rápido, seguro)
#   --deps          Remove dependências compiladas (recompila deps)
#   --full          Limpa target/ completo (cargo clean)
#   --all           Limpa tudo incluindo sccache
#
# Por padrão: --incremental (mais seguro)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_ROOT/target"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  🧹 Rust Build Cache Cleaner${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

get_size() {
    local path=$1
    if [ -d "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

show_current_usage() {
    echo -e "\n${YELLOW}📊 Uso atual do disco:${NC}"
    
    if [ -d "$TARGET_DIR" ]; then
        local total_size=$(get_size "$TARGET_DIR")
        echo -e "   target/              : ${total_size}"
        
        if [ -d "$TARGET_DIR/debug" ]; then
            echo -e "   ├── debug/          : $(get_size "$TARGET_DIR/debug")"
        fi
        if [ -d "$TARGET_DIR/release" ]; then
            echo -e "   ├── release/        : $(get_size "$TARGET_DIR/release")"
        fi
        if [ -d "$TARGET_DIR/dev-fast" ]; then
            echo -e "   ├── dev-fast/       : $(get_size "$TARGET_DIR/dev-fast")"
        fi
        
        # Incremental cache dirs
        local inc_size=0
        for inc_dir in "$TARGET_DIR"/*/incremental "$TARGET_DIR"/*/.fingerprint; do
            if [ -d "$inc_dir" ]; then
                inc_size=$((inc_size + $(du -s "$inc_dir" 2>/dev/null | cut -f1 || echo 0)))
            fi
        done
        echo -e "   └── incremental/     : $((inc_size / 1024)) MB (estimado)"
    else
        echo -e "   ${GREEN}✓ Nenhum cache de build encontrado${NC}"
    fi
    
    # sccache
    if command -v sccache &> /dev/null; then
        local sccache_dir="${SCCACHE_DIR:-$HOME/.cache/sccache}"
        if [ -d "$sccache_dir" ]; then
            echo -e "   sccache/            : $(get_size "$sccache_dir")"
        fi
    fi
    echo ""
}

clean_incremental() {
    echo -e "${YELLOW}🔄 Removendo cache incremental...${NC}"
    
    local freed=0
    for profile_dir in "$TARGET_DIR"/*/; do
        if [ -d "${profile_dir}incremental" ]; then
            local size=$(du -s "${profile_dir}incremental" 2>/dev/null | cut -f1 || echo 0)
            freed=$((freed + size))
            rm -rf "${profile_dir}incremental"
            echo -e "   ${GREEN}✓${NC} Removido: ${profile_dir}incremental"
        fi
        if [ -d "${profile_dir}.fingerprint" ]; then
            local size=$(du -s "${profile_dir}.fingerprint" 2>/dev/null | cut -f1 || echo 0)
            freed=$((freed + size))
            rm -rf "${profile_dir}.fingerprint"
            echo -e "   ${GREEN}✓${NC} Removido: ${profile_dir}.fingerprint"
        fi
    done
    
    echo -e "${GREEN}✓ Liberado aproximadamente $((freed / 1024)) MB${NC}"
}

clean_deps() {
    echo -e "${YELLOW}📦 Removendo dependências compiladas...${NC}"
    
    for profile_dir in "$TARGET_DIR"/*/; do
        if [ -d "${profile_dir}deps" ]; then
            rm -rf "${profile_dir}deps"
            echo -e "   ${GREEN}✓${NC} Removido: ${profile_dir}deps"
        fi
        if [ -d "${profile_dir}build" ]; then
            rm -rf "${profile_dir}build"
            echo -e "   ${GREEN}✓${NC} Removido: ${profile_dir}build"
        fi
    done
    
    echo -e "${GREEN}✓ Dependências removidas (serão recompiladas no próximo build)${NC}"
}

clean_full() {
    echo -e "${YELLOW}🗑️  Executando cargo clean...${NC}"
    cd "$PROJECT_ROOT"
    cargo clean
    echo -e "${GREEN}✓ target/ completamente removido${NC}"
}

clean_sccache() {
    if command -v sccache &> /dev/null; then
        echo -e "${YELLOW}🔥 Limpando sccache...${NC}"
        sccache --stop-server 2>/dev/null || true
        local sccache_dir="${SCCACHE_DIR:-$HOME/.cache/sccache}"
        if [ -d "$sccache_dir" ]; then
            rm -rf "$sccache_dir"
            echo -e "${GREEN}✓ sccache limpo${NC}"
        fi
    fi
}

clean_all() {
    clean_full
    clean_sccache
}

# =============================================================================
# Main
# =============================================================================

print_header

MODE="${1:---incremental}"

show_current_usage

case "$MODE" in
    --incremental)
        clean_incremental
        ;;
    --deps)
        clean_incremental
        clean_deps
        ;;
    --full)
        clean_full
        ;;
    --all)
        clean_all
        ;;
    -h|--help)
        echo "Uso: $0 [--incremental | --deps | --full | --all]"
        echo ""
        echo "Opções:"
        echo "  --incremental   Remove apenas cache incremental (padrão)"
        echo "  --deps          Remove cache incremental + dependências"
        echo "  --full          cargo clean (remove target/)"
        echo "  --all           Remove tudo incluindo sccache"
        exit 0
        ;;
    *)
        echo -e "${RED}Opção desconhecida: $MODE${NC}"
        echo "Use --help para ver opções disponíveis"
        exit 1
        ;;
esac

echo ""
show_current_usage

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✓ Limpeza concluída!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"











