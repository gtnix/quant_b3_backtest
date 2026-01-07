#!/bin/bash
# =============================================================================
# Stop Alpha Forge Dashboard
# =============================================================================

echo "Parando servicos..."

# Parar combiner se estiver rodando
pkill -f "combiner run" 2>/dev/null && echo "  Combiner parado"

# Parar API Server
pkill -f "node server.js" 2>/dev/null && echo "  API Server parado"

# Parar Frontend
pkill -f "vite" 2>/dev/null && echo "  Frontend parado"

echo ""
echo "Todos os servicos foram parados."




