#!/bin/bash
# Stop all Alpha Forge processes

echo "Stopping Alpha Forge..."

pkill -f "combiner run" 2>/dev/null
pkill -f "node server.js" 2>/dev/null
pkill -f "vite" 2>/dev/null

sleep 1
echo "✅ All processes stopped"
