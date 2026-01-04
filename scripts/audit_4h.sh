#!/bin/bash
# =============================================================================
# Auditoria 4 Horas - SCG Factory
# =============================================================================
# Técnica: sleep é morto automaticamente quando o processo principal termina
# =============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export NEON_DATABASE_URL="postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?channel_binding=require&sslmode=require"
export MACHINE_ORIGIN="local"

echo "=== SCG 4h Audit Started: $(date) ==="
echo "Runtime: 2 hours (7200s)"
echo "PID: $$"

# Roda o combiner em background
./target/release/combiner factory run --campaign configs/campaigns/scg_4h_audit.toml &
PID=$!

# Sleep em background (será morto quando combiner terminar)
sleep 7200 &
SLEEP_PID=$!

# Espera o combiner terminar
wait $PID
EXIT_CODE=$?

# Mata o sleep
kill $SLEEP_PID 2>/dev/null || true

echo "=== SCG 4h Audit Finished: $(date) ==="
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE

