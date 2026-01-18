#!/usr/bin/env bash
# Benchmark Regression Checker
# Parses Criterion JSON output and enforces a percentage threshold for regressions.
# Usage: ./check-benchmark-regression.sh [threshold_percent] [criterion_dir]
# Example: ./check-benchmark-regression.sh 5.0 target/criterion

set -euo pipefail

THRESHOLD="${1:-5.0}"
CRITERION_DIR="${2:-target/criterion}"
BASELINE_DIR="${3:-target/criterion-baseline}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "=== Benchmark Regression Check ==="
echo "Threshold: ${THRESHOLD}%"
echo ""

exit_code=0
checked=0
regressions=0

# Output table header
printf "%-50s %12s %12s %10s %8s\n" "Benchmark" "Baseline" "Current" "Change" "Status"
printf "%-50s %12s %12s %10s %8s\n" "---------" "--------" "-------" "------" "------"

# Find all benchmark estimate files
find "$CRITERION_DIR" -path "*/new/estimates.json" 2>/dev/null | while read -r new_file; do
    # Extract benchmark name from path
    bench_path="${new_file#$CRITERION_DIR/}"
    bench_name="${bench_path%%/new/estimates.json}"
    
    # Construct baseline path
    baseline_file="${BASELINE_DIR}/${bench_name}/base/estimates.json"
    
    # Skip if no baseline exists
    if [[ ! -f "$baseline_file" ]]; then
        continue
    fi
    
    # Extract mean point estimates (in nanoseconds)
    new_mean=$(jq -r '.mean.point_estimate // empty' "$new_file" 2>/dev/null)
    old_mean=$(jq -r '.mean.point_estimate // empty' "$baseline_file" 2>/dev/null)
    
    # Skip if we couldn't extract values
    if [[ -z "$new_mean" ]] || [[ -z "$old_mean" ]]; then
        continue
    fi
    
    # Calculate percentage change
    pct_change=$(echo "scale=4; ($new_mean - $old_mean) / $old_mean * 100" | bc -l)
    pct_display=$(printf "%.2f" "$pct_change")
    
    # Format times for display (convert ns to human readable)
    format_time() {
        local ns="$1"
        if (( $(echo "$ns >= 1000000000" | bc -l) )); then
            printf "%.2fs" "$(echo "$ns / 1000000000" | bc -l)"
        elif (( $(echo "$ns >= 1000000" | bc -l) )); then
            printf "%.2fms" "$(echo "$ns / 1000000" | bc -l)"
        elif (( $(echo "$ns >= 1000" | bc -l) )); then
            printf "%.2fμs" "$(echo "$ns / 1000" | bc -l)"
        else
            printf "%.0fns" "$ns"
        fi
    }
    
    old_formatted=$(format_time "$old_mean")
    new_formatted=$(format_time "$new_mean")
    
    # Check if regression exceeds threshold
    is_regression=$(echo "$pct_change > $THRESHOLD" | bc -l)
    
    if [[ "$is_regression" == "1" ]]; then
        status="${RED}FAIL${NC}"
        regressions=$((regressions + 1))
        exit_code=1
    elif (( $(echo "$pct_change > 0" | bc -l) )); then
        status="${YELLOW}WARN${NC}"
    else
        status="${GREEN}PASS${NC}"
    fi
    
    # Format change with sign
    if (( $(echo "$pct_change >= 0" | bc -l) )); then
        change_str="+${pct_display}%"
    else
        change_str="${pct_display}%"
    fi
    
    printf "%-50s %12s %12s %10s " "$bench_name" "$old_formatted" "$new_formatted" "$change_str"
    printf "%b\n" "$status"
    
    checked=$((checked + 1))
done

echo ""
echo "=== Summary ==="
echo "Benchmarks checked: $checked"
echo "Regressions (>${THRESHOLD}%): $regressions"

if [[ "$exit_code" -ne 0 ]]; then
    echo ""
    echo -e "${RED}FAILED: Performance regression detected exceeding ${THRESHOLD}% threshold${NC}"
    echo ""
    echo "To update the baseline (if regression is acceptable):"
    echo "  cargo bench --bench strategy_bench -- --save-baseline main"
    exit 1
fi

if [[ "$checked" -eq 0 ]]; then
    echo ""
    echo -e "${YELLOW}WARNING: No benchmarks with baselines found for comparison${NC}"
    echo "This is expected on first run. Baseline will be saved after this run."
    exit 0
fi

echo ""
echo -e "${GREEN}PASSED: No significant regressions detected${NC}"
exit 0
