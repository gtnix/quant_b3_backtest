from __future__ import annotations

def generate_execution_table(strategy_summary: dict) -> str:
    """
    Generate a clean and modern execution summary table for terminal display.
    Returns: formatted string
    """
    if not strategy_summary:
        return "No execution data available."

    rows = []
    daily_executions = strategy_summary.get('daily_executions', {})
    for attempt_type, metrics in daily_executions.items():
        rows.append({
            'type': attempt_type,
            'category': 'Daily',
            'attempts': metrics['attempts'],
            'successful': metrics['successful'],
            'failed': metrics['failed'],
            'fill_rate': metrics['fill_rate']
        })
    if not rows:
        return "No execution data available."

    total_attempts = sum(r['attempts'] for r in rows)
    total_successful = sum(r['successful'] for r in rows)
    total_failed = sum(r['failed'] for r in rows)
    overall_fill_rate = (total_successful / total_attempts) if total_attempts > 0 else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append("EXECUTION PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Execution Type':<15} {'Category':<10} {'Attempts':>10} {'Successful':>10} {'Failed':>10} {'Fill Rate':>10}")
    lines.append("-" * 80)
    for r in rows:
        fill_rate_pct = r['fill_rate'] * 100
        bars = int(fill_rate_pct / 10)
        progress_bar = "[" + "█" * bars + " " * (10 - bars) + "]"
        lines.append(
            f"{r['type']:<15} {r['category']:<10} {r['attempts']:>10} {r['successful']:>10} "
            f"{r['failed']:>10} {r['fill_rate']:>9.1%} {progress_bar}"
        )
    lines.append("-" * 80)
    overall_pct = overall_fill_rate * 100
    bars = int(overall_pct / 10)
    progress_bar = "[" + "█" * bars + " " * (10 - bars) + "]"
    lines.append(
        f"{'TOTAL':<15} {'ALL':<10} {total_attempts:>10} {total_successful:>10} "
        f"{total_failed:>10} {overall_fill_rate:>9.1%} {progress_bar}"
    )
    lines.append("=" * 80)
    return "\n".join(lines)


