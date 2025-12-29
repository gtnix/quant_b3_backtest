interface VaRGaugeProps {
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
}

export function VaRGauge({ var95, var99, cvar95, cvar99 }: VaRGaugeProps) {
  // Convert to percentage and absolute values
  const var95Pct = Math.abs(var95 * 100);
  const var99Pct = Math.abs(var99 * 100);
  const cvar95Pct = Math.abs(cvar95 * 100);
  const cvar99Pct = Math.abs(cvar99 * 100);

  // Max for scaling (cap at 5%)
  const maxValue = Math.max(5, var95Pct, var99Pct, cvar95Pct, cvar99Pct);

  const GaugeBar = ({
    label,
    value,
    sublabel,
    color,
  }: {
    label: string;
    value: number;
    sublabel: string;
    color: string;
  }) => {
    const width = (value / maxValue) * 100;
    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-terminal-muted">{label}</span>
          <span className="font-mono" style={{ color }}>
            -{value.toFixed(2)}%
          </span>
        </div>
        <div className="h-4 bg-terminal-surface rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${width}%`,
              backgroundColor: color,
            }}
          />
        </div>
        <div className="text-[10px] text-terminal-muted">{sublabel}</div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* VaR Section */}
      <div>
        <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent-yellow" />
          Value at Risk (VaR)
        </h4>
        <div className="space-y-4">
          <GaugeBar
            label="VaR 95%"
            value={var95Pct}
            sublabel="5% chance of losing more than this in a day"
            color="#fbbf24"
          />
          <GaugeBar
            label="VaR 99%"
            value={var99Pct}
            sublabel="1% chance of losing more than this in a day"
            color="#f97316"
          />
        </div>
      </div>

      {/* CVaR Section */}
      <div>
        <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-loss" />
          Expected Shortfall (CVaR)
        </h4>
        <div className="space-y-4">
          <GaugeBar
            label="CVaR 95%"
            value={cvar95Pct}
            sublabel="Average loss in worst 5% of days"
            color="#ef4444"
          />
          <GaugeBar
            label="CVaR 99%"
            value={cvar99Pct}
            sublabel="Average loss in worst 1% of days"
            color="#dc2626"
          />
        </div>
      </div>

      {/* Risk Summary */}
      <div className="p-4 bg-terminal-surface rounded-lg border border-terminal-border">
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-terminal-muted mb-1">Tail Risk Ratio</div>
            <div className="font-mono text-lg">
              {(cvar95Pct / (var95Pct || 1)).toFixed(2)}x
            </div>
            <div className="text-[10px] text-terminal-muted">CVaR/VaR (higher = fatter tails)</div>
          </div>
          <div>
            <div className="text-terminal-muted mb-1">Stress Multiplier</div>
            <div className="font-mono text-lg">
              {(var99Pct / (var95Pct || 1)).toFixed(2)}x
            </div>
            <div className="text-[10px] text-terminal-muted">VaR99/VaR95</div>
          </div>
        </div>
      </div>
    </div>
  );
}

