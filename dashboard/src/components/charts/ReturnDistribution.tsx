import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Cell,
} from 'recharts';

interface ReturnDistributionProps {
  returns: number[];
  bins?: number;
  showNormal?: boolean;
}

export function ReturnDistribution({ returns, bins = 30, showNormal = true }: ReturnDistributionProps) {
  const data = useMemo(() => {
    if (returns.length === 0) return [];

    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const range = max - min;
    const binWidth = range / bins;

    // Create histogram
    const histogram: { bin: number; count: number; pct: number; normal: number }[] = [];
    const counts = new Array(bins).fill(0);

    for (const r of returns) {
      const binIdx = Math.min(Math.floor((r - min) / binWidth), bins - 1);
      counts[binIdx]++;
    }

    // Calculate mean and std for normal overlay
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
    const std = Math.sqrt(variance);

    const total = returns.length;
    const maxCount = Math.max(...counts);

    for (let i = 0; i < bins; i++) {
      const binCenter = min + binWidth * (i + 0.5);
      const pct = (counts[i] / total) * 100;

      // Normal PDF scaled to histogram
      const z = (binCenter - mean) / std;
      const normalPdf = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
      const normalScaled = (normalPdf * binWidth * total / maxCount) * 100;

      histogram.push({
        bin: binCenter * 100, // Convert to percentage
        count: counts[i],
        pct,
        normal: showNormal ? normalScaled : 0,
      });
    }

    return histogram;
  }, [returns, bins, showNormal]);

  // Calculate statistics with proper guards
  const stats = useMemo(() => {
    if (returns.length < 4) return { mean: 0, std: 0, skew: 0, kurt: 0 };

    const n = returns.length;
    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
    const std = Math.sqrt(variance);

    // Skewness (requires n >= 3)
    const skew = std > 0 && n >= 3
      ? (returns.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) * n) / ((n - 1) * (n - 2))
      : 0;

    // Kurtosis (excess) - requires n >= 4
    const kurt = std > 0 && n >= 4
      ? (returns.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) * n * (n + 1)) /
        ((n - 1) * (n - 2) * (n - 3)) -
        (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
      : 0;

    return { mean: mean * 100, std: std * 100, skew, kurt };
  }, [returns]);

  if (returns.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        No return data available
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Stats bar */}
      <div className="flex items-center gap-6 mb-4 text-xs">
        <div>
          <span className="text-terminal-muted">Mean:</span>
          <span className={`ml-1 font-mono ${stats.mean >= 0 ? 'text-profit' : 'text-loss'}`}>
            {stats.mean.toFixed(3)}%
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Std:</span>
          <span className="ml-1 font-mono">{stats.std.toFixed(3)}%</span>
        </div>
        <div>
          <span className="text-terminal-muted">Skew:</span>
          <span className={`ml-1 font-mono ${stats.skew < 0 ? 'text-loss' : 'text-profit'}`}>
            {stats.skew.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Kurtosis:</span>
          <span className={`ml-1 font-mono ${stats.kurt > 3 ? 'text-accent-yellow' : ''}`}>
            {stats.kurt.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
            <XAxis
              dataKey="bin"
              tickFormatter={(v) => `${v.toFixed(1)}%`}
              fontSize={10}
              stroke="#6b7280"
              tickLine={false}
            />
            <YAxis
              fontSize={10}
              stroke="#6b7280"
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d0d0f',
                border: '1px solid #2a2a30',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              formatter={(value: number, name: string) => [
                name === 'pct' ? `${value.toFixed(2)}%` : `${value.toFixed(2)}`,
                name === 'pct' ? 'Frequency' : 'Normal',
              ]}
              labelFormatter={(label) => `Return: ${(label as number).toFixed(2)}%`}
            />
            <ReferenceLine x={0} stroke="#6b7280" strokeDasharray="3 3" />
            <Bar dataKey="pct" radius={[2, 2, 0, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.bin < 0 ? '#ff3366' : '#00ff88'}
                  opacity={0.7}
                />
              ))}
            </Bar>
            {showNormal && (
              <Area
                type="monotone"
                dataKey="normal"
                stroke="#ff6b6b"
                fill="transparent"
                strokeWidth={2}
                dot={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

