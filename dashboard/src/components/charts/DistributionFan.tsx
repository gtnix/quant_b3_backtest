import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';

interface ConfidenceBands {
  dates: string[];
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}

interface DistributionFanProps {
  confidenceBands: ConfidenceBands;
  actualEquity?: { date: string; value: number }[];
}

export function DistributionFan({ confidenceBands, actualEquity }: DistributionFanProps) {
  // Merge data
  const data = confidenceBands.dates.map((date, i) => ({
    date,
    p5: confidenceBands.p5[i],
    p25: confidenceBands.p25[i],
    p50: confidenceBands.p50[i],
    p75: confidenceBands.p75[i],
    p95: confidenceBands.p95[i],
    actual: actualEquity?.find(p => p.date === date)?.value,
    // For area ranges
    p5_25: [confidenceBands.p5[i], confidenceBands.p25[i]],
    p25_50: [confidenceBands.p25[i], confidenceBands.p50[i]],
    p50_75: [confidenceBands.p50[i], confidenceBands.p75[i]],
    p75_95: [confidenceBands.p75[i], confidenceBands.p95[i]],
  }));

  if (confidenceBands.dates.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        Run Monte Carlo simulation to see confidence bands
      </div>
    );
  }

  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <XAxis
            dataKey="date"
            tickFormatter={(v) => v.substring(5, 10)}
            fontSize={10}
            stroke="#6b7280"
            tickLine={false}
            minTickGap={50}
          />
          <YAxis
            fontSize={10}
            stroke="#6b7280"
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0d0d0f',
              border: '1px solid #2a2a30',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            formatter={(value: number, name: string) => {
              const label = {
                p5: '5th %ile',
                p25: '25th %ile',
                p50: 'Median',
                p75: '75th %ile',
                p95: '95th %ile',
                actual: 'Actual',
              }[name] || name;
              return [`$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, label];
            }}
            labelFormatter={(label) => `Date: ${label}`}
          />

          {/* P5-P25 band (outer) */}
          <Area
            type="monotone"
            dataKey="p5"
            stroke="transparent"
            fill="#00ff88"
            fillOpacity={0.1}
          />
          <Area
            type="monotone"
            dataKey="p25"
            stroke="transparent"
            fill="#0d0d0f"
            fillOpacity={1}
          />

          {/* P25-P75 band (inner) */}
          <Area
            type="monotone"
            dataKey="p25"
            stroke="transparent"
            fill="#00ff88"
            fillOpacity={0.2}
          />
          <Area
            type="monotone"
            dataKey="p75"
            stroke="transparent"
            fill="#0d0d0f"
            fillOpacity={1}
          />

          {/* P75-P95 band (outer) */}
          <Area
            type="monotone"
            dataKey="p75"
            stroke="transparent"
            fill="#00ff88"
            fillOpacity={0.1}
          />
          <Area
            type="monotone"
            dataKey="p95"
            stroke="transparent"
            fill="#0d0d0f"
            fillOpacity={1}
          />

          {/* Median line */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke="#00ff88"
            strokeWidth={2}
            dot={false}
          />

          {/* Actual equity (if provided) */}
          {actualEquity && actualEquity.length > 0 && (
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#ffffff"
              strokeWidth={2}
              dot={false}
              strokeDasharray="5 5"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-4 h-2 bg-profit/20 rounded" />
          <span className="text-terminal-muted">P25-P75</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-2 bg-profit/10 rounded" />
          <span className="text-terminal-muted">P5-P95</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-profit" />
          <span className="text-terminal-muted">Median</span>
        </div>
        {actualEquity && (
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-white border-dashed" style={{ borderTop: '2px dashed white' }} />
            <span className="text-terminal-muted">Actual</span>
          </div>
        )}
      </div>
    </div>
  );
}

