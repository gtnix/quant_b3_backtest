import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';

interface WalkForwardWindow {
  period_start: string;
  period_end: string;
  is_sharpe: number;
  oos_sharpe: number;
  is_return: number;
  oos_return: number;
}

interface WalkForwardChartProps {
  windows: WalkForwardWindow[];
  metric?: 'sharpe' | 'return';
}

export function WalkForwardChart({ windows, metric = 'sharpe' }: WalkForwardChartProps) {
  const data = windows.map((w, i) => ({
    period: `P${i + 1}`,
    periodLabel: `${w.period_start.substring(5, 10)} - ${w.period_end.substring(5, 10)}`,
    is: metric === 'sharpe' ? w.is_sharpe : w.is_return * 100,
    oos: metric === 'sharpe' ? w.oos_sharpe : w.oos_return * 100,
    degradation: metric === 'sharpe'
      ? (w.is_sharpe > 0 ? w.oos_sharpe / w.is_sharpe : 0)
      : (w.is_return > 0 ? w.oos_return / w.is_return : 0),
  }));

  // Calculate stats
  const avgIS = data.reduce((a, b) => a + b.is, 0) / data.length;
  const avgOOS = data.reduce((a, b) => a + b.oos, 0) / data.length;
  const avgDegradation = avgIS > 0 ? avgOOS / avgIS : 0;
  const profitPeriods = data.filter(d => d.oos > 0).length;

  if (windows.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        No walk-forward data available
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Stats bar */}
      <div className="flex items-center gap-6 mb-4 text-xs flex-wrap">
        <div>
          <span className="text-terminal-muted">Média IS:</span>
          <span className="ml-1 font-mono text-accent-cyan">
            {metric === 'sharpe' ? avgIS.toFixed(2) : `${avgIS.toFixed(1)}%`}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Média OOS:</span>
          <span className={`ml-1 font-mono ${avgOOS >= 0 ? 'text-profit' : 'text-loss'}`}>
            {metric === 'sharpe' ? avgOOS.toFixed(2) : `${avgOOS.toFixed(1)}%`}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">WFE:</span>
          <span className={`ml-1 font-mono font-bold ${avgDegradation >= 0.5 ? 'text-profit' : 'text-loss'}`}>
            {(avgDegradation * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Períodos Lucro:</span>
          <span className={`ml-1 font-mono ${profitPeriods > data.length / 2 ? 'text-profit' : 'text-loss'}`}>
            {profitPeriods}/{data.length}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
            <XAxis
              dataKey="period"
              fontSize={10}
              stroke="#6b7280"
              tickLine={false}
            />
            <YAxis
              fontSize={10}
              stroke="#6b7280"
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d0d0f',
                border: '1px solid #2a2a30',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              formatter={(value: number, name: string) => [
                metric === 'sharpe' ? value.toFixed(2) : `${value.toFixed(1)}%`,
                name === 'is' ? 'In-Sample' : 'Out-of-Sample',
              ]}
              labelFormatter={(_, payload) => {
                const item = payload?.[0]?.payload;
                return item ? item.periodLabel : '';
              }}
            />
            <Legend
              verticalAlign="top"
              height={36}
              iconType="rect"
              wrapperStyle={{ fontSize: '11px' }}
              formatter={(value) => (value === 'is' ? 'In-Sample' : 'Out-of-Sample')}
            />
            <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
            {/* 50% WFE Threshold Line - shows minimum acceptable OOS performance */}
            <ReferenceLine 
              y={avgIS * 0.5} 
              stroke="#f59e0b" 
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{ 
                value: '50% WFE', 
                position: 'right', 
                fill: '#f59e0b',
                fontSize: 10,
              }}
            />
            <Bar
              dataKey="is"
              fill="#00d4ff"
              opacity={0.6}
              radius={[4, 4, 0, 0]}
              name="is"
            />
            <Bar
              dataKey="oos"
              fill="#00ff88"
              radius={[4, 4, 0, 0]}
              name="oos"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

