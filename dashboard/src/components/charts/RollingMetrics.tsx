import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';

interface RollingPoint {
  date: string;
  value: number;
}

interface RollingMetricsProps {
  data: {
    label: string;
    points: RollingPoint[];
    color: string;
  }[];
  title?: string;
  yAxisLabel?: string;
  showZeroLine?: boolean;
}

export function RollingMetrics({ data, title, yAxisLabel, showZeroLine = true }: RollingMetricsProps) {
  // Merge all datasets by date
  const mergedData = (() => {
    if (data.length === 0) return [];

    const dateMap = new Map<string, Record<string, number>>();

    for (const series of data) {
      for (const point of series.points) {
        const existing = dateMap.get(point.date) || {};
        existing[series.label] = point.value;
        dateMap.set(point.date, existing);
      }
    }

    return Array.from(dateMap.entries())
      .map(([date, values]) => ({ date, ...values }))
      .sort((a, b) => a.date.localeCompare(b.date));
  })();

  if (mergedData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        No rolling metrics data available
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {title && <div className="text-sm font-medium mb-2">{title}</div>}
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mergedData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <XAxis
              dataKey="date"
              tickFormatter={(v) => v.substring(5, 10)} // MM-DD
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
              label={
                yAxisLabel
                  ? { value: yAxisLabel, angle: -90, position: 'insideLeft', fontSize: 10 }
                  : undefined
              }
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d0d0f',
                border: '1px solid #2a2a30',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              formatter={(value: number, name: string) => [value.toFixed(3), name]}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Legend
              verticalAlign="top"
              height={36}
              iconType="line"
              wrapperStyle={{ fontSize: '11px' }}
            />
            {showZeroLine && <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />}
            {data.map((series) => (
              <Line
                key={series.label}
                type="monotone"
                dataKey={series.label}
                stroke={series.color}
                dot={false}
                strokeWidth={1.5}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

