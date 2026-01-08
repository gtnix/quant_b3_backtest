import { useMemo } from 'react';

interface WeekdayHeatmapProps {
  dailyReturns: number[];
  dates: string[];
}

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const WEEKDAY_FULL = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];

export function WeekdayHeatmap({ dailyReturns, dates }: WeekdayHeatmapProps) {
  const weekdayStats = useMemo(() => {
    if (dailyReturns.length === 0 || dates.length === 0) return [];

    const stats: { weekday: number; returns: number[]; name: string; fullName: string }[] = 
      WEEKDAYS.map((name, i) => ({ weekday: i, returns: [], name, fullName: WEEKDAY_FULL[i] }));

    for (let i = 0; i < Math.min(dailyReturns.length, dates.length); i++) {
      const date = new Date(dates[i]);
      const weekday = date.getDay();
      // Skip weekends (0=Sunday, 6=Saturday)
      if (weekday === 0 || weekday === 6) continue;
      // Convert to 0-indexed Mon-Fri
      const idx = weekday - 1;
      if (idx >= 0 && idx < 5) {
        stats[idx].returns.push(dailyReturns[i] * 100);
      }
    }

    return stats.map(s => {
      const n = s.returns.length;
      if (n === 0) return { ...s, avgReturn: 0, winRate: 0, count: 0, totalReturn: 0 };
      const avgReturn = s.returns.reduce((a, b) => a + b, 0) / n;
      const winRate = (s.returns.filter(r => r > 0).length / n) * 100;
      const totalReturn = s.returns.reduce((a, b) => a + b, 0);
      return { ...s, avgReturn, winRate, count: n, totalReturn };
    });
  }, [dailyReturns, dates]);

  const getColorClass = (value: number): string => {
    if (value > 0.1) return 'bg-profit text-black';
    if (value > 0.05) return 'bg-profit/70 text-black';
    if (value > 0) return 'bg-profit/30 text-white';
    if (value > -0.05) return 'bg-loss/30 text-white';
    if (value > -0.1) return 'bg-loss/70 text-white';
    return 'bg-loss text-white';
  };

  if (weekdayStats.length === 0 || weekdayStats.every(s => s.count === 0)) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted">
        No weekday data available
      </div>
    );
  }

  const maxAvg = Math.max(...weekdayStats.map(s => Math.abs(s.avgReturn)), 0.1);
  const bestDay = weekdayStats.reduce((best, s) => s.avgReturn > best.avgReturn ? s : best, weekdayStats[0]);
  const worstDay = weekdayStats.reduce((worst, s) => s.avgReturn < worst.avgReturn ? s : worst, weekdayStats[0]);

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-profit/10 border border-profit/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Best Day</div>
          <div className="text-lg font-bold text-profit">{bestDay.fullName}</div>
          <div className="text-sm font-mono text-profit/80">+{bestDay.avgReturn.toFixed(3)}% avg</div>
        </div>
        <div className="p-4 rounded-xl bg-loss/10 border border-loss/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Worst Day</div>
          <div className="text-lg font-bold text-loss">{worstDay.fullName}</div>
          <div className="text-sm font-mono text-loss/80">{worstDay.avgReturn.toFixed(3)}% avg</div>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="space-y-3">
        {weekdayStats.map((stat) => (
          <div key={stat.weekday} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium w-12">{stat.name}</span>
              <div className="flex items-center gap-4 text-terminal-muted">
                <span>n={stat.count}</span>
                <span>Win: {stat.winRate.toFixed(0)}%</span>
                <span className={stat.avgReturn >= 0 ? 'text-profit font-mono' : 'text-loss font-mono'}>
                  {stat.avgReturn >= 0 ? '+' : ''}{stat.avgReturn.toFixed(3)}%
                </span>
              </div>
            </div>
            <div className="h-6 bg-terminal-surface rounded-lg overflow-hidden relative">
              {/* Center line */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-terminal-border z-10" />
              {/* Bar */}
              <div 
                className={`absolute top-0 bottom-0 transition-all ${
                  stat.avgReturn >= 0 
                    ? 'left-1/2 bg-profit rounded-r' 
                    : 'right-1/2 bg-loss rounded-l'
                }`}
                style={{ 
                  width: `${(Math.abs(stat.avgReturn) / maxAvg) * 50}%`
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-2 text-terminal-muted font-normal">Day</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Count</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Avg Return</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Win Rate</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Total Return</th>
            </tr>
          </thead>
          <tbody>
            {weekdayStats.map((stat) => (
              <tr key={stat.weekday} className="border-b border-terminal-border/30">
                <td className="py-2 px-2 font-medium">{stat.fullName}</td>
                <td className="py-2 px-2 text-right font-mono text-terminal-muted">{stat.count}</td>
                <td className="py-2 px-2 text-right">
                  <span className={`font-mono px-2 py-0.5 rounded ${getColorClass(stat.avgReturn)}`}>
                    {stat.avgReturn >= 0 ? '+' : ''}{stat.avgReturn.toFixed(3)}%
                  </span>
                </td>
                <td className="py-2 px-2 text-right">
                  <span className={`font-mono ${stat.winRate >= 50 ? 'text-profit' : 'text-loss'}`}>
                    {stat.winRate.toFixed(1)}%
                  </span>
                </td>
                <td className="py-2 px-2 text-right">
                  <span className={`font-mono ${stat.totalReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {stat.totalReturn >= 0 ? '+' : ''}{stat.totalReturn.toFixed(2)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
