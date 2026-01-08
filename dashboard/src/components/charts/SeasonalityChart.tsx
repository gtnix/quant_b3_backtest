import { useMemo } from 'react';
import { Calendar } from 'lucide-react';

interface SeasonalityChartProps {
  monthlyReturns: Array<{ year: number; month: number; return_pct: number }>;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTHS_FULL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

export function SeasonalityChart({ monthlyReturns }: SeasonalityChartProps) {
  const seasonalData = useMemo(() => {
    if (monthlyReturns.length === 0) return [];

    // Group returns by month (across all years)
    const byMonth: number[][] = Array.from({ length: 12 }, () => []);
    
    for (const m of monthlyReturns) {
      if (m.month >= 1 && m.month <= 12) {
        byMonth[m.month - 1].push(m.return_pct);
      }
    }

    return byMonth.map((returns, idx) => {
      const n = returns.length;
      if (n === 0) return { month: idx + 1, name: MONTHS[idx], fullName: MONTHS_FULL[idx], avgReturn: 0, winRate: 0, count: 0, median: 0, std: 0 };
      
      const avgReturn = returns.reduce((a, b) => a + b, 0) / n;
      const winRate = (returns.filter(r => r > 0).length / n) * 100;
      const sorted = [...returns].sort((a, b) => a - b);
      const median = n % 2 === 0 
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 
        : sorted[Math.floor(n / 2)];
      const variance = returns.reduce((sum, r) => sum + (r - avgReturn) ** 2, 0) / n;
      const std = Math.sqrt(variance);

      return { month: idx + 1, name: MONTHS[idx], fullName: MONTHS_FULL[idx], avgReturn, winRate, count: n, median, std };
    });
  }, [monthlyReturns]);

  if (seasonalData.length === 0 || seasonalData.every(s => s.count === 0)) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted">
        No seasonality data available
      </div>
    );
  }

  const maxAvg = Math.max(...seasonalData.map(s => Math.abs(s.avgReturn)), 2);
  const bestMonth = seasonalData.reduce((best, s) => s.avgReturn > best.avgReturn ? s : best, seasonalData[0]);
  const worstMonth = seasonalData.reduce((worst, s) => s.avgReturn < worst.avgReturn ? s : worst, seasonalData[0]);

  // Group by quarters
  const quarters = [
    { name: 'Q1', months: [0, 1, 2], color: 'text-blue-400' },
    { name: 'Q2', months: [3, 4, 5], color: 'text-green-400' },
    { name: 'Q3', months: [6, 7, 8], color: 'text-amber-400' },
    { name: 'Q4', months: [9, 10, 11], color: 'text-purple-400' },
  ].map(q => {
    const qReturns = q.months.map(m => seasonalData[m].avgReturn);
    const avgReturn = qReturns.reduce((a, b) => a + b, 0) / 3;
    return { ...q, avgReturn };
  });

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-profit/10 border border-profit/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Best Month</div>
          <div className="text-lg font-bold text-profit">{bestMonth.fullName}</div>
          <div className="text-sm font-mono text-profit/80">+{bestMonth.avgReturn.toFixed(2)}% avg</div>
        </div>
        <div className="p-4 rounded-xl bg-loss/10 border border-loss/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Worst Month</div>
          <div className="text-lg font-bold text-loss">{worstMonth.fullName}</div>
          <div className="text-sm font-mono text-loss/80">{worstMonth.avgReturn.toFixed(2)}% avg</div>
        </div>
        {quarters.slice(0, 2).map(q => (
          <div key={q.name} className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
            <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">{q.name} Avg</div>
            <div className={`text-lg font-bold ${q.avgReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
              {q.avgReturn >= 0 ? '+' : ''}{q.avgReturn.toFixed(2)}%
            </div>
          </div>
        ))}
      </div>

      {/* Monthly Bar Chart */}
      <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
        <div className="flex items-center gap-2 mb-4">
          <Calendar className="w-4 h-4 text-terminal-muted" />
          <h4 className="text-sm font-medium">Average Monthly Returns</h4>
        </div>
        
        <div className="flex items-end gap-1 h-32">
          {seasonalData.map((stat, idx) => {
            const height = (Math.abs(stat.avgReturn) / maxAvg) * 100;
            const isPositive = stat.avgReturn >= 0;
            const quarter = Math.floor(idx / 3);
            const quarterColors = ['bg-blue-500', 'bg-green-500', 'bg-amber-500', 'bg-purple-500'];
            
            return (
              <div key={stat.month} className="flex-1 flex flex-col items-center group relative">
                <div className="w-full h-24 flex flex-col justify-center relative">
                  {/* Zero line */}
                  <div className="absolute left-0 right-0 h-px bg-terminal-border" style={{ top: '50%' }} />
                  
                  {isPositive ? (
                    <div className="absolute bottom-1/2 left-0.5 right-0.5 flex flex-col justify-end">
                      <div
                        className={`w-full ${quarterColors[quarter]} rounded-t transition-all opacity-80 hover:opacity-100`}
                        style={{ height: `${Math.max(height / 2, 4)}%` }}
                      />
                    </div>
                  ) : (
                    <div className="absolute top-1/2 left-0.5 right-0.5 flex flex-col justify-start">
                      <div
                        className={`w-full ${quarterColors[quarter]} rounded-b transition-all opacity-80 hover:opacity-100`}
                        style={{ height: `${Math.max(height / 2, 4)}%` }}
                      />
                    </div>
                  )}
                </div>
                
                <span className="text-[10px] text-terminal-muted mt-1">{stat.name}</span>
                
                {/* Tooltip */}
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                  <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs whitespace-nowrap shadow-lg">
                    <div className="font-medium mb-1">{stat.fullName}</div>
                    <div className="space-y-0.5">
                      <div className="flex justify-between gap-3">
                        <span className="text-terminal-muted">Avg:</span>
                        <span className={`font-mono font-bold ${isPositive ? 'text-profit' : 'text-loss'}`}>
                          {isPositive ? '+' : ''}{stat.avgReturn.toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-terminal-muted">Median:</span>
                        <span className="font-mono">{stat.median >= 0 ? '+' : ''}{stat.median.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-terminal-muted">Win Rate:</span>
                        <span className={`font-mono ${stat.winRate >= 50 ? 'text-profit' : 'text-loss'}`}>
                          {stat.winRate.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-terminal-muted">Samples:</span>
                        <span className="font-mono">{stat.count}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        
        {/* Quarter Legend */}
        <div className="flex items-center justify-center gap-4 mt-4 text-xs">
          {quarters.map((q, idx) => (
            <div key={q.name} className="flex items-center gap-1">
              <div className={`w-3 h-3 rounded ${['bg-blue-500', 'bg-green-500', 'bg-amber-500', 'bg-purple-500'][idx]}`} />
              <span className="text-terminal-muted">{q.name}</span>
              <span className={`font-mono ${q.avgReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                {q.avgReturn >= 0 ? '+' : ''}{q.avgReturn.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-2 text-terminal-muted font-normal">Month</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Avg Return</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Median</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Std Dev</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Win Rate</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Years</th>
            </tr>
          </thead>
          <tbody>
            {seasonalData.map((stat) => (
              <tr key={stat.month} className="border-b border-terminal-border/30 hover:bg-terminal-surface/50">
                <td className="py-2 px-2 font-medium">{stat.fullName}</td>
                <td className="py-2 px-2 text-right">
                  <span className={`font-mono font-bold ${stat.avgReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {stat.avgReturn >= 0 ? '+' : ''}{stat.avgReturn.toFixed(2)}%
                  </span>
                </td>
                <td className="py-2 px-2 text-right font-mono">
                  {stat.median >= 0 ? '+' : ''}{stat.median.toFixed(2)}%
                </td>
                <td className="py-2 px-2 text-right font-mono text-terminal-muted">
                  {stat.std.toFixed(2)}%
                </td>
                <td className="py-2 px-2 text-right">
                  <span className={`font-mono ${stat.winRate >= 50 ? 'text-profit' : 'text-loss'}`}>
                    {stat.winRate.toFixed(0)}%
                  </span>
                </td>
                <td className="py-2 px-2 text-right font-mono text-terminal-muted">{stat.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
