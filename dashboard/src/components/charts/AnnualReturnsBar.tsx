import { useMemo } from 'react';

interface AnnualReturnsBarProps {
  monthlyReturns: Array<{ year: number; month: number; return_pct: number }>;
}

export function AnnualReturnsBar({ monthlyReturns }: AnnualReturnsBarProps) {
  const annualData = useMemo(() => {
    if (monthlyReturns.length === 0) return [];

    const byYear = new Map<number, number[]>();
    for (const m of monthlyReturns) {
      if (!byYear.has(m.year)) byYear.set(m.year, []);
      byYear.get(m.year)!.push(m.return_pct);
    }

    return Array.from(byYear.entries())
      .map(([year, returns]) => {
        const totalReturn = returns.reduce((a, b) => a + b, 0);
        const monthsWithData = returns.length;
        const positiveMonths = returns.filter(r => r > 0).length;
        const negativeMonths = returns.filter(r => r < 0).length;
        return { year, totalReturn, monthsWithData, positiveMonths, negativeMonths };
      })
      .sort((a, b) => a.year - b.year);
  }, [monthlyReturns]);

  if (annualData.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted">
        No annual data available
      </div>
    );
  }

  const maxReturn = Math.max(...annualData.map(d => Math.abs(d.totalReturn)), 10);
  const bestYear = annualData.reduce((best, y) => y.totalReturn > best.totalReturn ? y : best, annualData[0]);
  const worstYear = annualData.reduce((worst, y) => y.totalReturn < worst.totalReturn ? y : worst, annualData[0]);
  const avgReturn = annualData.reduce((sum, y) => sum + y.totalReturn, 0) / annualData.length;

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-profit/10 border border-profit/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Best Year</div>
          <div className="text-xl font-bold text-profit">{bestYear.year}</div>
          <div className="text-sm font-mono text-profit/80">+{bestYear.totalReturn.toFixed(1)}%</div>
        </div>
        <div className="p-4 rounded-xl bg-loss/10 border border-loss/30">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Worst Year</div>
          <div className="text-xl font-bold text-loss">{worstYear.year}</div>
          <div className="text-sm font-mono text-loss/80">{worstYear.totalReturn.toFixed(1)}%</div>
        </div>
        <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Average</div>
          <div className={`text-xl font-bold ${avgReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
            {avgReturn >= 0 ? '+' : ''}{avgReturn.toFixed(1)}%
          </div>
          <div className="text-sm text-terminal-muted">{annualData.length} years</div>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="space-y-2">
        <div className="flex items-end gap-2 h-48">
          {annualData.map((yearData) => {
            const height = (Math.abs(yearData.totalReturn) / maxReturn) * 100;
            const isPositive = yearData.totalReturn >= 0;
            
            return (
              <div key={yearData.year} className="flex-1 flex flex-col items-center group relative min-w-[40px]">
                <div className="w-full h-40 flex flex-col justify-center relative">
                  {/* Zero line */}
                  <div className="absolute left-0 right-0 h-px bg-terminal-border" style={{ top: '50%' }} />
                  
                  {isPositive ? (
                    <div className="absolute bottom-1/2 left-1 right-1 flex flex-col justify-end">
                      <div
                        className="w-full bg-profit rounded-t transition-all hover:bg-profit/80"
                        style={{ height: `${Math.max(height / 2, 4)}%` }}
                      />
                    </div>
                  ) : (
                    <div className="absolute top-1/2 left-1 right-1 flex flex-col justify-start">
                      <div
                        className="w-full bg-loss rounded-b transition-all hover:bg-loss/80"
                        style={{ height: `${Math.max(height / 2, 4)}%` }}
                      />
                    </div>
                  )}
                </div>
                
                {/* Year label */}
                <span className="text-xs text-terminal-muted mt-2">{yearData.year}</span>
                
                {/* Value label */}
                <span className={`text-xs font-mono ${isPositive ? 'text-profit' : 'text-loss'}`}>
                  {isPositive ? '+' : ''}{yearData.totalReturn.toFixed(1)}%
                </span>
                
                {/* Tooltip */}
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                  <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs whitespace-nowrap shadow-lg">
                    <div className="font-medium mb-1">{yearData.year}</div>
                    <div className="flex justify-between gap-3">
                      <span className="text-terminal-muted">Return:</span>
                      <span className={`font-mono font-bold ${isPositive ? 'text-profit' : 'text-loss'}`}>
                        {isPositive ? '+' : ''}{yearData.totalReturn.toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-terminal-muted">Months:</span>
                      <span className="font-mono">{yearData.monthsWithData}</span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-profit">↑ {yearData.positiveMonths}</span>
                      <span className="text-loss">↓ {yearData.negativeMonths}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detailed Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-2 text-terminal-muted font-normal">Year</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Return</th>
              <th className="text-center py-2 px-2 text-terminal-muted font-normal">Months</th>
              <th className="text-center py-2 px-2 text-terminal-muted font-normal">Win/Loss</th>
              <th className="text-right py-2 px-2 text-terminal-muted font-normal">Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {annualData.slice().reverse().map((yearData) => {
              const winRate = yearData.monthsWithData > 0 
                ? (yearData.positiveMonths / yearData.monthsWithData) * 100 
                : 0;
              const isPositive = yearData.totalReturn >= 0;
              
              return (
                <tr key={yearData.year} className="border-b border-terminal-border/30 hover:bg-terminal-surface/50">
                  <td className="py-2 px-2 font-medium">{yearData.year}</td>
                  <td className="py-2 px-2 text-right">
                    <span className={`font-mono font-bold ${isPositive ? 'text-profit' : 'text-loss'}`}>
                      {isPositive ? '+' : ''}{yearData.totalReturn.toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-2 px-2 text-center text-terminal-muted">{yearData.monthsWithData}</td>
                  <td className="py-2 px-2 text-center">
                    <span className="text-profit">{yearData.positiveMonths}</span>
                    <span className="text-terminal-muted mx-1">/</span>
                    <span className="text-loss">{yearData.negativeMonths}</span>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <span className={`font-mono ${winRate >= 50 ? 'text-profit' : 'text-loss'}`}>
                      {winRate.toFixed(0)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
