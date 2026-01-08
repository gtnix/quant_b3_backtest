import { useMemo, useState } from 'react';
import { BarChart3, Grid3X3 } from 'lucide-react';

interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

interface MonthlyHeatmapProps {
  data: MonthlyReturn[];
  defaultView?: 'heatmap' | 'bars';
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function MonthlyHeatmap({ data, defaultView = 'heatmap' }: MonthlyHeatmapProps) {
  const [viewMode, setViewMode] = useState<'heatmap' | 'bars'>(defaultView);
  const { years, grid, yearlyReturns } = useMemo(() => {
    if (data.length === 0) return { years: [], grid: new Map(), yearlyReturns: new Map() };

    const grid = new Map<string, number>();
    const yearlyReturns = new Map<number, number>();
    const yearSet = new Set<number>();

    for (const item of data) {
      yearSet.add(item.year);
      const key = `${item.year}-${item.month}`;
      grid.set(key, item.return_pct);

      // Accumulate yearly returns
      yearlyReturns.set(
        item.year,
        (yearlyReturns.get(item.year) || 0) + item.return_pct
      );
    }

    const years = Array.from(yearSet).sort((a, b) => b - a); // Most recent first
    return { years, grid, yearlyReturns };
  }, [data]);

  // Color scale function
  const getColor = (value: number): string => {
    if (value > 5) return 'bg-profit text-black';
    if (value > 2) return 'bg-profit/70 text-black';
    if (value > 0) return 'bg-profit/30 text-white';
    if (value > -2) return 'bg-loss/30 text-white';
    if (value > -5) return 'bg-loss/70 text-white';
    return 'bg-loss text-white';
  };

  // Find max absolute return for bar scaling
  const maxAbsReturn = useMemo(() => {
    if (data.length === 0) return 10;
    return Math.max(...data.map(d => Math.abs(d.return_pct)), 10);
  }, [data]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        No monthly return data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* View Toggle */}
      <div className="flex items-center justify-end gap-2">
        <button
          onClick={() => setViewMode('heatmap')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            viewMode === 'heatmap'
              ? 'bg-profit/20 text-profit border border-profit/30'
              : 'bg-terminal-surface border border-terminal-border hover:border-terminal-muted'
          }`}
        >
          <Grid3X3 className="w-3.5 h-3.5" />
          Heatmap
        </button>
        <button
          onClick={() => setViewMode('bars')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            viewMode === 'bars'
              ? 'bg-profit/20 text-profit border border-profit/30'
              : 'bg-terminal-surface border border-terminal-border hover:border-terminal-muted'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Bars
        </button>
      </div>

      {viewMode === 'heatmap' ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr>
                <th className="text-left py-2 px-2 text-terminal-muted font-normal">Year</th>
                {MONTHS.map((month) => (
                  <th key={month} className="py-2 px-1 text-terminal-muted font-normal text-center">
                    {month}
                  </th>
                ))}
                <th className="py-2 px-2 text-terminal-muted font-normal text-center">YTD</th>
              </tr>
            </thead>
            <tbody>
              {years.map((year) => (
                <tr key={year} className="border-t border-terminal-border/30">
                  <td className="py-1 px-2 text-terminal-muted">{year}</td>
                  {MONTHS.map((_, monthIdx) => {
                    const key = `${year}-${monthIdx + 1}`;
                    const value = grid.get(key);
                    const hasValue = value !== undefined;
                    
                    return (
                      <td key={monthIdx} className="py-1 px-1 text-center">
                        {hasValue ? (
                          <div
                            className={`rounded px-1 py-0.5 ${getColor(value)}`}
                            title={`${MONTHS[monthIdx]} ${year}: ${value.toFixed(2)}%`}
                          >
                            {value.toFixed(1)}
                          </div>
                        ) : (
                          <div className="text-terminal-muted/30">-</div>
                        )}
                      </td>
                    );
                  })}
                  <td className="py-1 px-2 text-center">
                    {yearlyReturns.has(year) ? (
                      <div
                        className={`rounded px-2 py-0.5 font-medium ${getColor(yearlyReturns.get(year)!)}`}
                      >
                        {yearlyReturns.get(year)!.toFixed(1)}%
                      </div>
                    ) : (
                      <span className="text-terminal-muted">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Legend */}
          <div className="flex items-center justify-center gap-3 mt-4 text-xs flex-wrap">
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-loss" />
              <span className="text-gray-300">{'< -5%'}</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-loss/70" />
              <span className="text-gray-300">-5% to -2%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-loss/30" />
              <span className="text-gray-300">-2% to 0%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-profit/30" />
              <span className="text-gray-300">0% to 2%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-profit/70" />
              <span className="text-gray-300">2% to 5%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-profit" />
              <span className="text-gray-300">{'> 5%'}</span>
            </div>
          </div>
        </div>
      ) : (
        /* Bar Chart View */
        <div className="space-y-6">
          {years.map((year) => (
            <div key={year} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-terminal-muted">{year}</span>
                {yearlyReturns.has(year) && (
                  <span className={`text-sm font-mono font-bold ${yearlyReturns.get(year)! >= 0 ? 'text-profit' : 'text-loss'}`}>
                    YTD: {yearlyReturns.get(year)! >= 0 ? '+' : ''}{yearlyReturns.get(year)!.toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="flex items-end gap-1 h-24">
                {MONTHS.map((month, monthIdx) => {
                  const key = `${year}-${monthIdx + 1}`;
                  const value = grid.get(key);
                  if (value === undefined) {
                    return (
                      <div key={monthIdx} className="flex-1 flex flex-col items-center">
                        <div className="w-full h-16" />
                        <span className="text-[9px] text-terminal-muted/30 mt-1">{month}</span>
                      </div>
                    );
                  }
                  const height = Math.abs(value) / maxAbsReturn * 100;
                  const isPositive = value >= 0;
                  return (
                    <div key={monthIdx} className="flex-1 flex flex-col items-center group relative">
                      <div className="w-full h-16 flex flex-col justify-center">
                        {isPositive ? (
                          <div className="w-full flex flex-col justify-end h-8">
                            <div
                              className="w-full bg-profit rounded-t transition-all group-hover:bg-profit/80"
                              style={{ height: `${Math.max(height / 2, 4)}%` }}
                            />
                          </div>
                        ) : (
                          <div className="w-full flex flex-col justify-start h-8">
                            <div
                              className="w-full bg-loss rounded-b transition-all group-hover:bg-loss/80"
                              style={{ height: `${Math.max(height / 2, 4)}%` }}
                            />
                          </div>
                        )}
                      </div>
                      <span className="text-[9px] text-terminal-muted mt-1">{month}</span>
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        <div className="bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs whitespace-nowrap shadow-lg">
                          <span className={isPositive ? 'text-profit' : 'text-loss'}>
                            {isPositive ? '+' : ''}{value.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

