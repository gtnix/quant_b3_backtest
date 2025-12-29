import { useMemo } from 'react';

interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

interface MonthlyHeatmapProps {
  data: MonthlyReturn[];
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function MonthlyHeatmap({ data }: MonthlyHeatmapProps) {
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

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        No monthly return data available
      </div>
    );
  }

  return (
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
      <div className="flex items-center justify-center gap-4 mt-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-loss" />
          <span className="text-terminal-muted">{'< -5%'}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-loss/50" />
          <span className="text-terminal-muted">-5% to 0%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-profit/30" />
          <span className="text-terminal-muted">0% to 2%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-profit" />
          <span className="text-terminal-muted">{'> 5%'}</span>
        </div>
      </div>
    </div>
  );
}

