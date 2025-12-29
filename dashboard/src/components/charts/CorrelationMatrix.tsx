import { useMemo, useState } from 'react';

interface CorrelationMatrixProps {
  labels: string[];
  matrix: number[][];
  onCellClick?: (i: number, j: number) => void;
}

export function CorrelationMatrix({ labels, matrix, onCellClick }: CorrelationMatrixProps) {
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null);

  const getColor = (value: number): string => {
    // Color scale from red (negative) through white (0) to green (positive)
    const abs = Math.abs(value);
    if (value > 0.8) return 'bg-profit text-black';
    if (value > 0.5) return 'bg-profit/70 text-black';
    if (value > 0.2) return 'bg-profit/30 text-white';
    if (value > -0.2) return 'bg-terminal-surface text-white';
    if (value > -0.5) return 'bg-loss/30 text-white';
    if (value > -0.8) return 'bg-loss/70 text-white';
    return 'bg-loss text-white';
  };

  const { minCorr, maxCorr, avgCorr } = useMemo(() => {
    let min = 1, max = -1, sum = 0, count = 0;
    
    for (let i = 0; i < matrix.length; i++) {
      for (let j = i + 1; j < matrix[i].length; j++) {
        const val = matrix[i][j];
        min = Math.min(min, val);
        max = Math.max(max, val);
        sum += val;
        count++;
      }
    }
    
    return {
      minCorr: min,
      maxCorr: max,
      avgCorr: count > 0 ? sum / count : 0,
    };
  }, [matrix]);

  if (labels.length === 0 || matrix.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted">
        Select candidates to compare
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      <div className="flex items-center gap-6 text-xs">
        <div>
          <span className="text-terminal-muted">Min Corr:</span>
          <span className={`ml-1 font-mono ${minCorr < 0 ? 'text-loss' : 'text-profit'}`}>
            {minCorr.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Max Corr:</span>
          <span className={`ml-1 font-mono ${maxCorr < 0.5 ? 'text-profit' : 'text-accent-yellow'}`}>
            {maxCorr.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Avg Corr:</span>
          <span className="ml-1 font-mono">{avgCorr.toFixed(2)}</span>
        </div>
      </div>

      {/* Matrix */}
      <div className="overflow-x-auto">
        <table className="text-xs font-mono">
          <thead>
            <tr>
              <th className="p-2"></th>
              {labels.map((label, i) => (
                <th
                  key={i}
                  className="p-2 text-terminal-muted font-normal max-w-[80px] truncate"
                  title={label}
                >
                  {label.length > 10 ? `${label.substring(0, 10)}...` : label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td
                  className="p-2 text-terminal-muted max-w-[80px] truncate"
                  title={labels[i]}
                >
                  {labels[i].length > 10 ? `${labels[i].substring(0, 10)}...` : labels[i]}
                </td>
                {row.map((value, j) => (
                  <td
                    key={j}
                    className="p-1"
                    onMouseEnter={() => setHoveredCell({ i, j })}
                    onMouseLeave={() => setHoveredCell(null)}
                    onClick={() => onCellClick?.(i, j)}
                  >
                    <div
                      className={`
                        w-12 h-10 flex items-center justify-center rounded cursor-pointer
                        transition-all duration-150
                        ${getColor(value)}
                        ${hoveredCell?.i === i && hoveredCell?.j === j ? 'ring-2 ring-white scale-105' : ''}
                        ${i === j ? 'opacity-50' : 'hover:scale-105'}
                      `}
                      title={`${labels[i]} × ${labels[j]}: ${value.toFixed(3)}`}
                    >
                      {value.toFixed(2)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-2 text-xs">
        <div className="w-6 h-4 rounded bg-loss" />
        <span className="text-terminal-muted">-1.0</span>
        <div className="w-16 h-4 rounded bg-gradient-to-r from-loss via-terminal-surface to-profit" />
        <span className="text-terminal-muted">+1.0</span>
        <div className="w-6 h-4 rounded bg-profit" />
      </div>
    </div>
  );
}

