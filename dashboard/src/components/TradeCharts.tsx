import { useMemo } from 'react';
import { 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ReferenceLine,
  CartesianGrid
} from 'recharts';
import type { TradeRecord } from '../stores/dataStore';
import type { TradeAnalytics } from './TradeBlotter';
import { formatCurrency, formatPct, formatHoldingPeriod } from './TradeBlotter';

interface TradeChartsProps {
  trades: TradeRecord[];
  analytics: TradeAnalytics;
  onTradeClick?: (trade: TradeRecord) => void;
}

export function TradeCharts({ trades, analytics, onTradeClick }: TradeChartsProps) {
  // Cumulative PnL data
  const cumulativePnLData = useMemo(() => {
    let cumPnL = 0;
    return trades.map((t, i) => {
      cumPnL += t.net_pnl;
      return {
        index: i + 1,
        pnl: cumPnL,
        trade: t
      };
    });
  }, [trades]);

  // PnL Distribution histogram
  const pnlDistribution = useMemo(() => {
    if (trades.length === 0) return [];
    
    const pnls = trades.map(t => t.net_pnl);
    const min = Math.min(...pnls);
    const max = Math.max(...pnls);
    const range = max - min;
    const binCount = 20;
    const binSize = range / binCount;
    
    const bins: { range: string; count: number; isPositive: boolean }[] = [];
    for (let i = 0; i < binCount; i++) {
      const binMin = min + i * binSize;
      const binMax = min + (i + 1) * binSize;
      const count = pnls.filter(p => p >= binMin && p < binMax).length;
      bins.push({
        range: `${(binMin / 1000).toFixed(0)}K`,
        count,
        isPositive: (binMin + binMax) / 2 >= 0
      });
    }
    return bins;
  }, [trades]);

  // Win rate by symbol
  const winRateBySymbol = useMemo(() => {
    return Object.entries(analytics.bySymbol)
      .map(([symbol, stats]) => ({
        symbol,
        winRate: stats.winRate * 100,
        trades: stats.trades,
        pnl: stats.pnl
      }))
      .sort((a, b) => b.pnl - a.pnl)
      .slice(0, 10);
  }, [analytics.bySymbol]);

  // Holding period vs Return scatter
  const holdingVsReturn = useMemo(() => {
    return trades.map(t => ({
      holdingHours: t.holding_period_hours,
      returnPct: t.return_pct,
      isWinner: t.is_winner,
      trade: t
    }));
  }, [trades]);

  // PnL by trade number (for equity curve)
  const equityCurve = useMemo(() => {
    let equity = 100000; // Starting capital
    return trades.map((t, i) => {
      equity += t.net_pnl;
      return {
        trade: i + 1,
        equity,
        pnl: t.net_pnl,
        isWinner: t.is_winner
      };
    });
  }, [trades]);

  // Direction breakdown
  const directionData = useMemo(() => {
    return Object.entries(analytics.byDirection).map(([dir, stats]) => ({
      direction: dir,
      trades: stats.trades,
      winRate: stats.winRate * 100,
      pnl: stats.pnl
    }));
  }, [analytics.byDirection]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Cumulative PnL Chart */}
      <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">
          Cumulative PnL
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={cumulativePnLData}>
              <defs>
                <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="index" 
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
              />
              <YAxis 
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
              />
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs">
                        <div>Trade #{data.index}</div>
                        <div className={data.pnl >= 0 ? 'text-profit' : 'text-loss'}>
                          PnL: {formatCurrency(data.pnl)}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area 
                type="monotone" 
                dataKey="pnl" 
                stroke="#00ff88" 
                fill="url(#pnlGradient)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* PnL Distribution */}
      <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">
          PnL Distribution
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={pnlDistribution}>
              <XAxis 
                dataKey="range" 
                tick={{ fill: '#6b7280', fontSize: 9 }}
                axisLine={{ stroke: '#374151' }}
                interval={3}
              />
              <YAxis 
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs">
                        <div>Range: {payload[0].payload.range}</div>
                        <div>Count: {payload[0].payload.count}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {pnlDistribution.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.isPositive ? '#00ff88' : '#ef4444'} 
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Win Rate by Symbol */}
      <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">
          PnL by Symbol
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={winRateBySymbol} layout="vertical">
              <XAxis 
                type="number"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
              />
              <YAxis 
                type="category"
                dataKey="symbol" 
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
                width={50}
              />
              <ReferenceLine x={0} stroke="#374151" />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs">
                        <div className="font-medium">{data.symbol}</div>
                        <div>Trades: {data.trades}</div>
                        <div>Win Rate: {data.winRate.toFixed(1)}%</div>
                        <div className={data.pnl >= 0 ? 'text-profit' : 'text-loss'}>
                          PnL: {formatCurrency(data.pnl)}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="pnl" radius={[0, 4, 4, 0]}>
                {winRateBySymbol.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.pnl >= 0 ? '#00ff88' : '#ef4444'} 
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Holding Period vs Return Scatter */}
      <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">
          Holding Period vs Return
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="holdingHours" 
                type="number"
                name="Holding (h)"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
                tickFormatter={(v) => v < 24 ? `${v}h` : `${Math.round(v/24)}d`}
              />
              <YAxis 
                dataKey="returnPct" 
                type="number"
                name="Return %"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                axisLine={{ stroke: '#374151' }}
                tickFormatter={(v) => `${v.toFixed(0)}%`}
              />
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs">
                        <div>Hold: {formatHoldingPeriod(data.holdingHours)}</div>
                        <div className={data.returnPct >= 0 ? 'text-profit' : 'text-loss'}>
                          Return: {formatPct(data.returnPct)}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
                cursor={{ strokeDasharray: '3 3' }}
              />
              <Scatter 
                data={holdingVsReturn} 
                fill="#00d4ff"
                onClick={(data) => onTradeClick && onTradeClick(data.trade)}
              >
                {holdingVsReturn.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`}
                    fill={entry.isWinner ? '#00ff88' : '#ef4444'}
                    fillOpacity={0.7}
                    cursor="pointer"
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Direction Performance */}
      <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4 lg:col-span-2">
        <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">
          Performance by Direction
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {directionData.map(d => (
            <div key={d.direction} className="flex items-center justify-between p-4 bg-terminal-bg/50 rounded-lg">
              <div className="flex items-center gap-3">
                <span className={`px-3 py-1 rounded font-medium ${
                  d.direction === 'Long' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                }`}>
                  {d.direction}
                </span>
                <div>
                  <div className="text-sm font-medium">{d.trades} trades</div>
                  <div className="text-xs text-terminal-muted">Win Rate: {d.winRate.toFixed(1)}%</div>
                </div>
              </div>
              <div className={`font-mono font-bold text-lg ${d.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                {formatCurrency(d.pnl)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
