import { useMemo } from 'react';
import { 
  X, 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  DollarSign, 
  Target,
  Calendar,
  Hash,
  ArrowRight,
  BarChart3,
  Percent,
  Zap
} from 'lucide-react';
import type { TradeRecord } from '../stores/dataStore';
import { formatHoldingPeriod, formatCurrency, formatPct } from './TradeBlotter';
import { format, parseISO } from 'date-fns';
import { ptBR } from 'date-fns/locale';

/** Format date - only date, no fake minute times */
function formatDateOnly(dateStr: string): string {
  try {
    const date = parseISO(dateStr.replace(' ', 'T'));
    return format(date, 'dd/MM/yyyy', { locale: ptBR });
  } catch {
    return dateStr.slice(0, 10);
  }
}

interface TradeDetailModalProps {
  trade: TradeRecord;
  allTrades: TradeRecord[];
  onClose: () => void;
}

export function TradeDetailModal({ trade, allTrades, onClose }: TradeDetailModalProps) {
  // Calculate cost breakdown in bps
  const notional = trade.quantity * trade.entry_price;
  const commissionBps = (trade.commission / notional) * 10000;
  const slippageBps = (trade.slippage / notional) * 10000;
  const totalCostBps = commissionBps + slippageBps;
  
  // Price movement
  const priceMove = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100;
  const priceMoveAdjusted = trade.direction === 'Long' ? priceMove : -priceMove;
  
  // Similar trades (same symbol)
  const similarTrades = useMemo(() => {
    return allTrades
      .filter(t => t.symbol === trade.symbol && t.trade_id !== trade.trade_id)
      .slice(0, 5);
  }, [allTrades, trade]);
  
  // Symbol stats
  const symbolStats = useMemo(() => {
    const symbolTrades = allTrades.filter(t => t.symbol === trade.symbol);
    const winners = symbolTrades.filter(t => t.is_winner);
    const totalPnL = symbolTrades.reduce((s, t) => s + t.net_pnl, 0);
    const avgReturn = symbolTrades.length > 0 
      ? symbolTrades.reduce((s, t) => s + t.return_pct, 0) / symbolTrades.length 
      : 0;
    return {
      totalTrades: symbolTrades.length,
      winRate: symbolTrades.length > 0 ? (winners.length / symbolTrades.length) * 100 : 0,
      totalPnL,
      avgReturn
    };
  }, [allTrades, trade]);

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-terminal-bg border border-terminal-border rounded-xl w-full max-w-[780px] max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-terminal-bg border-b border-terminal-border px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl ${trade.is_winner ? 'bg-profit/20' : 'bg-loss/20'}`}>
              {trade.is_winner ? (
                <TrendingUp className="w-6 h-6 text-profit" />
              ) : (
                <TrendingDown className="w-6 h-6 text-loss" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-bold">{trade.symbol}</h2>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  trade.direction === 'Long' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                }`}>
                  {trade.direction}
                </span>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  trade.is_winner ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                }`}>
                  {trade.is_winner ? 'WINNER' : 'LOSER'}
                </span>
              </div>
              <p className="text-sm text-terminal-muted font-mono">{trade.trade_id}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-terminal-surface rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {/* Main Metrics Grid */}
          <div className="grid grid-cols-4 gap-4">
            <MetricBox 
              icon={<DollarSign className="w-4 h-4" />}
              label="Net PnL"
              value={formatCurrency(trade.net_pnl)}
              color={trade.net_pnl >= 0 ? 'profit' : 'loss'}
              large
            />
            <MetricBox 
              icon={<Percent className="w-4 h-4" />}
              label="Return"
              value={formatPct(trade.return_pct)}
              color={trade.return_pct >= 0 ? 'profit' : 'loss'}
              large
            />
            <MetricBox 
              icon={<Clock className="w-4 h-4" />}
              label="Holding Period"
              value={formatHoldingPeriod(trade.holding_period_hours)}
            />
            <MetricBox 
              icon={<BarChart3 className="w-4 h-4" />}
              label="Quantity"
              value={trade.quantity.toLocaleString()}
            />
          </div>

          {/* Entry/Exit Timeline */}
          <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-5">
            <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-4">Trade Timeline</h3>
            <div className="flex items-center">
              <div className="flex-1">
                <div className="text-xs text-terminal-muted mb-1">ENTRY</div>
                <div className="font-mono text-sm">{formatDateOnly(trade.entry_date)}</div>
                <div className="font-mono text-lg font-bold mt-1">R$ {trade.entry_price.toFixed(2)}</div>
              </div>
              <div className="flex flex-col items-center px-6">
                <ArrowRight className="w-6 h-6 text-terminal-muted" />
                <div className={`text-sm font-mono mt-1 ${priceMoveAdjusted >= 0 ? 'text-profit' : 'text-loss'}`}>
                  {priceMoveAdjusted >= 0 ? '+' : ''}{priceMove.toFixed(2)}%
                </div>
              </div>
              <div className="flex-1 text-right">
                <div className="text-xs text-terminal-muted mb-1">EXIT</div>
                <div className="font-mono text-sm">{formatDateOnly(trade.exit_date)}</div>
                <div className="font-mono text-lg font-bold mt-1">R$ {trade.exit_price.toFixed(2)}</div>
              </div>
            </div>
          </div>

          {/* PnL Breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-3">PnL Breakdown</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Gross PnL</span>
                  <span className={`font-mono ${trade.gross_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {formatCurrency(trade.gross_pnl)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Commission</span>
                  <span className="font-mono text-loss">-{formatCurrency(trade.commission)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Slippage</span>
                  <span className="font-mono text-loss">-{formatCurrency(trade.slippage)}</span>
                </div>
                <div className="border-t border-terminal-border pt-2 flex justify-between font-medium">
                  <span>Net PnL</span>
                  <span className={`font-mono ${trade.net_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {formatCurrency(trade.net_pnl)}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-3">Cost Analysis</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Notional</span>
                  <span className="font-mono">{formatCurrency(notional)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Commission</span>
                  <span className="font-mono">{commissionBps.toFixed(1)} bps</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-terminal-muted">Slippage</span>
                  <span className="font-mono">{slippageBps.toFixed(1)} bps</span>
                </div>
                <div className="border-t border-terminal-border pt-2 flex justify-between font-medium">
                  <span>Total Cost</span>
                  <span className="font-mono text-loss">{totalCostBps.toFixed(1)} bps</span>
                </div>
              </div>
            </div>
          </div>

          {/* Symbol Performance */}
          <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
            <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-3">
              {trade.symbol} Performance Summary
            </h3>
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="text-xs text-terminal-muted">Total Trades</div>
                <div className="font-mono font-bold text-lg">{symbolStats.totalTrades}</div>
              </div>
              <div>
                <div className="text-xs text-terminal-muted">Win Rate</div>
                <div className={`font-mono font-bold text-lg ${symbolStats.winRate >= 50 ? 'text-profit' : 'text-loss'}`}>
                  {symbolStats.winRate.toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-terminal-muted">Total PnL</div>
                <div className={`font-mono font-bold text-lg ${symbolStats.totalPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
                  {formatCurrency(symbolStats.totalPnL)}
                </div>
              </div>
              <div>
                <div className="text-xs text-terminal-muted">Avg Return</div>
                <div className={`font-mono font-bold text-lg ${symbolStats.avgReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                  {formatPct(symbolStats.avgReturn)}
                </div>
              </div>
            </div>
          </div>

          {/* Similar Trades */}
          {similarTrades.length > 0 && (
            <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-terminal-muted uppercase tracking-wider mb-3">
                Recent {trade.symbol} Trades
              </h3>
              <div className="space-y-2">
                {similarTrades.map(t => (
                  <div key={t.trade_id} className="flex items-center justify-between py-2 border-b border-terminal-border/50 last:border-0">
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        t.direction === 'Long' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                      }`}>
                        {t.direction}
                      </span>
                      <span className="text-sm text-terminal-muted font-mono">{formatDateOnly(t.entry_date)}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className={`font-mono text-sm ${t.net_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {formatCurrency(t.net_pnl)}
                      </span>
                      <span className={`font-mono text-xs ${t.return_pct >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {formatPct(t.return_pct)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricBox({ 
  icon, 
  label, 
  value, 
  color,
  large
}: { 
  icon: React.ReactNode; 
  label: string; 
  value: string; 
  color?: 'profit' | 'loss';
  large?: boolean;
}) {
  const colorClass = color === 'profit' ? 'text-profit' : color === 'loss' ? 'text-loss' : 'text-white';
  return (
    <div className="bg-terminal-surface/50 border border-terminal-border rounded-xl p-4">
      <div className="flex items-center gap-2 text-terminal-muted mb-2">
        {icon}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <div className={`font-mono font-bold ${large ? 'text-xl' : 'text-lg'} ${colorClass}`}>
        {value}
      </div>
    </div>
  );
}
