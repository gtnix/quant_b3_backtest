import { useState, useEffect, useMemo, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { format, parseISO, differenceInHours, differenceInMinutes } from 'date-fns';
import { ptBR } from 'date-fns/locale';
import * as XLSX from 'xlsx';
import { 
  ChevronUp, 
  ChevronDown, 
  ChevronsUpDown, 
  Download, 
  Search,
  TrendingUp,
  TrendingDown,
  Clock,
  DollarSign,
  Target,
  BarChart3,
  RefreshCw,
  FileSpreadsheet,
  X,
  ChevronRight,
  Zap,
  Activity,
  Award,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  Calendar
} from 'lucide-react';
import { useDataStore } from '../stores/dataStore';
import type { TradeRecord, TradesSummary } from '../stores/dataStore';
import { config } from '../lib/platform';
import { TradeDetailModal } from './TradeDetailModal';
import { TradeCharts } from './TradeCharts';

interface TradeBlotterProps {
  candidateId: string;
}

type SortKey = keyof TradeRecord;
type SortDirection = 'asc' | 'desc';

// =============================================================================
// FORMATTING UTILITIES - Institutional Grade
// =============================================================================

/** Format date to Brazilian standard - only date for daily data */
function formatDateBR(dateStr: string, showTime = false): string {
  try {
    const date = parseISO(dateStr.replace(' ', 'T'));
    // For daily data, only show date (no fake minute times)
    return format(date, showTime ? 'dd/MM/yyyy HH:mm' : 'dd/MM/yyyy', { locale: ptBR });
  } catch {
    return dateStr.slice(0, 10);
  }
}

/** Format holding period as human readable */
function formatHoldingPeriod(hours: number): string {
  if (hours < 1) {
    const mins = Math.round(hours * 60);
    return `${mins}m`;
  }
  if (hours < 24) {
    const h = Math.floor(hours);
    const m = Math.round((hours - h) * 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  const days = Math.floor(hours / 24);
  const remainingHours = Math.round(hours % 24);
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

/** Format currency with smart notation */
function formatCurrency(v: number, compact = false): string {
  if (compact && Math.abs(v) >= 1000000) {
    return `R$ ${(v / 1000000).toFixed(1)}M`;
  }
  if (compact && Math.abs(v) >= 1000) {
    return `R$ ${(v / 1000).toFixed(1)}K`;
  }
  return v.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
}

/** Format percentage */
function formatPct(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

/** Format number with K/M notation */
function formatNumber(v: number): string {
  if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}K`;
  return v.toLocaleString();
}

// =============================================================================
// ANALYTICS CALCULATIONS
// =============================================================================

interface TradeAnalytics {
  bySymbol: Record<string, { trades: number; pnl: number; winRate: number }>;
  byDirection: Record<string, { trades: number; pnl: number; winRate: number }>;
  largestWin: TradeRecord | null;
  largestLoss: TradeRecord | null;
  maxWinStreak: number;
  maxLossStreak: number;
  currentStreak: { type: 'win' | 'loss'; count: number };
  avgHoldWinners: number;
  avgHoldLosers: number;
  cumulativePnL: number[];
}

function calculateAnalytics(trades: TradeRecord[]): TradeAnalytics {
  const bySymbol: Record<string, { trades: number; pnl: number; wins: number }> = {};
  const byDirection: Record<string, { trades: number; pnl: number; wins: number }> = {};
  
  let largestWin: TradeRecord | null = null;
  let largestLoss: TradeRecord | null = null;
  let maxWinStreak = 0;
  let maxLossStreak = 0;
  let currentWinStreak = 0;
  let currentLossStreak = 0;
  let currentStreakType: 'win' | 'loss' = 'win';
  let currentStreakCount = 0;
  
  const winnerHolds: number[] = [];
  const loserHolds: number[] = [];
  const cumulativePnL: number[] = [];
  let cumPnL = 0;
  
  for (const trade of trades) {
    // Cumulative PnL
    cumPnL += trade.net_pnl;
    cumulativePnL.push(cumPnL);
    
    // By Symbol
    if (!bySymbol[trade.symbol]) {
      bySymbol[trade.symbol] = { trades: 0, pnl: 0, wins: 0 };
    }
    bySymbol[trade.symbol].trades++;
    bySymbol[trade.symbol].pnl += trade.net_pnl;
    if (trade.is_winner) bySymbol[trade.symbol].wins++;
    
    // By Direction
    if (!byDirection[trade.direction]) {
      byDirection[trade.direction] = { trades: 0, pnl: 0, wins: 0 };
    }
    byDirection[trade.direction].trades++;
    byDirection[trade.direction].pnl += trade.net_pnl;
    if (trade.is_winner) byDirection[trade.direction].wins++;
    
    // Largest Win/Loss
    if (trade.net_pnl > 0 && (!largestWin || trade.net_pnl > largestWin.net_pnl)) {
      largestWin = trade;
    }
    if (trade.net_pnl < 0 && (!largestLoss || trade.net_pnl < largestLoss.net_pnl)) {
      largestLoss = trade;
    }
    
    // Streaks
    if (trade.is_winner) {
      currentWinStreak++;
      currentLossStreak = 0;
      maxWinStreak = Math.max(maxWinStreak, currentWinStreak);
      currentStreakType = 'win';
      currentStreakCount = currentWinStreak;
    } else {
      currentLossStreak++;
      currentWinStreak = 0;
      maxLossStreak = Math.max(maxLossStreak, currentLossStreak);
      currentStreakType = 'loss';
      currentStreakCount = currentLossStreak;
    }
    
    // Holding periods
    if (trade.is_winner) {
      winnerHolds.push(trade.holding_period_hours);
    } else {
      loserHolds.push(trade.holding_period_hours);
    }
  }
  
  // Calculate win rates
  const bySymbolWithRates = Object.fromEntries(
    Object.entries(bySymbol).map(([k, v]) => [k, { 
      trades: v.trades, 
      pnl: v.pnl, 
      winRate: v.trades > 0 ? v.wins / v.trades : 0 
    }])
  );
  
  const byDirectionWithRates = Object.fromEntries(
    Object.entries(byDirection).map(([k, v]) => [k, { 
      trades: v.trades, 
      pnl: v.pnl, 
      winRate: v.trades > 0 ? v.wins / v.trades : 0 
    }])
  );
  
  return {
    bySymbol: bySymbolWithRates,
    byDirection: byDirectionWithRates,
    largestWin,
    largestLoss,
    maxWinStreak,
    maxLossStreak,
    currentStreak: { type: currentStreakType, count: currentStreakCount },
    avgHoldWinners: winnerHolds.length > 0 ? winnerHolds.reduce((a, b) => a + b, 0) / winnerHolds.length : 0,
    avgHoldLosers: loserHolds.length > 0 ? loserHolds.reduce((a, b) => a + b, 0) / loserHolds.length : 0,
    cumulativePnL
  };
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function TradeBlotter({ candidateId }: TradeBlotterProps) {
  const { tradesResult, loadTrades } = useDataStore();
  const [sortKey, setSortKey] = useState<SortKey>('entry_date');
  const [sortDir, setSortDir] = useState<SortDirection>('desc');
  const [search, setSearch] = useState('');
  const [filterWinners, setFilterWinners] = useState<'all' | 'winners' | 'losers'>('all');
  const [filterDirection, setFilterDirection] = useState<'all' | 'Long' | 'Short'>('all');
  const [filterSymbol, setFilterSymbol] = useState<string>('all');
  const [limit, setLimit] = useState(100);
  const [selectedTrade, setSelectedTrade] = useState<TradeRecord | null>(null);
  const [showCharts, setShowCharts] = useState(true);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [loadingTrades, setLoadingTrades] = useState(false);

  useEffect(() => {
    setLoadingTrades(true);
    loadTrades(candidateId, limit).finally(() => setLoadingTrades(false));
  }, [candidateId, limit]);

  const trades = tradesResult?.trades || [];
  const summary = tradesResult?.summary;
  
  // Get unique symbols for filter
  const uniqueSymbols = useMemo(() => {
    return [...new Set(trades.map(t => t.symbol))].sort();
  }, [trades]);

  // Calculate analytics
  const analytics = useMemo(() => calculateAnalytics(trades), [trades]);

  // Filter and sort trades
  const filteredTrades = useMemo(() => {
    let result = [...trades];
    
    // Search filter
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(t => 
        t.symbol.toLowerCase().includes(q) || 
        t.trade_id.toLowerCase().includes(q)
      );
    }
    
    // Win/Loss filter
    if (filterWinners === 'winners') result = result.filter(t => t.is_winner);
    if (filterWinners === 'losers') result = result.filter(t => !t.is_winner);
    
    // Direction filter
    if (filterDirection !== 'all') result = result.filter(t => t.direction === filterDirection);
    
    // Symbol filter
    if (filterSymbol !== 'all') result = result.filter(t => t.symbol === filterSymbol);
    
    // Sort
    result.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal === bVal) return 0;
      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      return sortDir === 'asc' ? 1 : -1;
    });
    
    return result;
  }, [trades, search, filterWinners, filterDirection, filterSymbol, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const toggleRowExpand = (tradeId: string) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(tradeId)) {
        next.delete(tradeId);
      } else {
        next.add(tradeId);
      }
      return next;
    });
  };

  // Export CSV
  const exportCSV = () => {
    const headers = ['Trade ID', 'Entry Date', 'Exit Date', 'Symbol', 'Direction', 'Quantity', 
                     'Entry Price', 'Exit Price', 'Gross PnL', 'Commission', 'Slippage', 
                     'Net PnL', 'Return %', 'Holding Period', 'Winner'];
    const rows = filteredTrades.map(t => [
      t.trade_id, t.entry_date, t.exit_date, t.symbol, t.direction, t.quantity,
      t.entry_price, t.exit_price, t.gross_pnl, t.commission, t.slippage,
      t.net_pnl, t.return_pct, formatHoldingPeriod(t.holding_period_hours), t.is_winner ? 'Yes' : 'No'
    ]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades_${candidateId.slice(0, 8)}_${format(new Date(), 'yyyyMMdd')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export Excel
  const exportExcel = () => {
    const data = filteredTrades.map(t => ({
      'Trade ID': t.trade_id,
      'Entry Date': t.entry_date,
      'Exit Date': t.exit_date,
      'Symbol': t.symbol,
      'Direction': t.direction,
      'Quantity': t.quantity,
      'Entry Price': t.entry_price,
      'Exit Price': t.exit_price,
      'Gross PnL': t.gross_pnl,
      'Commission': t.commission,
      'Slippage': t.slippage,
      'Net PnL': t.net_pnl,
      'Return %': t.return_pct,
      'Holding Period': formatHoldingPeriod(t.holding_period_hours),
      'Winner': t.is_winner ? 'Yes' : 'No'
    }));
    
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Trades');
    
    // Add summary sheet
    if (summary) {
      const summaryData = [
        { Metric: 'Total Trades', Value: summary.total_trades },
        { Metric: 'Winners', Value: summary.winners },
        { Metric: 'Losers', Value: summary.losers },
        { Metric: 'Win Rate', Value: `${summary.win_rate.toFixed(1)}%` },
        { Metric: 'Total Net PnL', Value: formatCurrency(summary.total_net_pnl) },
        { Metric: 'Profit Factor', Value: summary.profit_factor.toFixed(2) },
        { Metric: 'Expectancy', Value: formatCurrency(summary.expectancy) },
        { Metric: 'Avg Win', Value: formatCurrency(summary.avg_win) },
        { Metric: 'Avg Loss', Value: formatCurrency(summary.avg_loss) },
      ];
      const summaryWs = XLSX.utils.json_to_sheet(summaryData);
      XLSX.utils.book_append_sheet(wb, summaryWs, 'Summary');
    }
    
    XLSX.writeFile(wb, `trades_${candidateId.slice(0, 8)}_${format(new Date(), 'yyyyMMdd')}.xlsx`);
  };

  // Quick sort presets
  const applySortPreset = (preset: 'best' | 'worst' | 'recent' | 'oldest') => {
    switch (preset) {
      case 'best':
        setSortKey('net_pnl');
        setSortDir('desc');
        break;
      case 'worst':
        setSortKey('net_pnl');
        setSortDir('asc');
        break;
      case 'recent':
        setSortKey('entry_date');
        setSortDir('desc');
        break;
      case 'oldest':
        setSortKey('entry_date');
        setSortDir('asc');
        break;
    }
  };

  if (loadingTrades) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
        <span className="ml-3 text-terminal-muted">Loading trades...</span>
      </div>
    );
  }

  // Calculate additional KPIs
  const payoffRatio = summary ? (summary.avg_loss !== 0 ? Math.abs(summary.avg_win / summary.avg_loss) : 0) : 0;

  return (
    <div className="space-y-6">
      {/* Summary Stats - Institutional Grade */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-3">
          <SummaryCard 
            icon={<BarChart3 className="w-4 h-4" />} 
            label="Total Trades" 
            value={formatNumber(summary.total_trades)} 
          />
          <SummaryCard 
            icon={<Target className="w-4 h-4" />} 
            label="Win Rate" 
            value={`${summary.win_rate.toFixed(1)}%`}
            color={summary.win_rate >= 50 ? 'profit' : 'loss'}
            subtext={`${summary.winners}W / ${summary.losers}L`}
          />
          <SummaryCard 
            icon={<DollarSign className="w-4 h-4" />} 
            label="Net PnL" 
            value={formatCurrency(summary.total_net_pnl, true)}
            color={summary.total_net_pnl >= 0 ? 'profit' : 'loss'}
          />
          <SummaryCard 
            icon={<Zap className="w-4 h-4" />} 
            label="Expectancy" 
            value={formatCurrency(summary.expectancy)}
            color={summary.expectancy >= 0 ? 'profit' : 'loss'}
            tooltip="Expected $ per trade"
          />
          <SummaryCard 
            icon={<Activity className="w-4 h-4" />} 
            label="Profit Factor" 
            value={summary.profit_factor.toFixed(2)}
            color={summary.profit_factor >= 1.5 ? 'profit' : summary.profit_factor >= 1 ? 'warning' : 'loss'}
          />
          <SummaryCard 
            icon={<Award className="w-4 h-4" />} 
            label="Payoff Ratio" 
            value={payoffRatio.toFixed(2)}
            color={payoffRatio >= 1.5 ? 'profit' : payoffRatio >= 1 ? 'warning' : 'loss'}
            tooltip="Avg Win / Avg Loss"
          />
          <SummaryCard 
            icon={<TrendingUp className="w-4 h-4" />} 
            label="Avg Win" 
            value={formatCurrency(summary.avg_win, true)}
            color="profit"
          />
          <SummaryCard 
            icon={<TrendingDown className="w-4 h-4" />} 
            label="Avg Loss" 
            value={formatCurrency(summary.avg_loss, true)}
            color="loss"
          />
        </div>
      )}

      {/* Extended Analytics Row */}
      {analytics && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <SummaryCard 
            icon={<ArrowUpRight className="w-4 h-4" />} 
            label="Largest Win" 
            value={analytics.largestWin ? formatCurrency(analytics.largestWin.net_pnl, true) : '-'}
            color="profit"
            subtext={analytics.largestWin?.symbol}
            onClick={() => analytics.largestWin && setSelectedTrade(analytics.largestWin)}
          />
          <SummaryCard 
            icon={<ArrowDownRight className="w-4 h-4" />} 
            label="Largest Loss" 
            value={analytics.largestLoss ? formatCurrency(analytics.largestLoss.net_pnl, true) : '-'}
            color="loss"
            subtext={analytics.largestLoss?.symbol}
            onClick={() => analytics.largestLoss && setSelectedTrade(analytics.largestLoss)}
          />
          <SummaryCard 
            icon={<Activity className="w-4 h-4" />} 
            label="Win Streak" 
            value={analytics.maxWinStreak.toString()}
            color="profit"
            subtext={`Current: ${analytics.currentStreak.type === 'win' ? analytics.currentStreak.count : 0}`}
          />
          <SummaryCard 
            icon={<AlertTriangle className="w-4 h-4" />} 
            label="Loss Streak" 
            value={analytics.maxLossStreak.toString()}
            color="loss"
            subtext={`Current: ${analytics.currentStreak.type === 'loss' ? analytics.currentStreak.count : 0}`}
          />
          <SummaryCard 
            icon={<Clock className="w-4 h-4" />} 
            label="Avg Hold Winners" 
            value={formatHoldingPeriod(analytics.avgHoldWinners)}
            color="profit"
          />
          <SummaryCard 
            icon={<Clock className="w-4 h-4" />} 
            label="Avg Hold Losers" 
            value={formatHoldingPeriod(analytics.avgHoldLosers)}
            color="loss"
          />
        </div>
      )}

      {/* Cost Breakdown */}
      {summary && (
        <div className="flex items-center gap-6 p-4 bg-terminal-surface/50 border border-terminal-border rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-xs text-terminal-muted uppercase">Total Costs:</span>
            <span className="font-mono font-bold text-loss">
              {formatCurrency(summary.total_commission + summary.total_slippage)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-terminal-muted">Commission:</span>
            <span className="font-mono text-sm">{formatCurrency(summary.total_commission)}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-terminal-muted">Slippage:</span>
            <span className="font-mono text-sm">{formatCurrency(summary.total_slippage)}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-terminal-muted">Cost/Trade:</span>
            <span className="font-mono text-sm">
              {summary.total_trades > 0 
                ? formatCurrency((summary.total_commission + summary.total_slippage) / summary.total_trades)
                : '-'
              }
            </span>
          </div>
        </div>
      )}

      {/* Charts Section */}
      {showCharts && trades.length > 0 && (
        <TradeCharts 
          trades={filteredTrades} 
          analytics={analytics}
          onTradeClick={setSelectedTrade}
        />
      )}

      {/* Filters & Actions */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-terminal-muted" />
          <input
            type="text"
            placeholder="Search symbol or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm focus:outline-none focus:border-profit"
          />
        </div>

        {/* Symbol Filter */}
        <select
          value={filterSymbol}
          onChange={(e) => setFilterSymbol(e.target.value)}
          className="px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm focus:outline-none focus:border-profit"
        >
          <option value="all">All Symbols</option>
          {uniqueSymbols.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        {/* Direction Filter */}
        <div className="flex rounded-lg overflow-hidden border border-terminal-border">
          {(['all', 'Long', 'Short'] as const).map((d) => (
            <button
              key={d}
              onClick={() => setFilterDirection(d)}
              className={`px-3 py-2 text-xs font-medium transition-colors ${
                filterDirection === d 
                  ? d === 'Long' ? 'bg-profit/20 text-profit' 
                    : d === 'Short' ? 'bg-loss/20 text-loss' 
                    : 'bg-terminal-muted/20 text-white'
                  : 'bg-terminal-surface text-terminal-muted hover:text-white'
              }`}
            >
              {d === 'all' ? 'All' : d}
            </button>
          ))}
        </div>

        {/* Win/Loss Filter */}
        <div className="flex rounded-lg overflow-hidden border border-terminal-border">
          {(['all', 'winners', 'losers'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilterWinners(f)}
              className={`px-3 py-2 text-xs font-medium transition-colors ${
                filterWinners === f 
                  ? f === 'winners' ? 'bg-profit/20 text-profit' 
                    : f === 'losers' ? 'bg-loss/20 text-loss' 
                    : 'bg-terminal-muted/20 text-white'
                  : 'bg-terminal-surface text-terminal-muted hover:text-white'
              }`}
            >
              {f === 'all' ? 'All' : f === 'winners' ? 'Winners' : 'Losers'}
            </button>
          ))}
        </div>

        {/* Quick Sort Presets */}
        <div className="flex items-center gap-1 border-l border-terminal-border pl-3">
          <span className="text-xs text-terminal-muted mr-1">Sort:</span>
          <button onClick={() => applySortPreset('best')} className="px-2 py-1 text-xs hover:bg-profit/20 rounded transition-colors">Best</button>
          <button onClick={() => applySortPreset('worst')} className="px-2 py-1 text-xs hover:bg-loss/20 rounded transition-colors">Worst</button>
          <button onClick={() => applySortPreset('recent')} className="px-2 py-1 text-xs hover:bg-terminal-muted/20 rounded transition-colors">Recent</button>
        </div>

        {/* Limit Selector */}
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm focus:outline-none focus:border-profit"
        >
          <option value={50}>50</option>
          <option value={100}>100</option>
          <option value={250}>250</option>
          <option value={500}>500</option>
        </select>

        {/* Toggle Charts */}
        <button
          onClick={() => setShowCharts(!showCharts)}
          className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm transition-colors ${
            showCharts ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-terminal-surface border border-terminal-border'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          Charts
        </button>

        {/* Export Buttons */}
        <div className="flex items-center gap-2 ml-auto">
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm hover:border-profit transition-colors"
          >
            <Download className="w-4 h-4" />
            CSV
          </button>
          <button
            onClick={exportExcel}
            className="flex items-center gap-2 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm hover:border-profit transition-colors"
          >
            <FileSpreadsheet className="w-4 h-4" />
            Excel
          </button>
        </div>
      </div>

      {/* Data Source Badge */}
      {tradesResult?.data_source === 'simulated' && (
        <div className="flex items-center gap-2 text-xs text-accent-yellow bg-accent-yellow/10 px-3 py-1.5 rounded-lg w-fit">
          <Filter className="w-3 h-3" />
          Simulated trades based on strategy metrics
        </div>
      )}

      {/* Trades Table with Virtual Scrolling */}
      <div className="border border-terminal-border rounded-lg overflow-hidden">
        <div className="overflow-auto" style={{ maxHeight: '600px' }}>
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-terminal-surface z-10 border-b border-terminal-border">
              <tr>
                <th className="w-8 px-2 py-3"></th>
                <SortableHeader label="ID" sortKey="trade_id" current={sortKey} dir={sortDir} onSort={handleSort} width="w-20" />
                <SortableHeader label="Entry" sortKey="entry_date" current={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableHeader label="Exit" sortKey="exit_date" current={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableHeader label="Symbol" sortKey="symbol" current={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableHeader label="Dir" sortKey="direction" current={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                <SortableHeader label="Qty" sortKey="quantity" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Entry $" sortKey="entry_price" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Exit $" sortKey="exit_price" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Gross" sortKey="gross_pnl" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Costs" sortKey="commission" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Net PnL" sortKey="net_pnl" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Return" sortKey="return_pct" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortableHeader label="Hold" sortKey="holding_period_hours" current={sortKey} dir={sortDir} onSort={handleSort} align="right" />
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map((trade) => (
                <>
                  <tr 
                    key={trade.trade_id} 
                    className={`border-b border-terminal-border/50 hover:bg-terminal-surface/80 cursor-pointer transition-colors ${
                      trade.is_winner ? 'bg-profit/5' : 'bg-loss/5'
                    }`}
                    onClick={() => setSelectedTrade(trade)}
                  >
                    <td className="px-2 py-2">
                      <button 
                        onClick={(e) => { e.stopPropagation(); toggleRowExpand(trade.trade_id); }}
                        className="p-1 hover:bg-terminal-muted/20 rounded transition-colors"
                      >
                        <ChevronRight className={`w-3 h-3 transition-transform ${expandedRows.has(trade.trade_id) ? 'rotate-90' : ''}`} />
                      </button>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-terminal-muted" title={trade.trade_id}>
                      {trade.trade_id.slice(-8)}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{formatDateBR(trade.entry_date)}</td>
                    <td className="px-3 py-2 font-mono text-xs">{formatDateBR(trade.exit_date)}</td>
                    <td className="px-3 py-2 font-medium">{trade.symbol}</td>
                    <td className="px-3 py-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        trade.direction === 'Long' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                      }`}>
                        {trade.direction}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono">{formatNumber(trade.quantity)}</td>
                    <td className="px-3 py-2 text-right font-mono">{trade.entry_price.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right font-mono">{trade.exit_price.toFixed(2)}</td>
                    <td className={`px-3 py-2 text-right font-mono ${trade.gross_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {formatCurrency(trade.gross_pnl)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-terminal-muted text-xs">
                      {formatCurrency(trade.commission + trade.slippage)}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono font-medium ${trade.net_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {formatCurrency(trade.net_pnl)}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${trade.return_pct >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {formatPct(trade.return_pct)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-terminal-muted">
                      {formatHoldingPeriod(trade.holding_period_hours)}
                    </td>
                  </tr>
                  {/* Expanded Row */}
                  {expandedRows.has(trade.trade_id) && (
                    <tr className="bg-terminal-bg/50">
                      <td colSpan={14} className="px-6 py-4">
                        <div className="grid grid-cols-4 gap-6 text-xs">
                          <div>
                            <span className="text-terminal-muted">Commission:</span>
                            <span className="ml-2 font-mono">{formatCurrency(trade.commission)}</span>
                            <span className="text-terminal-muted ml-1">
                              ({((trade.commission / (trade.quantity * trade.entry_price)) * 10000).toFixed(1)} bps)
                            </span>
                          </div>
                          <div>
                            <span className="text-terminal-muted">Slippage:</span>
                            <span className="ml-2 font-mono">{formatCurrency(trade.slippage)}</span>
                            <span className="text-terminal-muted ml-1">
                              ({((trade.slippage / (trade.quantity * trade.entry_price)) * 10000).toFixed(1)} bps)
                            </span>
                          </div>
                          <div>
                            <span className="text-terminal-muted">Notional:</span>
                            <span className="ml-2 font-mono">{formatCurrency(trade.quantity * trade.entry_price, true)}</span>
                          </div>
                          <div>
                            <span className="text-terminal-muted">Price Move:</span>
                            <span className={`ml-2 font-mono ${trade.exit_price > trade.entry_price ? 'text-profit' : 'text-loss'}`}>
                              {((trade.exit_price - trade.entry_price) / trade.entry_price * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex items-center justify-between text-xs text-terminal-muted">
        <span>Showing {filteredTrades.length} of {trades.length} trades</span>
        {summary && (
          <span className={`font-mono ${summary.total_net_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
            Total Net PnL: {formatCurrency(summary.total_net_pnl)}
          </span>
        )}
      </div>

      {/* Trade Detail Modal */}
      {selectedTrade && (
        <TradeDetailModal 
          trade={selectedTrade} 
          allTrades={trades}
          onClose={() => setSelectedTrade(null)} 
        />
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function SummaryCard({ 
  icon, 
  label, 
  value, 
  color, 
  subtext,
  tooltip,
  onClick
}: { 
  icon: React.ReactNode; 
  label: string; 
  value: string; 
  color?: 'profit' | 'loss' | 'warning';
  subtext?: string;
  tooltip?: string;
  onClick?: () => void;
}) {
  const colorClass = color === 'profit' ? 'text-profit' : color === 'loss' ? 'text-loss' : color === 'warning' ? 'text-accent-yellow' : 'text-white';
  return (
    <div 
      className={`p-3 bg-terminal-surface border border-terminal-border rounded-lg ${onClick ? 'cursor-pointer hover:border-profit transition-colors' : ''}`}
      onClick={onClick}
      title={tooltip}
    >
      <div className="flex items-center gap-2 text-terminal-muted mb-1">
        {icon}
        <span className="text-[10px] uppercase tracking-wide">{label}</span>
      </div>
      <div className={`font-mono font-bold text-lg ${colorClass}`}>{value}</div>
      {subtext && <div className="text-[10px] text-terminal-muted mt-0.5">{subtext}</div>}
    </div>
  );
}

function SortableHeader({ 
  label, 
  sortKey, 
  current, 
  dir, 
  onSort, 
  align = 'left',
  width
}: { 
  label: string; 
  sortKey: SortKey; 
  current: SortKey; 
  dir: SortDirection; 
  onSort: (k: SortKey) => void;
  align?: 'left' | 'center' | 'right';
  width?: string;
}) {
  const isActive = current === sortKey;
  const alignClass = align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start';
  
  return (
    <th 
      className={`px-3 py-3 text-xs font-medium text-terminal-muted uppercase tracking-wider cursor-pointer hover:text-white select-none ${
        align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : 'text-left'
      } ${width || ''}`}
      onClick={() => onSort(sortKey)}
    >
      <div className={`flex items-center gap-1 ${alignClass}`}>
        <span>{label}</span>
        {isActive ? (
          dir === 'asc' ? <ChevronUp className="w-3 h-3 text-profit" /> : <ChevronDown className="w-3 h-3 text-profit" />
        ) : (
          <ChevronsUpDown className="w-3 h-3 opacity-30" />
        )}
      </div>
    </th>
  );
}

export { formatHoldingPeriod, formatCurrency, formatDateBR, formatPct };
export type { TradeAnalytics };
