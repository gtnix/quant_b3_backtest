import { useState } from 'react';
import {
  X,
  Download,
  FileJson,
  FileText,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';

interface TearsheetData {
  candidate_id: string;
  display_name: string;
  generated_at: string;
  sharpe: number;
  cagr: number;
  max_drawdown: number;
  volatility: number;
  sortino: number;
  calmar: number;
  pbo: number;
  var_95: number;
  cvar_95: number;
  omega_ratio: number;
  tail_ratio: number;
  skewness: number;
  kurtosis: number;
  total_trades?: number;
  hit_rate?: number;
  profit_factor?: number;
  payoff_ratio: number;
  start_date: string;
  end_date: string;
  trading_days: number;
  strategy_blocks: string[];
  execution_config?: string;
  git_sha?: string;
  dataset_hash?: string;
  config_hash?: string;
}

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  candidateId: string;
  candidateName: string;
}

export function ExportModal({ isOpen, onClose, candidateId, candidateName }: ExportModalProps) {
  const [tearsheet, setTearsheet] = useState<TearsheetData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);

  const generateTearsheet = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await invoke<TearsheetData>('generate_tearsheet', { candidateId });
      setTearsheet(data);
    } catch (e) {
      setError(String(e));
    }
    setIsLoading(false);
  };

  const exportToJSON = async () => {
    if (!tearsheet) return;
    
    const outputPath = `${candidateId}_tearsheet.json`;
    try {
      await invoke('export_tearsheet_json', { candidateId, outputPath });
      setExportSuccess(true);
      setTimeout(() => setExportSuccess(false), 3000);
    } catch (e) {
      setError(String(e));
    }
  };

  const copyToClipboard = () => {
    if (!tearsheet) return;
    navigator.clipboard.writeText(JSON.stringify(tearsheet, null, 2));
    setExportSuccess(true);
    setTimeout(() => setExportSuccess(false), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      
      {/* Modal */}
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-terminal-surface border border-terminal-border rounded-lg overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-terminal-border">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-profit" />
            <div>
              <h2 className="font-semibold">Export Tearsheet</h2>
              <p className="text-xs text-terminal-muted font-mono">{candidateName}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-terminal-bg rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-terminal-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {!tearsheet ? (
            <div className="flex flex-col items-center justify-center py-12">
              {isLoading ? (
                <>
                  <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted mb-4" />
                  <p className="text-terminal-muted">Generating tearsheet...</p>
                </>
              ) : error ? (
                <>
                  <AlertTriangle className="w-12 h-12 text-accent-yellow mb-4" />
                  <p className="text-loss mb-4">{error}</p>
                  <button
                    onClick={generateTearsheet}
                    className="px-4 py-2 bg-terminal-bg border border-terminal-border rounded-lg hover:border-profit transition-colors"
                  >
                    Retry
                  </button>
                </>
              ) : (
                <>
                  <FileText className="w-16 h-16 text-terminal-muted mb-4" />
                  <p className="text-terminal-muted mb-4">
                    Generate a comprehensive tearsheet with all metrics and provenance data.
                  </p>
                  <button
                    onClick={generateTearsheet}
                    className="flex items-center gap-2 px-6 py-3 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors"
                  >
                    <FileText className="w-4 h-4" />
                    Generate Tearsheet
                  </button>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {/* Success Message */}
              {exportSuccess && (
                <div className="p-3 bg-profit/20 border border-profit/30 rounded-lg flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-profit" />
                  <span className="text-profit">Exported successfully!</span>
                </div>
              )}

              {/* Strategy Info */}
              <div className="card">
                <h3 className="font-semibold mb-3">Strategy</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-terminal-muted">Name:</span>
                    <span className="ml-2">{tearsheet.display_name}</span>
                  </div>
                  <div>
                    <span className="text-terminal-muted">Period:</span>
                    <span className="ml-2 font-mono">{tearsheet.start_date} to {tearsheet.end_date}</span>
                  </div>
                  <div>
                    <span className="text-terminal-muted">Trading Days:</span>
                    <span className="ml-2 font-mono">{tearsheet.trading_days}</span>
                  </div>
                  <div>
                    <span className="text-terminal-muted">Generated:</span>
                    <span className="ml-2 font-mono">{new Date(tearsheet.generated_at).toLocaleString()}</span>
                  </div>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="card">
                <h3 className="font-semibold mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                  <MetricDisplay label="Sharpe" value={tearsheet.sharpe.toFixed(2)} />
                  <MetricDisplay label="CAGR" value={`${(tearsheet.cagr * 100).toFixed(1)}%`} />
                  <MetricDisplay label="Max DD" value={`${(tearsheet.max_drawdown * 100).toFixed(1)}%`} isNegative />
                  <MetricDisplay label="Volatility" value={`${(tearsheet.volatility * 100).toFixed(1)}%`} />
                  <MetricDisplay label="Sortino" value={tearsheet.sortino.toFixed(2)} />
                  <MetricDisplay label="Calmar" value={tearsheet.calmar.toFixed(2)} />
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="card">
                <h3 className="font-semibold mb-3">Risk Metrics</h3>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                  <MetricDisplay label="PBO" value={`${(tearsheet.pbo * 100).toFixed(1)}%`} />
                  <MetricDisplay label="VaR 95%" value={`${(tearsheet.var_95 * 100).toFixed(2)}%`} isNegative />
                  <MetricDisplay label="CVaR 95%" value={`${(tearsheet.cvar_95 * 100).toFixed(2)}%`} isNegative />
                  <MetricDisplay label="Omega" value={tearsheet.omega_ratio.toFixed(2)} />
                  <MetricDisplay label="Skewness" value={tearsheet.skewness.toFixed(2)} />
                  <MetricDisplay label="Kurtosis" value={tearsheet.kurtosis.toFixed(2)} />
                </div>
              </div>

              {/* Trade Stats */}
              {tearsheet.total_trades && (
                <div className="card">
                  <h3 className="font-semibold mb-3">Trade Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricDisplay label="Total Trades" value={tearsheet.total_trades.toString()} />
                    <MetricDisplay label="Hit Rate" value={`${((tearsheet.hit_rate || 0) * 100).toFixed(0)}%`} />
                    <MetricDisplay label="Profit Factor" value={(tearsheet.profit_factor || 0).toFixed(2)} />
                    <MetricDisplay label="Payoff Ratio" value={tearsheet.payoff_ratio.toFixed(2)} />
                  </div>
                </div>
              )}

              {/* Provenance */}
              <div className="card">
                <h3 className="font-semibold mb-3">Provenance</h3>
                <div className="space-y-2 text-sm font-mono">
                  {tearsheet.git_sha && (
                    <div className="flex">
                      <span className="text-terminal-muted w-28">Git SHA:</span>
                      <span>{tearsheet.git_sha}</span>
                    </div>
                  )}
                  {tearsheet.dataset_hash && (
                    <div className="flex">
                      <span className="text-terminal-muted w-28">Dataset Hash:</span>
                      <span>{tearsheet.dataset_hash}</span>
                    </div>
                  )}
                  {tearsheet.config_hash && (
                    <div className="flex">
                      <span className="text-terminal-muted w-28">Config Hash:</span>
                      <span>{tearsheet.config_hash}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Strategy Blocks */}
              {tearsheet.strategy_blocks.length > 0 && (
                <div className="card">
                  <h3 className="font-semibold mb-3">Strategy Pipeline</h3>
                  <div className="flex flex-wrap gap-2">
                    {tearsheet.strategy_blocks.map((block, i) => (
                      <span
                        key={i}
                        className="px-3 py-1 bg-terminal-bg border border-terminal-border rounded text-sm font-mono"
                      >
                        {block}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        {tearsheet && (
          <div className="flex items-center justify-end gap-3 p-4 border-t border-terminal-border">
            <button
              onClick={copyToClipboard}
              className="flex items-center gap-2 px-4 py-2 bg-terminal-bg border border-terminal-border rounded-lg hover:border-profit transition-colors"
            >
              <FileJson className="w-4 h-4" />
              Copy JSON
            </button>
            <button
              onClick={exportToJSON}
              className="flex items-center gap-2 px-4 py-2 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors"
            >
              <Download className="w-4 h-4" />
              Export JSON
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricDisplay({ label, value, isNegative = false }: { label: string; value: string; isNegative?: boolean }) {
  return (
    <div>
      <div className="text-xs text-terminal-muted">{label}</div>
      <div className={`font-mono text-lg ${isNegative ? 'text-loss' : ''}`}>{value}</div>
    </div>
  );
}

