import { useState, useEffect } from 'react';
import {
  Bell,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingDown,
  Shield,
  Settings,
  X,
  RefreshCw,
} from 'lucide-react';

interface Alert {
  id: string;
  alert_type: string;
  message: string;
  severity: string;
  timestamp: string;
  candidate_id?: string;
}

interface AlertThresholds {
  max_pbo?: number;
  max_drawdown?: number;
  min_sharpe_for_alert?: number;
}

interface AlertsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AlertsPanel({ isOpen, onClose }: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [thresholds, setThresholds] = useState<AlertThresholds>({
    max_pbo: 0.15,
    max_drawdown: 0.25,
    min_sharpe_for_alert: 1.0,
  });

  const checkAlerts = async () => {
    setIsLoading(true);
    try {
      const result = await invoke<Alert[]>('check_alerts', { thresholds });
      setAlerts(result);
    } catch (error) {
      console.error('Failed to check alerts:', error);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    if (isOpen) {
      checkAlerts();
    }
  }, [isOpen]);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <XCircle className="w-4 h-4 text-loss" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-accent-yellow" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-profit" />;
      default:
        return <Bell className="w-4 h-4 text-terminal-muted" />;
    }
  };

  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'pbo_exceeded':
        return <Shield className="w-4 h-4" />;
      case 'drawdown_exceeded':
        return <TrendingDown className="w-4 h-4" />;
      case 'data_integrity':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Bell className="w-4 h-4" />;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      
      {/* Panel */}
      <div className="relative w-full max-w-md h-full bg-terminal-surface border-l border-terminal-border overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-terminal-border">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-profit" />
            <h2 className="font-semibold">Alerts</h2>
            {alerts.length > 0 && (
              <span className="px-2 py-0.5 bg-loss/20 text-loss rounded text-xs font-mono">
                {alerts.length}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-terminal-bg rounded-lg transition-colors"
            >
              <Settings className="w-4 h-4 text-terminal-muted" />
            </button>
            <button
              onClick={checkAlerts}
              className="p-2 hover:bg-terminal-bg rounded-lg transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''} text-terminal-muted`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-terminal-bg rounded-lg transition-colors"
            >
              <X className="w-4 h-4 text-terminal-muted" />
            </button>
          </div>
        </div>

        {/* Settings */}
        {showSettings && (
          <div className="p-4 border-b border-terminal-border bg-terminal-bg/50">
            <h3 className="text-sm font-medium mb-3">Alert Thresholds</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-terminal-muted">Max PBO</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0.05"
                    max="0.30"
                    step="0.01"
                    value={thresholds.max_pbo || 0.15}
                    onChange={(e) => setThresholds({ ...thresholds, max_pbo: Number(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="font-mono text-sm w-12">{((thresholds.max_pbo || 0.15) * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div>
                <label className="text-xs text-terminal-muted">Max Drawdown</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0.10"
                    max="0.50"
                    step="0.05"
                    value={thresholds.max_drawdown || 0.25}
                    onChange={(e) => setThresholds({ ...thresholds, max_drawdown: Number(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="font-mono text-sm w-12">{((thresholds.max_drawdown || 0.25) * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div>
                <label className="text-xs text-terminal-muted">Min Sharpe for Alert</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.1"
                    value={thresholds.min_sharpe_for_alert || 1.0}
                    onChange={(e) => setThresholds({ ...thresholds, min_sharpe_for_alert: Number(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="font-mono text-sm w-12">{(thresholds.min_sharpe_for_alert || 1.0).toFixed(1)}</span>
                </div>
              </div>
              <button
                onClick={checkAlerts}
                className="w-full mt-2 px-3 py-2 bg-profit text-black font-medium rounded-lg text-sm hover:bg-profit/90 transition-colors"
              >
                Apply & Refresh
              </button>
            </div>
          </div>
        )}

        {/* Alerts List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {isLoading && alerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-terminal-muted">
              <RefreshCw className="w-8 h-8 animate-spin mb-2" />
              <p>Checking alerts...</p>
            </div>
          ) : alerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-terminal-muted">
              <CheckCircle className="w-12 h-12 mb-4 text-profit opacity-50" />
              <p className="font-medium">No Active Alerts</p>
              <p className="text-sm mt-1">All candidates within thresholds</p>
            </div>
          ) : (
            alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border ${
                  alert.severity === 'error'
                    ? 'bg-loss/10 border-loss/30'
                    : alert.severity === 'warning'
                    ? 'bg-accent-yellow/10 border-accent-yellow/30'
                    : 'bg-profit/10 border-profit/30'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getSeverityIcon(alert.severity)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {getAlertTypeIcon(alert.alert_type)}
                      <span className="text-xs text-terminal-muted capitalize">
                        {alert.alert_type.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <p className="text-sm">{alert.message}</p>
                    {alert.candidate_id && (
                      <p className="text-xs font-mono text-terminal-muted mt-1">
                        {alert.candidate_id.substring(0, 20)}...
                      </p>
                    )}
                    <p className="text-xs text-terminal-muted mt-1">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

