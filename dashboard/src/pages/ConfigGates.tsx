/**
 * ConfigGates - Configure validation gates and promotion thresholds
 */

import { useState } from 'react';
import { Shield, Save, RefreshCw, CheckCircle2, XCircle, AlertTriangle, Trophy } from 'lucide-react';

interface GateConfig {
  enabled: boolean;
  threshold: number;
  operator: 'gte' | 'lte';
  label: string;
  description: string;
  unit: string;
}

interface GatesConfig {
  minOosSharpeNet: GateConfig;
  maxPbo: GateConfig;
  minDsr: GateConfig;
  maxDegradation: GateConfig;
  minTradesOos: GateConfig;
  maxDrawdownNet: GateConfig;
  minStressPassed: GateConfig;
}

interface PromotionConfig {
  enabled: boolean;
  minOosSharpeNet: number;
  maxPbo: number;
  minDsr: number;
  maxDrawdownNet: number;
  requireAllStressPassed: boolean;
  requireGatesPassed: boolean;
  autoCopyArtifacts: boolean;
}

const DEFAULT_GATES: GatesConfig = {
  minOosSharpeNet: {
    enabled: true,
    threshold: 0.5,
    operator: 'gte',
    label: 'OOS Sharpe NET',
    description: 'Minimum out-of-sample Sharpe ratio after costs',
    unit: '',
  },
  maxPbo: {
    enabled: true,
    threshold: 0.15,
    operator: 'lte',
    label: 'PBO',
    description: 'Maximum Probability of Backtest Overfitting',
    unit: '%',
  },
  minDsr: {
    enabled: true,
    threshold: 0.8,
    operator: 'gte',
    label: 'DSR',
    description: 'Minimum Deflated Sharpe Ratio',
    unit: '',
  },
  maxDegradation: {
    enabled: true,
    threshold: 0.40,
    operator: 'lte',
    label: 'IS→OOS Degradation',
    description: 'Maximum performance drop from in-sample to out-of-sample',
    unit: '%',
  },
  minTradesOos: {
    enabled: true,
    threshold: 30,
    operator: 'gte',
    label: 'OOS Trades',
    description: 'Minimum number of trades in out-of-sample period',
    unit: '',
  },
  maxDrawdownNet: {
    enabled: true,
    threshold: 0.25,
    operator: 'lte',
    label: 'Max Drawdown',
    description: 'Maximum drawdown after costs',
    unit: '%',
  },
  minStressPassed: {
    enabled: true,
    threshold: 4,
    operator: 'gte',
    label: 'Stress Tests',
    description: 'Minimum stress scenarios passed (out of 8)',
    unit: '/8',
  },
};

const DEFAULT_PROMOTION: PromotionConfig = {
  enabled: true,
  minOosSharpeNet: 1.0,
  maxPbo: 0.10,
  minDsr: 0.8,
  maxDrawdownNet: 0.20,
  requireAllStressPassed: true,
  requireGatesPassed: true,
  autoCopyArtifacts: true,
};

export function ConfigGates() {
  const [gates, setGates] = useState<GatesConfig>(DEFAULT_GATES);
  const [promotion, setPromotion] = useState<PromotionConfig>(DEFAULT_PROMOTION);
  const [saving, setSaving] = useState(false);
  
  const toggleGate = (key: keyof GatesConfig) => {
    setGates(prev => ({
      ...prev,
      [key]: { ...prev[key], enabled: !prev[key].enabled },
    }));
  };
  
  const updateGateThreshold = (key: keyof GatesConfig, value: number) => {
    setGates(prev => ({
      ...prev,
      [key]: { ...prev[key], threshold: value },
    }));
  };
  
  const handleSave = async () => {
    setSaving(true);
    console.log('Saving gates config:', { gates, promotion });
    await new Promise(r => setTimeout(r, 500));
    setSaving(false);
  };
  
  const formatValue = (gate: GateConfig) => {
    if (gate.unit === '%') return `${(gate.threshold * 100).toFixed(0)}%`;
    return `${gate.threshold}${gate.unit}`;
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Shield className="w-6 h-6 text-rose-400" />
              Validation Gates
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure quality gates and Hall of Fame promotion criteria
            </p>
          </div>
          
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Changes
          </button>
        </div>
        
        {/* Validation Gates */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Quality Gates
          </h3>
          <p className="text-sm text-slate-400 mb-4">
            Candidates must pass all enabled gates to be considered valid
          </p>
          
          <div className="space-y-3">
            {Object.entries(gates).map(([key, gate]) => (
              <div 
                key={key}
                className={`flex items-center justify-between p-4 rounded-lg border transition-colors ${
                  gate.enabled 
                    ? 'bg-slate-800/50 border-slate-700' 
                    : 'bg-slate-900/30 border-slate-800 opacity-60'
                }`}
              >
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => toggleGate(key as keyof GatesConfig)}
                    className={`w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
                      gate.enabled 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-slate-700 text-slate-500'
                    }`}
                  >
                    {gate.enabled ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                  </button>
                  <div>
                    <p className="font-medium text-white">{gate.label}</p>
                    <p className="text-xs text-slate-500">{gate.description}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-400">
                    {gate.operator === 'gte' ? '≥' : '≤'}
                  </span>
                  <input
                    type="number"
                    step={gate.unit === '%' ? 0.01 : 1}
                    value={gate.unit === '%' ? (gate.threshold * 100) : gate.threshold}
                    onChange={e => {
                      const val = parseFloat(e.target.value);
                      updateGateThreshold(key as keyof GatesConfig, gate.unit === '%' ? val / 100 : val);
                    }}
                    disabled={!gate.enabled}
                    className="w-20 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-center focus:outline-none focus:border-rose-500 disabled:opacity-50"
                  />
                  <span className="text-sm text-slate-400 w-8">{gate.unit}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Hall of Fame Promotion */}
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            Hall of Fame Promotion
          </h3>
          <p className="text-sm text-slate-400 mb-4">
            Stricter criteria for automatic promotion to the Hall of Fame
          </p>
          
          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={() => setPromotion(prev => ({ ...prev, enabled: !prev.enabled }))}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                promotion.enabled 
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50' 
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}
            >
              {promotion.enabled ? 'Auto-Promotion Enabled' : 'Auto-Promotion Disabled'}
            </button>
          </div>
          
          {promotion.enabled && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Min Sharpe NET</label>
                <input
                  type="number"
                  step="0.1"
                  value={promotion.minOosSharpeNet}
                  onChange={e => setPromotion(prev => ({ ...prev, minOosSharpeNet: parseFloat(e.target.value) || 0 }))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Max PBO (%)</label>
                <input
                  type="number"
                  step="1"
                  value={(promotion.maxPbo * 100).toFixed(0)}
                  onChange={e => setPromotion(prev => ({ ...prev, maxPbo: parseFloat(e.target.value) / 100 || 0 }))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Min DSR</label>
                <input
                  type="number"
                  step="0.1"
                  value={promotion.minDsr}
                  onChange={e => setPromotion(prev => ({ ...prev, minDsr: parseFloat(e.target.value) || 0 }))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Max Drawdown (%)</label>
                <input
                  type="number"
                  step="1"
                  value={(promotion.maxDrawdownNet * 100).toFixed(0)}
                  onChange={e => setPromotion(prev => ({ ...prev, maxDrawdownNet: parseFloat(e.target.value) / 100 || 0 }))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>
          )}
          
          {promotion.enabled && (
            <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-slate-700">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={promotion.requireAllStressPassed}
                  onChange={e => setPromotion(prev => ({ ...prev, requireAllStressPassed: e.target.checked }))}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500"
                />
                <span className="text-sm text-slate-300">Require all stress tests passed</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={promotion.requireGatesPassed}
                  onChange={e => setPromotion(prev => ({ ...prev, requireGatesPassed: e.target.checked }))}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500"
                />
                <span className="text-sm text-slate-300">Require all gates passed</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={promotion.autoCopyArtifacts}
                  onChange={e => setPromotion(prev => ({ ...prev, autoCopyArtifacts: e.target.checked }))}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500"
                />
                <span className="text-sm text-slate-300">Auto-copy artifacts on promotion</span>
              </label>
            </div>
          )}
        </div>
        
        {/* Summary */}
        <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 text-sm text-slate-400">
          <p>
            <strong className="text-white">Active Gates:</strong>{' '}
            {Object.values(gates).filter(g => g.enabled).length} of {Object.keys(gates).length}
            {' • '}
            <strong className="text-white">Promotion:</strong>{' '}
            {promotion.enabled ? 'Auto-enabled' : 'Disabled'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConfigGates;






