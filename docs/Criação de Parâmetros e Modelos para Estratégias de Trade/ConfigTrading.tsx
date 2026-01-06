/**
 * ConfigTrading - Configure trading parameters (fees, slippage, sizing)
 */

import { useState } from 'react';
import { DollarSign, Save, RefreshCw, Info } from 'lucide-react';
import { QuickTooltip } from '../components/ui/TooltipInfo';

interface TradingParams {
  market: 'br' | 'us';
  feeTier: string;
  feeRate: number;
  slippageBps: number;
  delayMs: number;
  positionSizing: 'equal' | 'volatility' | 'risk_parity';
  maxPositionPct: number;
  maxDrawdownPct: number;
  lotSize: number;
}

/**
 * B3 Fee Structure (2024/2025):
 * - Emolumentos: 0.003186% (negociação) + 0.0025% (registro)
 * - Taxa de Liquidação: 0.0275% para swing trade, 0.02% para day trade
 * - ISS: 5% sobre corretagem
 * - Total típico retail: ~0.030% por operação (ida e volta = 0.06%)
 * 
 * Corretoras Zero: XP, Clear, Rico, Inter (sem corretagem, apenas taxas B3)
 * Corretoras Tradicionais: ~R$15-20 por ordem
 */
const FEE_TIERS: Record<string, { label: string; rate: number; market: string; details: string }> = {
  'b3-retail': { 
    label: 'B3 Retail (Zero)', 
    rate: 0.0003, 
    market: 'br',
    details: 'Corretoras zero: emolumentos 0.0032% + liquidação 0.0275% ≈ 0.030%'
  },
  'b3-day-trade': { 
    label: 'B3 Day Trade', 
    rate: 0.00022, 
    market: 'br',
    details: 'Emolumentos 0.0032% + liquidação reduzida 0.020% ≈ 0.022%'
  },
  'b3-institutional': { 
    label: 'B3 Institutional', 
    rate: 0.0002, 
    market: 'br',
    details: 'Volume alto: emolumentos reduzidos + liquidação ≈ 0.020%'
  },
  'us-retail': { 
    label: 'US Retail (Zero)', 
    rate: 0.0, 
    market: 'us',
    details: 'Robinhood, Webull: zero commission, PFOF model'
  },
  'us-ibkr-lite': { 
    label: 'IBKR Lite', 
    rate: 0.0, 
    market: 'us',
    details: 'Interactive Brokers Lite: zero commission on US stocks'
  },
  'us-ibkr-pro': { 
    label: 'IBKR Pro', 
    rate: 0.0005, 
    market: 'us',
    details: '$0.005/share, min $1. Best execution, no PFOF.'
  },
};

const POSITION_SIZING: Record<string, { label: string; description: string }> = {
  equal: { label: 'Equal Weight', description: 'Distribute capital equally across positions' },
  volatility: { label: 'Inverse Volatility', description: 'Weight inversely by volatility (lower vol = higher weight)' },
  risk_parity: { label: 'Risk Parity', description: 'Equal risk contribution per position' },
};

export function ConfigTrading() {
  const [params, setParams] = useState<Record<string, TradingParams>>({
    br: {
      market: 'br',
      feeTier: 'b3-retail',
      feeRate: 0.0003,
      slippageBps: 10,
      delayMs: 0,
      positionSizing: 'equal',
      maxPositionPct: 20,
      maxDrawdownPct: 25,
      lotSize: 100,
    },
    us: {
      market: 'us',
      feeTier: 'us-retail',
      feeRate: 0.0,
      slippageBps: 5,
      delayMs: 0,
      positionSizing: 'equal',
      maxPositionPct: 10,
      maxDrawdownPct: 20,
      lotSize: 1,
    },
  });
  
  const [activeMarket, setActiveMarket] = useState<'br' | 'us'>('br');
  const [saving, setSaving] = useState(false);
  
  const current = params[activeMarket];
  
  const updateParam = <K extends keyof TradingParams>(key: K, value: TradingParams[K]) => {
    setParams(prev => ({
      ...prev,
      [activeMarket]: { ...prev[activeMarket], [key]: value },
    }));
  };
  
  const selectFeeTier = (tier: string) => {
    const tierInfo = FEE_TIERS[tier];
    if (tierInfo) {
      updateParam('feeTier', tier);
      updateParam('feeRate', tierInfo.rate);
    }
  };
  
  const handleSave = async () => {
    setSaving(true);
    console.log('Saving trading params:', params);
    await new Promise(r => setTimeout(r, 500));
    setSaving(false);
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <DollarSign className="w-6 h-6 text-emerald-400" />
              Trading Parameters
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure fees, slippage, position sizing, and risk limits
            </p>
          </div>
          
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Changes
          </button>
        </div>
        
        {/* Market Tabs */}
        <div className="flex rounded-lg overflow-hidden border border-slate-700 w-fit">
          <button
            onClick={() => setActiveMarket('br')}
            className={`px-6 py-2 text-sm font-medium transition-colors ${
              activeMarket === 'br' 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            🇧🇷 B3 Brasil
          </button>
          <button
            onClick={() => setActiveMarket('us')}
            className={`px-6 py-2 text-sm font-medium transition-colors ${
              activeMarket === 'us' 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            🇺🇸 US Equities
          </button>
        </div>
        
        {/* Parameters Grid */}
        <div className="grid gap-6">
          
          {/* Fee Tier */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Fee Structure</h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {Object.entries(FEE_TIERS)
                .filter(([, info]) => info.market === activeMarket)
                .map(([key, info]) => (
                  <button
                    key={key}
                    onClick={() => selectFeeTier(key)}
                    className={`p-4 rounded-lg border text-left transition-colors ${
                      current.feeTier === key
                        ? 'bg-emerald-500/20 border-emerald-500/50 text-white'
                        : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
                    }`}
                  >
                    <p className="font-medium">{info.label}</p>
                    <p className="text-sm text-slate-400 mt-1">
                      {info.rate === 0 ? 'Zero commission' : `${(info.rate * 100).toFixed(3)}% per trade`}
                    </p>
                    <p className="text-xs text-slate-500 mt-2 leading-relaxed">{info.details}</p>
                  </button>
                ))}
            </div>
            
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Custom Fee Rate (%)</label>
                <input
                  type="number"
                  step="0.001"
                  value={(current.feeRate * 100).toFixed(4)}
                  onChange={e => updateParam('feeRate', parseFloat(e.target.value) / 100 || 0)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 flex items-center">
                  Slippage (bps)
                  <QuickTooltip termKey="slippage" size="sm" />
                </label>
                <input
                  type="number"
                  value={current.slippageBps}
                  onChange={e => updateParam('slippageBps', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>
          </div>
          
          {/* Position Sizing */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Position Sizing</h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {Object.entries(POSITION_SIZING).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => updateParam('positionSizing', key as TradingParams['positionSizing'])}
                  className={`p-4 rounded-lg border text-left transition-colors ${
                    current.positionSizing === key
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
                  }`}
                >
                  <p className="font-medium">{info.label}</p>
                  <p className="text-xs text-slate-400 mt-1">{info.description}</p>
                </button>
              ))}
            </div>
          </div>
          
          {/* Risk Limits */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Risk Limits</h3>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Max Position (%)</label>
                <input
                  type="number"
                  value={current.maxPositionPct}
                  onChange={e => updateParam('maxPositionPct', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 flex items-center">
                  Max Drawdown (%)
                  <QuickTooltip termKey="max_drawdown" size="sm" />
                </label>
                <input
                  type="number"
                  value={current.maxDrawdownPct}
                  onChange={e => updateParam('maxDrawdownPct', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Lot Size</label>
                <input
                  type="number"
                  value={current.lotSize}
                  onChange={e => updateParam('lotSize', parseInt(e.target.value) || 1)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Execution Delay (ms)</label>
                <input
                  type="number"
                  value={current.delayMs}
                  onChange={e => updateParam('delayMs', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Info */}
        <div className="flex items-start gap-3 p-4 rounded-lg bg-slate-800/50 border border-slate-700 text-sm text-slate-400">
          <Info className="w-5 h-5 text-slate-500 mt-0.5 flex-shrink-0" />
          <p>
            These parameters affect how backtests calculate costs and manage positions.
            {activeMarket === 'br' && ' B3 requires trades in multiples of 100 shares (lote padrão).'}
            {activeMarket === 'us' && ' US markets allow single-share trades.'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConfigTrading;











