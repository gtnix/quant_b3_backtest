/**
 * ConfigUniverse - Configure asset universes (B3 + US Equities)
 */

import { useEffect, useState } from 'react';
import { Globe, CheckCircle2, XCircle, RefreshCw, Plus, Trash2, Save } from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';

interface MarketConfig {
  enabled: boolean;
  name: string;
  universe: string;
  symbols?: string[];
  calendar: string;
  currency: string;
  lot_size: number;
}

const DEFAULT_MARKETS: Record<string, MarketConfig> = {
  br: {
    enabled: true,
    name: 'B3 - Brasil',
    universe: 'ibov',
    calendar: 'b3',
    currency: 'BRL',
    lot_size: 100,
  },
  us: {
    enabled: true,
    name: 'US Equities',
    universe: 'sp500',
    calendar: 'nyse',
    currency: 'USD',
    lot_size: 1,
  },
};

const PRESET_UNIVERSES: Record<string, { label: string; description: string; symbols?: number }> = {
  // B3
  ibov: { label: 'IBOV', description: 'Índice Bovespa (~90 stocks)', symbols: 90 },
  ibrx100: { label: 'IBrX 100', description: 'Brasil 100 Index', symbols: 100 },
  small: { label: 'SMLL', description: 'Small Caps Index', symbols: 100 },
  custom_br: { label: 'Custom', description: 'Custom symbol list' },
  // US
  sp500: { label: 'S&P 500', description: 'Standard & Poor\'s 500', symbols: 500 },
  nasdaq100: { label: 'NASDAQ 100', description: 'NASDAQ 100 Index', symbols: 100 },
  djia: { label: 'DJIA', description: 'Dow Jones Industrial', symbols: 30 },
  custom_us: { label: 'Custom', description: 'Custom symbol list' },
};

export function ConfigUniverse() {
  const { config, fetchConfig } = useOmpStore();
  const [markets, setMarkets] = useState<Record<string, MarketConfig>>(DEFAULT_MARKETS);
  const [saving, setSaving] = useState(false);
  const [customSymbols, setCustomSymbols] = useState<Record<string, string>>({ br: '', us: '' });
  
  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);
  
  useEffect(() => {
    if (config?.markets) {
      setMarkets(prev => ({
        ...prev,
        ...(config.markets as unknown as Record<string, MarketConfig>),
      }));
    }
  }, [config]);
  
  const toggleMarket = (market: string) => {
    setMarkets(prev => ({
      ...prev,
      [market]: { ...prev[market], enabled: !prev[market].enabled },
    }));
  };
  
  const updateUniverse = (market: string, universe: string) => {
    setMarkets(prev => ({
      ...prev,
      [market]: { ...prev[market], universe },
    }));
  };
  
  const handleSave = async () => {
    setSaving(true);
    // In a real implementation, this would save to the server
    console.log('Saving market config:', markets);
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
              <Globe className="w-6 h-6 text-blue-400" />
              Universe Configuration
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure which markets and asset universes to include in strategy mining
            </p>
          </div>
          
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Changes
          </button>
        </div>
        
        {/* Market Cards */}
        <div className="grid gap-6">
          {Object.entries(markets).map(([key, market]) => (
            <div 
              key={key}
              className={`rounded-xl border p-6 transition-colors ${
                market.enabled 
                  ? 'bg-slate-800/50 border-slate-700' 
                  : 'bg-slate-900/30 border-slate-800 opacity-60'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => toggleMarket(key)}
                    className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                      market.enabled 
                        ? key === 'br' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'
                        : 'bg-slate-700 text-slate-500'
                    }`}
                  >
                    {market.enabled ? <CheckCircle2 className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
                  </button>
                  <div>
                    <h3 className="text-lg font-semibold text-white">{market.name}</h3>
                    <p className="text-sm text-slate-400">
                      {market.currency} • {market.calendar.toUpperCase()} Calendar • Lot Size: {market.lot_size}
                    </p>
                  </div>
                </div>
                
                <span className={`px-3 py-1 text-xs rounded-full font-medium ${
                  market.enabled 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-slate-700 text-slate-500'
                }`}>
                  {market.enabled ? 'ENABLED' : 'DISABLED'}
                </span>
              </div>
              
              {market.enabled && (
                <div className="mt-6 space-y-4">
                  <div>
                    <label className="text-sm text-slate-400 mb-2 block">Universe Preset</label>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {Object.entries(PRESET_UNIVERSES)
                        .filter(([k]) => k.endsWith(key) || (!k.includes('_') && (key === 'br' ? ['ibov', 'ibrx100', 'small'].includes(k) : ['sp500', 'nasdaq100', 'djia'].includes(k))))
                        .map(([universeKey, universeInfo]) => (
                          <button
                            key={universeKey}
                            onClick={() => updateUniverse(key, universeKey.replace(`_${key}`, ''))}
                            className={`p-3 rounded-lg border text-left transition-colors ${
                              market.universe === universeKey.replace(`_${key}`, '')
                                ? 'bg-blue-500/20 border-blue-500/50 text-white'
                                : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
                            }`}
                          >
                            <p className="font-medium text-sm">{universeInfo.label}</p>
                            <p className="text-xs text-slate-500 mt-0.5">
                              {universeInfo.symbols ? `${universeInfo.symbols} stocks` : universeInfo.description}
                            </p>
                          </button>
                        ))}
                    </div>
                  </div>
                  
                  {market.universe === 'custom' && (
                    <div>
                      <label className="text-sm text-slate-400 mb-2 block">Custom Symbols (comma-separated)</label>
                      <textarea
                        value={customSymbols[key]}
                        onChange={e => setCustomSymbols(prev => ({ ...prev, [key]: e.target.value }))}
                        placeholder={key === 'br' ? 'PETR4, VALE3, ITUB4...' : 'AAPL, MSFT, GOOGL...'}
                        className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 resize-none"
                        rows={3}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Info */}
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30 text-sm text-blue-300">
          <p>
            <strong>Note:</strong> Universe configuration affects which symbols are considered during strategy mining.
            Each backtest will use the appropriate market settings (fees, slippage, calendar) automatically.
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConfigUniverse;

