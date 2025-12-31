/**
 * ConfigBudget - Configure compute budget for strategy mining
 */

import { useState } from 'react';
import { Cpu, Save, RefreshCw, Zap, Clock, Hash } from 'lucide-react';

interface BudgetConfig {
  maxRuntimeSeconds: number;
  workers: number;
  seeds: number[];
  populationSize: number;
  maxGenerations: number;
  convergenceGenerations: number;
  topK: number;
  stageATopN: number;
  intensityLevel: 'quick' | 'standard' | 'thorough' | 'exhaustive';
}

const INTENSITY_PRESETS: Record<string, { label: string; description: string; config: Partial<BudgetConfig> }> = {
  quick: {
    label: '⚡ Quick',
    description: '5 min, 1 seed, fast discovery',
    config: { maxRuntimeSeconds: 300, seeds: [42], populationSize: 50, maxGenerations: 25 },
  },
  standard: {
    label: '🔄 Standard',
    description: '15 min, 3 seeds, balanced',
    config: { maxRuntimeSeconds: 900, seeds: [42, 123, 456], populationSize: 100, maxGenerations: 50 },
  },
  thorough: {
    label: '🔬 Thorough',
    description: '30 min, 5 seeds, detailed search',
    config: { maxRuntimeSeconds: 1800, seeds: [42, 123, 456, 789, 1011], populationSize: 150, maxGenerations: 75 },
  },
  exhaustive: {
    label: '🏆 Exhaustive',
    description: '1 hour, 10 seeds, maximum coverage',
    config: { maxRuntimeSeconds: 3600, seeds: [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066], populationSize: 200, maxGenerations: 100 },
  },
};

export function ConfigBudget() {
  const [config, setConfig] = useState<BudgetConfig>({
    maxRuntimeSeconds: 900,
    workers: 8,
    seeds: [42, 123, 456],
    populationSize: 100,
    maxGenerations: 50,
    convergenceGenerations: 10,
    topK: 10,
    stageATopN: 100,
    intensityLevel: 'standard',
  });
  
  const [saving, setSaving] = useState(false);
  const [seedInput, setSeedInput] = useState(config.seeds.join(', '));
  
  const selectIntensity = (level: keyof typeof INTENSITY_PRESETS) => {
    const preset = INTENSITY_PRESETS[level];
    setConfig(prev => ({
      ...prev,
      ...preset.config,
      intensityLevel: level as BudgetConfig['intensityLevel'],
    }));
    if (preset.config.seeds) {
      setSeedInput(preset.config.seeds.join(', '));
    }
  };
  
  const updateSeeds = (input: string) => {
    setSeedInput(input);
    const seeds = input.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    if (seeds.length > 0) {
      setConfig(prev => ({ ...prev, seeds }));
    }
  };
  
  const handleSave = async () => {
    setSaving(true);
    console.log('Saving budget config:', config);
    await new Promise(r => setTimeout(r, 500));
    setSaving(false);
  };
  
  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Cpu className="w-6 h-6 text-violet-400" />
              Compute Budget
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure compute resources, runtime, and evolution parameters
            </p>
          </div>
          
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Changes
          </button>
        </div>
        
        {/* Intensity Presets */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            Intensity Level
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(INTENSITY_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => selectIntensity(key as keyof typeof INTENSITY_PRESETS)}
                className={`p-4 rounded-lg border text-left transition-colors ${
                  config.intensityLevel === key
                    ? 'bg-violet-500/20 border-violet-500/50 text-white'
                    : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
                }`}
              >
                <p className="font-medium text-lg">{preset.label}</p>
                <p className="text-xs text-slate-400 mt-1">{preset.description}</p>
              </button>
            ))}
          </div>
        </div>
        
        {/* Runtime & Workers */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-400" />
            Runtime & Resources
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Max Runtime</label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={config.maxRuntimeSeconds}
                  onChange={e => setConfig(prev => ({ ...prev, maxRuntimeSeconds: parseInt(e.target.value) || 300 }))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
                />
                <span className="text-xs text-slate-500 whitespace-nowrap">{formatTime(config.maxRuntimeSeconds)}</span>
              </div>
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Workers</label>
              <input
                type="number"
                min="1"
                max="32"
                value={config.workers}
                onChange={e => setConfig(prev => ({ ...prev, workers: parseInt(e.target.value) || 1 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Top K (Stage B)</label>
              <input
                type="number"
                value={config.topK}
                onChange={e => setConfig(prev => ({ ...prev, topK: parseInt(e.target.value) || 10 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Stage A Top N</label>
              <input
                type="number"
                value={config.stageATopN}
                onChange={e => setConfig(prev => ({ ...prev, stageATopN: parseInt(e.target.value) || 100 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>
        </div>
        
        {/* Evolution Parameters */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Hash className="w-5 h-5 text-emerald-400" />
            Evolution Parameters
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Population Size</label>
              <input
                type="number"
                value={config.populationSize}
                onChange={e => setConfig(prev => ({ ...prev, populationSize: parseInt(e.target.value) || 50 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Max Generations</label>
              <input
                type="number"
                value={config.maxGenerations}
                onChange={e => setConfig(prev => ({ ...prev, maxGenerations: parseInt(e.target.value) || 50 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Convergence Gens</label>
              <input
                type="number"
                value={config.convergenceGenerations}
                onChange={e => setConfig(prev => ({ ...prev, convergenceGenerations: parseInt(e.target.value) || 10 }))}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Seeds</label>
              <input
                type="text"
                value={seedInput}
                onChange={e => updateSeeds(e.target.value)}
                placeholder="42, 123, 456"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>
        </div>
        
        {/* Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/30 text-center">
            <p className="text-2xl font-bold text-violet-400">{formatTime(config.maxRuntimeSeconds)}</p>
            <p className="text-xs text-slate-400 mt-1">Max Runtime</p>
          </div>
          <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/30 text-center">
            <p className="text-2xl font-bold text-blue-400">{config.seeds.length}</p>
            <p className="text-xs text-slate-400 mt-1">Seeds</p>
          </div>
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
            <p className="text-2xl font-bold text-emerald-400">{config.populationSize}</p>
            <p className="text-xs text-slate-400 mt-1">Population</p>
          </div>
          <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-center">
            <p className="text-2xl font-bold text-amber-400">{config.workers}</p>
            <p className="text-xs text-slate-400 mt-1">Workers</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfigBudget;






