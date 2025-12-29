/**
 * Cockpit - Main control panel for SCG runs
 * 
 * Sections:
 * A) Compute Budget - Time, intensity, seeds
 * B) Risk & Robustness Gates - Thresholds for validation
 * C) Ranking & Prioritization - How to sort candidates
 * D) Controls - Start/Stop, progress, log
 * E) Top Strategies - Live ranking with drilldown
 */

import React, { useEffect } from 'react';
import { useCockpitStore } from '../stores/cockpitStore';
import type { RankedCandidate } from '../stores/cockpitStore';
import type { PresetKey, RankingMethodKey } from '../config/defaults';
import {
  COCKPIT_PRESETS,
  RANKING_METHODS,
  TIME_PRESETS,
  INTENSITY_LEVELS,
  GATES_CONFIG,
} from '../config/defaults';
import { InfoIcon, TOOLTIPS } from '../components/ui/TooltipInfo';

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

interface SectionProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}

function Section({ title, subtitle, children, className = '' }: SectionProps) {
  return (
    <div className={`bg-slate-800/50 border border-slate-700/50 rounded-lg p-5 ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
        {subtitle && <p className="text-sm text-slate-400 mt-1">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

function Toggle({ label, checked, onChange, disabled }: ToggleProps) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <div className="relative">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
        />
        <div className={`w-10 h-6 rounded-full transition-colors ${
          checked ? 'bg-cyan-500' : 'bg-slate-600'
        } ${disabled ? 'opacity-50' : ''}`}>
          <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
            checked ? 'translate-x-4' : ''
          }`} />
        </div>
      </div>
      <span className="text-sm text-slate-300">{label}</span>
    </label>
  );
}

// =============================================================================
// PRESET SELECTOR
// =============================================================================

function PresetSelector() {
  const { config, setPreset } = useCockpitStore();
  
  return (
    <div className="grid grid-cols-3 gap-3">
      {Object.values(COCKPIT_PRESETS).map((preset) => (
        <button
          key={preset.key}
          onClick={() => setPreset(preset.key)}
          className={`p-4 rounded-lg border-2 transition-all text-left ${
            config.preset === preset.key
              ? 'border-cyan-500 bg-cyan-500/10'
              : 'border-slate-600 hover:border-slate-500 bg-slate-800/50'
          }`}
        >
          <div className="text-2xl mb-2">{preset.icon}</div>
          <div className="font-medium text-slate-100">{preset.name}</div>
          <div className="text-xs text-slate-400 mt-1">{preset.description}</div>
        </button>
      ))}
    </div>
  );
}

// =============================================================================
// TIME SLIDER
// =============================================================================

function TimeSlider() {
  const { config, updateConfig } = useCockpitStore();
  
  const currentIndex = TIME_PRESETS.findIndex(
    (p) => p.seconds >= config.maxRuntimeSeconds
  );
  const index = currentIndex === -1 ? TIME_PRESETS.length - 1 : currentIndex;
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = parseInt(e.target.value);
    updateConfig({ maxRuntimeSeconds: TIME_PRESETS[newIndex].seconds });
  };
  
  const current = TIME_PRESETS[index];
  const progress = (index / (TIME_PRESETS.length - 1)) * 100;
  
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm text-slate-300 flex items-center">
          Tempo Máximo
          <InfoIcon tooltipKey="max_runtime" />
        </label>
        <span className="text-lg font-mono text-cyan-400 transition-all duration-300">{current.label}</span>
      </div>
      
      {/* Custom smooth slider */}
      <div className="relative h-8 flex items-center">
        {/* Track background */}
        <div className="absolute inset-x-0 h-2 bg-slate-700 rounded-full" />
        
        {/* Filled track with smooth transition */}
        <div 
          className="absolute left-0 h-2 bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        />
        
        {/* Invisible range input for interaction */}
        <input
          type="range"
          min={0}
          max={TIME_PRESETS.length - 1}
          value={index}
          onChange={handleChange}
          className="absolute inset-x-0 w-full h-8 opacity-0 cursor-pointer z-10"
        />
        
        {/* Custom thumb with smooth transition */}
        <div 
          className="absolute w-5 h-5 bg-white rounded-full shadow-lg shadow-cyan-500/50 border-2 border-cyan-400 transition-all duration-300 ease-out pointer-events-none"
          style={{ left: `calc(${progress}% - 10px)` }}
        />
        
        {/* Tick marks */}
        <div className="absolute inset-x-0 flex justify-between pointer-events-none">
          {TIME_PRESETS.map((_, i) => (
            <div 
              key={i} 
              className={`w-1 h-1 rounded-full transition-colors duration-200 ${
                i <= index ? 'bg-cyan-400' : 'bg-slate-600'
              }`}
            />
          ))}
        </div>
      </div>
      
      {/* Labels with smooth highlight transition */}
      <div className="flex justify-between text-xs">
        {TIME_PRESETS.map((p, i) => (
          <span 
            key={i} 
            className={`transition-all duration-200 ${
              i === index 
                ? 'text-cyan-400 font-medium transform scale-110' 
                : 'text-slate-500'
            }`}
          >
            {p.label}
          </span>
        ))}
      </div>
      
      <p className="text-xs text-slate-500 italic transition-opacity duration-300">{current.description}</p>
    </div>
  );
}

// =============================================================================
// INTENSITY SELECTOR
// =============================================================================

function IntensitySelector() {
  const { config, updateConfig } = useCockpitStore();
  
  return (
    <div className="space-y-3">
      <label className="text-sm text-slate-300 flex items-center">
        Intensidade (CPU)
        <InfoIcon tooltipKey="workers" />
      </label>
      
      <div className="grid grid-cols-4 gap-2">
        {INTENSITY_LEVELS.map((level) => (
          <button
            key={level.key}
            onClick={() => updateConfig({ workers: level.workers })}
            className={`p-2 rounded text-center text-sm transition-colors ${
              config.workers === level.workers
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
            }`}
            title={level.description}
          >
            {level.name}
          </button>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// GATES CONFIGURATION
// =============================================================================

function GatesConfig() {
  const { config, updateConfig, viewMode } = useCockpitStore();
  
  if (viewMode === 'basic') {
    return (
      <div className="space-y-4">
        <Toggle
          label="Stress Testing habilitado"
          checked={config.stressTestingEnabled}
          onChange={(checked) => updateConfig({ stressTestingEnabled: checked })}
        />
        
        <div className="text-sm text-slate-400">
          <p>Gates ativos com defaults institucionais:</p>
          <ul className="mt-2 space-y-1 text-slate-500">
            <li>• Sharpe OOS NET ≥ {config.minOosSharpeNet}</li>
            <li>• PBO ≤ {(config.maxPbo * 100).toFixed(0)}%</li>
            {config.stressTestingEnabled && (
              <li>• Stress tests ≥ {config.minStressPassed}/8</li>
            )}
          </ul>
        </div>
      </div>
    );
  }
  
  // Advanced mode
  return (
    <div className="space-y-5">
      {GATES_CONFIG.map((gate) => {
        if (gate.type === 'boolean') {
          return (
            <div key={gate.key} className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="text-sm text-slate-300">{gate.name}</span>
                <InfoIcon tooltipKey={gate.key as keyof typeof TOOLTIPS} />
              </div>
              <Toggle
                label=""
                checked={(config as any)[gate.key] ?? gate.defaultValue}
                onChange={(checked) => updateConfig({ [gate.key]: checked })}
              />
            </div>
          );
        }
        
        return (
          <div key={gate.key} className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm text-slate-300 flex items-center">
                {gate.name}
                {gate.description && (
                  <span className="ml-2 text-xs text-slate-500">({gate.description})</span>
                )}
                <InfoIcon tooltipKey={gate.key as keyof typeof TOOLTIPS} />
              </label>
              <span className="font-mono text-cyan-400">
                {((config as any)[gate.key] ?? gate.defaultValue).toFixed(gate.step && gate.step < 1 ? 2 : 0)}
                {gate.unit && <span className="text-slate-500 ml-1">{gate.unit}</span>}
              </span>
            </div>
            
            <input
              type="range"
              min={gate.min}
              max={gate.max}
              step={gate.step}
              value={(config as any)[gate.key] ?? gate.defaultValue}
              onChange={(e) => updateConfig({ [gate.key]: parseFloat(e.target.value) })}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>
        );
      })}
    </div>
  );
}

// =============================================================================
// RANKING SELECTOR
// =============================================================================

function RankingSelector() {
  const { rankingMethod, setRankingMethod } = useCockpitStore();
  
  return (
    <div className="space-y-3">
      {Object.values(RANKING_METHODS).map((method) => (
        <button
          key={method.key}
          onClick={() => setRankingMethod(method.key)}
          className={`w-full p-3 rounded-lg border text-left transition-all ${
            rankingMethod === method.key
              ? 'border-cyan-500 bg-cyan-500/10'
              : 'border-slate-600 hover:border-slate-500 bg-slate-800/50'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-100">{method.name}</span>
            {rankingMethod === method.key && (
              <span className="text-cyan-400">✓</span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">{method.description}</p>
        </button>
      ))}
    </div>
  );
}

// =============================================================================
// PROGRESS DISPLAY
// =============================================================================

function ProgressDisplay() {
  const { progress, runStatus } = useCockpitStore();
  
  if (!progress || runStatus === 'idle') {
    return (
      <div className="text-center py-8 text-slate-500">
        Configure os parâmetros e clique START para iniciar
      </div>
    );
  }
  
  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins}:${s.toString().padStart(2, '0')}`;
  };
  
  // Safely access progress values with defaults
  const percent = progress.percentComplete ?? 0;
  const elapsed = progress.elapsedSeconds ?? 0;
  const maxRuntime = progress.maxRuntimeSeconds ?? 0;
  const currentGen = progress.currentGeneration ?? 0;
  const maxGens = progress.maxGenerations ?? 0;
  const candidates = progress.candidatesEvaluated ?? 0;
  
  return (
    <div className="space-y-4">
      {/* Progress bar with smooth animation */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Progresso</span>
          <span className="text-cyan-400 font-mono tabular-nums transition-all duration-500">{percent.toFixed(1)}%</span>
        </div>
        <div className="h-4 bg-slate-700/50 rounded-full overflow-hidden relative">
          {/* Background glow */}
          <div 
            className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 blur-sm transition-all duration-1000 ease-out"
            style={{ width: `${Math.min(percent + 10, 100)}%` }}
          />
          {/* Main progress bar with smooth easing */}
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-cyan-400 to-emerald-400 relative overflow-hidden transition-all duration-1000 ease-out"
            style={{ 
              width: `${percent}%`,
              boxShadow: '0 0 20px rgba(34, 211, 238, 0.4)'
            }}
          >
            {/* Animated shimmer effect */}
            <div 
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"
              style={{
                animation: 'shimmer 2s infinite linear',
              }}
            />
          </div>
          {/* Pulse dot at the end */}
          {percent > 0 && percent < 100 && (
            <div 
              className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-white rounded-full shadow-lg transition-all duration-1000 ease-out animate-pulse"
              style={{ left: `calc(${percent}% - 4px)` }}
            />
          )}
        </div>
      </div>
      
      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs text-slate-500 uppercase">Tempo</div>
          <div className="text-lg font-mono text-slate-200">
            {formatTime(elapsed)} / {formatTime(maxRuntime)}
          </div>
        </div>
        
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs text-slate-500 uppercase">Geração</div>
          <div className="text-lg font-mono text-slate-200">
            {currentGen} / {maxGens}
          </div>
        </div>
        
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs text-slate-500 uppercase">Melhor Sharpe</div>
          <div className="text-lg font-mono text-cyan-400">
            {progress.bestSharpe?.toFixed(3) ?? '—'}
          </div>
        </div>
        
        <div className="bg-slate-800 rounded p-3">
          <div className="text-xs text-slate-500 uppercase">Candidatos</div>
          <div className="text-lg font-mono text-slate-200">
            {candidates}
          </div>
        </div>
      </div>
      
      {/* Latest log */}
      {progress.latestLog && (
        <div className="p-3 bg-slate-900 rounded border border-slate-700 font-mono text-xs text-slate-400">
          {progress.latestLog}
        </div>
      )}
      
      {/* Error message */}
      {progress.errorMessage && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
          {progress.errorMessage}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// CONTROL BUTTONS
// =============================================================================

function ControlButtons() {
  const { runStatus, startRun, stopRun } = useCockpitStore();
  
  const isRunning = runStatus === 'running' || runStatus === 'starting';
  const isStopping = runStatus === 'stopping';
  
  return (
    <div className="flex gap-3">
      {!isRunning ? (
        <button
          onClick={startRun}
          disabled={isStopping}
          className="flex-1 py-4 px-6 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white font-bold rounded-lg hover:from-cyan-400 hover:to-emerald-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-lg"
        >
          ▶ START
        </button>
      ) : (
        <button
          onClick={stopRun}
          disabled={isStopping}
          className="flex-1 py-4 px-6 bg-gradient-to-r from-red-500 to-orange-500 text-white font-bold rounded-lg hover:from-red-400 hover:to-orange-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-lg"
        >
          {isStopping ? '⏳ PARANDO...' : '⏹ STOP'}
        </button>
      )}
    </div>
  );
}

// =============================================================================
// TOP STRATEGIES TABLE
// =============================================================================

function TopStrategiesTable() {
  const { topCandidates, selectCandidate, selectedCandidateId } = useCockpitStore();
  
  if (topCandidates.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        Nenhum candidato ainda. Execute uma simulação para ver resultados.
      </div>
    );
  }
  
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-500 uppercase border-b border-slate-700">
            <th className="py-2 px-2">#</th>
            <th className="py-2 px-2">Estratégia</th>
            <th className="py-2 px-2 text-right">Sharpe</th>
            <th className="py-2 px-2 text-right">PBO</th>
            <th className="py-2 px-2 text-right">MaxDD</th>
            <th className="py-2 px-2 text-center">Gates</th>
            <th className="py-2 px-2">Por que no topo?</th>
          </tr>
        </thead>
        <tbody>
          {topCandidates.slice(0, 10).map((candidate) => (
            <tr
              key={candidate.candidateId}
              onClick={() => selectCandidate(candidate.candidateId)}
              className={`border-b border-slate-800 cursor-pointer transition-colors ${
                selectedCandidateId === candidate.candidateId
                  ? 'bg-cyan-500/10'
                  : 'hover:bg-slate-800/50'
              }`}
            >
              <td className="py-3 px-2 font-mono text-slate-500">
                {candidate.rank}
              </td>
              <td className="py-3 px-2">
                <button
                  className="text-cyan-400 hover:text-cyan-300 hover:underline text-left"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Navigate to backtest page with candidate
                    window.dispatchEvent(new CustomEvent('navigate', { detail: 'backtest' }));
                  }}
                >
                  {candidate.displayName}
                </button>
              </td>
              <td className="py-3 px-2 text-right font-mono">
                <span className={candidate.oosSharpeNet >= 1 ? 'text-emerald-400' : 
                  candidate.oosSharpeNet >= 0.5 ? 'text-cyan-400' : 'text-slate-400'}>
                  {candidate.oosSharpeNet.toFixed(3)}
                </span>
              </td>
              <td className="py-3 px-2 text-right font-mono">
                <span className={candidate.pbo <= 0.10 ? 'text-emerald-400' : 
                  candidate.pbo <= 0.15 ? 'text-cyan-400' : 'text-amber-400'}>
                  {(candidate.pbo * 100).toFixed(1)}%
                </span>
              </td>
              <td className="py-3 px-2 text-right font-mono text-red-400">
                {candidate.maxDrawdownNet.toFixed(1)}%
              </td>
              <td className="py-3 px-2 text-center">
                {candidate.gatesPassed ? (
                  <span className="text-emerald-400">✓</span>
                ) : (
                  <span className="text-slate-600">✗</span>
                )}
              </td>
              <td className="py-3 px-2">
                <div className="flex flex-wrap gap-1">
                  {candidate.rankReasons.slice(0, 2).map((reason, i) => (
                    <span
                      key={i}
                      className="px-1.5 py-0.5 bg-slate-700 text-slate-400 text-xs rounded"
                    >
                      {reason}
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// =============================================================================
// MAIN COCKPIT PAGE
// =============================================================================

export function Cockpit() {
  const { viewMode, setViewMode, subscribeToProgress } = useCockpitStore();
  
  // Subscribe to progress updates
  useEffect(() => {
    const unsubscribe = subscribeToProgress();
    return unsubscribe;
  }, [subscribeToProgress]);
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Cockpit</h1>
          <p className="text-slate-400 mt-1">
            Painel de controle para descoberta de estratégias
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* View mode toggle */}
          <div className="flex bg-slate-800 rounded-lg p-1">
            <button
              onClick={() => setViewMode('basic')}
              className={`px-4 py-2 rounded text-sm transition-colors ${
                viewMode === 'basic'
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Básico
            </button>
            <button
              onClick={() => setViewMode('advanced')}
              className={`px-4 py-2 rounded text-sm transition-colors ${
                viewMode === 'advanced'
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Avançado
            </button>
          </div>
          
          {/* Glossary hint */}
          <div className="text-xs text-slate-500">
            Pressione <kbd className="px-1.5 py-0.5 bg-slate-700 rounded">?</kbd> para glossário
          </div>
        </div>
      </div>
      
      {/* Main grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left column - Configuration */}
        <div className="col-span-8 space-y-6">
          {/* Preset selection */}
          <Section title="Preset de Execução" subtitle="Escolha um perfil ou personalize">
            <PresetSelector />
          </Section>
          
          {/* Compute Budget */}
          <Section
            title="Compute Budget"
            subtitle="Quanto tempo e recursos alocar"
          >
            <div className="space-y-6">
              <TimeSlider />
              
              {viewMode === 'advanced' && (
                <>
                  <IntensitySelector />
                  
                  <div className="space-y-3">
                    <label className="text-sm text-slate-300 flex items-center">
                      Seeds (reprodutibilidade)
                      <InfoIcon tooltipKey="seeds" />
                    </label>
                    <p className="text-xs text-slate-500">
                      {useCockpitStore.getState().config.seeds.length || 'Auto'} seed(s) configurada(s)
                    </p>
                  </div>
                </>
              )}
            </div>
          </Section>
          
          {/* Gates */}
          <Section
            title="Risk & Robustness Gates"
            subtitle="Thresholds para filtrar estratégias"
          >
            <GatesConfig />
          </Section>
        </div>
        
        {/* Right column - Controls & Results */}
        <div className="col-span-4 space-y-6">
          {/* Ranking method */}
          <Section title="Ranking" subtitle="Como ordenar candidatos">
            <RankingSelector />
          </Section>
          
          {/* Controls */}
          <Section title="Controles">
            <div className="space-y-4">
              <ControlButtons />
              <ProgressDisplay />
            </div>
          </Section>
        </div>
      </div>
      
      {/* Bottom - Results table */}
      <Section title="Top Estratégias" subtitle="Clique para ver detalhes">
        <TopStrategiesTable />
      </Section>
    </div>
  );
}

export default Cockpit;

