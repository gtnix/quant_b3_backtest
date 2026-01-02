/**
 * Bloomberg-Style Tooltip Component
 * 
 * Rich tooltips with multiple data rows, sparklines, and professional styling.
 */

import { ReactNode, useState } from 'react';
import { Info } from 'lucide-react';
import { Sparkline } from '../charts/Sparkline';

interface TooltipRowProps {
  label: string;
  value: string | number;
  sublabel?: string;
  color?: 'profit' | 'loss' | 'warning' | 'default';
}

interface BloombergTooltipProps {
  title: string;
  description?: string;
  rows?: TooltipRowProps[];
  sparkData?: number[];
  formula?: string;
  benchmark?: string;
  interpretation?: string;
  children?: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export function BloombergTooltip({
  title,
  description,
  rows = [],
  sparkData,
  formula,
  benchmark,
  interpretation,
  children,
  position = 'top'
}: BloombergTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2'
  };

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-terminal-surface',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-terminal-surface',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-terminal-surface',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-terminal-surface'
  };

  const colorMap = {
    profit: 'text-profit',
    loss: 'text-loss',
    warning: 'text-accent-yellow',
    default: 'text-white'
  };

  return (
    <div 
      className="relative inline-flex"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children || (
        <button className="p-0.5 rounded hover:bg-terminal-surface/50 transition-colors">
          <Info className="w-3.5 h-3.5 text-terminal-muted hover:text-accent-cyan transition-colors" />
        </button>
      )}
      
      {isVisible && (
        <div 
          className={`absolute z-50 ${positionClasses[position]} min-w-[240px] max-w-[320px]`}
        >
          {/* Tooltip Card */}
          <div className="bg-terminal-surface border border-terminal-border rounded-lg shadow-2xl shadow-black/50 overflow-hidden">
            {/* Header */}
            <div className="px-3 py-2 bg-gradient-to-r from-terminal-bg to-terminal-surface border-b border-terminal-border">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-white">{title}</h4>
                {sparkData && sparkData.length > 0 && (
                  <Sparkline data={sparkData} width={50} height={18} />
                )}
              </div>
              {description && (
                <p className="text-[10px] text-terminal-muted mt-0.5">{description}</p>
              )}
            </div>
            
            {/* Content */}
            <div className="px-3 py-2 space-y-2">
              {/* Formula */}
              {formula && (
                <div className="p-2 bg-terminal-bg/50 rounded border border-terminal-border/50">
                  <div className="text-[10px] text-terminal-muted uppercase tracking-wider mb-1">Formula</div>
                  <code className="text-xs font-mono text-accent-cyan">{formula}</code>
                </div>
              )}
              
              {/* Data Rows */}
              {rows.length > 0 && (
                <div className="space-y-1">
                  {rows.map((row, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-terminal-muted">
                        {row.label}
                        {row.sublabel && (
                          <span className="text-[10px] ml-1">({row.sublabel})</span>
                        )}
                      </span>
                      <span className={`font-mono font-medium ${colorMap[row.color || 'default']}`}>
                        {row.value}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Benchmark */}
              {benchmark && (
                <div className="flex items-center justify-between text-xs pt-1 border-t border-terminal-border/30">
                  <span className="text-terminal-muted">Benchmark</span>
                  <span className="font-mono text-accent-yellow">{benchmark}</span>
                </div>
              )}
              
              {/* Interpretation */}
              {interpretation && (
                <div className="pt-2 border-t border-terminal-border/30">
                  <div className="text-[10px] leading-relaxed text-terminal-muted">
                    💡 {interpretation}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Arrow */}
          <div 
            className={`absolute w-0 h-0 border-4 border-transparent ${arrowClasses[position]}`} 
          />
        </div>
      )}
    </div>
  );
}

/**
 * MetricTooltip - Tooltips predefinidos para métricas financeiras comuns
 */
export const MetricTooltips = {
  sharpe: (value: number) => ({
    title: 'Sharpe Ratio',
    description: 'Retorno ajustado ao risco',
    formula: '(Rp - Rf) / σp',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: value >= 1.0 ? 'profit' as const : value >= 0.5 ? 'warning' as const : 'loss' as const },
      { label: 'Anualizado', value: 'Sim', sublabel: '252 dias' },
    ],
    benchmark: '≥ 1.0 (bom), ≥ 2.0 (excelente)',
    interpretation: 'Valores maiores indicam melhores retornos ajustados ao risco. >1 é bom, >2 é excelente.'
  }),
  
  sortino: (value: number) => ({
    title: 'Sortino Ratio',
    description: 'Retorno ajustado ao risco de baixa',
    formula: '(Rp - Rf) / σd',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: value >= 1.5 ? 'profit' as const : value >= 1.0 ? 'warning' as const : 'loss' as const },
      { label: 'Usa', value: 'Apenas desvio de baixa' },
    ],
    benchmark: '≥ 1.5 (bom), ≥ 2.0 (excelente)',
    interpretation: 'Melhor que Sharpe para distribuições de retorno assimétricas. Ignora volatilidade de alta.'
  }),
  
  calmar: (value: number) => ({
    title: 'Calmar Ratio',
    description: 'Retorno vs drawdown máximo',
    formula: 'CAGR / Drawdown Máximo',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: value >= 1.0 ? 'profit' as const : value >= 0.5 ? 'warning' as const : 'loss' as const },
    ],
    benchmark: '≥ 1.0 (bom), ≥ 3.0 (excelente)',
    interpretation: 'Valores maiores significam melhor recompensa relativa à pior perda histórica.'
  }),
  
  omega: (value: number) => ({
    title: 'Omega Ratio',
    description: 'Ganhos/perdas ponderados por probabilidade',
    formula: '∫ ganhos / ∫ perdas (acima/abaixo do limiar)',
    rows: [
      { label: 'Atual', value: value === Infinity ? '∞' : value.toFixed(3), color: value >= 1.5 ? 'profit' as const : 'warning' as const },
      { label: 'Limiar', value: '0%', sublabel: 'livre de risco' },
    ],
    benchmark: '≥ 1.5 (bom), ≥ 2.0 (excelente)',
    interpretation: '>1 significa mais ganhos que perdas. Captura toda a distribuição de retornos.'
  }),
  
  var95: (value: number) => ({
    title: 'Value at Risk (95%)',
    description: 'Perda máxima diária esperada com 95% de confiança',
    formula: 'Percentil 5 dos retornos diários',
    rows: [
      { label: 'VaR Diário', value: `${(value * 100).toFixed(2)}%`, color: 'loss' as const },
      { label: 'Confiança', value: '95%' },
    ],
    interpretation: 'Em 95% dos dias, as perdas não excederão esse valor. Não captura risco de cauda.'
  }),
  
  cvar95: (value: number) => ({
    title: 'CVaR Condicional (95%)',
    description: 'Perda esperada nos piores 5% dos casos',
    formula: 'E[perda | perda > VaR95]',
    rows: [
      { label: 'CVaR Diário', value: `${(value * 100).toFixed(2)}%`, color: 'loss' as const },
      { label: 'Também chamado', value: 'Expected Shortfall' },
    ],
    interpretation: 'Média das piores perdas. Captura melhor o risco de cauda que o VaR.'
  }),
  
  pbo: (value: number) => ({
    title: 'Probabilidade de Overfitting',
    description: 'Probabilidade de performance OOS ser pior que IS',
    formula: 'Baseado na distribuição de degradação CPCV',
    rows: [
      { label: 'Atual', value: `${(value * 100).toFixed(1)}%`, color: value <= 0.15 ? 'profit' as const : value <= 0.30 ? 'warning' as const : 'loss' as const },
      { label: 'Status', value: value <= 0.15 ? 'APROVADO' : 'REPROVADO' },
    ],
    benchmark: '≤ 15% (validado)',
    interpretation: 'Menor é melhor. Acima de 15% sugere que a estratégia pode estar sobreajustada.'
  }),
  
  dsr: (value: number) => ({
    title: 'Sharpe Ratio Deflacionado',
    description: 'Sharpe ajustado para múltiplos testes',
    formula: 'SR × √(1 - Var(SR) / SR²)',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: value >= 0.5 ? 'profit' as const : 'warning' as const },
      { label: 'Tentativas', value: 'Ajustado' },
    ],
    benchmark: '≥ 0.5 (validado)',
    interpretation: 'Compensa a sorte de testar muitas estratégias. Use em vez do Sharpe bruto.'
  }),
  
  tstat: (value: number) => ({
    title: 'Estatística T',
    description: 'Significância estatística do Sharpe Ratio',
    formula: 'SR × √(n / 252)',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: Math.abs(value) >= 2.0 ? 'profit' as const : 'warning' as const },
      { label: 'Limiar', value: '≥ 2.0 para IC 95%' },
    ],
    interpretation: 't ≥ 2.0 significa <5% de chance do Sharpe ser só sorte.'
  }),
  
  pvalue: (value: number) => ({
    title: 'Valor-P',
    description: 'Probabilidade dos retornos serem por acaso',
    formula: '2 × (1 - Φ(|t-stat|))',
    rows: [
      { label: 'Atual', value: value < 0.001 ? '<0.001' : value.toFixed(4), color: value <= 0.05 ? 'profit' as const : 'loss' as const },
      { label: 'Status', value: value <= 0.05 ? 'Significativo' : 'Não significativo' },
    ],
    benchmark: '≤ 0.05 (95% confiança)',
    interpretation: 'Menor é melhor. <0.05 significa estatisticamente significativo.'
  }),
  
  skewness: (value: number) => ({
    title: 'Assimetria (Skewness)',
    description: 'Assimetria da distribuição de retornos',
    formula: 'E[(X - μ)³] / σ³',
    rows: [
      { label: 'Atual', value: value.toFixed(3), color: value >= 0 ? 'profit' as const : 'warning' as const },
      { label: 'Distribuição', value: value > 0 ? 'Assimétrica à direita' : value < 0 ? 'Assimétrica à esquerda' : 'Simétrica' },
    ],
    interpretation: 'Assimetria positiva = mais ganhos extremos. Negativa = mais perdas extremas.'
  }),
  
  kurtosis: (value: number) => ({
    title: 'Curtose Excessiva',
    description: 'Espessura das caudas vs distribuição normal',
    formula: 'E[(X - μ)⁴] / σ⁴ - 3',
    rows: [
      { label: 'Atual', value: value.toFixed(3) },
      { label: 'Caudas', value: value > 0 ? 'Caudas gordas' : value < 0 ? 'Caudas finas' : 'Normal' },
    ],
    interpretation: 'Positivo = mais eventos extremos que o normal. Comum em finanças.'
  }),
};

