/**
 * TooltipInfo - Informative tooltip component with structured content
 * 
 * Provides context-aware help for UI controls with:
 * - What: One-line description
 * - Impact: How it affects generation/validation
 * - When: When to adjust
 * - Example: Practical example
 */

import React, { useState, useRef, useEffect } from 'react';
import { Info } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export interface TooltipContent {
  what: string;
  impact: string;
  when: string;
  example: string;
}

export interface QuickTooltipContent {
  term: string;
  definition: string;
  formula?: string;
  benchmark?: string;
  interpretation?: string;
}

interface TooltipInfoProps {
  content: TooltipContent;
  children?: React.ReactNode;
}

interface SimpleTooltipProps {
  text: string;
  children: React.ReactNode;
}

interface QuickTooltipProps {
  termKey: keyof typeof QUANT_TOOLTIPS;
  position?: 'top' | 'bottom' | 'left' | 'right';
  size?: 'sm' | 'md';
}

// =============================================================================
// QUANT TOOLTIPS DATABASE - English explanations for all quant terms
// =============================================================================

export const QUANT_TOOLTIPS: Record<string, QuickTooltipContent> = {
  // ═══════════════════════════════════════════════════════════════════════════
  // MÉTRICAS DO MINERADOR
  // ═══════════════════════════════════════════════════════════════════════════
  loops: {
    term: 'Ciclos de Mineração',
    definition: 'Quantidade de ciclos completos de mineração executados. Cada ciclo verifica recursos disponíveis, inicia campanhas quando possível e monitora o progresso.',
    interpretation: 'Mais ciclos = mais atividade de mineração. Os ciclos rodam a cada 30 segundos quando a mineração está ativa.'
  },
  uptime: {
    term: 'Tempo de Execução',
    definition: 'Tempo total que o orquestrador de mineração está rodando desde a última inicialização.',
    interpretation: 'Maior tempo = mineração mais contínua. Zera quando você para/inicia a mineração.'
  },
  candidates_24h: {
    term: 'Candidatos Gerados (24h)',
    definition: 'Total de estratégias candidatas criadas nas últimas 24 horas em todas as campanhas.',
    benchmark: '1000+ por dia indica atividade saudável',
    interpretation: 'Mais candidatos = mais estratégias avaliadas. Porém, qualidade importa mais que quantidade!'
  },
  promotions_24h: {
    term: 'Promoções (24h)',
    definition: 'Estratégias que passaram em todos os gates de validação e foram promovidas ao Hall da Fama nas últimas 24 horas.',
    benchmark: '5-50 promoções por dia dependendo da rigidez dos gates',
    interpretation: 'Poucas promoções = gates rigorosos (bom para qualidade). Zero = pode precisar ajustar gates ou rodar mais tempo.'
  },
  hall_of_fame_count: {
    term: 'Tamanho do Hall da Fama',
    definition: 'Total de estratégias de elite que passaram em todos os critérios institucionais de validação.',
    interpretation: 'São estratégias prontas para produção com alta confiança de performance real.'
  },
  throughput_min: {
    term: 'Throughput por Minuto',
    definition: 'Quantidade de genomas de estratégia avaliados por minuto. Mede a velocidade de mineração.',
    benchmark: '10-100 genomas/min é típico dependendo do hardware',
    interpretation: 'Maior = exploração mais rápida. Limitado pela CPU, complexidade do backtest e tamanho dos dados.'
  },
  cpu_usage: {
    term: 'Uso de CPU',
    definition: 'Percentual da CPU sendo utilizado pelo processo de mineração.',
    benchmark: '80-100% é normal durante mineração ativa',
    interpretation: 'Alta CPU = mineração trabalhando forte. Baixa CPU durante mineração pode indicar espera por I/O.'
  },
  memory_usage: {
    term: 'Uso de Memória',
    definition: 'Percentual da memória do sistema sendo utilizado.',
    benchmark: '<80% é saudável. >90% pode causar problemas.',
    interpretation: 'A memória cresce com o tamanho da população e backtests em cache.'
  },
  disk_free: {
    term: 'Disco Livre',
    definition: 'Espaço em disco disponível no servidor de mineração.',
    benchmark: '>5 GB necessário para operação segura',
    interpretation: 'Pouco disco = artefatos podem falhar ao salvar. Limpe outputs antigos periodicamente.'
  },
  campaign_queue: {
    term: 'Fila de Campanhas',
    definition: 'Número de campanhas aguardando para serem executadas.',
    interpretation: 'A fila processa uma campanha por vez. Adicione campanhas para automatizar mineração noturna.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MÉTRICAS DE PERFORMANCE
  // ═══════════════════════════════════════════════════════════════════════════
  sharpe: {
    term: 'Sharpe Ratio',
    definition: 'Retorno ajustado ao risco. Mede quanto retorno extra você ganha por cada unidade de volatilidade assumida. Quanto maior, melhor a relação risco-retorno.',
    formula: '(Retorno - Taxa Livre de Risco) / Volatilidade',
    benchmark: '≥1.0 é bom, ≥2.0 é excelente',
    interpretation: 'Sharpe 1.5 significa 1.5% de retorno extra para cada 1% de risco. É a métrica mais usada para comparar estratégias.'
  },
  sharpe_oos: {
    term: 'Sharpe Real (Out-of-Sample)',
    definition: 'Sharpe Ratio calculado em dados que a estratégia NUNCA viu durante a otimização. Este é o teste REAL - representa a performance que você terá ao operar ao vivo.',
    formula: 'Média dos Sharpes de todos os períodos OOS',
    benchmark: 'Deve ser próximo ao Sharpe IS. WFE > 50% é robusto.',
    interpretation: 'Se IS=1.0 e OOS=0.7, o WFE é 70% - a estratégia mantém boa parte da performance. Se OOS=0.3, WFE é 30% - provavelmente overfit.'
  },
  sharpe_net: {
    term: 'Sharpe Ratio Líquido',
    definition: 'Sharpe Ratio após descontar todos os custos de operação (taxas, slippage). É a performance real que você teria.',
    interpretation: 'Sempre use NET para decisões de trading. GROSS é enganoso pois ignora custos.'
  },
  cagr: {
    term: 'CAGR (Taxa de Crescimento Anual Composta)',
    definition: 'Retorno anualizado considerando juros compostos. Quanto seu investimento cresce por ano em média.',
    formula: '(ValorFinal/ValorInicial)^(1/anos) - 1',
    benchmark: '15%+ é forte para renda variável',
    interpretation: '15% CAGR dobra o dinheiro em ~5 anos. 25% dobra em ~3 anos.'
  },
  max_drawdown: {
    term: 'Drawdown Máximo (MDD)',
    definition: 'Maior queda do pico ao vale antes da recuperação. É a pior perda que você teria experimentado.',
    formula: 'Max((Pico - Vale) / Pico)',
    benchmark: '<20% é conservador, <30% é moderado',
    interpretation: '-25% MDD significa que no pior momento você estava 25% abaixo do seu pico.'
  },
  volatility: {
    term: 'Volatilidade (Anualizada)',
    definition: 'Desvio padrão dos retornos anualizado. Mede quanto os retornos oscilam em torno da média.',
    formula: 'DesvioPadrão(RetornosDiários) × √252',
    benchmark: '10-20% é típico para estratégias de ações',
    interpretation: '15% de vol significa que os retornos tipicamente ficam dentro de ±15% do esperado.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // ÍNDICES AJUSTADOS AO RISCO
  // ═══════════════════════════════════════════════════════════════════════════
  sortino: {
    term: 'Sortino Ratio',
    definition: 'Similar ao Sharpe, mas só penaliza volatilidade negativa. Ignora o "risco" de subida, que na verdade é bom.',
    formula: '(Retorno - Livre de Risco) / Desvio Negativo',
    benchmark: '≥1.5 é bom, ≥2.0 é excelente',
    interpretation: 'Melhor que Sharpe para estratégias com retornos assimétricos (mais ganhos que perdas extremas).'
  },
  calmar: {
    term: 'Calmar Ratio',
    definition: 'Retorno anual dividido pelo drawdown máximo. Mede recompensa por unidade de pior perda possível.',
    formula: 'CAGR / |DrawdownMáximo|',
    benchmark: '≥1.0 é bom, ≥3.0 é excelente',
    interpretation: 'Calmar 2.0 significa que você ganhou 2% para cada 1% de risco de perda máxima.'
  },
  omega: {
    term: 'Omega Ratio',
    definition: 'Razão ponderada por probabilidade entre ganhos e perdas acima de um limite. Captura toda a distribuição de retornos.',
    formula: '∫(ganhos acima do limite) / ∫(perdas abaixo do limite)',
    benchmark: '≥1.5 é bom, ≥2.0 é excelente',
    interpretation: 'Omega 1.8 significa que os ganhos são 80% maiores que as perdas em média.'
  },
  profit_factor: {
    term: 'Fator de Lucro',
    definition: 'Soma de todas as operações vencedoras dividida pela soma das perdedoras. Medida simples de lucratividade.',
    formula: 'Σ(Ganhos) / Σ(Perdas)',
    benchmark: '≥1.5 é bom, ≥2.0 é excelente',
    interpretation: 'PF 2.0 significa que você ganha R$2 para cada R$1 que perde.'
  },
  win_rate: {
    term: 'Taxa de Acerto',
    definition: 'Percentual de operações que são lucrativas. Deve ser considerado junto com o tamanho médio de ganho/perda.',
    formula: 'OperaçõesVencedoras / TotalOperações × 100',
    benchmark: 'Depende do payoff. 40% pode ser excelente com relação 3:1.',
    interpretation: 'Alta taxa com ganhos pequenos pode render menos que baixa taxa com ganhos grandes.'
  },
  payoff_ratio: {
    term: 'Payoff Ratio (Risco/Retorno)',
    definition: 'Tamanho médio da operação vencedora dividido pelo tamanho médio da perdedora.',
    formula: 'GanhoMédio / PerdaMédia',
    benchmark: '≥1.5 é bom, ≥2.0 é excelente',
    interpretation: 'Combinado com taxa de acerto, determina se a estratégia é lucrativa.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // VALUE AT RISK
  // ═══════════════════════════════════════════════════════════════════════════
  var_95: {
    term: 'Value at Risk (95%)',
    definition: 'Perda máxima diária esperada com 95% de confiança. Em 95% dos dias, as perdas não ultrapassarão esse valor.',
    formula: 'Percentil 5 dos retornos diários',
    benchmark: 'Depende da tolerância ao risco',
    interpretation: 'VaR 2% significa que 1 em cada 20 dias você pode perder mais de 2%.'
  },
  var_99: {
    term: 'Value at Risk (99%)',
    definition: 'Perda máxima diária esperada com 99% de confiança. Mais conservador que VaR95.',
    formula: 'Percentil 1 dos retornos diários',
    interpretation: 'Captura eventos de cauda mais extremos que o VaR95.'
  },
  cvar_95: {
    term: 'CVaR / Expected Shortfall (95%)',
    definition: 'Perda média nos piores 5% dos casos. Captura melhor o risco de cauda que o VaR.',
    formula: 'E[Perda | Perda > VaR95]',
    benchmark: 'Deve ser ~1.5x o VaR para distribuição normal',
    interpretation: 'Se VaR é 2%, CVaR pode ser 3% - a perda média nos dias ruins.'
  },
  cvar_99: {
    term: 'CVaR / Expected Shortfall (99%)',
    definition: 'Perda média nos piores 1% dos casos. A medida de risco de cauda mais conservadora.',
    formula: 'E[Perda | Perda > VaR99]',
    interpretation: 'Use para stress testing de cenários de pior caso.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // ESTATÍSTICAS DE DISTRIBUIÇÃO
  // ═══════════════════════════════════════════════════════════════════════════
  skewness: {
    term: 'Assimetria (Skewness)',
    definition: 'Assimetria da distribuição de retornos. Positivo = mais ganhos extremos, Negativo = mais perdas extremas.',
    formula: 'E[(X-μ)³] / σ³',
    benchmark: 'Positivo é preferível (assimetria à direita)',
    interpretation: 'Assimetria negativa é comum em venda de opções, positiva em compra.'
  },
  kurtosis: {
    term: 'Curtose Excessiva',
    definition: 'Espessura das caudas comparada à distribuição normal. Positivo = caudas gordas (mais eventos extremos).',
    formula: 'E[(X-μ)⁴] / σ⁴ - 3',
    benchmark: '>0 é comum em finanças (caudas gordas)',
    interpretation: 'Alta curtose significa que eventos "cisne negro" são mais prováveis do que o normal sugere.'
  },
  tail_ratio: {
    term: 'Razão de Cauda',
    definition: 'Razão entre ganho do percentil 95 e perda do percentil 5. Mede extremos de alta vs baixa.',
    formula: 'Percentil95 / |Percentil5|',
    benchmark: '>1.0 significa caudas de alta maiores que de baixa',
    interpretation: 'Razão 1.5 significa ganhos extremos 50% maiores que perdas extremas.'
  },
  stability: {
    term: 'Estabilidade da Série',
    definition: 'R² da regressão linear nos retornos acumulados. Mede quão consistentemente a estratégia cresce.',
    formula: 'R² de RetornosAcumulados ~ Tempo',
    benchmark: '>0.9 muito estável, <0.7 irregular',
    interpretation: 'Alta estabilidade = composição consistente. Baixa = curva de equity volátil.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MÉTRICAS DE DRAWDOWN
  // ═══════════════════════════════════════════════════════════════════════════
  longest_dd: {
    term: 'Duração do Maior Drawdown',
    definition: 'Número máximo de dias abaixo do pico anterior. Quanto tempo até a recuperação.',
    benchmark: '<180 dias é preferível',
    interpretation: '400 dias submerso é psicologicamente brutal, mesmo que eventualmente lucrativo.'
  },
  avg_dd_duration: {
    term: 'Duração Média de Drawdown',
    definition: 'Tempo médio para se recuperar de drawdowns. Menor é melhor para eficiência de capital.',
    benchmark: '<60 dias é bom',
    interpretation: 'Recuperações rápidas significam que o capital não fica preso em posições perdedoras.'
  },
  time_underwater: {
    term: 'Tempo Submerso',
    definition: 'Percentual do tempo abaixo do pico anterior. Com que frequência você está em drawdown.',
    benchmark: '<50% é bom',
    interpretation: '70% submerso significa que você geralmente está perdendo - difícil de manter.'
  },
  gain_to_pain: {
    term: 'Razão Ganho/Dor',
    definition: 'Soma de todos os retornos dividida pela soma absoluta dos retornos negativos. Razão geral de recompensa/sofrimento.',
    formula: 'Σ(Retornos) / Σ|RetornosNegativos|',
    benchmark: '>1.0 necessário para lucrar, >2.0 é excelente',
    interpretation: 'G2P 1.5 significa que você ganha 1.5x mais do que perde no total.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MÉTRICAS DE VALIDAÇÃO
  // ═══════════════════════════════════════════════════════════════════════════
  pbo: {
    term: 'Probabilidade de Overfitting (PBO)',
    definition: 'Probabilidade estatística de que a estratégia está ajustada a ruído histórico em vez de padrões genuínos.',
    formula: 'Baseado na distribuição de degradação CPCV',
    benchmark: '<15% é seguro, <10% é excelente',
    interpretation: 'PBO 8% = 8% de chance de ser apenas sorte. 30% = muito preocupante.'
  },
  dsr: {
    term: 'Sharpe Ratio Deflacionado (DSR)',
    definition: 'Sharpe Ratio ajustado para viés de múltiplos testes. Compensa testar muitas estratégias.',
    formula: 'SR × fator_correção(tentativas)',
    benchmark: '>0.5 após deflação é bom',
    interpretation: 'Se você testou 100 estratégias, DSR ajusta para a "melhor" ser sortuda.'
  },
  t_stat: {
    term: 'Estatística T',
    definition: 'Significância estatística do Sharpe Ratio. Maior significa menos provável de ser por acaso.',
    formula: 'SR × √(n/252)',
    benchmark: '≥2.0 para 95% de confiança',
    interpretation: 't-stat 2.5 significa <1% de chance desse Sharpe ser sorte aleatória.'
  },
  p_value: {
    term: 'Valor-P',
    definition: 'Probabilidade de que os retornos observados poderiam ocorrer por acaso. Menor é mais significativo.',
    formula: '2 × (1 - Φ(|t-stat|))',
    benchmark: '<0.05 para 95% de confiança',
    interpretation: 'valor-p 0.01 significa apenas 1% de chance de ser sorte.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // ANÁLISE WALK-FORWARD
  // ═══════════════════════════════════════════════════════════════════════════
  wfa: {
    term: 'Análise Walk-Forward (WFA)',
    definition: 'Técnica padrão-ouro de validação que divide os dados em janelas sequenciais. Cada janela treina (IS) em dados passados e testa (OOS) no período seguinte - simulando exatamente como você operaria ao vivo.',
    benchmark: 'WFE > 50% indica estratégia robusta',
    interpretation: 'Baseado no trabalho de Robert Pardo. Se uma estratégia mantém boa performance em múltiplos períodos OOS, você pode confiar que ela funcionará ao vivo. Se só vai bem no IS, ela está apenas "decorando" os dados.'
  },
  is_oos: {
    term: 'In-Sample (IS) / Out-of-Sample (OOS)',
    definition: 'IS = período de TREINO onde a estratégia "aprende" os parâmetros. OOS = período de TESTE com dados que a estratégia NUNCA viu - simula trading real.',
    benchmark: 'Ideal: OOS retém >50% da performance IS',
    interpretation: 'Se a estratégia vai bem no IS mas mal no OOS, ela "decorou" os dados históricos em vez de aprender padrões reais. A performance OOS é o que você terá ao operar de verdade.'
  },
  degradation_ratio: {
    term: 'Taxa de Degradação',
    definition: 'Quanto da performance IS é mantida no OOS. Mede severidade do overfitting.',
    formula: 'Sharpe_OOS / Sharpe_IS × 100%',
    benchmark: '>50% é robusto, <30% é preocupante',
    interpretation: '70% de degradação = OOS mantém 70% da performance IS. Bom sinal.'
  },
  wfe: {
    term: 'WFE (Walk-Forward Efficiency)',
    definition: 'Métrica de Robert Pardo que mede quanto da performance In-Sample se mantém Out-of-Sample. Indica robustez da estratégia.',
    formula: 'WFE = Média(Sharpe_OOS) / Média(Sharpe_IS) × 100%',
    benchmark: '>50% é robusto, 30-50% zona de alerta, <30% overfit',
    interpretation: 'WFE 80% significa que a estratégia mantém 80% da performance quando testada em dados novos. É a métrica mais importante para validar se a estratégia vai funcionar ao vivo.'
  },
  consistency_score: {
    term: 'Consistência Temporal',
    definition: 'Percentual de janelas Walk-Forward onde o período OOS foi lucrativo. Mede estabilidade da estratégia ao longo do tempo.',
    formula: 'Períodos OOS Positivos / Total de Períodos × 100%',
    benchmark: '>60% é bom, >80% é excelente',
    interpretation: 'Consistência 75% = a estratégia foi lucrativa em 3 de cada 4 períodos testados. Baixa consistência indica que a estratégia depende de condições específicas de mercado.'
  },
  wfa_window: {
    term: 'Tamanho da Janela WFA',
    definition: 'Duração do período de treino in-sample. Maior = mais dados mas menos testes.',
    benchmark: '12-24 meses é típico',
    interpretation: 'Equilíbrio entre dados suficientes para aprender e testes suficientes para validar.'
  },
  wfa_step: {
    term: 'Passo do WFA',
    definition: 'Quanto avançar entre janelas WFA. Menor = mais testes mas mais sobreposição.',
    benchmark: '3-6 meses é típico',
    interpretation: 'Passos menores dão mais dados mas podem ser correlacionados.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // ANÁLISE DE REGIMES
  // ═══════════════════════════════════════════════════════════════════════════
  regime_analysis: {
    term: 'Análise de Regimes',
    definition: 'Decomposição da performance da estratégia por condições de mercado. Classifica cada dia em um regime (Trend + Volatilidade) e calcula métricas separadas para cada um.',
    benchmark: 'Estratégia robusta tem Sharpe positivo em todos os regimes',
    interpretation: 'Responde a pergunta crucial: "Em quais condições de mercado minha estratégia funciona?" Uma estratégia com Sharpe 1.5 agregado mas -0.5 em Bear+HighVol vai te destruir em crashes.'
  },
  regime_heatmap: {
    term: 'Matriz de Performance 3×5',
    definition: 'Visualização em grid mostrando a performance em cada uma das 15 combinações de regime (3 trends × 5 níveis de volatilidade). Cores indicam qualidade do Sharpe.',
    interpretation: 'Verde = regime favorável (Sharpe > 1). Vermelho = regime desfavorável (Sharpe < 0). Clique em uma célula para ver detalhes e filtrar a timeline.'
  },
  regime_timeline: {
    term: 'Timeline de Regimes',
    definition: 'Visualização temporal mostrando as transições entre regimes ao longo do backtest. Cores representam a combinação trend+volatilidade.',
    interpretation: 'Identifica padrões de duração de regimes e transições. Regimes curtos frequentes podem indicar mercado instável.'
  },
  trend_state: {
    term: 'Estado de Tendência',
    definition: 'Classificação do mercado baseada na direção dos retornos acumulados. Usa regressão linear normalizada pela volatilidade.',
    formula: 'slope = Cov(t, cumret) / Var(t), normalizado por vol',
    benchmark: 'Uptrend, Sideways, ou Downtrend',
    interpretation: 'Uptrend = mercado subindo consistentemente. Downtrend = queda. Sideways = sem direção clara.'
  },
  vol_quantile: {
    term: 'Quantil de Volatilidade',
    definition: 'Classificação da volatilidade atual em quintis (Q1-Q5) baseado no histórico. Usa janela expansiva para evitar look-ahead bias.',
    formula: 'Percentil da vol atual vs histórico → Q1 (0-20%) a Q5 (80-100%)',
    benchmark: 'Q1 = muito calmo, Q3 = normal, Q5 = muito volátil',
    interpretation: 'Q1-Q2 geralmente são melhores para estratégias de momentum. Q4-Q5 podem ser melhores para mean-reversion.'
  },
  allocation_recommendation: {
    term: 'Recomendação de Alocação',
    definition: 'Sugestão de sizing baseada na performance histórica da estratégia no regime atual. Considera Sharpe e quantidade de dados disponíveis.',
    benchmark: '100% = Full, 75% = Moderada, 50% = Reduzida, 25% = Cautela',
    interpretation: 'Alta confiança requer Sharpe > 1 E pelo menos 50 dias de dados no regime. Poucos dados = baixa confiança mesmo com bom Sharpe.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // SIMULAÇÃO MONTE CARLO
  // ═══════════════════════════════════════════════════════════════════════════
  monte_carlo: {
    term: 'Simulação Monte Carlo',
    definition: 'Técnica estatística que usa amostragem aleatória para estimar distribuições de resultados possíveis.',
    interpretation: 'Mostra o leque de futuros possíveis, não apenas um caminho de backtest.'
  },
  bootstrap: {
    term: 'Reamostragem Bootstrap',
    definition: 'Técnica que embaralha retornos históricos para gerar cenários alternativos. Preserva propriedades estatísticas.',
    interpretation: 'Cria milhares de curvas de equity possíveis a partir dos mesmos retornos.'
  },
  block_size: {
    term: 'Tamanho do Bloco (Bootstrap)',
    definition: 'Tamanho dos blocos de retorno na reamostragem. Preserva autocorrelação dentro dos blocos.',
    benchmark: '5-21 dias é típico',
    interpretation: 'Bloco=1 é IID (independente). Bloco=21 preserva padrões mensais.'
  },
  confidence_interval: {
    term: 'Intervalo de Confiança',
    definition: 'Faixa de valores que provavelmente contém o parâmetro verdadeiro com probabilidade especificada.',
    benchmark: 'IC 95% é padrão',
    interpretation: 'IC 95% [0.8, 1.2] significa que o Sharpe verdadeiro provavelmente está entre 0.8 e 1.2.'
  },
  percentile_p5: {
    term: 'P5 (Percentil 5)',
    definition: 'Valor abaixo do qual 5% dos resultados caem. Representa cenário pessimista.',
    interpretation: 'Use P5 para planejamento pessimista. 95% dos resultados são melhores que isso.'
  },
  percentile_p50: {
    term: 'P50 (Mediana)',
    definition: 'Valor onde metade dos resultados está acima e metade abaixo. Estimativa central robusta.',
    interpretation: 'Mais robusto que a média para distribuições assimétricas.'
  },
  percentile_p95: {
    term: 'P95 (Percentil 95)',
    definition: 'Valor abaixo do qual 95% dos resultados caem. Representa cenário otimista.',
    interpretation: 'Não planeje com P95 - só 5% de chance de alcançá-lo.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // ALGORITMO GENÉTICO / EVOLUÇÃO
  // ═══════════════════════════════════════════════════════════════════════════
  genetic_algorithm: {
    term: 'Algoritmo Genético',
    definition: 'Otimização inspirada na evolução. Estratégias "cruzam" e mutam, os mais aptos sobrevivem.',
    interpretation: 'Explora vastos espaços de estratégias eficientemente através de seleção natural.'
  },
  population: {
    term: 'Tamanho da População',
    definition: 'Número de estratégias evoluindo simultaneamente. Maior = mais diversidade mas mais lento.',
    benchmark: '100-200 é típico',
    interpretation: 'População 100 significa 100 estratégias competindo a cada geração.'
  },
  generation: {
    term: 'Geração',
    definition: 'Um ciclo de avaliação, seleção e cruzamento. A evolução progride através de gerações.',
    benchmark: '30-100 gerações é típico',
    interpretation: 'Cada geração deve mostrar melhoria no melhor/média de fitness.'
  },
  fitness: {
    term: 'Função de Fitness',
    definition: 'Score que determina quais estratégias sobrevivem e reproduzem. Geralmente Sharpe ou métrica composta.',
    interpretation: 'Maior fitness = melhor estratégia. A evolução maximiza isso.'
  },
  pareto_frontier: {
    term: 'Fronteira de Pareto',
    definition: 'Conjunto de estratégias onde nenhuma é melhor em todos os objetivos. Representa trade-offs ótimos.',
    interpretation: 'Todas as estratégias da fronteira são escolhas válidas dependendo das suas prioridades.'
  },
  convergence: {
    term: 'Convergência',
    definition: 'Quando a população para de melhorar significativamente. Pode indicar ótimo encontrado ou travado.',
    interpretation: 'Convergência precoce pode significar ótimo local. Tente população maior.'
  },
  seeds: {
    term: 'Seeds (Sementes Aleatórias)',
    definition: 'Sementes de números aleatórios para reprodutibilidade. Cada seed gera uma sequência diferente de mutações e crossovers no algoritmo genético.',
    benchmark: '3 seeds mínimo para produção, 5+ para validação institucional',
    interpretation: 'Estratégia que performa bem em múltiplas seeds é robusta. Se só funciona com 1 seed, é sorte, não skill.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // TESTE DE ESTRESSE
  // ═══════════════════════════════════════════════════════════════════════════
  stress_test: {
    term: 'Teste de Estresse',
    definition: 'Testa a estratégia contra cenários históricos extremos como crashes e alta volatilidade.',
    interpretation: 'Estratégias que passam em testes de estresse têm mais chance de sobreviver a crises reais.'
  },
  stress_scenario: {
    term: 'Cenário de Estresse',
    definition: 'Condição de mercado específica usada para teste de estresse (ex: crise 2008, crash COVID).',
    interpretation: 'Cada cenário testa um tipo diferente de estresse de mercado.'
  },
  stress_degradation: {
    term: 'Degradação sob Estresse',
    definition: 'Quanto a performance (Sharpe) cai sob condições de estresse vs condições normais. Mostra resiliência da estratégia.',
    formula: '(Sharpe_Estresse / Sharpe_Base) × 100%',
    benchmark: '>50% retenção é bom, >70% é excelente',
    interpretation: 'Uma barra de 70% significa que a estratégia mantém 70% do seu Sharpe original sob estresse. Barras curtas indicam vulnerabilidade.'
  },
  sharpe_impact: {
    term: 'Impacto no Sharpe',
    definition: 'Visualização de como cada cenário de estresse afeta o Sharpe Ratio. A barra mostra a porcentagem do Sharpe original que sobrevive.',
    benchmark: 'Barra maior = mais resiliência',
    interpretation: 'A linha amarela marca o threshold mínimo. Estratégias que ficam acima passam no cenário.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // GATES DE VALIDAÇÃO
  // ═══════════════════════════════════════════════════════════════════════════
  gates_passed: {
    term: 'Gates de Validação Aprovados',
    definition: 'Se a estratégia passou todos os thresholds mínimos para estar pronta para produção.',
    interpretation: 'Gates incluem Sharpe, PBO, testes de estresse e requisitos de consistência.'
  },
  validated: {
    term: 'Status Validado',
    definition: 'Estratégia passou em todos os critérios institucionais de validação e está pronta para produção.',
    interpretation: 'Estratégias validadas têm alta confiança de performance no mundo real.'
  },
  research: {
    term: 'Status Pesquisa',
    definition: 'Estratégia mostra promessa mas ainda não passou em todos os gates de validação.',
    interpretation: 'Precisa de mais testes ou ajuste de parâmetros antes de operar.'
  },
  cpcv: {
    term: 'CPCV (Validação Cruzada Combinatória Purgada)',
    definition: 'Validação avançada que gera múltiplas combinações IS/OOS para calcular PBO estatisticamente.',
    interpretation: 'Mais rigoroso que divisão simples treino/teste. Padrão da indústria.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // EXECUÇÃO E CUSTOS
  // ═══════════════════════════════════════════════════════════════════════════
  net_vs_gross: {
    term: 'NET vs GROSS',
    definition: 'NET = após todos os custos (taxas, slippage). GROSS = antes dos custos. Sempre use NET para decisões.',
    interpretation: 'Um Sharpe 2.0 GROSS pode ser 0.5 NET após custos - grande diferença!'
  },
  slippage: {
    term: 'Slippage',
    definition: 'Diferença entre preço esperado e preço real de execução. Causado por latência e impacto de mercado.',
    benchmark: '1-5 bps para ações líquidas',
    interpretation: 'Estratégias de alta frequência são muito sensíveis ao slippage.'
  },
  delay_bars: {
    term: 'Barras de Delay',
    definition: 'Número de barras entre sinal e execução. Simula latência do mundo real.',
    benchmark: '1 barra é conservador',
    interpretation: 'Delay=0 assume execução instantânea - irrealista para a maioria dos traders.'
  },
  turnover: {
    term: 'Turnover Anual',
    definition: 'Quantas vezes o portfólio é completamente trocado por ano. Maior = mais custos.',
    formula: 'Valor total operado / Valor médio do portfólio',
    benchmark: '<12x para eficiência de custos',
    interpretation: 'Turnover 24x significa operar duas vezes por mês em média.'
  },
  capacity: {
    term: 'Capacidade da Estratégia',
    definition: 'Capital máximo que a estratégia pode gerir antes que impacto de mercado degrade retornos.',
    interpretation: 'Capacidade R$10M significa que performance degrada acima desse AUM.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PORTFÓLIO E COMPARAÇÃO
  // ═══════════════════════════════════════════════════════════════════════════
  correlation_matrix: {
    term: 'Matriz de Correlação',
    definition: 'Tabela mostrando correlações par-a-par entre retornos de estratégias. Baixa correlação = melhor diversificação.',
    benchmark: '<0.5 para boa diversificação',
    interpretation: 'Estratégias com correlação 0.2 fornecem melhor performance combinada.'
  },
  diversification_ratio: {
    term: 'Razão de Diversificação',
    definition: 'Razão entre soma das volatilidades individuais e volatilidade do portfólio. Mede benefício da diversificação.',
    formula: 'Σ(pesos × volatilidades) / VolatilidadePortfólio',
    benchmark: '>1.5 é boa diversificação',
    interpretation: 'Razão 2.0 significa que a diversificação corta o risco pela metade.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DIVERSOS
  // ═══════════════════════════════════════════════════════════════════════════
  best_day: {
    term: 'Melhor Retorno Diário',
    definition: 'Melhor retorno em um único dia no período de backtest.',
    interpretation: 'Mostra potencial de alta. Valores muito altos podem indicar dependência de outliers.'
  },
  worst_day: {
    term: 'Pior Retorno Diário',
    definition: 'Pior retorno em um único dia no período de backtest. Seu cenário de pesadelo.',
    interpretation: 'Você consegue emocionalmente lidar com essa perda em um único dia?'
  },
  best_month: {
    term: 'Melhor Retorno Mensal',
    definition: 'Melhor retorno em um único mês no período de backtest.',
    interpretation: 'Meses fortes não devem ser essenciais para a lucratividade geral.'
  },
  worst_month: {
    term: 'Pior Retorno Mensal',
    definition: 'Pior retorno em um único mês no período de backtest.',
    interpretation: 'A maioria dos investidores avalia mensalmente - você consegue explicar isso para clientes?'
  },
  rolling_sharpe: {
    term: 'Sharpe Móvel (252 dias)',
    definition: 'Sharpe calculado nos últimos 252 dias úteis, rolando para frente. Mostra estabilidade.',
    interpretation: 'Oscilações grandes no Sharpe móvel indicam sensibilidade a regimes de mercado.'
  },
  rolling_volatility: {
    term: 'Volatilidade Móvel (252 dias)',
    definition: 'Volatilidade calculada no último ano, rolando para frente. Mostra variação de risco.',
    interpretation: 'Picos de volatilidade móvel durante crises são normais mas informativos.'
  },
};

// =============================================================================
// TOOLTIPS LEGADO (para página Cockpit)
// =============================================================================

export const TOOLTIPS: Record<string, TooltipContent> = {
  // Orçamento de Computação
  max_runtime: {
    what: 'Tempo máximo que o sistema usará para descobrir estratégias',
    impact: 'Mais tempo = mais estratégias avaliadas = maior chance de encontrar boas.',
    when: 'Aumente para exploração profunda, diminua para testes rápidos',
    example: '15 min para exploração inicial. 1h para análise profunda.',
  },
  population_size: {
    what: 'Número de estratégias evoluindo simultaneamente',
    impact: 'População maior = mais diversidade genética = encontra soluções em espaços maiores.',
    when: 'Aumente se estratégias convergirem muito cedo',
    example: '100 para produção, 200 para exploração exaustiva',
  },
  max_generations: {
    what: 'Número máximo de ciclos evolutivos',
    impact: 'Mais gerações = mais refinamento. Retornos decrescentes após ~50.',
    when: 'Deixe o padrão ou aumente se o tempo permitir',
    example: '50 gerações geralmente suficientes para convergência',
  },
  workers: {
    what: 'Threads paralelas para avaliação de estratégias',
    impact: 'Mais workers = mais rápido, mas usa mais CPU/memória. Ideal: núcleos físicos.',
    when: 'Reduza se o sistema ficar lento para outras tarefas',
    example: '8 workers em CPU 8-core usa 100% da capacidade',
  },
  seeds: {
    what: 'Seeds para reprodutibilidade de experimentos',
    impact: 'Múltiplas seeds = resultados mais robustos (menos dependência de sorte).',
    when: 'Use 3-5 seeds para validação institucional',
    example: '3 seeds = 3 execuções independentes, resultado é a média',
  },
  
  // Gates
  min_oos_sharpe: {
    what: 'Sharpe Ratio mínimo no período Out-of-Sample',
    impact: 'Gate filtrando estratégias com performance insuficiente.',
    when: 'Ajuste baseado no benchmark. Mercados mais voláteis podem ter thresholds menores.',
    example: 'Sharpe 0.5 = 50% mais retorno que risco. 1.0 = excelente.',
  },
  max_pbo: {
    what: 'Probabilidade de Overfitting no Backtest',
    impact: 'Mede chance da estratégia ser "sortuda" vs genuinamente boa.',
    when: 'Mantenha ≤0.15 para estratégias de produção',
    example: 'PBO 0.08 = 8% de chance de overfitting. 0.30 = preocupante.',
  },
  min_stress_passed: {
    what: 'Testes de estresse mínimos que a estratégia deve passar',
    impact: 'Testa robustez em cenários históricos extremos.',
    when: 'Use 4+ para produção. 0 para exploração rápida.',
    example: '4 de 8 testes = estratégia sobrevive à maioria dos crashes',
  },
  stress_testing: {
    what: 'Simula cenários extremos de mercado',
    impact: 'Testa cada estratégia contra volatilidade 2x, gaps de preço, drawdowns prolongados.',
    when: 'Sempre habilite para produção. Desabilite só para testes rápidos.',
    example: 'Estratégia que passa nos stress tests sobreviveu a 2008 e COVID',
  },
  
  // Ranking
  ranking_institutional: {
    what: 'Ranking ponderado multi-critério (padrão institucional)',
    impact: 'Pondera Sharpe OOS (40%), PBO (25%), stress (20%), gates (15%).',
    when: 'Use como padrão para produção',
    example: 'Sharpe 1.2 + PBO 0.05 pontua mais que Sharpe 1.5 + PBO 0.25',
  },
  ranking_pareto: {
    what: 'Fronteira de Pareto (estratégias não-dominadas)',
    impact: 'Mostra estratégias ótimas em pelo menos uma dimensão.',
    when: 'Use para explorar trade-offs (ex: risco vs retorno)',
    example: '5 estratégias na fronteira = 5 escolhas válidas dependendo da preferência',
  },
  ranking_sharpe: {
    what: 'Ordena apenas por Sharpe Ratio OOS NET',
    impact: 'Simples mas pode premiar overfitting. Ignora PBO e stress.',
    when: 'Use para análise inicial ou quando PBO já validado',
    example: 'Top 1 por Sharpe pode ter PBO alto - verifique!',
  },
  ranking_riskadjusted: {
    what: 'Sharpe dividido por Drawdown Máximo',
    impact: 'Penaliza estratégias com quedas grandes mesmo com bom Sharpe.',
    when: 'Use se drawdown é prioridade (aversão a perdas)',
    example: 'Sharpe 1.0 com DD 10% > Sharpe 1.5 com DD 30%',
  },
};

// =============================================================================
// COMPONENTS
// =============================================================================

/**
 * QuickTooltip - Compact inline tooltip for quant terms
 * Shows definition, formula, benchmark on hover with (?) icon
 * Auto-detects best position based on viewport bounds
 */
export function QuickTooltip({ termKey, position: preferredPosition = 'top', size = 'sm' }: QuickTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [actualPosition, setActualPosition] = useState<'top' | 'bottom' | 'left' | 'right'>(preferredPosition);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const content = QUANT_TOOLTIPS[termKey];
  
  // Auto-detect best position based on viewport bounds
  useEffect(() => {
    if (isOpen && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      const tooltipHeight = 180; // approximate tooltip height
      const tooltipWidth = 288; // w-72 = 18rem = 288px
      
      // Check if there's room at preferred position
      if (preferredPosition === 'top' && rect.top < tooltipHeight + 20) {
        setActualPosition('bottom');
      } else if (preferredPosition === 'bottom' && window.innerHeight - rect.bottom < tooltipHeight + 20) {
        setActualPosition('top');
      } else if (preferredPosition === 'left' && rect.left < tooltipWidth + 20) {
        setActualPosition('right');
      } else if (preferredPosition === 'right' && window.innerWidth - rect.right < tooltipWidth + 20) {
        setActualPosition('left');
      } else {
        setActualPosition(preferredPosition);
      }
    }
  }, [isOpen, preferredPosition]);
  
  if (!content) return null;
  
  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  
  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-b-transparent border-t-slate-800',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-t-transparent border-b-slate-800',
    left: 'left-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-r-transparent border-l-slate-800',
    right: 'right-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-l-transparent border-r-slate-800',
  };
  
  const sizeClasses = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4';
  
  return (
    <span className="relative inline-flex items-center ml-1">
      <button
        ref={buttonRef}
        type="button"
        className={`inline-flex items-center justify-center ${sizeClasses} text-cyan-400/70 hover:text-cyan-400 transition-colors`}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onClick={(e) => { e.stopPropagation(); setIsOpen(!isOpen); }}
        aria-label={`What is ${content.term}?`}
      >
        <Info className={sizeClasses} />
      </button>
      
      {isOpen && (
        <div className={`absolute z-[100] ${positionClasses[actualPosition]} w-72 pointer-events-none`}>
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl shadow-black/40 overflow-hidden">
            {/* Header */}
            <div className="px-3 py-2 bg-slate-700/50 border-b border-slate-600">
              <div className="font-semibold text-sm text-white">{content.term}</div>
            </div>
            
            {/* Content */}
            <div className="px-3 py-2.5 space-y-2">
              <p className="text-xs text-slate-300 leading-relaxed">{content.definition}</p>
              
              {content.formula && (
                <div className="flex items-start gap-2 text-xs">
                  <span className="text-slate-500 shrink-0">Formula:</span>
                  <code className="text-cyan-400 font-mono text-[11px]">{content.formula}</code>
                </div>
              )}
              
              {content.benchmark && (
                <div className="flex items-start gap-2 text-xs">
                  <span className="text-slate-500 shrink-0">Benchmark:</span>
                  <span className="text-emerald-400">{content.benchmark}</span>
                </div>
              )}
              
              {content.interpretation && (
                <div className="pt-1.5 border-t border-slate-700">
                  <p className="text-[11px] text-slate-400 italic leading-relaxed">
                    {content.interpretation}
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {/* Arrow */}
          <div className={`absolute w-0 h-0 border-[6px] ${arrowClasses[actualPosition]}`} />
        </div>
      )}
    </span>
  );
}

/**
 * TermWithTooltip - Label text with inline tooltip
 */
interface TermWithTooltipProps {
  termKey: keyof typeof QUANT_TOOLTIPS;
  label?: string;
  className?: string;
}

export function TermWithTooltip({ termKey, label, className = '' }: TermWithTooltipProps) {
  const content = QUANT_TOOLTIPS[termKey];
  const displayLabel = label || content?.term || termKey;
  
  return (
    <span className={`inline-flex items-center ${className}`}>
      {displayLabel}
      <QuickTooltip termKey={termKey} />
    </span>
  );
}

export function TooltipInfo({ content, children }: TooltipInfoProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative inline-block">
      <button
        type="button"
        className="inline-flex items-center justify-center w-4 h-4 ml-1 text-xs text-cyan-400 hover:text-cyan-300 rounded-full border border-cyan-400/30 hover:border-cyan-400/60 transition-colors"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="More info"
      >
        ?
      </button>
      
      {isOpen && (
        <div className="absolute z-50 w-80 p-4 mt-2 left-0 bg-slate-900 border border-cyan-500/30 rounded-lg shadow-xl shadow-cyan-500/10">
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-cyan-400 font-mono text-xs uppercase tracking-wider">What</span>
              <p className="text-slate-200 mt-1">{content.what}</p>
            </div>
            
            <div>
              <span className="text-amber-400 font-mono text-xs uppercase tracking-wider">Impact</span>
              <p className="text-slate-300 mt-1">{content.impact}</p>
            </div>
            
            <div>
              <span className="text-emerald-400 font-mono text-xs uppercase tracking-wider">When to adjust</span>
              <p className="text-slate-300 mt-1">{content.when}</p>
            </div>
            
            <div className="pt-2 border-t border-slate-700">
              <span className="text-slate-500 font-mono text-xs uppercase tracking-wider">Example</span>
              <p className="text-slate-400 mt-1 italic">{content.example}</p>
            </div>
          </div>
          
          {/* Arrow */}
          <div className="absolute -top-2 left-4 w-4 h-4 bg-slate-900 border-l border-t border-cyan-500/30 transform rotate-45" />
        </div>
      )}
      
      {children}
    </div>
  );
}

export function SimpleTooltip({ text, children }: SimpleTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      {children}
      
      {isOpen && (
        <div className="absolute z-50 px-3 py-2 mt-1 left-1/2 transform -translate-x-1/2 bg-slate-800 border border-slate-600 rounded-lg text-[11px] leading-relaxed text-slate-200 max-w-[280px] text-left whitespace-normal shadow-lg">
          {text}
          <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-slate-800 border-l border-t border-slate-600 rotate-45" />
        </div>
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENT
// =============================================================================

interface InfoIconProps {
  tooltipKey: keyof typeof TOOLTIPS;
}

export function InfoIcon({ tooltipKey }: InfoIconProps) {
  const content = TOOLTIPS[tooltipKey];
  if (!content) return null;
  
  return <TooltipInfo content={content} />;
}

export default TooltipInfo;




