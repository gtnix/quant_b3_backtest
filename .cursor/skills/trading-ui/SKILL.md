---
name: trading-ui
description: UI/UX para sistemas de trading - design system, padrões de interação e componentes
---

# Trading UI Expert

Especialista em interfaces de usuário para sistemas de trading e backtesting quantitativo.

## Quando Usar

- Desenvolvimento de novas telas do dashboard
- Refatoração de componentes existentes
- Problemas de usabilidade ou feedback visual
- Escala de dados (paginação, virtualização)

## Design System

### Cores Semânticas

```typescript
// tailwind.config.js - cores do sistema
const colors = {
  // Estados de P&L
  profit: '#22c55e',      // Verde - ganhos, sucesso
  loss: '#ef4444',        // Vermelho - perdas, erro
  neutral: '#94a3b8',     // Cinza - neutro, inativo
  
  // Métricas
  'accent-cyan': '#22d3ee',    // Métricas primárias (Sharpe, CAGR)
  'accent-amber': '#f59e0b',   // Destaques, mineração
  'accent-violet': '#8b5cf6',  // Configuração, universo
  
  // Severidade
  warning: '#f59e0b',     // Amarelo - atenção
  info: '#3b82f6',        // Azul - informativo
  
  // Terminal
  'terminal-bg': '#0f172a',
  'terminal-surface': '#1e293b',
  'terminal-border': '#334155',
  'terminal-muted': '#64748b',
};
```

### Estados Visuais

Todo componente interativo deve ter estes estados:

1. **Default** - Estado normal
2. **Hover** - Mouse sobre (transição 200ms)
3. **Active/Pressed** - Clicado (scale-95)
4. **Loading** - Spinner ou skeleton
5. **Disabled** - Opacidade 50%, cursor not-allowed
6. **Error** - Borda vermelha, ícone de erro
7. **Success** - Borda verde, ícone de check

## Padrões de Interação

### 1. Feedback Imediato

**NUNCA** deixe o usuário sem feedback após uma ação:

```tsx
// CORRETO - feedback completo
const handleStop = async () => {
  setStopping(true);
  try {
    await api.stop();
    toast.success('Mineração parada');
  } catch (err) {
    toast.error('Falha ao parar: ' + err.message);
  } finally {
    setStopping(false);
  }
};
```

### 2. Estados de Transição

Para operações que demoram > 500ms, mostre estado intermediário:

```tsx
type ActionState = 'idle' | 'loading' | 'success' | 'error';
```

### 3. Confirmação de Operações Destrutivas

Operações que não podem ser desfeitas precisam de confirmação modal.

## Padrões de Dados

### 1. Listas Pequenas (< 20 itens)
Renderizar diretamente sem paginação.

### 2. Listas Médias (20-100 itens)
Usar paginação tradicional ou "Mostrar mais".

### 3. Listas Grandes (> 100 itens)
**OBRIGATÓRIO**: Usar virtualização (react-window).

### 4. Seções Colapsáveis
Para agrupar conteúdo extenso, usar accordion.

## Componentes Padrão

### StatusBadge com Tooltip

Para exibir estados com explicação clara:

```tsx
interface StatusBadgeProps {
  status: 'valid' | 'warning' | 'error' | 'pending';
  label: string;
  details?: string[];  // Lista de issues para tooltip
}
```

## Checklists por Página

### MinerControl
- [ ] Botão PARAR mostra estado "Parando..." com spinner
- [ ] Feedback visual quando mineração inicia/para
- [ ] Contagem de tempo atualiza a cada segundo

### HallOfFame
- [ ] Badge de "Incompleto" mostra tooltip com lista de issues
- [ ] Métricas inválidas destacadas em cor de warning

### StrategySelector (Catálogo)
- [ ] Famílias colapsadas por padrão
- [ ] Contador de selecionadas por família
- [ ] Performance suave com 100+ estratégias

### ConfigUniverse
- [ ] Presets rápidos no topo
- [ ] Incompatibilidades explicadas em tooltip

### ConfigTrading
- [ ] Simulador de custo mostra impacto por trade

## Regras Críticas

### 1. NUNCA mostrar dados técnicos sem explicação

Termos que SEMPRE precisam de tooltip:
- Sharpe Ratio, CAGR, Drawdown
- PBO, DSR, OOS
- Walk-Forward, Monte Carlo
- Kelly Fraction, Slippage, bps

### 2. SEMPRE usar português para labels

### 3. SEMPRE mostrar loading states

### 4. Números formatados conforme contexto

```typescript
format.percent(0.1234) → "12.34%"
format.sharpe(1.2345) → "1.2345"
format.brl(10000) → "R$ 10.000,00"
format.duration(3665) → "1h 1m 5s"
```

## Arquivos de Referência

- Design tokens: dashboard/tailwind.config.js
- Componentes UI: dashboard/src/components/ui/
- Stores: dashboard/src/stores/
- Pages: dashboard/src/pages/
