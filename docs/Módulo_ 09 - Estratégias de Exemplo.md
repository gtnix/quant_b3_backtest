# Módulo: 09 - Estratégias de Exemplo

---

## Sumário

1. [Como Ler Este Módulo](#1-como-ler-este-modulo)
2. [Regras de Ouro para Exemplos](#2-regras-de-ouro-para-exemplos)
3. [Template de Especificação de Estratégia](#3-template-de-especificacao-de-estrategia)
4. [Estratégia 1: Tendência Diária (Swing)](#4-estrategia-1-tendencia-diaria-swing)
5. [Estratégia 2: Reversão à Média Intraday (Net Zero)](#5-estrategia-2-reversao-a-media-intraday-net-zero)
6. [Estratégia 3: Pairs/Spread Intraday (Net Zero)](#6-estrategia-3-pairsspread-intraday-net-zero)
7. [Matriz de Cobertura de Funcionalidades](#7-matriz-de-cobertura-de-funcionalidades)
8. [Modos de Falha e Diagnóstico](#8-modos-de-falha-e-diagnostico)
9. [Checklist de Aceite do Módulo](#9-checklist-de-aceite-do-modulo)
10. [Próximo Módulo Sugerido](#10-proximo-modulo-sugerido)

---

## 1. Como Ler Este Módulo

Este documento não é um tutorial de trading, mas sim um manual de engenharia. Os exemplos a seguir são "blueprints" operacionais que demonstram como usar a API de Estratégia (Módulo 08) de forma correta, performática e segura. O foco está em ilustrar a disciplina de hot-path, as barreiras anti-look-ahead e a conformidade com os contratos de determinismo e performance. Cada estratégia serve tanto como um guia de implementação quanto como um benchmark de referência para validação do sistema (Módulo 06).

## 2. Regras de Ouro para Exemplos

- **Anti-Look-Ahead Aplicado**: A lógica de sinal só pode usar dados de barras com `timestamp <= t` do evento atual.
- **Determinismo é Contrato**: Toda operação (cálculo, ordenação, emissão) deve ser executada em uma ordem estável e reprodutível.
- **RNG Somente com Seed**: Qualquer aleatoriedade deve vir de um gerador semeado pelo motor, garantindo reprodutibilidade.
- **Zero Alocação por Evento**: Os hooks `on_bar` e `on_session_close` são zonas de zero alocação. Buffers devem ser pré-alocados no `on_init`.
- **Sem I/O, Sem Logs no Hot Path**: Nenhum acesso a disco ou rede. Logging é feito via API e escrito fora do loop quente.
- **Execução é Enviesada**: A estratégia deve ser robusta o suficiente para funcionar mesmo com os custos, slippage e latência do Módulo 05.
- **Estado é Mínimo**: A estratégia deve manter o mínimo estado interno necessário, preferencialmente em estruturas de dados `data-oriented` (e.g., `Vec<T>`).
- **Falha Rápida e Clara**: Tentar uma operação proibida (I/O, alocação) deve resultar em um erro explícito, não em comportamento indefinido.

## 3. Template de Especificação de Estratégia

Cada estratégia a seguir obedece a este template padrão.

- **Objetivo e Quando Usar**: Propósito da estratégia e em que tipo de mercado/cenário ela se aplica.
- **Dados Necessários**: Granularidade (diário/intraday) e janela de dados históricos requerida.
- **Lifecycle Hooks Usados**: Quais hooks da API (M08) são implementados e por quê.
- **Sinal**: Definição formal da lógica que gera a intenção de negociação.
- **Geração de Ordens**: Contrato de como os sinais são convertidos em `OrderRequest`s de forma determinística.
- **Política de Sizing**: Regra para determinar a quantidade de cada ordem.
- **Regras de Encerramento**: Lógica para zerar posições (crítico para estratégias `net zero`).
- **Guardrails**: Limites de risco simples implementados pela estratégia.
- **Interação com Execução Enviesada**: Como a estratégia deve se comportar sabendo que a execução não é perfeita.
- **Determinismo & Performance**: Pontos de atenção específicos da estratégia para manter a reprodutibilidade e a performance.
- **Validação**: Checklist de testes específicos para garantir a corretude da estratégia.
- **Benchmark**: Qual cenário-base do Módulo 06 usar para medir a performance desta estratégia.

## 4. Estratégia 1: Tendência Diária (Swing)

- **Objetivo**: Capturar movimentos direcionais de médio prazo, mantendo posições por vários dias (swing trade).
- **Dados Necessários**: Barras diárias (OHLCV), com uma janela histórica de pelo menos 50 dias para cálculo de médias.
- **Lifecycle Hooks Usados**:
    - `on_init`: Para pré-alocar buffers para médias móveis.
    - `on_bar`: Para atualizar as médias, calcular o sinal e emitir ordens a cada novo dia.
- **Sinal**: Cruzamento de duas médias móveis simples (MMS) de fechamento. 
    - Sinal de Compra: MMS de 20 dias cruza para cima da MMS de 50 dias.
    - Sinal de Venda: MMS de 20 dias cruza para baixo da MMS de 50 dias.
- **Geração de Ordens**: Ao receber um sinal de compra e não ter posição, emite uma `OrderRequest` de compra a mercado. Ao receber um sinal de venda e ter posição comprada, emite uma ordem de venda para zerar.
- **Política de Sizing**: Alocação de uma fração fixa do NAV (e.g., 10%) por posição.
- **Regras de Encerramento**: A posição é encerrada quando o sinal reverte (cruzamento oposto das médias).
- **Guardrails**: Limite de uma posição aberta por vez.
- **Interação com Execução Enviesada**: A estratégia deve ser robusta a slippage, que pode corroer a vantagem do sinal. O `sizing` deve considerar que o preço de execução não será o preço de fechamento do dia anterior.
- **Determinismo & Performance**: As médias móveis devem ser calculadas com agregações de ponto flutuante em ordem fixa para garantir o determinismo. O estado (as médias) é atualizado incrementalmente a cada barra para performance O(1).
- **Validação**:
    - [ ] O cálculo do sinal usa apenas dados de barras até `t-1` para decidir a ordem em `t`?
    - [ ] A estratégia não gera novas ordens se já tiver uma posição aberta?
- **Benchmark**: Cenário "Diário Swing Trade" do Módulo 06.

## 5. Estratégia 2: Reversão à Média Intraday (Net Zero)

- **Objetivo**: Explorar desvios de preço de curto prazo em relação a uma média, com a obrigação de zerar todas as posições no final do dia.
- **Dados Necessários**: Barras intraday (e.g., 1-minuto), com janela de 1 sessão para cálculo de VWAP aproximado.
- **Lifecycle Hooks Usados**:
    - `on_bar`: Para calcular o sinal e emitir ordens de entrada/saída.
    - `on_session_close`: Para emitir ordens que forçam o zeramento de qualquer posição remanescente.
- **Sinal**: Desvio do preço de fechamento em relação a um VWAP (Volume-Weighted Average Price) aproximado, calculado com dados OHLCV. 
    - Sinal de Compra: `close < vwap * (1 - threshold)`.
    - Sinal de Venda: `close > vwap * (1 + threshold)`.
- **Geração de Ordens**: Emite ordens a mercado para entrar na posição quando o sinal é ativado e para sair quando o preço retorna à média.
- **Política de Sizing**: Quantidade fixa por operação (e.g., 100 unidades).
- **Regras de Encerramento**: A posição é zerada se o preço cruzar a VWAP de volta, ou compulsoriamente no hook `on_session_close`.
- **Guardrails**: Limite máximo de operações por dia para evitar `overtrading`.
- **Interação com Execução Enviesada**: A latência é crítica. Um atraso na execução pode fazer com que a oportunidade de reversão à média desapareça. A estratégia deve ter um `threshold` de sinal largo o suficiente para absorver custos e slippage.
- **Determinismo & Performance**: O VWAP deve ser calculado com agregações em ordem fixa. O estado é resetado a cada nova sessão.
- **Validação**:
    - [ ] A estratégia emite ordens para zerar todas as posições no `on_session_close`?
    - [ ] O cálculo do VWAP não usa informações da barra atual (look-ahead intra-barra)?
- **Benchmark**: Cenário "Intraday Net Zero" do Módulo 06.

## 6. Estratégia 3: Pairs/Spread Intraday (Net Zero)

- **Objetivo**: Negociar o spread (diferença de preço) entre dois ativos correlacionados, mantendo a exposição de mercado próxima de zero.
- **Dados Necessários**: Barras intraday para um par de ativos (e.g., `AssetA`, `AssetB`).
- **Lifecycle Hooks Usados**: Mesmos da Estratégia 2.
- **Sinal**: Desvio padrão do spread (`price_A - price_B`) em relação à sua média móvel.
    - Sinal de Entrada: `spread > media_spread + 2 * desvio_padrao` → Vender A, Comprar B.
    - Sinal de Saída: O spread retorna à sua média.
- **Geração de Ordens**: Ao receber um sinal, emite **duas** `OrderRequest`s (uma para cada ativo) na mesma chamada do `on_bar`. A ordem de emissão no vetor de retorno deve ser determinística (e.g., ordenada por `AssetId`).
- **Política de Sizing**: Sizing relativo para garantir que o valor financeiro (notional) das duas pernas seja aproximadamente igual (`qty_A * price_A ≈ qty_B * price_B`).
- **Regras de Encerramento**: Zera ambas as posições quando o spread se normaliza ou compulsoriamente no `on_session_close`.
- **Guardrails**: Limite no desvio padrão máximo do spread para evitar entrar em uma "quebra de correlação".
- **Interação com Execução Enviesada**: O risco de execução de apenas uma das pernas (`leg`) é real. A estratégia deve ser capaz de lidar com um estado onde apenas uma das ordens foi executada, ajustando sua lógica no próximo evento.
- **Determinismo & Performance**: O cálculo do spread e seus indicadores deve ser determinístico. A emissão das duas ordens deve ser em ordem estável.
- **Validação**:
    - [ ] A estratégia emite as duas ordens em ordem determinística?
    - [ ] A lógica de sizing relativo é robusta a preços muito diferentes?
- **Benchmark**: Cenário "Stress de Universo" (com 2 ativos) do Módulo 06.

## 7. Matriz de Cobertura de Funcionalidades

| Funcionalidade | Estratégia 1 (Tendência) | Estratégia 2 (Reversão) | Estratégia 3 (Pairs) |
| :--- | :---: | :---: | :---: |
| **Hook `on_init`** | ✅ | ✅ | ✅ |
| **Hook `on_bar`** | ✅ | ✅ | ✅ |
| **Hook `on_session_close`** | ❌ | ✅ | ✅ |
| **Dados Diários** | ✅ | ❌ | ❌ |
| **Dados Intraday** | ❌ | ✅ | ✅ |
| **Modo Swing** | ✅ | ❌ | ❌ |
| **Modo Net Zero** | ❌ | ✅ | ✅ |
| **Exercita Custos/Slippage** | ✅ | ✅ | ✅ |
| **Exercita Latência** | Menos sensível | Muito sensível | Muito sensível |
| **Invariantes de Portfólio** | ✅ | ✅ | ✅ |
| **Ordem de Emissão Determinística** | N/A | ❌ | ✅ |

## 8. Modos de Falha e Diagnóstico

| Sintoma | Causa Provável | Como Verificar (Ref. Módulo 06) |
| :--- | :--- | :--- |
| **PnL "Bom Demais"** | Viés de look-ahead (usando preço futuro no sinal). | Executar a suite `test-suite-anti-look-ahead`. Revisar a lógica de acesso a dados da estratégia. |
| **Resultados Não Reprodutíveis** | Não-determinismo (RNG não-seedado, ordem de iteração de `HashMap`). | Executar a suite `test-suite-determinism`. Revisar a estratégia em busca de fontes de não-determinismo. |
| **Performance Caiu Drasticamente** | Alocação de memória ou I/O no hot path (`on_bar`). | Executar o benchmark da estratégia e usar `heaptrack` e `strace` para encontrar a violação. |
| **Drawdown Impossível** | Bug no sizing ou na lógica de encerramento. | Teste de integração com cenário controlado para validar o estado do portfólio a cada passo. |

## 9. Checklist de Aceite do Módulo

- [ ] O documento define 3 estratégias de exemplo distintas?
- [ ] Cada exemplo segue o template de especificação padrão?
- [ ] Os exemplos cobrem os modos diário, intraday, swing e net zero?
- [ ] A disciplina de hot-path (zero alocação, sem I/O) é reforçada em cada exemplo?
- [ ] As regras de anti-look-ahead e determinismo são aplicadas na prática?
- [ ] A interação com o modelo de execução enviesado é considerada?
- [ ] A matriz de cobertura demonstra que os exemplos exercitam as principais funcionalidades do sistema?
- [ ] A seção de modos de falha fornece um guia prático para diagnóstico?

## 10. Próximo Módulo Sugerido

Este é o último módulo de especificação de engenharia. O próximo passo lógico é a criação da documentação voltada para o usuário final.

**`10_user_manual_and_quickstart.md`**

- Fornecerá um guia de início rápido para que um novo usuário possa instalar, configurar e executar um backtest de exemplo.
- Explicará a estrutura de configuração (arquivos TOML) de forma clara e com exemplos.
- Descreverá como interpretar os resultados de um backtest (métricas, gráficos) e como usar os logs para diagnóstico. 
