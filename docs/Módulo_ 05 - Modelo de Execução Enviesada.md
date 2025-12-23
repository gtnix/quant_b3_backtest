# Módulo: 05 - Modelo de Execução Enviesada

---

## Sumário

1. [Papel do Execution Model no Hot Path](#1-papel-do-execution-model-no-hot-path)
2. [Contrato de Entrada/Saída](#2-contrato-de-entrada-saida)
3. [Taxonomia de Ordens Suportadas](#3-taxonomia-de-ordens-suportadas)
4. [Modelo de Custos (Fees)](#4-modelo-de-custos-fees)
5. [Modelo de Slippage: Políticas Determinísticas](#5-modelo-de-slippage-politicas-deterministicas)
6. [Modelo de Latência Simplificada](#6-modelo-de-latencia-simplificada)
7. [Regras de Fill com Dados OHLCV](#7-regras-de-fill-com-dados-ohlcv)
8. [Fills Parciais e Restrições de Liquidez](#8-fills-parciais-e-restricoes-de-liquidez)
9. [Calibração e Parâmetros de Configuração](#9-calibracao-e-parametros-de-configuracao)
10. [Performance: Contratos de Throughput](#10-performance-contratos-de-throughput)
11. [Plano de Validação do Modelo](#11-plano-de-validacao-do-modelo)
12. [Checklist de Aceite do Módulo](#12-checklist-de-aceite-do-modulo)
13. [Próximo Módulo Sugerido](#13-proximo-modulo-sugerido)

---

## 1. Papel do Execution Model no Hot Path

O Modelo de Execução (Execution Model) é o componente que injeta realismo no backtest, servindo como a ponte entre a intenção de negociação (`OrderEvent`) e o resultado transacional (`FillEvent`). Ele opera dentro do hot path do Motor de Simulação (Módulo 03), sendo invocado a cada ordem gerada. Sua responsabilidade é simular os atritos do mercado — custos, slippage e latência — de forma determinística e performática, sem recorrer à complexidade de um book de ofertas. O objetivo não é a perfeição, mas uma **modelagem suficientemente realista** que permita a diferenciação robusta entre estratégias boas e ruins, impactando diretamente o PnL final (Módulo 04).

## 2. Contrato de Entrada/Saída

- **Entrada**: A função de execução recebe dois argumentos imutáveis:
    1.  `order: &OrderEvent`: A ordem a ser executada.
    2.  `market_state: &MarketState`: O estado de mercado no tempo `t` da decisão, para servir de referência de preço.
- **Saída**: A função retorna `Option<FillEvent>`. 
    - `Some(FillEvent)`: Se a ordem foi executada. O `FillEvent` contém o `fill_price` (preço efetivo), `filled_qty` (quantidade executada), e `costs` (taxas/comissões).
    - `None`: Se a ordem não pôde ser executada (e.g., uma ordem limite fora do range da barra).

**Invariantes:**
- **Determinismo**: Para a mesma `OrderEvent`, `MarketState` e configuração, a saída deve ser sempre idêntica (bit-a-bit).
- **Monotonicidade Temporal**: O `timestamp` de um `FillEvent` deve ser `>=` ao `timestamp` da `OrderEvent` correspondente.
- **Ausência de Look-Ahead**: O modelo só pode usar informações do `MarketState` em `t` ou anterior.
- **Custo Único**: O campo `costs` no `FillEvent` é a única fonte de taxas de transação. O Módulo 04 garante que este valor seja aplicado uma única vez ao caixa.

## 3. Taxonomia de Ordens Suportadas

O sistema suporta um conjunto mínimo e contratual de tipos de ordem.

- **Market Order (Ordem a Mercado)**: É o tipo de ordem base e obrigatório. A ordem é para ser executada ao melhor preço disponível, que na simulação se traduz para `preço_referência + slippage`.
- **Limit Order (Ordem Limite)**: Suportada com regras de execução baseadas estritamente em dados OHLCV. A lógica de preenchimento é detalhada na Seção 7.

**Não-Requisitos:**
- Ordens `Stop`, `Stop-Limit`, `TWAP/VWAP`, `Iceberg` e outras ordens algorítmicas complexas estão **fora do escopo**. A sua implementação é proibida.

## 4. Modelo de Custos (Fees)

Os custos são configuráveis e adicionados ao campo `costs` do `FillEvent`.

| Tipo de Custo | Fórmula Conceitual | Quando é Cobrado | Impacto no Sistema |
| :--- | :--- | :--- | :--- |
| **Fixo por Ordem** | `C` (constante) | No `FillEvent` de cada ordem executada. | Reduz o caixa. Impacta diretamente o NAV. |
| **Proporcional ao Notional** | `notional * bps` (onde `notional = qty * price`) | No `FillEvent`. | Reduz o caixa. Impacta diretamente o NAV. |
| **Por Quantidade** | `qty * cost_per_unit` | No `FillEvent`. | Reduz o caixa. Impacta diretamente o NAV. |

O `costs` final no `FillEvent` é a soma de todos os componentes de custo aplicáveis. Conforme o Módulo 04, este valor é subtraído do caixa no momento da atualização do portfólio.

## 5. Modelo de Slippage: Políticas Determinísticas

Slippage é a diferença entre o preço de referência da ordem e o preço efetivo de execução. É modelado como uma função determinística. O preço de referência é, por contrato, o `close` da barra do `MarketEvent` que gerou a ordem.

**Famílias de Modelos de Slippage:**

| Família | Inputs Principais | Custo Computacional | Prós / Contras | Risco de Viés |
| :--- | :--- | :--- | :--- | :--- |
| **1. Constante** | `bps` (parâmetro fixo) | Muito Baixo (O(1)) | **Pró**: Simples, rápido, estável. **Contra**: Pouco realista, não reage às condições de mercado. | Pode subestimar o impacto em mercados voláteis ou com ordens grandes. |
| **2. Linear por Volume** | `order_qty`, `bar_volume`, `coef` | Baixo (O(1)) | **Pró**: Captura o impacto básico do tamanho da ordem na liquidez. **Contra**: `bar_volume` é um proxy grosseiro para liquidez. | Pode ser irrealista se a relação não for linear. Requer calibração cuidadosa do `coef`. |
| **3. Volatilidade (Bar Range)** | `(high - low)`, `coef` | Baixo (O(1)) | **Pró**: Captura a intuição de que o slippage é maior em mercados voláteis. **Contra**: Não diferencia a causa da volatilidade. | Pode superestimar o slippage em movimentos de preço direcionais com alta liquidez. |

O modelo a ser usado e seus parâmetros são definidos na configuração do backtest.

## 6. Modelo de Latência Simplificada

Latência, neste contexto, **não é uma simulação de matching engine**. É um modelo determinístico para simular o atraso entre a geração da ordem e sua execução.

- **Definição**: A latência é configurada como um atraso de `K` eventos de mercado (ou `N` barras). Uma `OrderEvent` gerada no tempo `t` só será processada pelo Modelo de Execução no tempo `t + K`.
- **Implementação**: O Motor de Simulação (Módulo 03) deve conter uma fila secundária para `OrderEvent`s com latência. Ao receber uma ordem, ele a insere em uma `priority_queue` ordenada pelo `timestamp` de execução futuro. O motor então processa ordens dessa fila quando seu `current_time` alcança o `timestamp` da ordem.
- **Determinismo**: A ordem de processamento de múltiplas ordens com o mesmo `timestamp` de execução futuro deve seguir a mesma regra de desempate estável do sistema (e.g., `AssetId`).

## 7. Regras de Fill com Dados OHLCV

- **Market Order**: `fill_price = reference_price +/- slippage`. A ordem sempre executa (`filled_qty = order_qty`), a menos que restrições de liquidez se apliquem (Seção 8).
- **Limit Order**: A execução depende do range da barra seguinte ao evento da ordem.
    - **Buy Limit Order**: Executa se `limit_price >= bar.low`. 
    - **Sell Limit Order**: Executa se `limit_price <= bar.high`.
- **Ambiguidade Intra-Barra**: Como o caminho do preço dentro da barra é desconhecido, adotamos uma política **pessimista** para evitar otimismo:
    - Para uma ordem de compra limite, o `fill_price` é o `limit_price`, assumindo que o preço tocou o limite e voltou.
    - Para uma ordem de venda limite, o `fill_price` é o `limit_price`.

**Checklist Anti-Otimismo:**
- [ ] A regra de fill para ordens limite usa o range da barra *seguinte* à geração da ordem?
- [ ] O preço de fill para ordens limite é o preço da ordem (pior caso), e não um preço mais favorável dentro da barra?
- [ ] O slippage é sempre aplicado de forma a prejudicar o resultado (adicionado ao preço de compra, subtraído do de venda)?

## 8. Fills Parciais e Restrições de Liquidez

Para evitar a suposição irrealista de liquidez infinita, o sistema suporta um modelo simples de restrição de volume, que pode resultar em *fills* parciais. Esta funcionalidade é considerada parte do escopo mínimo.

- **Política**: A quantidade máxima que pode ser executada em uma única barra é uma fração configurável do volume daquela barra. `max_fill_qty = bar.volume * max_participation_rate`.
- **Cálculo do Fill**: `filled_qty = min(order_qty, max_fill_qty)`.
- **Resíduo**: A quantidade não executada da ordem é descartada. O sistema não suporta ordens que persistem por múltiplas barras (Good-Til-Canceled).

## 9. Calibração e Parâmetros de Configuração

O Modelo de Execução é parametrizável. A configuração do backtest deve incluir uma seção `[execution]` com:
- `fee_model`: (e.g., `{ type = "proportional", bps = 1.5 }`)
- `slippage_model`: (e.g., `{ type = "volatility", coef = 0.1 }`)
- `latency_model`: (e.g., `{ type = "events", delay = 2 }`)
- `liquidity_model`: (e.g., `{ max_participation_rate = 0.1 }`)

Para garantir o determinismo (AC-03), um hash desta configuração deve ser combinado com o hash dos dados de entrada para formar a assinatura final da execução.

## 10. Performance: Contratos de Throughput

| Operação | Complexidade | Hot/Slow Path | Como Medir |
| :--- | :--- | :--- | :--- |
| **Cálculo de Custo** | **O(1)** | Hot | Microbenchmark da função de custo. |
| **Cálculo de Slippage** | **O(1)** | Hot | Microbenchmark para cada família de modelo. |
| **Verificação de Fill (Limit)** | **O(1)** | Hot | Microbenchmark da lógica de verificação contra o range da barra. |

O modelo é projetado para ser `branchless` sempre que possível e sem alocações, garantindo que ele não seja um gargalo de performance no loop de eventos.

## 11. Plano de Validação do Modelo

- **Testes de Determinismo**: Executar o mesmo cenário 100x e garantir que o `FillEvent` resultante seja idêntico em todas as execuções.
- **Testes de Invariantes**: Testes unitários que verificam se `costs >= 0`, `0 <= filled_qty <= order_qty`, e `fill.timestamp >= order.timestamp`.
- **Testes de Sensibilidade**: Testes que validam se, ao aumentar o parâmetro de slippage, o `fill_price` piora de forma monotônica. Validar se o aumento dos custos reduz o NAV de forma correspondente.
- **Testes Anti-Look-Ahead**: Criar um cenário onde o preço da barra `t+1` é favorável. Uma ordem limite em `t` não pode ser preenchida com base nessa informação futura.

## 12. Checklist de Aceite do Módulo

- [ ] O contrato de entrada (`OrderEvent`) e saída (`FillEvent`) está definido e é respeitado?
- [ ] Os modelos de custo, slippage e latência são determinísticos?
- [ ] A simulação de ordens limite é baseada estritamente em dados OHLCV e segue uma política anti-otimismo?
- [ ] O escopo é respeitado (sem book replay, sem microestrutura)?
- [ ] A integração com o Motor (Módulo 03) e o Portfólio (Módulo 04) está clara?
- [ ] O modelo de `partial fill` baseado em volume está especificado?
- [ ] As diretrizes de performance (O(1), sem alocação) são explícitas?
- [ ] O plano de validação cobre determinismo, invariantes e sensibilidade?
- [ ] Os parâmetros de configuração são rastreáveis para garantir a reprodutibilidade?

## 13. Próximo Módulo Sugerido

**`06_performance_and_benchmarking.md`**

- Consolidará todas as estratégias de performance definidas nos módulos anteriores em um único guia de engenharia.
- Detalhará a metodologia e as ferramentas para profiling (CPU, memória) e benchmarking do sistema.
- Estabelecerá os cenários de teste concretos para validar os critérios de aceite de performance (AC-01, AC-02) e os procedimentos para detectar regressão de performance. 
performance. 
  para regressão de performance.
