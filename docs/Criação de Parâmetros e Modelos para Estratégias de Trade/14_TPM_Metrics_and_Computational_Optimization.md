# Especificação Lógica 14: Métricas e Otimização Computacional

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação aborda dois aspectos críticos do sistema de backtesting: as **métricas de performance** usadas para avaliar as estratégias e as **técnicas de otimização computacional** empregadas para garantir que o processo de geração e validação seja executado com a máxima eficiência. A precisão das métricas é fundamental para a validade dos resultados, enquanto a performance computacional é crucial para a usabilidade da plataforma, especialmente durante o processo iterativo do algoritmo genético (GA).

O motor de backtesting, sendo implementado em Rust, já possui uma vantagem de performance significativa. [1] Esta especificação detalha como iremos alavancar ainda mais os recursos da linguagem e do hardware para entregar resultados de forma rápida e precisa.

## 2. Métricas de Performance da Estratégia

O relatório de performance final, bem como a função de fitness do GA, serão baseados em um conjunto abrangente de métricas padrão da indústria. Elas serão calculadas para o período total, para o período in-sample (treino) e para o período out-of-sample (teste).

### 2.1. Métricas de Retorno e Risco

| Métrica | Descrição | Importância |
| :--- | :--- | :--- |
| **CAGR (Compound Annual Growth Rate)** | A taxa de crescimento anual composta do capital. | Mede o retorno anualizado de forma geométrica. |
| **Annualized Volatility** | O desvio padrão anualizado dos retornos diários. | Mede o risco total ou a "turbulência" da estratégia. |
| **Sharpe Ratio** | (CAGR - Taxa Livre de Risco) / Volatilidade Anualizada. | Mede o retorno ajustado ao risco. É a métrica mais comum. |
| **Max Drawdown** | A maior queda percentual do pico ao fundo da curva de capital. | Mede a maior perda que um investidor teria experimentado. Crucial para o risco. |
| **Calmar Ratio** | CAGR / Max Drawdown. | Mede o retorno em relação ao maior risco incorrido. |
| **Sortino Ratio** | Similar ao Sharpe, mas usa apenas a volatilidade dos retornos negativos (downside deviation). | Mede o retorno ajustado ao risco "ruim". |
| **Profit Factor** | Ganhos brutos / Perdas brutas. | Mede quanto a estratégia ganha para cada dólar que perde. |

### 2.2. Métricas de Trade

| Métrica | Descrição |
| :--- | :--- |
| **Total Trades** | Número total de operações executadas. |
| **Win Rate (%)** | Percentual de operações que resultaram em lucro. |
| **Payoff Ratio** | Ganho médio / Perda média. |
| **Average Holding Period** | Tempo médio de duração de uma operação. |
| **Profit per Trade** | Lucro/Prejuízo médio por operação. |
| **Max Consecutive Wins/Losses** | Maior sequência de ganhos ou perdas consecutivas. |

## 3. Função de Fitness do Algoritmo Genético

A função de fitness, que guia a evolução das estratégias, não será baseada em uma única métrica. Será uma **soma ponderada** de várias métricas, permitindo uma otimização mais balanceada e customizável. Os pesos serão definidos na seção `[optimization]` do arquivo TOML.

**Fórmula da Fitness:**

`Fitness = (w_sharpe * Sharpe) + (w_cagr * CAGR) - (w_drawdown * MaxDrawdown) + (w_calmar * Calmar)`

-   Os pesos (`w_sharpe`, etc.) permitem priorizar diferentes aspectos. Por exemplo, uma estratégia conservadora pode ter um peso de drawdown (`w_drawdown`) muito alto.
-   A função também aplicará **penalidades** para evitar soluções super otimizadas (overfitting) ou irrealistas:
    -   **Penalidade por Baixo Número de Trades**: Estratégias com poucos trades não são estatisticamente significantes. Se `Total Trades < min_trades_for_no_penalty`, a fitness é drasticamente reduzida.
    -   **Penalidade por Turnover Extremo**: Estratégias que operam excessivamente incorrem em altos custos. Se o turnover anual for maior que `max_turnover_annual`, a fitness é penalizada.

## 4. Otimização Computacional

O processo de otimização do GA envolve a execução de milhares de backtests. A velocidade deste processo é o principal gargalo de performance do sistema. As seguintes estratégias serão empregadas para acelerar o cálculo:

### 4.1. Paralelização Massiva com Rayon

O laço de avaliação da fitness de uma população do GA é um problema "embaraçosamente paralelo". Cada cromossomo (estratégia) pode ser testado de forma completamente independente dos outros. A biblioteca **Rayon** em Rust será usada para paralelizar a avaliação da população em todos os núcleos de CPU disponíveis. [2]

```rust
// Exemplo de paralelização da avaliação da população
use rayon::prelude::*;

let population: Vec<Chromosome> = ...;

let fitness_scores: Vec<f64> = population.par_iter()
    .map(|chromosome| {
        // run_backtest é uma função computacionalmente intensiva
        run_backtest(chromosome)
    })
    .collect();
```

### 4.2. Otimizações de Dados e Memória

-   **Formato de Dados Binário**: Os dados de mercado (OHLCV) não serão lidos de arquivos CSV a cada backtest. Na primeira vez, eles serão lidos, processados e salvos em um formato binário otimizado (ex: usando `bincode` ou `Apache Arrow`) para leitura subsequente ultrarrápida.

-   **Mapeamento de Memória (Memory Mapping)**: Para datasets muito grandes, a técnica de `mmap` (memory mapping) será usada para mapear o arquivo de dados binário diretamente no espaço de endereço virtual do processo. Isso permite que o sistema operacional gerencie o carregamento de partes do arquivo em memória de forma eficiente, evitando o custo de ler o arquivo inteiro de uma vez.

-   **Estruturas de Dados Eficientes**: O uso de `structs` em Rust em vez de `classes` (como em Python) e a alocação de dados em arrays contíguos na memória (ex: `Vec<f64>`) melhoram a localidade de cache da CPU, resultando em um processamento de dados mais rápido.

### 4.3. Pré-cálculo de Indicadores

Em vez de recalcular indicadores técnicos (médias móveis, RSI, etc.) para cada um dos milhares de backtests, o sistema pode identificar todos os indicadores e períodos necessários a partir da configuração TOML e pré-calculá-los uma única vez para o dataset completo. O processo de backtest então apenas lê os valores pré-calculados, transformando um cálculo caro em uma simples leitura de array.

### 4.4. Compilação em Modo `release`

O código Rust final será compilado com o perfil `release` e otimizações específicas (`lto = "fat"`, `codegen-units = 1`), conforme já configurado no `Cargo.toml` do projeto. Isso instrui o compilador LLVM a aplicar as otimizações mais agressivas, resultando em um binário com a máxima performance possível.

## 5. Conclusão

A combinação de métricas de performance robustas e otimizações computacionais agressivas é o que permitirá que a plataforma seja, ao mesmo tempo, poderosa e interativa. A precisão das métricas garante que o GA esteja otimizando para o objetivo correto, enquanto a performance do motor de backtesting garante que o processo de otimização seja concluído em um tempo razoável para o usuário. A escolha de Rust como linguagem base é um facilitador fundamental para alcançar ambos os objetivos.

A próxima e última especificação abordará o **Guia de Implementação e Roadmap**, fornecendo um plano de ação para construir o TPM.

## Referências

[1] The Rust Programming Language. *Benchmarks*. Disponível em: <https://www.rust-lang.org/benchmarks>. Acessado em: 05 de janeiro de 2026.
[2] The Rayon Community. *Rayon - A data parallelism library for Rust*. Disponível em: <https://github.com/rayon-rs/rayon>. Acessado em: 05 de janeiro de 2026.
