# Módulo: 02 - Modelo de Dados e Ingestão

---

## Sumário

1. [Objetivos do Modelo de Dados](#1-objetivos-do-modelo-de-dados)
2. [Entidades Canônicas](#2-entidades-canonicas)
3. [Granularidade e Calendário](#3-granularidade-e-calendario)
4. [Formatos de Entrada](#4-formatos-de-entrada)
5. [Pipeline de Ingestão e Normalização](#5-pipeline-de-ingestao-e-normalizacao)
6. [Tratamento de Dados Ausentes e Qualidade](#6-tratamento-de-dados-ausentes-e-qualidade)
7. [Ajustes de Preço (Splits/Dividendos)](#7-ajustes-de-preco-splitsdividendos)
8. [Garantias de Determinismo no Dado](#8-garantias-de-determinismo-no-dado)
9. [Performance no I/O e no Pré-processamento](#9-performance-no-io-e-no-pre-processamento)
10. [Checklist de Aceite do Módulo](#10-checklist-de-aceite-do-modulo)
11. [Próximo Módulo Sugerido](#11-proximo-modulo-sugerido)

---

## 1. Objetivos do Modelo de Dados

Este documento especifica o modelo de dados canônico e o pipeline de ingestão, que servem como a fundação para todo o sistema de backtesting. O design aqui apresentado é um contrato que visa cumprir os requisitos de performance, determinismo e robustez definidos nos Módulos 00 e 01.

Os objetivos centrais são:
- **Eficiência de Performance**: O modelo de dados é estruturado seguindo um **design orientado a dados (Data-Oriented Design)**, priorizando o layout de **Structure of Arrays (SoA)** sobre o de Array of Structures (AoS). Isso maximiza a localidade de cache da CPU e abre caminho para otimizações de vetorização (SIMD), sendo crucial para atender aos critérios de performance NFR01, AC-01 e AC-02.
- **Representação Canônica e Inequívoca**: Todas as entidades de dados, especialmente `Timestamp` e `AssetId`, possuem uma única representação interna. Timestamps são sempre **UTC**, e identificadores de ativos são sempre **inteiros**, eliminando ambiguidades e a necessidade de parsing ou conversões em caminhos críticos de execução.
- **Garantia de Ordenação Cronológica**: O pipeline de ingestão é o único responsável por entregar ao motor de simulação um fluxo de eventos de mercado **perfeitamente ordenado no tempo**, prevenindo estruturalmente o viés de look-ahead.

## 2. Entidades Canônicas

As seguintes entidades formam o núcleo do modelo de dados. Suas especificações são contratuais.

### Timestamp

| Propriedade | Especificação |
| :--- | :--- |
| **Representação Interna** | `i64` (inteiro de 64 bits), representando o número de **nanossegundos** desde a época UNIX (01/01/1970 00:00:00 UTC). |
| **Unidade** | Nanossegundos. |
| **Timezone** | **UTC**, obrigatoriamente. Nenhuma outra representação de timezone é permitida no núcleo do sistema. |
| **Invariantes** | - Um timestamp deve ser sempre positivo.\n- A representação é inequívoca e globalmente consistente. |
| **Erros/Validações** | A camada de parsing deve rejeitar timestamps malformados, ambíguos (e.g., com timezones locais) ou fora de uma faixa razoável. |
| **Impacto de Performance** | O uso de `i64` permite comparações e aritmética de forma extremamente eficiente (uma única instrução de CPU), crucial para a ordenação e o processamento no loop de eventos. |

### AssetId

| Propriedade | Especificação |
| :--- | :--- |
| **Representação Interna** | `u32` (inteiro de 32 bits sem sinal). |
| **Mapeamento** | Ocorre na camada de **Normalização**. Um mapa global (e.g., `HashMap<String, u32>`) é criado para associar o ticker (string) de cada ativo a um `AssetId` único e sequencial. |
| **Invariantes** | - Cada ticker de ativo corresponde a um e somente um `AssetId` durante uma execução.\n- O `AssetId` é estável durante toda a simulação. |
| **Erros/Validações** | O pipeline deve detectar e rejeitar tickers duplicados ou não reconhecidos no universo de ativos definido para o backtest. |
| **Impacto de Performance** | Usar `u32` como índice para acessar arrays de dados (e.g., `precos_fechamento[asset_id]`) é ordens de magnitude mais rápido do que usar strings como chaves em um hash map dentro do loop quente. |

### Bar OHLCV

| Propriedade | Especificação |
| :--- | :--- |
| **Campos** | `asset_id: AssetId`, `timestamp: Timestamp`, `open: f64`, `high: f64`, `low: f64`, `close: f64`, `volume: u64`. |
| **Invariantes** | - `high` deve ser o maior ou igual a todos os outros preços.\n- `low` deve ser o menor ou igual a todos os outros preços.\n- `open`, `high`, `low`, `close` e `volume` devem ser não-negativos. |
| **Erros/Validações** | A camada de parsing deve validar os invariantes acima e a consistência da barra. Barras inválidas devem ser descartadas ou corrigidas conforme a política de qualidade de dados. |
| **Impacto de Performance** | A representação em `f64` é um compromisso entre precisão e performance. Para performance máxima, os dados de barras são armazenados em arrays SoA (e.g., `Vec<f64>` para `close`), não como um `Vec<Bar>`. |

### Event

O `Event` é um `enum` que encapsula todos os tipos de eventos que fluem pelo sistema.

| Tipo de Evento | Campos Essenciais | Descrição |
| :--- | :--- | :--- |
| **MarketEvent** | `asset_id: AssetId`, `timestamp: Timestamp`, `data: Bar` | Representa a chegada de uma nova barra de mercado para um ativo. É o principal insumo para o motor. |
| **SignalEvent** | `asset_id: AssetId`, `timestamp: Timestamp`, `direction: SignalDirection`, `strength: f64` | Gerado pela Estratégia, indica uma intenção de negociação (e.g., Comprar, Vender) com uma certa força. |
| **OrderEvent** | `asset_id: AssetId`, `timestamp: Timestamp`, `quantity: i64`, `type: OrderType` | Gerado pelo Roteador de Ordens, representa uma ordem concreta a ser enviada para execução. |
| **FillEvent** | `asset_id: AssetId`, `timestamp: Timestamp`, `quantity: i64`, `price: f64`, `costs: f64` | Gerado pelo Modelo de Execução, confirma a execução de uma ordem, incluindo o preço efetivo e os custos. |

## 3. Granularidade e Calendário

- **Diário vs. Intraday**: O **Motor de Simulação** é agnóstico à granularidade. A distinção é feita na **Normalização**: barras diárias recebem um timestamp fixo (e.g., 23:59:59.999999999 UTC do dia correspondente), enquanto barras intraday mantêm seu timestamp original. Para o motor, ambos são apenas `MarketEvent`s a serem processados em ordem cronológica.
- **Timezone e Sessão**: Toda a lógica interna opera em **UTC**. A noção de 
sessão de mercado (e.g., horário de abertura/fechamento da bolsa) é aplicada durante a Normalização para validar barras ou gerar eventos de `SessionOpen` / `SessionClose`, se necessário, mas o timestamp do evento permanece em UTC.
- **Feriados e Dias Sem Pregão**: O pipeline de Normalização deve usar um calendário de mercado para identificar e descartar barras que caiam em feriados ou fins de semana. Se houver um buraco na série temporal de um ativo onde deveria haver dados, isso é tratado pela política de qualidade de dados.

## 4. Formatos de Entrada

O sistema deve suportar, no mínimo, os seguintes formatos de entrada. A extensibilidade para outros formatos é um objetivo de design, mas não de implementação inicial.

### CSV (Comma-Separated Values)

- **Esquema Mínimo Esperado**:

| Coluna | Tipo de Dados | Exemplo | Obrigatório |
| :--- | :--- | :--- | :--- |
| `timestamp` | String (ISO 8601) ou Int (época) | `2023-10-27T10:00:00Z` | Sim |
| `ticker` | String | `PETR4` | Sim |
| `open` | Float | `35.20` | Sim |
| `high` | Float | `35.40` | Sim |
| `low` | Float | `35.10` | Sim |
| `close` | Float | `35.35` | Sim |
| `volume` | Integer | `5000000` | Sim |

- **Regras de Parsing**: O parser deve ser robusto a diferentes formatos de timestamp (desde que bem especificados) e validar que todos os campos numéricos são válidos.

### Apache Parquet

- **Esquema Mínimo Esperado**: O esquema do Parquet deve ser análogo ao do CSV, com tipos de dados mais estritos (e.g., `Int64` para timestamp, `Float64` para preços).
- **Vantagens**: O uso de Parquet é preferível para grandes datasets, pois oferece compressão colunar e leitura muito mais performática, alinhando-se diretamente com a estratégia de I/O em batch e o layout SoA.

O suporte a outros formatos (`etc.`) é considerado uma extensão futura e não será detalhado ou implementado na versão inicial.

## 5. Pipeline de Ingestão e Normalização

O pipeline é uma sequência de passos executada **fora do loop quente** do backtest, garantindo que o motor receba dados limpos e prontos para processamento de alta velocidade.

| Passo | Descrição | Garantias | Falhas Comuns | Estratégia de Performance |
| :--- | :--- | :--- | :--- | :--- |
| **1. Leitura em Batch** | Ler blocos de dados brutos do arquivo de origem (e.g., um arquivo Parquet inteiro ou grandes chunks de um CSV). | Os dados estão na memória. | Arquivo não encontrado, formato corrompido. | Minimiza o número de chamadas de I/O. Usa I/O assíncrono se possível. |
| **2. Parsing/Decoding** | Converter os dados brutos em representações de `Bar` em memória. | Os dados estão em um formato estruturado, mas ainda não validados. | Erros de tipo, campos ausentes. | Usar parsers otimizados (e.g., o ecossistema `arrow` em Rust para Parquet). |
| **3. Validações e Saneamento** | Validar cada barra (invariantes de OHLC), verificar a consistência dos timestamps. | Barras inválidas são descartadas. | Barras com `high < low`, timestamps fora de ordem. | Feito em paralelo por ativo ou por chunk de dados, se possível. |
| **4. Normalização Canônica** | Converter timestamps para UTC `i64`, mapear tickers para `AssetId`. | Todas as entidades estão em seu formato canônico interno. | Ticker não encontrado no mapa de universo. | O mapa de `AssetId` é construído uma vez e depois apenas consultado. |
| **5. Ordenação Cronológica** | Ordenar **todos** os eventos de mercado de **todos** os ativos em um único fluxo cronológico. | O fluxo de eventos está globalmente ordenado e pronto para o motor. Garante o não-look-ahead. | Algoritmo de ordenação instável. | Usar um algoritmo de ordenação estável (e.g., `sort_by_key` em Rust) com uma chave primária (`timestamp`) e uma chave secundária (`AssetId`) para desempate determinístico. |
| **6. Materialização** | Entregar o fluxo de `MarketEvent`s ordenado ao Motor de Simulação. | O motor recebe um iterador ou um `Vec` de eventos pronto para consumo. | N/A | A estrutura de dados final é um `Vec<Event>` ou similar, que permite acesso sequencial de altíssima velocidade. |

## 6. Tratamento de Dados Ausentes e Qualidade

Políticas de tratamento de dados devem ser **explícitas e determinísticas**. O sistema deve suportar, via configuração, as seguintes políticas mínimas:

- **Buracos (Missing Bars)**: 
  - `Descartar Ativo`: Se um ativo tiver dados ausentes, ele é removido da simulação.
  - `Forward-Fill (Preenchimento para Frente)`: Preencher a barra ausente repetindo os valores da barra anterior. Esta é uma política comum, mas deve ser usada com cautela, pois pode introduzir vieses.
- **Duplicatas**: Barras com o mesmo `AssetId` e `timestamp` devem ser detectadas. A política padrão é manter a primeira e descartar as subsequentes.
- **Outliers**: O sistema não implementará modelos sofisticados de detecção de outliers na versão inicial. A qualidade dos dados de entrada é uma premissa.

## 7. Ajustes de Preço (Splits/Dividendos)

O suporte a ajustes de preço é uma responsabilidade da camada de **Normalização**. O pipeline deve ser capaz de consumir um arquivo secundário de eventos de ajuste (splits e dividendos) e aplicá-los aos dados brutos de OHLC antes da ordenação.

- **Política**: Os preços históricos são ajustados para trás (`backward-adjusted`). Por exemplo, em um split 2-por-1, todos os preços históricos antes do evento são divididos por 2. Isso garante que os retornos percentuais permaneçam corretos, o que é essencial para a lógica da estratégia.
- **Impacto**: Esta etapa é crucial para a corretude do backtest. A não aplicação de ajustes levaria a saltos de preço artificiais que invalidariam completamente os resultados.

## 8. Garantias de Determinismo no Dado

O determinismo (NFR02) começa no dado. Este módulo garante:

- **Ordenação Estável**: Conforme descrito na Seção 5, o uso de uma chave de ordenação secundária (`AssetId`) garante que, se dois eventos tiverem o mesmo timestamp, sua ordem relativa será sempre a mesma, evitando não-determinismo.
- **Ausência de Aleatoriedade**: O pipeline de ingestão e normalização é 100% determinístico. Nenhuma operação utiliza números aleatórios.
- **Hashing do Dataset**: Após a normalização, um hash (e.g., SHA-256) do fluxo de eventos final pode ser calculado e armazenado. Isso permite verificar se o conjunto de dados de entrada para o motor é idêntico entre duas execuções, facilitando a depuração de divergências (AC-03).

## 9. Performance no I/O e no Pré-processamento

As diretrizes de performance do Módulo 01 são implementadas aqui da seguinte forma:

- **Batch I/O**: A leitura de arquivos é feita em grandes blocos, não linha a linha. Formatos como Parquet são ideais para isso.
- **Layout SoA**: Durante a normalização, os dados são transformados de um formato de entrada (tipicamente AoS) para um layout SoA em memória. Ou seja, em vez de `Vec<Bar>`, teremos `(Vec<i64>, Vec<f64>, ...)` para timestamps, closes, etc. Isso é fundamental para o desempenho das estratégias que analisam séries temporais.
- **Pré-alocação**: Os vetores que armazenarão os dados normalizados são pré-alocados com o tamanho conhecido (se possível) para evitar realocações de memória durante o processamento.
- **Separação do Loop Quente**: Todo o pipeline descrito nesta seção é executado **uma vez**, antes do início do loop de simulação. O custo do pré-processamento é amortizado e não impacta a latência por evento do backtest, contribuindo para atender aos critérios AC-01 e AC-02.

## 10. Checklist de Aceite do Módulo

- [ ] A representação interna de `Timestamp` é `i64` (nanos) e sempre UTC.
- [ ] `AssetId` é um `u32` mapeado a partir de um ticker.
- [ ] A estrutura de `Bar` e `Event` está definida contratualmente.
- [ ] O pipeline de ingestão está definido em passos claros (Leitura → Parsing → Validação → Normalização → Ordenação).
- [ ] A ordenação de eventos é cronologicamente estrita e estável.
- [ ] A diferença entre dados diários e intraday é tratada na Normalização.
- [ ] Os formatos de entrada (CSV, Parquet) têm esquemas mínimos definidos.
- [ ] Políticas para dados ausentes e duplicatas são explícitas e determinísticas.
- [ ] A política de ajuste de preços (splits/dividendos) está definida como parte da Normalização.
- [ ] Mecanismos para garantir o determinismo dos dados (ordenação estável, hashing) estão especificados.
- [ ] A estratégia de performance (SoA, batch I/O, pré-processamento) está alinhada com os Módulos 00 e 01.

## 11. Próximo Módulo Sugerido

**`03_event_model_and_simulation_engine.md`**

- Detalhará a implementação do loop de eventos (o "coração" do backtester), explicando como ele consome o fluxo de eventos ordenado.
- Especificará as APIs e traits que o motor expõe para os componentes `Strategy`, `ExecutionModel` e `Portfolio`.
- Descreverá como o estado da simulação é gerenciado e como o motor garante a prevenção de look-ahead em tempo de execução.
