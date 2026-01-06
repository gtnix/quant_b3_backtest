# Especificação Lógica 1: Visão Geral e Arquitetura do TPM

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

O **Módulo de Parâmetros de Trade (Trade Parameters Module - TPM)** é um componente central do sistema de geração de estratégias, projetado para atuar como a **fonte da verdade** para a configuração de backtests e a geração de novas estratégias pelo algoritmo genético (GA). O objetivo principal é guiar o processo de descoberta, saindo de um universo genérico e irrestrito para um espaço de busca focado em metodologias de trading reconhecidas e comprovadas pelo mercado.

Este módulo resolve um problema fundamental: a ineficiência da geração de estratégias puramente aleatórias. Ao fornecer um catálogo de configurações pré-definidas, o TPM assegura que o algoritmo genético trabalhe dentro de limites lógicos e estruturados, aumentando drasticamente a qualidade, a relevância e a velocidade de convergência para estratégias potencialmente lucrativas. O sistema é desenhado para um usuário com pouco conhecimento técnico, que busca facilidade de uso sem sacrificar a versatilidade e a sofisticação das estratégias geradas.

## 2. Princípios de Design

O design do TPM é guiado pelos seguintes princípios fundamentais:

- **Simplicidade para o Usuário**: A complexidade inerente às centenas de parâmetros de uma estratégia de trading é completamente abstraída. O usuário interage com conceitos de alto nível, como "*Swing Trading Moderado*" ou "*Pair Trading Agressivo*", e o TPM se encarrega de traduzir essa escolha em uma configuração detalhada e completa.

- **Modularidade e Desacoplamento**: O TPM é um componente independente. As configurações são armazenadas em arquivos de texto simples no formato **TOML (Tom's Obvious, Minimal Language)**, que são fáceis de ler, editar e versionar. O core do sistema de backtesting e o dashboard consomem esses arquivos, mas não dependem da lógica interna do TPM.

- **Extensibilidade**: O sistema é projetado para ser facilmente expansível. Adicionar uma nova estratégia ou uma variação de uma existente é tão simples quanto criar um novo arquivo `.toml` no diretório de configurações. Nenhuma alteração no código do motor de backtesting é necessária.

- **Fonte da Verdade Única (Single Source of Truth)**: O TPM garante consistência total entre a interface do usuário (dashboard) e o motor de backtesting. Os parâmetros exibidos e selecionados no frontend são os mesmos que alimentam o processo de geração e validação de estratégias no backend, eliminando discrepâncias.

- **Performance**: O carregamento e a validação das configurações são operações críticas. Um *loader* dedicado, implementado em **Rust**, garantirá que a leitura, o parsing e a validação dos arquivos TOML sejam executados com máxima eficiência e segurança de tipos.

## 3. Arquitetura Lógica

A arquitetura do TPM pode ser dividida em três camadas principais: a Camada de Armazenamento, a Camada de Lógica e a Camada de Consumo.

| Camada | Componente | Tecnologia | Responsabilidade |
| :--- | :--- | :--- | :--- |
| **Armazenamento** | Repositório de Configurações | TOML Files | Armazenar de forma persistente e legível todas as 100+ configurações de estratégias. |
| **Lógica** | Carregador do TPM (TPM Loader) | Rust Crate | Carregar, validar, fazer cache e fornecer acesso programático às configurações TOML. |
| **Consumo** | Motor de Backtesting (GA) | Rust | Utilizar as configurações do TPM para definir o espaço de busca do algoritmo genético. |
| **Consumo** | Dashboard (UI) | React/TypeScript | Exibir as estratégias disponíveis, permitir a seleção e customização pelo usuário. |

### Diagrama de Fluxo de Dados

O fluxo de dados começa com a seleção do usuário no dashboard e termina com a geração de uma estratégia otimizada pelo motor de backtesting.

```mermaid
graph TD
    A[Dashboard UI] -- 1. Usuário seleciona "Swing Trading" --> B(TPM Loader);
    B -- 2. Carrega `swing_trading.toml` --> C{Configuração TOML};
    C -- 3. Parâmetros populam UI --> A;
    A -- 4. Usuário inicia geração --> D[Motor de Backtesting];
    B -- 5. Fornece configuração validada --> D;
    D -- 6. GA usa parâmetros como base --> E(Geração de Genomas);
    E -- 7. Gera estratégia otimizada --> F[Relatório de Performance];
    F -- 8. Exibe resultado --> A;
```

### Componentes Detalhados

1.  **Repositório de Configurações**: Um diretório no sistema de arquivos (ex: `/configs/strategies/`) contendo todos os arquivos `.toml`. A estrutura de diretórios pode ser organizada por família de estratégias para facilitar a manutenção:

    ```
    strategies/
    ├── intraday/
    │   ├── orb_conservative.toml
    │   └── vwap_moderate.toml
    ├── swing/
    │   └── ma_crossover_moderate.toml
    └── pair_trading/
        └── cointegration_conservative.toml
    ```

2.  **TPM Loader (Rust Crate)**: Este será um novo crate no workspace do `quant_b3_backtest` chamado `tpm_loader`. Suas principais funções serão:
    - `list_strategies()`: Retorna uma lista de todos os `strategy_id` disponíveis.
    - `load_config(strategy_id)`: Carrega um arquivo TOML específico, faz o parsing para uma struct Rust (`StrategyConfig`), valida contra o schema e retorna a struct ou um erro.
    - `get_metadata(strategy_id)`: Uma função otimizada que lê apenas a seção `[metadata]` de um arquivo para popular rapidamente a UI sem carregar o arquivo inteiro.
    - **Cache**: Implementa um cache em memória para evitar leituras repetidas do disco.

3.  **Integração com o Motor de GA**: O motor do algoritmo genético, ao ser iniciado, receberá um objeto `StrategyConfig` do TPM Loader. Ele usará os parâmetros contidos neste objeto (ex: `ma_fast_period = [10, 30]`, `rsi_oversold = [20, 35]`) para definir os limites (mínimo e máximo) para a geração dos genomas, em vez de usar limites genéricos e amplos.

4.  **Integração com o Dashboard**: A UI chamará uma API que, por sua vez, usará o `TPM Loader` para listar as estratégias. Ao selecionar uma estratégia, a UI receberá os parâmetros padrão e os limites de otimização para exibir ao usuário, permitindo ajustes finos antes de iniciar a geração.

## 4. Próximos Passos

A próxima especificação detalhará a **Taxonomia Completa de Estratégias**, classificando e organizando todos os 116 tipos de trade que formarão a base do catálogo do TPM.
