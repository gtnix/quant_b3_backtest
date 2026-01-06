# Especificação Lógica 7: UX do Dashboard - Seleção de Estratégias

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação detalha a experiência do usuário (UX) para a tela de seleção de estratégias no dashboard do `quant_b3_backtest`. O objetivo primordial desta interface é traduzir a complexidade do **Módulo de Parâmetros de Trade (TPM)** em uma experiência de descoberta simples, visualmente atraente e intuitiva para um usuário leigo. O design deve abstrair a complexidade dos mais de 100 modelos pré-configurados, permitindo que o usuário explore e selecione uma base para sua estratégia sem precisar entender os detalhes técnicos subjacentes.

A inspiração estética e funcional para esta interface vem de plataformas fintech de ponta, como a Revolut, que se destacam pela clareza, design minimalista e foco na jornada do usuário. [1]

## 2. Princípios de Design da UX

A interface de seleção será guiada pelos seguintes princípios:

-   **Descoberta Guiada (Guided Discovery)**: O usuário não precisa saber o que procurar. A interface deve funcionar como um "sommelier" de estratégias, fazendo perguntas implícitas através de filtros e categorias para ajudar o usuário a encontrar o que melhor se adapta ao seu perfil.

-   **Divulgação Progressiva (Progressive Disclosure)**: A tela principal exibe apenas informações de alto nível. Detalhes complexos são revelados apenas sob demanda, quando o usuário demonstra interesse em uma estratégia específica. Isso evita a sobrecarga de informação e a paralisia por análise.

-   **Feedback Visual Imediato**: O uso de ícones, tags coloridas e uma hierarquia visual clara permite que o usuário compreenda a natureza de uma estratégia (risco, timeframe, tipo) instantaneamente, antes mesmo de ler os detalhes.

-   **Abordagem Didática**: Cada termo técnico ou parâmetro deve ser acompanhado por uma explicação clara e concisa, acessível através de um tooltip (dica de ferramenta). O objetivo é que o usuário aprenda sobre os diferentes métodos de trading enquanto utiliza a plataforma.

## 3. Componentes da Tela de Seleção

A tela será dividida em duas áreas principais: uma barra lateral de filtros e uma área de conteúdo principal que exibe as estratégias em formato de cartões.

### 3.1. Painel de Filtros e Busca

Localizado na lateral esquerda, este painel permite ao usuário refinar o universo de estratégias.

-   **Barra de Busca**: Um campo de texto proeminente no topo para buscar estratégias por nome ou tags. Ex: "momentum", "cruzamento", "pares".

-   **Filtros Principais**: Botões ou dropdowns para filtrar pelas principais categorias da taxonomia do TPM:
    -   **Perfil de Risco**: Botões de seleção única (`Conservador`, `Moderado`, `Agressivo`).
    -   **Família da Estratégia**: Um dropdown de múltipla seleção (`Swing Trading`, `Pair Trading`, `Portfolio`, etc.).
    -   **Horizonte de Tempo**: Botões de seleção única (`Intraday`, `Curto Prazo`, `Médio Prazo`, `Longo Prazo`).

### 3.2. Grade de Estratégias (Strategy Cards)

A área principal exibirá as estratégias que correspondem aos filtros selecionados em uma grade responsiva de cartões. Cada cartão é um componente de visualização que resume uma única estratégia.

#### Design do Cartão de Estratégia

O cartão é o elemento central da descoberta. Ele deve ser visualmente limpo e informativo.

**Estrutura do Cartão:**

-   **Título (`metadata.name`)**: Nome completo da estratégia em destaque. Ex: "**Swing Momentum - MA Crossover**".
-   **Tags de Classificação**: Tags visuais no topo do cartão:
    -   **Família (`metadata.family`)**: Uma tag com ícone. Ex: `[Ícone de Gráfico de Linha] Swing Trading`.
    -   **Perfil de Risco (`metadata.risk_profile`)**: Uma tag colorida para feedback imediato:
        -   `Conservador`: Verde
        -   `Moderado`: Amarelo
        -   `Agressivo`: Vermelho
-   **Descrição Curta (`metadata.description`)**: Uma sinopse de uma ou duas linhas explicando a lógica principal da estratégia.
-   **Tags Adicionais (`metadata.tags`)**: Uma lista de tags de palavras-chave para contexto adicional. Ex: `momentum`, `crossover`, `volume`.
-   **Botão de Ação**: Um botão claro, como "**Ver Detalhes**" ou "**Selecionar**", que leva ao próximo passo.

## 4. Detalhes da Estratégia (Modal ou Painel Lateral)

Ao clicar em um cartão, uma visão detalhada da estratégia é apresentada, sem levar o usuário para uma nova página. Isso pode ser um *modal* que sobrepõe a tela ou um painel que desliza pela direita.

**Conteúdo da Visão de Detalhes:**

-   **Cabeçalho**: Repete o nome da estratégia e suas tags principais.
-   **Descrição Completa**: O texto completo do campo `metadata.description`.
-   **Parâmetros Chave**: Uma lista curada dos parâmetros mais importantes da estratégia, apresentados de forma legível e com explicações.
    -   **NÃO** exibir uma lista de todos os parâmetros do TOML.
    -   **SIM** agrupar logicamente e usar linguagem natural. Ex:
        -   **Sinal de Entrada**: "Cruzamento da Média Móvel de 20 dias sobre a de 50 dias."
        -   **Gestão de Risco**: "Stop loss posicionado a 2x o ATR(14)."
        -   **Alvo de Lucro**: "Alvo de 3x o risco inicial."
    -   Cada parâmetro ou termo técnico (ex: "ATR") deve ter um ícone de informação `(?)` que, ao ser sobrevoado, exibe um tooltip com uma explicação detalhada.
-   **Botão de Ação Primário**: Um botão proeminente para confirmar a seleção. Ex: "**Usar este Modelo**" ou "**Próximo: Configurar Otimização**".

## 5. Fluxo do Usuário

O fluxo de seleção é projetado para ser linear e intuitivo.

```mermaid
flowchart TD
    A[Tela de Seleção de Estratégia] -->|Aplica Filtro "Risco Moderado"| B(Grade de Cartões é atualizada);
    B -->|Clica no cartão "Swing Momentum"| C{Abre Modal de Detalhes};
    C -->|Lê a descrição e parâmetros chave| C;
    C -->|Clica em "Usar este Modelo"| D(Navega para a Tela de Configuração);
    C -->|Fecha o Modal| B;
```

## 6. Conclusão

Esta abordagem de UX para a seleção de estratégias coloca o usuário no centro da experiência. Ao priorizar a descoberta guiada, a clareza visual e a educação contextual, o sistema capacita um usuário não-técnico a tomar decisões informadas sobre a base de sua estratégia de trading. A complexidade do TPM é elegantemente abstraída, transformando uma tarefa potencialmente intimidante em um processo de exploração engajador e produtivo.

A próxima especificação detalhará a **UX do Dashboard para a Configuração de Parâmetros**, a tela para a qual o usuário é levado após selecionar um modelo de estratégia.

## Referências

[1] Norman, D. (2013). *The Design of Everyday Things: Revised and Expanded Edition*. Basic Books. (Princípios de design centrado no usuário, como feedback e divulgação progressiva).
