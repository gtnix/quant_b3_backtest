_# Especificação Lógica 8: UX do Dashboard - Configuração de Parâmetros

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação detalha a experiência do usuário (UX) para a tela de **Configuração de Parâmetros**, a segunda etapa no fluxo de geração de estratégias. Após o usuário selecionar um modelo de estratégia na tela de seleção, ele é direcionado para esta interface. O objetivo aqui é permitir que o usuário, mesmo que leigo, possa revisar e, opcionalmente, ajustar os parâmetros que serão usados pelo algoritmo genético (GA) para a otimização.

A filosofia de design continua a ser a de **abstração e simplicidade**. A interface não deve expor a complexidade bruta do arquivo TOML. Em vez disso, deve traduzir os parâmetros em controles interativos e compreensíveis, fornecendo contexto e orientação em cada passo. O usuário deve se sentir no controle, sem se sentir sobrecarregado.

## 2. Princípios de Design da UX

-   **Controle com Segurança (Safe Control)**: O usuário pode fazer ajustes, mas dentro de limites seguros e lógicos definidos pelo TPM. A interface deve impedir a inserção de valores inválidos ou ilógicos (ex: período de média móvel rápida maior que a lenta).

-   **Visualização de Causa e Efeito**: A interface deve, sempre que possível, dar uma indicação do que cada parâmetro faz. O uso de mini-gráficos, descrições claras e agrupamentos lógicos é fundamental.

-   **Modos Básico e Avançado**: Para atender tanto ao usuário leigo quanto ao avançado, a interface operará em dois modos. O modo **Básico** exibe apenas os 3-5 parâmetros mais impactantes e intuitivos. O modo **Avançado** revela todos os parâmetros otimizáveis, agrupados por seção (risco, entrada, saída, etc.).

-   **Padrões Inteligentes (Smart Defaults)**: A tela já vem pré-preenchida com os valores e ranges definidos no arquivo TOML do modelo selecionado. O usuário não é obrigado a mudar nada; ele pode simplesmente clicar em "Iniciar Otimização" para usar a configuração padrão.

## 3. Layout e Componentes da Tela

A tela é organizada em um layout de duas colunas ou em seções verticais claras para guiar o olhar do usuário.

### 3.1. Painel de Resumo da Estratégia (Topo da Tela)

Um cabeçalho fixo no topo da página relembra ao usuário qual modelo de estratégia ele está configurando. Este painel exibe:

-   **Nome da Estratégia**: Ex: "Configurando **Swing Momentum - MA Crossover**".
-   **Tags Principais**: As mesmas tags de risco e família da tela anterior.
-   **Botão de Trocar Modelo**: Um link discreto para voltar à tela de seleção caso o usuário mude de ideia.

### 3.2. Seção de Configuração de Parâmetros

Esta é a área principal da tela, onde os parâmetros são exibidos e podem ser editados.

-   **Toggle Básico/Avançado**: Um interruptor proeminente que permite ao usuário alternar entre os dois modos de visualização.

#### Modo Básico

Exibe apenas os parâmetros mais cruciais em uma única lista simples. Cada item da lista contém:

-   **Nome do Parâmetro**: Ex: "Período da Média Móvel Lenta".
-   **Controle Interativo**: O componente de UI apropriado para o tipo de parâmetro (ver seção 3.3).
-   **Ícone de Ajuda `(?)`**: Um tooltip que explica o que o parâmetro faz em linguagem simples. Ex: *"Define a janela de tempo para a média móvel de longo prazo. Valores maiores indicam uma tendência mais longa."*

#### Modo Avançado

Organiza todos os parâmetros otimizáveis em seções colapsáveis (accordions), espelhando a estrutura do arquivo TOML.

-   **`[parameters]` (Parâmetros da Estratégia)**
-   **`[exit_rules]` (Regras de Saída)**
-   **`[position_sizing]` (Tamanho da Posição)**
-   **`[risk_management]` (Gestão de Risco)**

Dentro de cada seção, os parâmetros são listados da mesma forma que no modo Básico.

### 3.3. Componentes de UI por Tipo de Parâmetro

A interface renderizará um controle diferente com base no formato do valor do parâmetro no arquivo TOML.

| Formato no TOML | Componente de UI | Comportamento |
| :--- | :--- | :--- |
| `[min, max]` (numérico) | **Slider de Range Duplo** | Permite ao usuário ajustar o valor mínimo e máximo do range que o GA irá explorar. |
| `[val1, val2, ...]` (string) | **Dropdown de Múltipla Seleção** | Permite ao usuário selecionar quais das opções categóricas o GA poderá usar. |
| Valor Único (fixo) | **Texto Não Editável** | Exibe o valor fixo para informação, mas não permite edição. |
| Valor Único (editável) | **Campo de Input Numérico** | Permite ao usuário definir um valor fixo diferente do padrão. |

**Exemplo Visual de um Slider de Range:**

**Período da Média Móvel Rápida** `(?)`

`[ 10 ] <-------------------[o]----[o]-------------------> [ 40 ]`

O usuário pode arrastar as duas alças para definir um novo range de busca para o GA, por exemplo, de `15` a `25`.

### 3.4. Painel de Ações (Fixo na Base da Tela)

Um painel fixo na parte inferior da tela contém as ações principais:

-   **Botão Primário**: "**Iniciar Otimização**". Este botão inicia o processo de backtesting e otimização com o motor genético.
-   **Botão Secundário**: "**Salvar como Template Customizado**". Permite que o usuário salve suas modificações como uma nova versão pessoal da estratégia.
-   **Link/Botão Terciário**: "**Restaurar Padrões**". Reverte todas as alterações feitas pelo usuário para os valores originais do modelo.

## 4. Fluxo do Usuário

```mermaid
graph TD
    A[Chega na tela de Configuração] --> B{Vê os parâmetros no modo Básico};
    B --> C{Ajusta o range de um parâmetro usando um slider};
    B --> D[Clica em "Iniciar Otimização"];
    C --> D;
    B --> E{Clica no toggle "Avançado"};
    E --> F{Expande a seção "Gestão de Risco"};
    F --> G{Altera o valor do "Drawdown Máximo"};
    G --> D;
    D --> H[Tela de Acompanhamento do Backtest];
```

## 5. Conclusão

A tela de configuração de parâmetros é uma ponte crítica entre a simplicidade desejada pelo usuário e a complexidade necessária para uma otimização de estratégia eficaz. Ao usar padrões inteligentes, divulgação progressiva (Básico/Avançado) e controles interativos e didáticos, a interface capacita o usuário a participar do processo de configuração de forma segura e informada. Isso desmistifica a otimização de parâmetros e a torna acessível, cumprindo o requisito central de facilidade de uso do projeto.

A próxima especificação abordará o **Sistema de Templates e Presets**, detalhando como os usuários podem salvar e gerenciar suas próprias configurações customizadas.
