# Especificação Lógica 10: Fluxo de Geração de Estratégias

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação consolida as especificações anteriores para descrever a jornada completa do usuário no processo de geração de uma estratégia de trading otimizada. O **Fluxo de Geração de Estratégias** é a sequência de ponta a ponta, desde a tela inicial do dashboard até a visualização dos resultados do backtest de uma nova estratégia criada pelo algoritmo genético (GA).

O objetivo é mapear cada passo, interação e estado do sistema, garantindo uma experiência de usuário coesa, transparente e livre de atritos. O fluxo é projetado para ser linear e progressivo, guiando o usuário naturalmente através de quatro etapas principais: **Seleção**, **Configuração**, **Otimização** e **Análise**.

## 2. As Quatro Etapas do Fluxo de Geração

O processo completo pode ser visualizado como um funil, onde o usuário começa com um universo de possibilidades e termina com uma única estratégia otimizada e validada.

| Etapa | Tela Principal | Objetivo do Usuário | Resultado da Etapa |
| :--- | :--- | :--- | :--- |
| 1. **Seleção** | Catálogo de Estratégias | "Quero encontrar um ponto de partida para minha ideia." | Um Modelo Base ou Template Customizado é selecionado. |
| 2. **Configuração** | Configuração de Parâmetros | "Quero revisar e talvez ajustar os limites da otimização." | Uma configuração final é enviada para o motor de GA. |
| 3. **Otimização** | Monitor de Otimização | "Quero acompanhar o progresso da busca pela melhor estratégia." | O processo do GA é executado. |
| 4. **Análise** | Relatório de Performance | "Quero entender se a estratégia encontrada é boa e como ela funciona." | Um relatório detalhado do backtest da melhor estratégia encontrada. |

## 3. Diagrama do Fluxo de Ponta a Ponta

O diagrama a seguir ilustra a jornada do usuário através das diferentes telas e estados do sistema.

```mermaid
flowchart TD
    subgraph Etapa 1: Seleção
        A[Tela: Catálogo de Estratégias] -->|Filtra e explora| A;
        A -->|Seleciona um cartão| B[Tela: Configuração de Parâmetros];
    end

    subgraph Etapa 2: Configuração
        B -->|Ajusta os ranges| B;
        B -->|Clica em "Salvar Template"| B;
        B -->|Clica em "Iniciar Otimização"| C[Tela: Monitor de Otimização];
    end

    subgraph Etapa 3: Otimização
        C -->> D{Processo em Background: GA};
        D -- Em andamento --> C;
        style C fill:#f9f,stroke:#333,stroke-width:2px
    end

    subgraph Etapa 4: Análise
        D -- Concluído --> E[Tela: Relatório de Performance];
        D -- Falha --> F[Notificação de Erro];
    end

    E -->|Clica em "Salvar Estratégia"| G[Estratégia salva no portfólio do usuário];
    E -->|Clica em "Rodar Novamente"| B;
```

## 4. Detalhamento das Etapas

### Etapa 1: Seleção

-   **Ponto de Partida**: O usuário acessa a página "Gerador de Estratégias" no dashboard.
-   **Ação**: O usuário utiliza os filtros (risco, família) e a busca para encontrar um modelo que lhe interesse. Ele pode alternar entre os "Modelos Base" do sistema e seus "Templates Customizados".
-   **Interação**: Clica no cartão da estratégia desejada.
-   **Transição**: O sistema carrega a configuração TOML correspondente e navega para a tela de Configuração de Parâmetros.

### Etapa 2: Configuração

-   **Ponto de Partida**: A tela exibe os parâmetros do modelo selecionado, com os ranges de otimização pré-definidos.
-   **Ação**: O usuário pode (opcionalmente) ajustar os ranges dos parâmetros que o GA irá explorar. Ele pode usar o modo Básico ou Avançado.
-   **Interação Principal**: Clica no botão "**Iniciar Otimização**".
-   **Transição**: A UI envia a configuração final (o objeto TOML modificado em memória) para o backend, que enfileira uma nova tarefa de otimização para o motor genético. O frontend navega para a tela de Monitor de Otimização.

### Etapa 3: Otimização (Processo Assíncrono)

-   **Ponto de Partida**: A tela de Monitor de Otimização exibe um estado de "*Processando...*" ou "*Buscando a melhor estratégia...*".
-   **Feedback para o Usuário**: A tela deve fornecer feedback em tempo real sobre o progresso do GA, para que o usuário saiba que o sistema está trabalhando. As seguintes informações devem ser atualizadas periodicamente via WebSocket ou polling:
    -   **Geração Atual**: `Geração 5 de 100`.
    -   **Melhor Fitness Até Agora**: `Sharpe Ratio: 1.25`.
    -   **Tempo Decorrido / Estimativa Restante**.
    -   Um gráfico simples mostrando a evolução do fitness da melhor estratégia ao longo das gerações.
-   **Estados Possíveis**:
    -   `Em Fila`: A tarefa foi recebida, mas aguarda recursos computacionais.
    -   `Rodando`: O GA está em execução.
    -   `Concluído`: O GA terminou e encontrou uma solução.
    -   `Falhou`: Ocorreu um erro irrecuperável durante o processo.
-   **Transição**: Quando o backend informa que o processo foi concluído, o frontend navega automaticamente para a tela de Relatório de Performance, passando o ID do resultado do backtest.

### Etapa 4: Análise

-   **Ponto de Partida**: A tela de Relatório de Performance exibe os resultados detalhados do backtest da **melhor estratégia** encontrada pelo GA.
-   **Conteúdo**: Esta tela é o resultado final do fluxo. Ela deve conter:
    -   Um resumo dos parâmetros da estratégia vencedora.
    -   Métricas de performance chave (Sharpe, Calmar, Drawdown, etc.).
    -   Gráfico da curva de capital.
    -   Gráfico de drawdown.
    -   Estatísticas de trades (taxa de acerto, payoff, etc.).
    -   Comparativo de performance In-Sample vs. Out-of-Sample.
-   **Ações do Usuário**:
    -   **Salvar Estratégia**: Adiciona a estratégia otimizada a uma lista de "Minhas Estratégias Salvas" para futuro monitoramento ou execução.
    -   **Exportar Relatório**: Gera um PDF com os resultados do backtest.
    -   **Rodar Novamente com Ajustes**: Leva o usuário de volta para a tela de Configuração (Etapa 2), com os parâmetros da estratégia vencedora já carregados, permitindo um novo ciclo de refinamento.

## 5. Conclusão

O Fluxo de Geração de Estratégias é projetado para ser uma jornada lógica e transparente. Ao dividir o processo em quatro etapas claras e fornecer feedback constante ao usuário, o sistema transforma uma operação de backend complexa e demorada em uma experiência interativa e gerenciável. Isso não apenas melhora a usabilidade, mas também aumenta a confiança do usuário nos resultados, pois ele pode acompanhar todo o processo de descoberta.

A próxima especificação abordará o **Sistema de Busca e Filtros**, detalhando a implementação técnica por trás da funcionalidade de descoberta na primeira etapa do fluxo.
