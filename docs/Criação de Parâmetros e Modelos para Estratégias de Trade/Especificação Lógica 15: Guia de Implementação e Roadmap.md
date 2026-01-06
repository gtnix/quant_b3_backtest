# Especificação Lógica 15: Guia de Implementação e Roadmap

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação final serve como um guia prático e um roadmap para a implementação do **Módulo de Parâmetros de Trade (TPM)** e sua integração com o ecossistema `quant_b3_backtest`. O objetivo é traduzir as 14 especificações lógicas anteriores em um plano de ação sequencial e faseado, que pode ser usado pela equipe de desenvolvimento para construir e entregar o projeto de forma incremental.

O roadmap é dividido em três fases principais, cada uma entregando um conjunto de funcionalidades que agregam valor e podem ser testadas de forma independente: **Fundação**, **Integração com o Motor** e **Interface do Usuário**.

## 2. Roadmap de Implementação

### Fase 1: A Fundação do TPM (Backend)

O objetivo desta fase é construir o núcleo do TPM como um componente de backend funcional e testável, sem qualquer interface de usuário.

| Tarefa | Descrição | Dependências | Critério de Aceitação |
| :--- | :--- | :--- | :--- |
| **1.1. Criar o Catálogo TOML** | Criar manualmente os 116 arquivos `.toml` com base na Especificação 12 e salvá-los em `/configs/strategies/`. | Nenhuma | Todos os 116 arquivos estão criados e passam em um validador TOML básico. |
| **1.2. Desenvolver o Crate `tpm_loader`** | Criar um novo crate Rust no workspace. Implementar as structs (`StrategyConfig`, `Metadata`, etc.) e a lógica de deserialização com `serde`. | Tarefa 1.1 | O crate compila com sucesso. |
| **1.3. Implementar a Validação** | Adicionar o Sistema de Validação (Especificação 5) ao `tpm_loader`, incluindo as validações lógicas. | Tarefa 1.2 | Testes de unidade cobrem todos os cenários de validação (válidos e inválidos). |
| **1.4. Implementar o Indexador** | Criar a função `list_all_metadata()` que lê parcialmente os arquivos e constrói o índice em memória. | Tarefa 1.2 | A função retorna um `Vec<Metadata>` com 116 itens. |
| **1.5. Criar a API Básica** | Expor os endpoints `GET /api/strategies` e `GET /api/strategies/{id}` usando `axum`. O primeiro endpoint deve usar o índice da Tarefa 1.4. | Tarefa 1.3, 1.4 | Os endpoints estão funcionais e podem ser testados com `curl` ou Postman. |

**Entrega da Fase 1**: Um conjunto de endpoints de API que permitem consultar o catálogo de estratégias do TPM de forma programática.

### Fase 2: Integração com o Motor Genético (Backend)

O objetivo desta fase é conectar o TPM ao algoritmo genético (GA), permitindo a geração de estratégias guiadas.

| Tarefa | Descrição | Dependências | Critério de Aceitação |
| :--- | :--- | :--- | :--- |
| **2.1. Modificar o Motor de GA** | Refatorar o motor de GA para aceitar um objeto `StrategyConfig` como entrada. | Fase 1 | O motor de GA tem uma nova função de inicialização que recebe a configuração. |
| **2.2. Implementar a Leitura de Limites** | Fazer com que o GA leia os ranges e valores da seção `[parameters]` do `StrategyConfig` para definir o espaço de busca dos genes. | Tarefa 2.1 | A população inicial do GA é gerada com valores dentro dos limites especificados. |
| **2.3. Implementar a Função de Fitness Ponderada** | Implementar a função de fitness com pesos e penalidades, lendo os parâmetros da seção `[optimization]` do `StrategyConfig`. | Tarefa 2.1 | A avaliação de fitness de um cromossomo considera os pesos definidos no TOML. |
| **2.4. Criar a API de Otimização** | Implementar os endpoints `POST /api/optimizations` e `GET /api/optimizations/{id}`. O endpoint POST deve iniciar um processo de otimização assíncrono. | Tarefa 2.2, 2.3 | É possível iniciar uma otimização via API e consultar seu status. |

**Entrega da Fase 2**: A capacidade de iniciar uma otimização de estratégia guiada pelo TPM através de uma chamada de API.

### Fase 3: Interface do Usuário (Frontend)

O objetivo desta fase é construir a experiência do usuário no dashboard para interagir com o TPM.

| Tarefa | Descrição | Dependências | Critério de Aceitação |
| :--- | :--- | :--- | :--- |
| **3.1. Tela de Seleção de Estratégias** | Desenvolver a página do catálogo com a grade de cartões e o painel de filtros, consumindo a API `GET /api/strategies`. | Tarefa 1.5 | A tela exibe os 116 modelos e permite filtrá-los e buscá-los. |
| **3.2. Modal de Detalhes da Estratégia** | Implementar o modal que exibe os detalhes de uma estratégia ao clicar em um cartão. | Tarefa 3.1 | O modal exibe a descrição e os parâmetros chave de forma legível. |
| **3.3. Tela de Configuração de Parâmetros** | Desenvolver a tela que permite ao usuário ajustar os ranges de otimização, com os modos Básico e Avançado. | Tarefa 3.2 | O usuário pode modificar os parâmetros e os valores são atualizados no estado do componente. |
| **3.4. Integração do Fluxo de Otimização** | Conectar o botão "Iniciar Otimização" ao endpoint `POST /api/optimizations`. Desenvolver a tela de monitoramento que consome o status via WebSocket. | Tarefa 2.4, 3.3 | O usuário pode iniciar uma otimização a partir da UI e acompanhar seu progresso. |
| **3.5. Tela de Relatório de Performance** | Desenvolver a tela que exibe os resultados finais da otimização, consumindo o endpoint `GET /api/optimizations/{id}/result`. | Tarefa 3.4 | O relatório completo do backtest da melhor estratégia é exibido ao final do processo. |
| **3.6. Sistema de Templates Customizados** | Implementar a lógica de UI e os endpoints de API (recurso `/users/.../templates`) para salvar e gerenciar os presets do usuário. | Fase 1, 2, 3 | O usuário pode salvar, carregar e gerenciar suas próprias configurações. |

**Entrega da Fase 3**: Uma experiência de usuário completa e funcional para gerar estratégias de trading otimizadas usando o TPM.

## 3. Stack de Tecnologia Recomendada

-   **Backend**: Rust, com o framework web `axum` para a API, `serde` para serialização, `tokio` para o runtime assíncrono e `rayon` para paralelismo.
-   **Frontend**: TypeScript, com o framework `React` (ou o que já estiver em uso no projeto `dashboard`) e uma biblioteca de componentes como `MUI` ou `Chakra UI` para construir a UI rapidamente.
-   **Comunicação em Tempo Real**: WebSockets para a tela de monitoramento de otimização.

## 4. Conclusão Final do Projeto de Especificação

Este conjunto de 15 especificações lógicas fornece um blueprint completo e detalhado para a construção do Módulo de Parâmetros de Trade. Ele cobre desde a visão arquitetural de alto nível até os detalhes de implementação da API, a lógica de integração com o motor de backtesting e a experiência do usuário final.

Ao seguir este guia, a equipe de desenvolvimento terá um caminho claro para construir um sistema que é, ao mesmo tempo, poderoso para o motor de busca e simples para o usuário final. O resultado será uma ferramenta que cumpre a visão original do projeto: capacitar usuários leigos a gerar estratégias de trading sofisticadas e otimizadas, aproveitando o conhecimento de mercado encapsulado no catálogo do TPM para guiar a inteligência artificial do algoritmo genético. A implementação deste módulo representa um salto qualitativo na capacidade e na usabilidade da plataforma `quant_b3_backtest`.
