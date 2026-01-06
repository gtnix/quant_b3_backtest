# Especificação Lógica 13: API e Interfaces de Integração

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação define o contrato técnico entre o frontend (Dashboard) e o backend (Motor Rust) através de uma **API RESTful**. O objetivo é detalhar cada endpoint, seu propósito, os dados que ele espera (request) e os dados que ele retorna (response). Uma API bem definida é crucial para o desacoplamento entre as camadas, permitindo que o desenvolvimento do frontend e do backend ocorra em paralelo e de forma independente.

A API será projetada seguindo as melhores práticas REST, utilizando os verbos HTTP (`GET`, `POST`, `PUT`, `DELETE`) de forma semântica e retornando códigos de status HTTP padrão para indicar o resultado das operações. Todos os dados serão trocados no formato **JSON**.

## 2. Visão Geral da API

A API será organizada em torno de três recursos principais:

1.  **Estratégias (`/strategies`)**: Para interagir com o catálogo de modelos base do TPM.
2.  **Templates (`/users/{user_id}/templates`)**: Para gerenciar os templates customizados de um usuário específico.
3.  **Otimizações (`/optimizations`)**: Para iniciar e monitorar os processos de otimização do algoritmo genético.

## 3. Definição dos Endpoints

### 3.1. Recurso: Estratégias

Este recurso fornece acesso de leitura ao catálogo global de modelos de estratégia.

#### **`GET /api/strategies`**

-   **Descrição**: Retorna uma lista de metadados de estratégias, com suporte para busca e filtragem.
-   **Parâmetros de Query**: Conforme definido na Especificação 11 (Sistema de Busca e Filtros).
    -   `q` (string), `risk_profile` (string[]), `family` (string[]), `sort_by` (string), `order` (string).
-   **Resposta de Sucesso (200 OK)**: Um array de objetos `Metadata`.
    ```json
    [
        {
            "strategy_id": "swing_ma_crossover_moderate",
            "name": "Swing Momentum - MA Crossover",
            "description": "...",
            "risk_profile": "moderate",
            "family": "swing",
            "tags": ["momentum", "crossover"]
        }
    ]
    ```

#### **`GET /api/strategies/{strategy_id}`**

-   **Descrição**: Retorna a configuração completa de um único modelo de estratégia.
-   **Parâmetros de URL**: `strategy_id` (string) - O ID da estratégia. Ex: `swing_ma_crossover_moderate`.
-   **Resposta de Sucesso (200 OK)**: Um objeto `StrategyConfig` completo, correspondente ao arquivo TOML.
-   **Resposta de Erro (404 Not Found)**: Se o `strategy_id` não existir.

### 3.2. Recurso: Templates Customizados

Estes endpoints são autenticados e associados a um `user_id`.

#### **`GET /api/users/{user_id}/templates`**

-   **Descrição**: Retorna a lista de metadados de todos os templates customizados de um usuário.
-   **Resposta de Sucesso (200 OK)**: Um array de objetos `Metadata` (incluindo `base_strategy_id`).

#### **`POST /api/users/{user_id}/templates`**

-   **Descrição**: Cria um novo template customizado.
-   **Corpo da Requisição (JSON)**: Um objeto contendo a configuração completa do novo template e os dados para o `[metadata]` (nome, descrição).
-   **Resposta de Sucesso (201 Created)**: Retorna o metadado do template recém-criado.
-   **Resposta de Erro (400 Bad Request)**: Se os dados forem inválidos.

#### **`PUT /api/users/{user_id}/templates/{template_id}`**

-   **Descrição**: Atualiza o nome e/ou a descrição de um template existente.
-   **Corpo da Requisição (JSON)**: `{ "name": "...", "description": "..." }`.
-   **Resposta de Sucesso (200 OK)**: Retorna o metadado atualizado.

#### **`DELETE /api/users/{user_id}/templates/{template_id}`**

-   **Descrição**: Exclui um template customizado.
-   **Resposta de Sucesso (204 No Content)**: Retorna uma resposta vazia.

### 3.3. Recurso: Otimizações

Este recurso gerencia o ciclo de vida dos processos de otimização.

#### **`POST /api/optimizations`**

-   **Descrição**: Inicia um novo processo de otimização.
-   **Corpo da Requisição (JSON)**: O objeto `StrategyConfig` completo (seja de um modelo base ou de um template customizado, possivelmente modificado na UI) que será usado para a otimização.
-   **Resposta de Sucesso (202 Accepted)**: A requisição foi aceita e enfileirada. O corpo da resposta contém um `optimization_id` para monitorar o processo.
    ```json
    {
        "optimization_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "status": "queued"
    }
    ```

#### **`GET /api/optimizations/{optimization_id}`**

-   **Descrição**: Retorna o status atual de um processo de otimização.
-   **Resposta de Sucesso (200 OK)**: Um objeto com o status e o progresso.
    ```json
    {
        "optimization_id": "...",
        "status": "running",
        "progress": {
            "current_generation": 50,
            "total_generations": 100,
            "best_fitness": 1.35
        }
    }
    ```
-   **Interface WebSocket**: Para uma experiência de usuário mais fluida, em vez de polling neste endpoint, o frontend deve se conectar a um endpoint WebSocket (`/ws/optimizations/{optimization_id}`) para receber atualizações de status em tempo real.

#### **`GET /api/optimizations/{optimization_id}/result`**

-   **Descrição**: Retorna o resultado final de uma otimização concluída.
-   **Resposta de Sucesso (200 OK)**: Um objeto JSON contendo o relatório de performance completo da melhor estratégia encontrada.
-   **Resposta de Erro (404 Not Found)**: Se a otimização ainda não estiver concluída ou se o ID for inválido.

## 4. Contratos de Dados (Data Contracts)

As estruturas de dados trocadas (JSON) devem ser estritamente definidas. Em um ambiente Rust/TypeScript, isso pode ser garantido usando ferramentas como `ts-rs`, que gera definições TypeScript a partir de structs Rust. Isso garante que o frontend e o backend estejam sempre sincronizados em relação aos formatos de dados.

## 5. Autenticação e Autorização

-   Endpoints sob `/api/users/{user_id}/` devem ser protegidos e requerem autenticação (ex: via JWT - JSON Web Token).
-   O backend deve verificar se o usuário autenticado tem permissão para acessar os recursos do `user_id` especificado na URL, prevenindo que um usuário acesse os templates de outro.

## 6. Conclusão

Uma API RESTful bem definida, como a especificada acima, serve como uma espinha dorsal robusta para a comunicação entre o frontend e o backend. Ela estabelece limites claros de responsabilidade, promove o desenvolvimento desacoplado e garante que a interação entre as diferentes partes do sistema seja previsível e confiável. A utilização de contratos de dados e um tratamento de erros consistente são fundamentais para a manutenibilidade e a escalabilidade da plataforma.

A próxima especificação abordará as **Métricas e Otimização Computacional**, detalhando as métricas de performance que serão calculadas e como o sistema será otimizado para performance.
