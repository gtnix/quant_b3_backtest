# Especificação Lógica 9: Sistema de Templates e Presets

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação detalha o sistema que permite aos usuários salvar, gerenciar e reutilizar suas próprias configurações de estratégia customizadas. Este recurso, referido como **Sistema de Templates e Presets**, é fundamental para a personalização da plataforma e para a eficiência do fluxo de trabalho do usuário.

O objetivo é dar ao usuário a capacidade de ir além do catálogo de modelos pré-definidos do **Módulo de Parâmetros de Trade (TPM)**. Após ajustar os parâmetros de um modelo base para encontrar uma configuração que lhe agrade, o usuário pode salvá-la como um *template pessoal*. Isso cria uma biblioteca de estratégias customizadas, prontas para serem reutilizadas em futuras otimizações, sem a necessidade de reconfigurar tudo a cada vez.

## 2. Conceitos e Terminologia

-   **Modelo Base**: Uma das 116+ configurações de estratégia originais, somente leitura, fornecidas pelo TPM. São a base para toda a customização.
-   **Template Customizado (ou Preset)**: Uma cópia de um Modelo Base que foi modificada e salva pelo usuário. É um arquivo de configuração pessoal, editável e gerenciável pelo seu criador.

## 3. Arquitetura e Armazenamento

Os templates customizados são dados específicos do usuário e devem ser armazenados de forma isolada e segura.

-   **Formato de Armazenamento**: Os templates serão salvos como arquivos `.toml`, mantendo a consistência com os modelos base do TPM.
-   **Localização**: Será criada uma estrutura de diretórios para armazenar os dados de cada usuário. Os templates serão salvos em um subdiretório específico.

    ```
    /user_data/
    └── {user_id}/
        ├── templates/
        │   ├── meu_swing_agressivo.toml
        │   └── pair_petr_vale_conservador.toml
        └── backtest_results/
            └── ...
    ```

-   **Identificação**: Cada usuário será identificado por um `user_id` único, garantindo que o acesso aos templates seja restrito ao seu proprietário.

### Estrutura do TOML do Template Customizado

O arquivo TOML de um template customizado terá a mesma estrutura de um modelo base, com uma adição crucial na seção `[metadata]` para rastrear sua origem.

| Campo Adicional | Seção | Tipo | Obrigatório | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `base_strategy_id` | `metadata` | String | Sim | O `strategy_id` do Modelo Base a partir do qual este template foi criado. |
| `is_custom_template` | `metadata` | Boolean | Sim | Um flag `true` para identificar facilmente que este é um template de usuário. |

**Exemplo de `[metadata]` de um template customizado:**

```toml
[metadata]
# Campos definidos pelo usuário ao salvar
strategy_id = "meu_swing_agressivo_v1"
name = "Meu Swing Agressivo v1"
description = "Versão modificada do MA Crossover com stop mais curto."

# Campos herdados e de rastreamento
version = "1.0.0"
base_strategy_id = "swing_momentum_ma_crossover"
is_custom_template = true

# ... demais campos de metadados
```

## 4. Experiência do Usuário (UX)

### 4.1. Salvando um Novo Template

1.  Na tela de **Configuração de Parâmetros**, após fazer os ajustes desejados, o usuário clica no botão "**Salvar como Template**".
2.  Um modal é exibido, solicitando ao usuário que forneça:
    -   **Nome do Template** (obrigatório): Um nome legível para o seu preset. Ex: "Meu Setup de Reversão à Média".
    -   **Descrição** (opcional): Um campo para notas pessoais sobre o template.
3.  Ao confirmar, o sistema gera um novo arquivo `.toml` no diretório do usuário com as informações fornecidas.

### 4.2. Acessando e Usando Templates

1.  Na tela principal de **Seleção de Estratégias**, haverá uma nova aba ou um filtro proeminente no topo: `[ Modelos Base ] [ Meus Templates ]`.
2.  Ao selecionar "**Meus Templates**", a grade de cartões é atualizada para exibir apenas os templates salvos pelo usuário.
3.  O design do cartão para um template customizado será ligeiramente diferente para distingui-lo de um modelo base. Ele incluirá uma tag ou um texto indicando sua origem. Ex: "*Baseado em: Swing Momentum - MA Crossover*".
4.  A partir daqui, o fluxo é o mesmo: o usuário pode selecionar seu template, ir para a tela de configuração (que já estará com seus parâmetros salvos) e iniciar uma nova otimização.

### 4.3. Gerenciando Templates

Cada cartão na aba "Meus Templates" terá um menu de contexto (ícone de três pontos `...`) com as seguintes opções:

-   **Editar**: Abre um modal para alterar o nome e a descrição do template.
-   **Duplicar**: Cria uma cópia do template, permitindo que o usuário crie variações rapidamente.
-   **Excluir**: Remove permanentemente o arquivo do template, após uma confirmação para evitar exclusões acidentais.

## 5. Lógica de Backend

-   **API Endpoints**: O backend precisará de um novo conjunto de endpoints RESTful para gerenciar os templates, associados ao `user_id`:
    -   `GET /api/users/{user_id}/templates`: Lista todos os templates de um usuário.
    -   `POST /api/users/{user_id}/templates`: Cria um novo template.
    -   `GET /api/users/{user_id}/templates/{template_id}`: Carrega um template específico.
    -   `PUT /api/users/{user_id}/templates/{template_id}`: Atualiza o nome/descrição de um template.
    -   `DELETE /api/users/{user_id}/templates/{template_id}`: Exclui um template.

-   **Modificações no TPM Loader**: O `TPM Loader` pode ser estendido ou uma nova estrutura pode ser criada para lidar especificamente com os templates de usuário, lendo a partir dos diretórios de dados do usuário em vez do diretório global de modelos base.

## 6. Conclusão

O Sistema de Templates e Presets transforma a plataforma de uma ferramenta estática para um ambiente de trabalho dinâmico e personalizado. Ele valoriza o tempo e o esforço do usuário, permitindo que ele construa sua própria biblioteca de estratégias e refine suas ideias ao longo do tempo. Esta funcionalidade é essencial para aumentar o engajamento do usuário e tornar a ferramenta uma parte indispensável de seu processo de pesquisa e desenvolvimento de estratégias de trading.

A próxima especificação abordará o **Fluxo de Geração de Estratégias**, detalhando a jornada completa do usuário desde a seleção até a visualização dos resultados.
