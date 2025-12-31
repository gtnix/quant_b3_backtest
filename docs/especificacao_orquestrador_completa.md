# Especificação Técnica Completa: Sistema de Mineração Perpétua de Estratégias Quantitativas

**Autor**: Manus AI  
**Versão**: 1.0  
**Data**: 29 de Dezembro de 2025  
**Status**: Especificação de Arquitetura para Implementação

---

## Sumário Executivo

Este documento apresenta a especificação técnica completa do **Orquestrador de Mineração Perpétua (OMP)**, um sistema de software projetado para operar continuamente (24 horas por dia, 7 dias por semana) sobre a infraestrutura de backtesting quantitativo existente do projeto *Quant B3 Backtester*. O objetivo principal do OMP é industrializar o processo de descoberta de estratégias de trading, transformando-o de um processo manual baseado em campanhas discretas para uma operação de fabricação automatizada e perpétua, inspirada nas práticas dos fundos quantitativos mais bem-sucedidos do mundo.

A arquitetura proposta integra-se perfeitamente ao ecossistema tecnológico existente em Rust, aproveitando o Sistema Combinador Generativo (SCG), o Strategy Factory e a base de dados Neon PostgreSQL, enquanto introduz uma camada de orquestração de nível superior para operação contínua. O sistema foi projetado para maximizar a utilização dos recursos computacionais disponíveis na VPS Linux, implementar validação rigorosa anti-overfitting e promover automaticamente as estratégias de elite para um "Hall da Fama" auditável.

---

## 1. Contexto e Motivação

### 1.1. Estado Atual do Sistema

O projeto *Quant B3 Backtester* representa um sistema institucional de backtesting para os mercados B3 (Brasil) e US, construído em Rust com foco em determinismo, performance e precisão. O sistema atual possui dois subsistemas principais:

**Backtester Engine**: Motor de simulação determinístico de alta performance, capaz de processar estratégias com speedups de até 124x via otimizações Structure-of-Arrays (SoA) e zero-allocation no hot path.

**Sistema Combinador Generativo (SCG)**: Framework de descoberta evolutiva de estratégias utilizando algoritmos genéticos, torneios evolutivos e otimização multi-objetivo baseada em fronteiras de Pareto. O SCG opera através do `combiner_cli` e do `Strategy Factory`, que gerenciam campanhas de mineração com múltiplas seeds, validação walk-forward e stress testing integrado.

O sistema atual já possui capacidades sofisticadas de descoberta de estratégias, mas opera em modo "batch" ou "campanha", onde um operador humano precisa iniciar manualmente cada run de mineração. Esta abordagem apresenta limitações significativas para a operação de um fundo quantitativo moderno:

**Subutilização de recursos**: A VPS permanece ociosa entre campanhas, desperdiçando poder computacional valioso.

**Latência na descoberta**: O tempo entre a conclusão de uma campanha e o início da próxima representa oportunidades perdidas de descobrir novas estratégias.

**Falta de adaptabilidade**: O sistema não pode responder automaticamente a mudanças nas condições de mercado ou ajustar dinamicamente seus parâmetros de busca.

**Gestão manual de campeãs**: A identificação e promoção de estratégias de elite requer intervenção manual, introduzindo atrasos e potencial para erro humano.

### 1.2. Inspiração da Indústria

A pesquisa sobre as práticas dos fundos quantitativos de elite revela padrões consistentes que fundamentam o design do OMP:

**Renaissance Technologies** desenvolveu um processo sistemático de 3 passos para descoberta de "tradeable effects": (1) identificar anomalias em dados históricos, (2) validar significância estatística e consistência temporal, (3) garantir explicabilidade parcial. O Medallion Fund, operando com este processo, gerou retornos anualizados brutos de 66% por 30 anos consecutivos, com um win rate de apenas 50.75% mas magnitude assimétrica favorável [1]. A chave do sucesso foi a operação contínua e sistemática, não a busca por estratégias perfeitas.

**Two Sigma** implementou otimização de parâmetros em escala industrial, utilizando Bayesian optimization e paralelização assíncrona em clusters massivos. A abordagem assíncrona, onde novos jobs são despachados assim que um worker fica livre (em vez de aguardar conclusão de batches síncronos), resultou em speedups de 8x no tempo de tuning de modelos complexos [2]. Esta arquitetura maximiza a utilização de recursos computacionais, um princípio fundamental para o OMP.

**Infraestrutura institucional** moderna de hedge funds, conforme documentado pela Arcesium, enfatiza data platforms com 4 pilares: qualidade de dados, observabilidade, descoberta e governança. Sistemas de produção operam com ingestão contínua de dados, monitoramento 24/7 de pipelines e validação automática de integridade [3]. O OMP adota estes princípios para garantir robustez operacional.

### 1.3. Objetivos do Orquestrador

O Orquestrador de Mineração Perpétua foi projetado para atingir os seguintes objetivos:

**Operação Contínua**: Executar ciclos de mineração de estratégias 24/7 sem intervenção manual, maximizando a descoberta de padrões lucrativos.

**Gestão Inteligente de Recursos**: Monitorar e controlar dinamicamente o uso de CPU, memória e disco da VPS, ajustando a intensidade das campanhas para evitar sobrecarga do sistema.

**Promoção Automatizada**: Identificar e promover automaticamente estratégias que atendam critérios rigorosos de robustez (Sharpe Ratio, PBO, DSR, stress tests) para um "Hall da Fama" auditável.

**Configuração Dinâmica**: Permitir ajustes em tempo real dos parâmetros de mineração (universos de ativos, tipos de blocos, orçamentos computacionais) sem reiniciar o daemon.

**Auditabilidade Total**: Garantir rastreabilidade completa de cada estratégia promovida, com proveniência imutável (genome hash, seed, config hash, Git SHA).

**Resiliência**: Recuperar-se automaticamente de falhas (interrupções de rede, erros em backtests individuais) sem perder o progresso da mineração.

---

## 2. Princípios de Arquitetura

O design do OMP é fundamentado em princípios de engenharia de software robustos e práticas comprovadas da indústria quantitativa.

### 2.1. Princípios Fundamentais

| Princípio | Descrição | Justificativa |
|---|---|---|
| **Operação Perpétua** | O sistema deve ser um daemon de longa duração, projetado para funcionar 24/7 sem intervenção manual. | Maximiza a descoberta de estratégias ao eliminar períodos ociosos entre campanhas. Inspirado na operação contínua do Medallion Fund. |
| **Arquitetura Assíncrona** | Execução assíncrona de tarefas de backtesting, onde novos jobs são despachados assim que workers ficam livres. | Maximiza utilização de recursos computacionais. Baseado nas práticas de Two Sigma que resultaram em speedups de 8x [2]. |
| **Gestão Inteligente de Recursos** | Monitoramento ativo de CPU, memória e disco com ajuste dinâmico da carga de trabalho. | Previne sobrecarga do sistema e garante estabilidade operacional em ambientes com recursos limitados (VPS). |
| **Robustez e Resiliência** | Capacidade de recuperação automática de falhas, isolamento de erros e retomada de campanhas interrompidas. | Essencial para operação 24/7 sem supervisão humana constante. Falhas individuais não devem parar o processo geral. |
| **Configuração Dinâmica** | Ajuste de parâmetros em tempo real através de arquivos de configuração, sem reiniciar o daemon. | Permite adaptação rápida a mudanças nas condições de mercado ou ajustes na estratégia de busca. |
| **Promoção Automatizada** | Pipeline automatizado para identificar, validar e promover estratégias de elite com base em critérios rigorosos. | Elimina latência na identificação de campeãs e garante aplicação consistente de critérios de qualidade. |
| **Auditabilidade Completa** | Proveniência imutável de cada estratégia promovida, com rastreabilidade total (genome hash, seed, config hash, Git SHA). | Fundamental para reprodutibilidade científica e compliance regulatório. Permite reconstrução exata de qualquer estratégia. |
| **Separação de Responsabilidades** | O orquestrador não reimplementa lógica de backtesting ou evolução genética, apenas coordena componentes existentes. | Reduz complexidade, facilita manutenção e aproveita código já validado e otimizado. |

### 2.2. Decisões de Design Críticas

**Rust como Linguagem de Implementação**: O OMP será implementado em Rust para garantir segurança de memória, performance nativa e integração perfeita com o ecossistema existente. Rust é a escolha natural para ferramentas financeiras core que requerem cálculos robustos e determinismo [4].

**Daemon vs. Cron Jobs**: A arquitetura de daemon foi escolhida em vez de cron jobs porque permite controle fino sobre a execução, monitoramento contínuo de recursos e resposta rápida a eventos. Cron jobs introduziriam latência desnecessária e não permitiriam ajuste dinâmico de carga.

**Fila Baseada em Arquivo vs. Message Queue**: Uma fila de campanhas baseada em arquivo JSON foi escolhida por simplicidade e transparência. Para um sistema single-node (VPS), uma message queue distribuída (Redis, RabbitMQ) seria over-engineering. O arquivo JSON pode ser editado manualmente e versionado no Git.

**Neon PostgreSQL para Persistência**: A escolha de continuar usando Neon PostgreSQL (já utilizado pelo Strategy Factory) garante consistência de dados e aproveita a infraestrutura existente. A nova tabela `hall_of_fame` será adicionada ao schema existente.

**PM2 para Gestão de Processo**: O uso de PM2 como gerenciador de processos é recomendado por sua robustez, capacidade de auto-restart em caso de falha e facilidade de integração com logs e monitoramento.

---

## 3. Arquitetura do Sistema

### 3.1. Visão Geral da Arquitetura

O OMP opera como uma camada de orquestração sobre o ecossistema existente, coordenando a execução de campanhas do SCG, monitorando recursos e promovendo estratégias de elite. A arquitetura é composta por cinco componentes principais:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VPS LINUX (24/7)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              ORQUESTRADOR DE MINERAÇÃO PERPÉTUA                  │  │
│  │                  (orchestrator_daemon)                           │  │
│  │                                                                  │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │  │
│  │  │ Loop Principal │  │ Gestor de      │  │ Monitor de     │    │  │
│  │  │ (Orquestração) │  │ Recursos       │  │ Campanhas      │    │  │
│  │  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘    │  │
│  │           │                   │                   │             │  │
│  │           └───────────────────┴───────────────────┘             │  │
│  │                              │                                  │  │
│  │  ┌───────────────────────────┴──────────────────────────────┐  │  │
│  │  │              Pipeline de Promoção                        │  │
│  │  │  (Análise de Candidatos → Validação → Hall da Fama)     │  │
│  │  └──────────────────────────────────────────────────────────┘  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CAMADA DE CONTROLE                            │  │
│  │  ┌────────────────────┐        ┌────────────────────┐            │  │
│  │  │ campaign_queue.json│◄───────┤  Dashboard         │            │  │
│  │  │ (Fila de Campanhas)│        │  (Cockpit)         │            │  │
│  │  └────────────────────┘        └────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                CAMADA DE EXECUÇÃO (EXISTENTE)                    │  │
│  │  ┌────────────────────┐        ┌────────────────────┐            │  │
│  │  │  combiner_cli      │───────►│  Backtester        │            │  │
│  │  │  (factory run)     │        │  Paralelo (Rayon)  │            │  │
│  │  └────────────────────┘        └────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  CAMADA DE PERSISTÊNCIA                          │  │
│  │                    (Neon PostgreSQL)                             │  │
│  │  ┌────────────────────┐        ┌────────────────────┐            │  │
│  │  │ Tabelas Existentes │        │ Nova Tabela:       │            │  │
│  │  │ - scg_campaigns    │        │ - hall_of_fame     │            │  │
│  │  │ - scg_runs         │        │                    │            │  │
│  │  │ - scg_candidates   │        │                    │            │  │
│  │  └────────────────────┘        └────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Componentes Detalhados

#### 3.2.1. Daemon Orquestrador (`orchestrator_daemon`)

O daemon orquestrador é o componente central do OMP, implementado como um binário Rust de longa duração.

**Responsabilidades**:
- Executar o loop principal de orquestração
- Gerenciar a fila de campanhas
- Monitorar recursos do sistema
- Invocar e monitorar execuções do `combiner_cli`
- Analisar resultados de campanhas
- Acionar o pipeline de promoção

**Estrutura do Loop Principal**:

```rust
// Pseudocódigo do loop principal
loop {
    // 1. Verificar fila de campanhas
    let next_campaign = campaign_queue.peek();
    
    // 2. Verificar recursos disponíveis
    let resources = resource_manager.check_availability();
    
    // 3. Decidir ação
    if resources.can_start_campaign() && next_campaign.is_some() {
        // 4. Lançar campanha
        let campaign = campaign_queue.pop();
        let process = launch_campaign(campaign);
        active_campaigns.push(process);
    }
    
    // 5. Monitorar campanhas ativas
    for campaign in &mut active_campaigns {
        if campaign.is_finished() {
            // 6. Analisar resultados
            let results = analyze_campaign_results(campaign);
            
            // 7. Promover campeãs
            promotion_pipeline.process(results);
            
            // Remover campanha concluída
            active_campaigns.remove(campaign);
        }
    }
    
    // 8. Aguardar próximo ciclo (e.g., 30 segundos)
    sleep(Duration::from_secs(30));
}
```

**Gestão de Processos Filhos**:

O daemon utilizará a biblioteca `tokio` para gerenciamento assíncrono de processos filhos. Cada invocação de `combiner factory run` será executada como um processo filho, com captura de `stdout` e `stderr` para parsing de logs JSON em tempo real.

```rust
use tokio::process::Command;

async fn launch_campaign(config_path: &str) -> Result<CampaignProcess> {
    let mut child = Command::new("combiner")
        .args(&["factory", "run", "--campaign", config_path, "--json-logs"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    // Capturar stdout para parsing de logs
    let stdout = child.stdout.take().unwrap();
    
    Ok(CampaignProcess {
        child,
        stdout,
        start_time: Instant::now(),
    })
}
```

#### 3.2.2. Gestor de Recursos

O gestor de recursos monitora continuamente a utilização de CPU, memória e disco da VPS, implementando políticas de controle para evitar sobrecarga.

**Bibliotecas Utilizadas**:
- `sysinfo`: Para coleta de métricas de sistema (CPU, memória, disco)
- `tokio`: Para operações assíncronas de monitoramento

**Políticas de Controle** (configuráveis via arquivo TOML):

```toml
[resource_limits]
max_cpu_util_pct = 85.0          # Não iniciar novas campanhas se CPU > 85%
min_mem_available_mb = 512       # Não iniciar se memória livre < 512MB
min_disk_free_gb = 10.0          # Pausar mineração se disco < 10GB
max_concurrent_campaigns = 1     # Número máximo de campanhas simultâneas
check_interval_secs = 30         # Intervalo de verificação de recursos
```

**Lógica de Decisão**:

```rust
impl ResourceManager {
    fn can_start_campaign(&self) -> bool {
        let cpu_ok = self.cpu_usage_pct < self.config.max_cpu_util_pct;
        let mem_ok = self.available_mem_mb > self.config.min_mem_available_mb;
        let disk_ok = self.free_disk_gb > self.config.min_disk_free_gb;
        let concurrency_ok = self.active_campaigns < self.config.max_concurrent_campaigns;
        
        cpu_ok && mem_ok && disk_ok && concurrency_ok
    }
}
```

#### 3.2.3. Fila de Campanhas (`campaign_queue.json`)

A fila de campanhas é um arquivo JSON no disco que atua como um buffer de comando, permitindo controle dinâmico do que o orquestrador executará.

**Localização**: `/home/ubuntu/quant-b3/orchestrator/campaign_queue.json`

**Estrutura**:

```json
{
  "version": "1.0",
  "campaigns": [
    {
      "id": "camp_001",
      "name": "B3 Momentum Deep Dive",
      "config_path": "/home/ubuntu/quant-b3/configs/campaigns/b3_momentum_deep.toml",
      "priority": 1,
      "enabled": true,
      "repeat": false,
      "tags": ["b3", "momentum"]
    },
    {
      "id": "camp_002",
      "name": "US Mean Reversion",
      "config_path": "/home/ubuntu/quant-b3/configs/campaigns/us_mean_reversion.toml",
      "priority": 2,
      "enabled": true,
      "repeat": true,
      "tags": ["us", "mean-reversion"]
    }
  ]
}
```

**Operações Suportadas**:
- **Adicionar**: Inserir nova campanha na fila
- **Remover**: Deletar campanha da fila
- **Reordenar**: Ajustar prioridades
- **Habilitar/Desabilitar**: Controlar quais campanhas estão ativas
- **Repeat Mode**: Campanhas com `repeat: true` são automaticamente re-adicionadas à fila após conclusão

**Sincronização**: O daemon recarrega o arquivo a cada ciclo do loop, permitindo modificações em tempo real sem reiniciar o processo.

#### 3.2.4. Monitor de Campanhas

O monitor de campanhas rastreia o progresso de execuções ativas, parseando logs JSON em tempo real e atualizando métricas.

**Métricas Rastreadas**:
- Geração atual do algoritmo genético
- Número de seeds completadas
- Melhor fitness encontrado até o momento
- Tempo decorrido
- Taxa de conclusão de backtests

**Parsing de Logs JSON**:

O `combiner_cli` com flag `--json-logs` emite eventos estruturados que podem ser parseados:

```json
{"timestamp":"2025-12-29T21:00:00Z","level":"INFO","event":"campaign_started","campaign_id":"camp_001"}
{"timestamp":"2025-12-29T21:05:00Z","level":"INFO","event":"generation_completed","generation":10,"best_fitness":1.234}
{"timestamp":"2025-12-29T21:30:00Z","level":"INFO","event":"campaign_completed","campaign_id":"camp_001","duration_secs":1800}
```

O monitor parseia estes eventos e atualiza um dashboard interno de métricas.

#### 3.2.5. Pipeline de Promoção para o Hall da Fama

O pipeline de promoção é acionado automaticamente após a conclusão bem-sucedida de uma campanha. Ele identifica estratégias que atendem critérios rigorosos de robustez e as promove para um repositório de elite.

**Fluxo do Pipeline**:

1. **Exportar Top Candidatos**:
   ```bash
   combiner factory export-top --run <run_id> --top 100 --format json
   ```
   Gera um ranking determinístico dos melhores candidatos.

2. **Variance Sanity Gate (SEV-0)** *(implementado v2.2.0)*:
   
   Antes de processar candidatos, verifica se métricas não colapsaram:
   
   ```javascript
   // Bloqueia se variância ~0 (indica bug ou dados corrompidos)
   if (sharpeVar < 1e-6 || pboVar < 1e-8 || dsrVar < 1e-6) {
     return { blocked: true, reason: 'metrics_collapsed' };
   }
   ```
   
   **Endpoint**: `GET /api/omp/promote-check` para verificar sem promover.

3. **Aplicar Critérios de Promoção**:
   Para cada candidato no top 100, verificar:
   
   | Critério | Threshold | Justificativa |
   |---|---|---|
   | `oos_sharpe_net` | >= 1.0 | Sharpe Ratio líquido robusto no período out-of-sample |
   | `pbo` (Probability of Backtest Overfitting) | <= 0.10 | Baixa probabilidade de overfitting |
   | `dsr` (Deflated Sharpe Ratio) | >= 0.8 | Sharpe Ratio ajustado para múltiplos testes |
   | `max_drawdown_net` | <= 20% | Drawdown máximo aceitável |
   | `stress_passed` | == `stress_total` | Passar em todos os cenários de stress test |
   | `gates_passed` | == true | Passar em todos os institutional gates |
   | `genome_hash` | Único | Evitar duplicatas no Hall da Fama |

4. **Copiar Artefatos**:
   Se todos os critérios forem atendidos:
   ```bash
   cp -r artifacts/candidates/<candidate_id> hall_of_fame/<candidate_id>
   ```

5. **Inserir Registro no Banco de Dados**:
   ```sql
   INSERT INTO hall_of_fame (
       candidate_id, campaign_id, run_id, genome_hash,
       oos_sharpe_net, pbo, max_drawdown_net, cagr_net,
       artifacts_path, promoted_at, git_sha
   ) VALUES (...);
   ```

**Notificação**: Após promoção, o sistema pode enviar notificações via webhook, email ou integração com Slack/Discord para alertar o operador sobre novas estratégias de elite.

---

## 4. Estrutura de Dados

### 4.1. Nova Tabela: `hall_of_fame`

A tabela `hall_of_fame` armazena as estratégias promovidas, garantindo auditabilidade e rastreabilidade completa.

**Schema SQL**:

```sql
CREATE TABLE hall_of_fame (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id VARCHAR(255) NOT NULL,
    campaign_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) NOT NULL,
    genome_hash VARCHAR(255) NOT NULL UNIQUE,
    
    -- Métricas de Performance
    oos_sharpe_net DECIMAL(10, 4) NOT NULL,
    pbo DECIMAL(10, 4) NOT NULL,
    dsr DECIMAL(10, 4),
    max_drawdown_net DECIMAL(10, 4) NOT NULL,
    cagr_net DECIMAL(10, 4) NOT NULL,
    
    -- Métricas de Risco
    var_95 DECIMAL(10, 4),
    cvar_95 DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    
    -- Validação
    stress_passed INTEGER NOT NULL,
    stress_total INTEGER NOT NULL,
    gates_passed BOOLEAN NOT NULL,
    
    -- Proveniência
    artifacts_path VARCHAR(1024) NOT NULL,
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    git_sha VARCHAR(40) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    dataset_hash VARCHAR(64),
    
    -- Metadados
    tags TEXT[],
    notes TEXT,
    
    -- Índices para queries rápidas
    CONSTRAINT unique_genome UNIQUE (genome_hash)
);

CREATE INDEX idx_hall_of_fame_sharpe ON hall_of_fame (oos_sharpe_net DESC);
CREATE INDEX idx_hall_of_fame_promoted_at ON hall_of_fame (promoted_at DESC);
CREATE INDEX idx_hall_of_fame_campaign ON hall_of_fame (campaign_id);
```

### 4.2. Arquivo de Configuração do Orquestrador

O orquestrador é configurado via arquivo TOML, permitindo ajustes sem recompilação.

**Localização**: `/home/ubuntu/quant-b3/orchestrator/config.toml`

**Estrutura**:

```toml
[orchestrator]
name = "Quant B3 Strategy Miner"
version = "1.0.0"
loop_interval_secs = 30
log_level = "INFO"

[resource_limits]
max_cpu_util_pct = 85.0
min_mem_available_mb = 512
min_disk_free_gb = 10.0
max_concurrent_campaigns = 1
check_interval_secs = 30

[campaign_queue]
path = "/home/ubuntu/quant-b3/orchestrator/campaign_queue.json"
auto_reload = true

[promotion]
enabled = true
min_oos_sharpe_net = 1.0
max_pbo = 0.10
min_dsr = 0.8
max_drawdown_net = 0.20
require_all_stress_passed = true
require_gates_passed = true

[database]
connection_string = "${NEON_DATABASE_URL}"
max_connections = 10
connection_timeout_secs = 30

[notifications]
enabled = false
webhook_url = ""
slack_channel = ""

[artifacts]
base_path = "/home/ubuntu/quant-b3/artifacts"
hall_of_fame_path = "/home/ubuntu/quant-b3/hall_of_fame"
max_disk_usage_gb = 100.0
```

---

## 5. Fluxos de Operação

### 5.1. Inicialização do Sistema

**Passo 1: Preparação do Ambiente**

```bash
# Criar diretórios necessários
mkdir -p /home/ubuntu/quant-b3/orchestrator
mkdir -p /home/ubuntu/quant-b3/hall_of_fame
mkdir -p /home/ubuntu/quant-b3/logs

# Criar arquivo de configuração inicial
cp orchestrator/config.example.toml orchestrator/config.toml

# Criar fila de campanhas inicial
echo '{"version":"1.0","campaigns":[]}' > orchestrator/campaign_queue.json

# Configurar variável de ambiente do banco de dados
export NEON_DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
```

**Passo 2: Criar Tabela no Banco de Dados**

```bash
# Executar script SQL de criação da tabela hall_of_fame
psql $NEON_DATABASE_URL -f orchestrator/schema/hall_of_fame.sql
```

**Passo 3: Compilar e Instalar o Orquestrador**

```bash
# Compilar em modo release
cargo build --release --bin orchestrator_daemon

# Copiar binário para diretório de execução
cp target/release/orchestrator_daemon /usr/local/bin/
```

**Passo 4: Iniciar com PM2**

```bash
# Instalar PM2 (se necessário)
npm install -g pm2

# Iniciar daemon
pm2 start orchestrator_daemon \
    --name strategy-miner \
    --cwd /home/ubuntu/quant-b3 \
    --log /home/ubuntu/quant-b3/logs/orchestrator.log \
    --time

# Configurar auto-restart em reboot
pm2 startup
pm2 save
```

### 5.2. Adição de Campanha à Fila

O operador adiciona uma nova campanha editando o arquivo `campaign_queue.json`:

```bash
# Editar fila de campanhas
nano /home/ubuntu/quant-b3/orchestrator/campaign_queue.json
```

Adicionar nova entrada:

```json
{
  "id": "camp_003",
  "name": "B3 Low Volatility",
  "config_path": "/home/ubuntu/quant-b3/configs/campaigns/b3_low_vol.toml",
  "priority": 1,
  "enabled": true,
  "repeat": false,
  "tags": ["b3", "low-vol"]
}
```

O orquestrador detectará a mudança no próximo ciclo do loop (30 segundos) e iniciará a campanha quando recursos estiverem disponíveis.

### 5.3. Monitoramento de Progresso

**Via Logs**:

```bash
# Visualizar logs em tempo real
pm2 logs strategy-miner

# Visualizar logs com filtro
pm2 logs strategy-miner | grep "campaign_completed"
```

**Via Dashboard**:

O Cockpit do dashboard existente pode ser estendido para exibir:
- Campanhas ativas e seu progresso
- Histórico de campanhas completadas
- Estratégias promovidas ao Hall da Fama
- Utilização de recursos (CPU, memória, disco)

**Via Banco de Dados**:

```sql
-- Verificar campanhas recentes
SELECT * FROM scg_campaigns 
ORDER BY created_at DESC 
LIMIT 10;

-- Verificar estratégias no Hall da Fama
SELECT candidate_id, oos_sharpe_net, pbo, promoted_at 
FROM hall_of_fame 
ORDER BY promoted_at DESC 
LIMIT 20;
```

### 5.4. Promoção de Estratégia

Quando uma campanha é concluída, o pipeline de promoção é acionado automaticamente:

1. **Análise de Candidatos**: O orquestrador executa `combiner factory export-top` para obter ranking.
2. **Validação de Critérios**: Cada candidato no top 100 é verificado contra os critérios de promoção.
3. **Cópia de Artefatos**: Estratégias aprovadas têm seus artefatos copiados para `hall_of_fame/`.
4. **Registro no Banco**: Um registro é inserido na tabela `hall_of_fame`.
5. **Notificação**: (Opcional) Webhook ou notificação Slack é enviada.

**Exemplo de Log de Promoção**:

```json
{
  "timestamp": "2025-12-29T22:00:00Z",
  "level": "INFO",
  "event": "strategy_promoted",
  "candidate_id": "cand_abc123",
  "genome_hash": "sha256:def456",
  "oos_sharpe_net": 1.234,
  "pbo": 0.08,
  "hall_of_fame_id": "hof_xyz789"
}
```

### 5.5. Recuperação de Falhas

O orquestrador é projetado para ser resiliente a falhas:

**Falha de Processo Filho (combiner_cli)**:
- O daemon detecta a falha via código de saída não-zero.
- A campanha é marcada como "failed" no log.
- A campanha pode ser automaticamente re-tentada (configurável).
- Outras campanhas na fila continuam normalmente.

**Falha do Daemon**:
- PM2 automaticamente reinicia o processo.
- O daemon verifica campanhas em andamento ao reiniciar.
- Campanhas interrompidas podem ser retomadas usando `combiner factory resume`.

**Falha de Conexão com Banco de Dados**:
- O daemon implementa retry logic com backoff exponencial.
- Operações de escrita são enfileiradas e re-tentadas.
- Alertas são emitidos se a conexão não for restabelecida em 5 minutos.

---

## 6. Implementação Técnica

### 6.1. Estrutura do Projeto

```
quant-b3-backtester/
├── orchestrator_daemon/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                 # Entry point do daemon
│   │   ├── orchestrator.rs         # Loop principal de orquestração
│   │   ├── resource_manager.rs     # Gestão de recursos
│   │   ├── campaign_queue.rs       # Gerenciamento da fila
│   │   ├── campaign_monitor.rs     # Monitoramento de campanhas
│   │   ├── promotion_pipeline.rs   # Pipeline de promoção
│   │   ├── database.rs             # Operações de banco de dados
│   │   ├── config.rs               # Parsing de configuração
│   │   └── types.rs                # Tipos e estruturas de dados
│   └── tests/
│       └── integration_tests.rs
├── orchestrator/
│   ├── config.toml                 # Configuração do orquestrador
│   ├── campaign_queue.json         # Fila de campanhas
│   └── schema/
│       └── hall_of_fame.sql        # Schema da tabela
└── hall_of_fame/                   # Diretório de estratégias promovidas
```

### 6.2. Dependências Rust

**Cargo.toml**:

```toml
[package]
name = "orchestrator_daemon"
version = "1.0.0"
edition = "2021"

[dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-native-tls", "uuid", "chrono"] }
sysinfo = "0.30"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
anyhow = "1.0"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.6", features = ["serde", "v4"] }
```

### 6.3. Exemplo de Código: Loop Principal

```rust
use tokio::time::{sleep, Duration};
use anyhow::Result;

pub struct Orchestrator {
    config: OrchestratorConfig,
    resource_manager: ResourceManager,
    campaign_queue: CampaignQueue,
    campaign_monitor: CampaignMonitor,
    promotion_pipeline: PromotionPipeline,
    db_pool: sqlx::PgPool,
}

impl Orchestrator {
    pub async fn run(&mut self) -> Result<()> {
        tracing::info!("Orquestrador iniciado");
        
        loop {
            // 1. Recarregar fila de campanhas (se modificada)
            self.campaign_queue.reload_if_changed().await?;
            
            // 2. Verificar recursos disponíveis
            let resources = self.resource_manager.check_availability().await?;
            
            // 3. Decidir se pode iniciar nova campanha
            if resources.can_start_campaign() {
                if let Some(campaign) = self.campaign_queue.pop_next() {
                    tracing::info!(
                        campaign_id = %campaign.id,
                        "Iniciando campanha"
                    );
                    
                    // 4. Lançar campanha
                    let process = self.launch_campaign(&campaign).await?;
                    self.campaign_monitor.add(process);
                }
            }
            
            // 5. Monitorar campanhas ativas
            let finished = self.campaign_monitor.check_finished().await?;
            
            for campaign_result in finished {
                tracing::info!(
                    campaign_id = %campaign_result.id,
                    duration_secs = campaign_result.duration.as_secs(),
                    "Campanha concluída"
                );
                
                // 6. Analisar resultados e promover campeãs
                self.promotion_pipeline
                    .process(&campaign_result, &self.db_pool)
                    .await?;
            }
            
            // 7. Aguardar próximo ciclo
            sleep(Duration::from_secs(self.config.loop_interval_secs)).await;
        }
    }
    
    async fn launch_campaign(&self, campaign: &Campaign) -> Result<CampaignProcess> {
        let mut cmd = tokio::process::Command::new("combiner");
        cmd.args(&[
            "factory", "run",
            "--campaign", &campaign.config_path,
            "--json-logs"
        ]);
        
        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        Ok(CampaignProcess {
            id: campaign.id.clone(),
            child,
            start_time: Instant::now(),
        })
    }
}
```

### 6.4. Exemplo de Código: Pipeline de Promoção

```rust
pub struct PromotionPipeline {
    config: PromotionConfig,
}

impl PromotionPipeline {
    pub async fn process(
        &self,
        campaign_result: &CampaignResult,
        db_pool: &sqlx::PgPool,
    ) -> Result<Vec<PromotedStrategy>> {
        // 1. Exportar top candidatos
        let top_candidates = self.export_top_candidates(&campaign_result.run_id).await?;
        
        let mut promoted = Vec::new();
        
        // 2. Aplicar critérios de promoção
        for candidate in top_candidates {
            if self.meets_promotion_criteria(&candidate) {
                // 3. Verificar se genome_hash já existe no Hall da Fama
                let exists = sqlx::query_scalar::<_, bool>(
                    "SELECT EXISTS(SELECT 1 FROM hall_of_fame WHERE genome_hash = $1)"
                )
                .bind(&candidate.genome_hash)
                .fetch_one(db_pool)
                .await?;
                
                if !exists {
                    // 4. Copiar artefatos
                    self.copy_artifacts(&candidate).await?;
                    
                    // 5. Inserir no banco de dados
                    let hof_id = self.insert_hall_of_fame(db_pool, &candidate).await?;
                    
                    tracing::info!(
                        candidate_id = %candidate.id,
                        genome_hash = %candidate.genome_hash,
                        oos_sharpe_net = candidate.oos_sharpe_net,
                        "Estratégia promovida ao Hall da Fama"
                    );
                    
                    promoted.push(PromotedStrategy {
                        hof_id,
                        candidate,
                    });
                }
            }
        }
        
        Ok(promoted)
    }
    
    fn meets_promotion_criteria(&self, candidate: &Candidate) -> bool {
        candidate.oos_sharpe_net >= self.config.min_oos_sharpe_net
            && candidate.pbo <= self.config.max_pbo
            && candidate.dsr.unwrap_or(0.0) >= self.config.min_dsr
            && candidate.max_drawdown_net.abs() <= self.config.max_drawdown_net
            && (candidate.stress_passed == candidate.stress_total || !self.config.require_all_stress_passed)
            && (candidate.gates_passed || !self.config.require_gates_passed)
    }
}
```

---

## 7. Operação e Manutenção

### 7.1. Comandos de Gestão

**Iniciar o Orquestrador**:
```bash
pm2 start strategy-miner
```

**Parar o Orquestrador**:
```bash
pm2 stop strategy-miner
```

**Reiniciar o Orquestrador**:
```bash
pm2 restart strategy-miner
```

**Ver Logs**:
```bash
pm2 logs strategy-miner --lines 100
```

**Ver Status**:
```bash
pm2 status strategy-miner
```

**Ver Métricas de Recursos**:
```bash
pm2 monit
```

### 7.2. Ajustes de Configuração

Para ajustar parâmetros do orquestrador em tempo real:

1. Editar o arquivo de configuração:
   ```bash
   nano /home/ubuntu/quant-b3/orchestrator/config.toml
   ```

2. Modificar os parâmetros desejados (e.g., `max_cpu_util_pct`, `min_oos_sharpe_net`)

3. O daemon recarregará a configuração no próximo ciclo do loop (30 segundos)

**Nota**: Algumas mudanças (como `database.connection_string`) requerem reiniciar o daemon:
```bash
pm2 restart strategy-miner
```

### 7.3. Gestão da Fila de Campanhas

**Adicionar Campanha**:
```bash
# Editar fila
nano /home/ubuntu/quant-b3/orchestrator/campaign_queue.json

# Adicionar nova entrada no array "campaigns"
```

**Remover Campanha**:
```bash
# Editar fila e deletar a entrada correspondente
nano /home/ubuntu/quant-b3/orchestrator/campaign_queue.json
```

**Reordenar Prioridades**:
```bash
# Ajustar o campo "priority" das campanhas (menor = maior prioridade)
nano /home/ubuntu/quant-b3/orchestrator/campaign_queue.json
```

**Desabilitar Campanha Temporariamente**:
```json
{
  "id": "camp_001",
  "enabled": false,  // Campanha não será executada
  ...
}
```

### 7.4. Consultas ao Hall da Fama

**Top 10 Estratégias por Sharpe Ratio**:
```sql
SELECT candidate_id, oos_sharpe_net, pbo, max_drawdown_net, promoted_at
FROM hall_of_fame
ORDER BY oos_sharpe_net DESC
LIMIT 10;
```

**Estratégias Promovidas nas Últimas 24 Horas**:
```sql
SELECT candidate_id, oos_sharpe_net, pbo, promoted_at
FROM hall_of_fame
WHERE promoted_at > NOW() - INTERVAL '24 hours'
ORDER BY promoted_at DESC;
```

**Estatísticas do Hall da Fama**:
```sql
SELECT 
    COUNT(*) as total_strategies,
    AVG(oos_sharpe_net) as avg_sharpe,
    AVG(pbo) as avg_pbo,
    AVG(max_drawdown_net) as avg_drawdown
FROM hall_of_fame;
```

### 7.5. Backup e Recuperação

**Backup de Artefatos**:
```bash
# Backup diário do diretório hall_of_fame
tar -czf hall_of_fame_backup_$(date +%Y%m%d).tar.gz /home/ubuntu/quant-b3/hall_of_fame

# Mover para storage remoto (e.g., S3)
aws s3 cp hall_of_fame_backup_$(date +%Y%m%d).tar.gz s3://quant-b3-backups/
```

**Backup do Banco de Dados**:
```bash
# Exportar tabela hall_of_fame
pg_dump $NEON_DATABASE_URL -t hall_of_fame > hall_of_fame_backup_$(date +%Y%m%d).sql
```

**Recuperação**:
```bash
# Restaurar artefatos
tar -xzf hall_of_fame_backup_20251229.tar.gz -C /home/ubuntu/quant-b3/

# Restaurar tabela
psql $NEON_DATABASE_URL < hall_of_fame_backup_20251229.sql
```

---

## 8. Extensões Futuras

### 8.1. Otimização de Parâmetros Bayesiana

Integrar biblioteca de Bayesian optimization (e.g., `optuna` via PyO3) para ajustar dinamicamente os parâmetros das campanhas com base em resultados históricos.

**Objetivo**: Descobrir automaticamente quais configurações de campanha (e.g., tamanho de população, taxa de mutação) produzem estratégias de maior qualidade.

### 8.2. Paralelização Multi-VPS

Estender o orquestrador para coordenar múltiplas VPS, distribuindo campanhas entre elas para acelerar a mineração.

**Arquitetura**: Implementar um "coordinator node" que distribui trabalho para múltiplos "worker nodes", utilizando Redis ou RabbitMQ para coordenação.

### 8.3. Aceleração GPU

Integrar aceleração GPU para o backtesting paralelo, inspirado em práticas de FinRL que demonstram speedups de >1000x.

**Tecnologia**: Utilizar CUDA via Rust bindings ou OpenCL para executar simulações de estratégias em GPU.

### 8.4. Adaptação Dinâmica a Regimes de Mercado

Implementar detecção automática de mudanças de regime de mercado (e.g., bull/bear, alta/baixa volatilidade) e ajustar automaticamente os universos de ativos e tipos de blocos nas campanhas.

**Técnica**: Utilizar Hidden Markov Models (HMM) ou Regime-Switching Models para detectar regimes.

### 8.5. Ensemble de Estratégias

Criar automaticamente ensembles (portfolios) de estratégias do Hall da Fama, otimizando para correlação baixa e Sharpe Ratio máximo do portfolio.

**Método**: Utilizar Mean-Variance Optimization ou Risk Parity para construir portfolios de estratégias.

---

## 9. Considerações de Segurança

### 9.1. Acesso ao Banco de Dados

- A connection string do Neon PostgreSQL deve ser armazenada como variável de ambiente, não hardcoded.
- Utilizar SSL/TLS para todas as conexões com o banco de dados (`sslmode=require`).
- Implementar princípio de least privilege: o usuário do banco deve ter apenas as permissões necessárias (SELECT, INSERT, UPDATE nas tabelas relevantes).

### 9.2. Isolamento de Processos

- O daemon deve rodar com um usuário não-privilegiado (não `root`).
- Processos filhos (`combiner_cli`) devem ser executados com limites de recursos (ulimit) para prevenir consumo excessivo de memória/CPU.

### 9.3. Validação de Inputs

- Todos os caminhos de arquivo na fila de campanhas devem ser validados para prevenir path traversal attacks.
- Parsing de JSON deve ser robusto contra inputs malformados.

### 9.4. Auditoria

- Todos os eventos críticos (início de campanha, promoção de estratégia, falhas) devem ser logados com timestamps precisos.
- Logs devem ser imutáveis e armazenados em local seguro para auditoria futura.

---

## 10. Métricas de Sucesso

O sucesso do Orquestrador de Mineração Perpétua será medido pelas seguintes métricas:

| Métrica | Target | Descrição |
|---|---|---|
| **Uptime do Daemon** | >= 99.5% | Percentual de tempo que o daemon está operacional em um mês |
| **Utilização de Recursos** | 70-85% CPU | Utilização média de CPU da VPS (não muito baixa = desperdício, não muito alta = risco de sobrecarga) |
| **Taxa de Descoberta** | >= 5 estratégias/semana | Número de estratégias promovidas ao Hall da Fama por semana |
| **Qualidade das Campeãs** | Sharpe >= 1.0, PBO <= 0.10 | Métricas médias das estratégias no Hall da Fama |
| **Tempo de Recuperação** | <= 5 minutos | Tempo médio para recuperação após falha do daemon |
| **Latência de Promoção** | <= 10 minutos | Tempo entre conclusão de campanha e promoção de estratégias |

---

## 11. Conclusão

O Orquestrador de Mineração Perpétua representa uma evolução fundamental na arquitetura do projeto *Quant B3 Backtester*, transformando-o de um sistema de pesquisa baseado em campanhas discretas para uma plataforma de descoberta contínua e industrial de estratégias quantitativas. Inspirado nas práticas dos fundos quantitativos de elite como Renaissance Technologies e Two Sigma, o OMP implementa princípios comprovados de operação 24/7, otimização de recursos, validação rigorosa e auditabilidade completa.

A arquitetura proposta é robusta, escalável e extensível, aproveitando o ecossistema existente em Rust enquanto introduz uma camada de orquestração inteligente. O sistema foi projetado para operar de forma autônoma em uma VPS Linux, maximizando a utilização de recursos computacionais e promovendo automaticamente estratégias de elite que atendem critérios rigorosos de robustez.

Com a implementação do OMP, o projeto estará posicionado para competir com os melhores fundos quantitativos do mundo, operando uma "máquina de fazer dinheiro" que funciona incansavelmente, 24 horas por dia, 7 dias por semana, minerando as estratégias mais lucrativas e robustas nos mercados B3 e US.

---

## Referências

[1] Zuckerman, G. (2019). *The Man Who Solved the Market: How Jim Simons Launched the Quant Revolution*. Portfolio/Penguin. Disponível em: https://www.readtrung.com/p/jim-simons-and-the-making-of-renaissance

[2] Adereth, M. (2019). *Why Two Sigma is using SigOpt for Automated Parameter Tuning*. Two Sigma. Disponível em: https://www.twosigma.com/articles/why-two-sigma-is-using-sigopt-for-automated-parameter-tuning/

[3] Arcesium. (2024). *Building the Ideal Hedge Fund Infrastructure: A Blueprint for Success*. Disponível em: https://www.arcesium.com/blog/ideal-hedge-fund-infrastructure-blueprint

[4] Documentação do Projeto Quant B3 Backtester. (2025). *System Overview e Architecture Documentation*. Versão 3.2.0.

---

**Apêndice A: Glossário de Termos**

| Termo | Definição |
|---|---|
| **OMP** | Orquestrador de Mineração Perpétua - o sistema proposto nesta especificação |
| **SCG** | Sistema Combinador Generativo - framework de algoritmos genéticos existente |
| **PBO** | Probability of Backtest Overfitting - métrica de overfitting |
| **DSR** | Deflated Sharpe Ratio - Sharpe Ratio ajustado para múltiplos testes |
| **Hall da Fama** | Repositório de estratégias de elite que passaram por validação rigorosa |
| **Daemon** | Processo de longa duração que roda em background |
| **VPS** | Virtual Private Server - servidor virtual onde o sistema opera |

---

**Apêndice B: Checklist de Implementação**

- [ ] Criar estrutura de diretórios do projeto
- [ ] Implementar crate `orchestrator_daemon` em Rust
- [ ] Criar schema SQL da tabela `hall_of_fame`
- [ ] Implementar loop principal de orquestração
- [ ] Implementar gestor de recursos
- [ ] Implementar gerenciamento da fila de campanhas
- [ ] Implementar monitor de campanhas
- [ ] Implementar pipeline de promoção
- [ ] Criar arquivo de configuração TOML
- [ ] Escrever testes de integração
- [ ] Configurar PM2 para gestão de processo
- [ ] Criar scripts de backup automatizado
- [ ] Documentar procedimentos operacionais
- [ ] Estender dashboard (Cockpit) para exibir Hall da Fama
- [ ] Implementar notificações (webhook/Slack)
- [ ] Realizar testes de stress e resiliência
- [ ] Deploy em ambiente de produção (VPS)
- [ ] Monitorar métricas de sucesso por 30 dias
