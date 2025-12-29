# Survivorship Bias

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O survivorship bias ocorre quando backtests incluem apenas ativos que "sobreviveram" até o presente, ignorando ativos que faliram, foram deslistados ou adquiridos.

---

## Problema

Incluir ativos que não existiam em datas passadas infla resultados:
- Ativos delisted por falência não aparecem
- IPOs recentes aparecem em datas anteriores à listagem
- Mudanças de índice não são respeitadas

---

## Solução: Universe Eligibility

### V1: UniverseRangeProvider

Usa `min_date`/`max_date` de dados disponíveis como proxy para existência.

**Fonte**: `cache/universe.csv`

```csv
symbol,avg_volume,bar_count,min_date,max_date
PETR4,50000000,2500,2015-01-02,2024-12-27
OIBR3,1000000,1800,2015-01-02,2022-06-15
```

**Regra de Elegibilidade**:
```
min_date ≤ rebalance_date ≤ max_date
```

### V2: TimelineEligibilityProvider

Usa `listing_date`/`delisting_date` de eventos reais (database).

**Schema**:
```sql
ALTER TABLE provider_universe 
ADD COLUMN listing_date DATE,
ADD COLUMN delisting_date DATE,
ADD COLUMN eligibility_source VARCHAR(20);
```

**Precedência**:
1. V2 Timeline (se DB tem listing_date) → Usar
2. V1 Range (se CSV tem min_date) → Fallback
3. Unknown → Excluir

---

## Invariantes

### INV-001: No Resurrection

Ativo com `max_date = 2020-12-31` NÃO pode aparecer em rebalances após 2020.

### INV-002: No Time Travel

Ativo com `min_date = 2021-08-05` NÃO pode aparecer em rebalances antes de 2021-08-05.

### INV-003: Selected ⊆ Eligible

Todo ativo selecionado satisfaz:
```
min_date ≤ rebalance_date ≤ max_date
```

---

## Integração

Universe validation é o **primeiro check** no gating pipeline:

```
Universe Check → Tradeability → Price Days → Price Level → Liquidity → Fundamentals → Dividends
```

Se falhar, nenhum outro check é executado.

---

## Configuração

### V1: CSV

```rust
let provider = UniverseRangeProvider::from_csv("cache/universe.csv")?;
let config = EntryEngineConfig {
    eligibility_provider: Some(provider.into_arc()),
    ..Default::default()
};
```

### V2: DB + Fallback

```rust
let v1_fallback = UniverseRangeProvider::from_csv("cache/universe.csv")?.into_arc();
let v2_timelines = load_from_database().await?;
let provider = TimelineEligibilityProvider::from_maps(v2_timelines, v1_fallback);

let config = EntryEngineConfig {
    eligibility_provider: Some(provider.into_arc()),
    ..Default::default()
};
```

---

## EligibilityResult

```rust
pub enum EligibilityResult {
    Eligible,
    OutsideDateRange,
    SymbolNotInUniverse,
}
```

---

## ExclusionReason

```rust
pub enum ExclusionReason {
    OutsideUniverseDateRange,
    NoUniverseRangeData,
    // ... outros
}
```

---

## Audit Trail

Exclusões aparecem no log:

```json
{
  "exclusions": [
    { "symbol": "OIBR3", "reason": "OutsideUniverseDateRange", "stage": "Gating" },
    { "symbol": "RAIZ4", "reason": "OutsideUniverseDateRange", "stage": "Gating" }
  ]
}
```

---

## Telemetria (V2)

```rust
let stats = provider.stats();
println!("V2 hits: {}", stats.v2_hits);
println!("V1 fallbacks: {}", stats.v1_fallbacks);
println!("Not found: {}", stats.not_found);
println!("V2 coverage: {:.1}%", stats.v2_percentage() * 100.0);
```

---

## Limitações (V1)

| Limitação | Impacto | Mitigação Futura |
|-----------|---------|------------------|
| `min_date` é primeiro dado, não IPO | Pode excluir early trading | Integrar dados de IPO |
| `max_date` é último dado, não delisting | Pode incluir após delisting | Integrar dados de delisting |
| Sem membership de índice histórico | Não reconstrói mudanças de índice | Construir tabela `universe_membership` |
| Unknown symbols excluídos | Novos símbolos excluídos até refresh | Auto-refresh do CSV |

---

## Como Validar

1. **Checar logs** para exclusões `OutsideUniverseDateRange`
2. **Rodar testes**: `cargo test universe_gating`
3. **Verificar**: Nenhum candidato tem `rebalance_date` fora do range CSV
4. **Comparar**: Com/sem provider para ver impacto de survivorship

---

## Testes

```bash
# Testes de parsing CSV
cargo test -p backtester_intelligence universe_range

# Testes de eligibility
cargo test -p backtester_intelligence eligibility

# Testes de integração
cargo test -p backtester_intelligence universe_gating

# Invariantes
cargo test no_resurrection
cargo test no_time_travel
```

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| UniverseRangeProvider | `backtester_intelligence/src/entry/universe_range.rs` |
| TimelineEligibilityProvider | `backtester_intelligence/src/entry/timeline_eligibility.rs` |
| EligibilityProvider trait | `backtester_intelligence/src/entry/eligibility.rs` |
| EligibilityResult | `backtester_intelligence/src/entry/types.rs` |
| ExclusionReason | `backtester_intelligence/src/entry/types.rs` |






