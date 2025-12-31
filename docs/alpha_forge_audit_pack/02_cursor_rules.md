# Cursor Rules (opcional)

1. **NÃO** trocar o stack Rust nem reescrever o motor.  
2. Toda correção deve vir com **critério de aceite testável** (ver checklist do prompt).  
3. Prioridade absoluta: **Hall of Fame correto** (SEV-0).  
4. Remover quaisquer textos e fluxos “manual/estimated”.  
5. Não introduzir polling agressivo; preferir SSE com fallback e **diff incremental**.  
6. Dados exibidos devem ser auditáveis: sempre carregar `candidate_id`, `run_id`, `campaign_id`, `genome_hash`, `git_sha`, `config_hash`, `dataset_hash`.  
7. Se encontrar métricas colapsadas (variância ~0), tratar como **incidente** e implementar “Sanity Gate”.  
8. Manter performance: evitar re-render global e loops pesados no frontend.  
