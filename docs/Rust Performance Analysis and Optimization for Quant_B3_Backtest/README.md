# Contexto de Otimização de Performance para Cursor IDE

Este diretório contém a especificação completa de otimização de performance para o sistema `quant_b3_backtest`, projetada para ser utilizada como contexto no Cursor IDE.

## Estrutura de Arquivos

- **`quant_b3_backtest_performance_spec.md`**: Documento principal consolidado contendo todas as 15 análises de performance e recomendações de otimização.

- **`analysis_reports/`**: Diretório contendo os 15 relatórios individuais de análise, organizados por área:
  - `0_*_cache_analysis.md`: Otimização de Cache e Memória (CPU Cache)
  - `1_*_unified_engine.md`: Otimização Algorítmica do UnifiedEngine
  - `2_*_performance_analysis_report.md`: Paralelização Avançada e SIMD
  - `3_*_performance_analysis_report.md`: Comparação Arquitetural com Frameworks de Backtesting
  - `4_*_backtest_analysis.md`: Comparação Conceitual com Backtrader
  - `5_*_combiner_engine_performance_analysis.md`: Otimização do Algoritmo Genético
  - `6_*_backtest_analysis.md`: Integração de Bibliotecas de Alta Performance
  - `7_*_performance_analysis_io.md`: Otimização de I/O e Serialização
  - `8_*_memory_allocation_analysis.md`: Análise de Padrões de Alocação de Memória
  - `9_*_simd_performance_analysis.md`: Análise Detalhada de SIMD
  - `10_*_lock_free_optimization_report.md`: Algoritmos Lock-Free e Wait-Free
  - `11_*_compilation_analysis.md`: Otimização de Compilação e Linking
  - `12_*_async_analysis.md`: Potencial de Async/Await para I/O
  - `13_*_gpu_acceleration_analysis.md`: Análise de Aceleração por GPU
  - `14_*_obfs_persistence_analysis.md`: Otimização da Camada de Persistência

## Como Usar no Cursor

1. **Adicionar como Contexto**: Arraste a pasta `cursor_context` inteira para o Cursor IDE ou adicione o arquivo `quant_b3_backtest_performance_spec.md` ao contexto da conversa.

2. **Implementação Incremental**: Comece pelas otimizações de maior impacto:
   - Substituir `CliExecutor` por chamadas em-processo (Relatório 6)
   - Implementar Symbol ID Mapping (Relatórios 1 e 4)
   - Paralelizar avaliação de genomas (Relatórios 5 e 6)
   - Otimizar I/O e parsing de CSV (Relatório 7)

3. **Referência Técnica**: Use os relatórios individuais para detalhes específicos de implementação, incluindo localizações exatas no código, estimativas de impacto e análise de trade-offs.

## Objetivo

Alcançar um ganho de performance de **100x a 1000x** na geração e cálculo de estratégias de backtesting, explorando o máximo potencial da linguagem Rust para computação de alta performance.

## Áreas de Otimização

- **Arquitetura**: Redesign do modelo de execução para chamadas em-processo e motor vetorizado
- **Dados**: Data-Oriented Design com Symbol ID Mapping e Structure-of-Arrays
- **Memória**: Arena allocation, SmallVec, eliminação de clones desnecessários
- **Paralelismo**: Rayon avançado, SIMD (wide/std::simd), AVX2/AVX512
- **I/O**: Zero-copy parsing, otimização de compressão e integridade
- **Concorrência**: Lock-free structures (arc_swap, crossbeam-queue)
- **Compilação**: PGO, LTO, CPU-specific optimizations

---

**Autor**: Manus AI  
**Data**: Janeiro 2026  
**Versão**: 1.0
