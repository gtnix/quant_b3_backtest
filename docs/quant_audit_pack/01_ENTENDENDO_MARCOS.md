# Entendendo os 6 Maros da Auditoria (explicado para júnior)

Este documento traduz os “marcos” da auditoria para linguagem prática.

> **Resumo mental:**  
> Marco 0–1: “O terreno é confiável?” (config e dados)  
> Marco 2: “A fábrica está produzindo coisas de verdade?” (evolução)  
> Marco 3: “Isso funciona fora da amostra ou é ilusão?” (validação)  
> Marco 4–5: “O que eu vou promover e como eu provo?” (gates + artefatos)

---

## Marco 0 — Inicialização (setup)

### O que é
É a checagem se o **experimento começou do jeito certo**: datas, configs, seeds, base de dados, caminho de output e versionamento.

### Por que existe
Se o setup estiver errado, o resto vira “lixo com rigor”. Ex.: rodar com data errada, com seed variável, ou misturar outputs.

### O que tem que “existir”
- arquivo de campanha/config
- assinaturas/hash de config e dataset
- `git sha` e build profile
- datas e timezone
- seed(s) explícitas

### Se falhar, o que significa
Você **não consegue reproduzir** o resultado. Ou pior: você acha que comparou duas coisas, mas comparou maçã com banana.

---

## Marco 1 — Integridade de Dados

### O que é
É a checagem se o seu backtest **não está trapaceando** por causa de dados.

### 3 bugs clássicos que isso pega
1. **Lookahead bias:** usar informação do futuro sem querer (ex.: sinal calculado no fechamento e executado no mesmo candle).  
2. **Survivorship bias:** universo “só com os vivos” hoje (IBOV atual), que faz o passado parecer melhor.  
3. **Ajuste de preços errado:** split/dividendo bagunçado gera retornos falsos.

### Se falhar, o que significa
Qualquer performance pode ser “fake”. Você tem que corrigir dados/modelo antes de confiar em Sharpe, CAGR etc.

---

## Marco 2 — Evolução (SCG)

### O que é (em português bem direto)
O SCG é uma “fábrica” que tenta criar muitas estratégias automaticamente e **vai selecionando as melhores** ao longo de gerações.

### Por que isso existe
Porque o espaço de combinações é enorme. Mas isso tem um perigo: a fábrica pode “hackear a nota” (fitness) e gerar coisas absurdas.

### O que tem que ser checado
- **Diversidade:** a população não pode virar clones (senão não explora nada).
- **Fitness coerente:** estratégia que não tradeia não pode “parecer genial”.
- **Convergência real:** o “melhor” melhora por motivo econômico, não por bug.
- **Penalidades aplicadas:** poucos trades, turnover absurdo e volatilidade estranha devem ser punidos.

### Se falhar, o que significa
O motor evolutivo pode estar:
- aprendendo a explorar um bug (ex.: Sharpe inflado por vol quase zero)
- quebrado (ex.: sempre o mesmo genoma)
- premiando estratégias vazias

---

## Marco 3 — Validação (o mais importante)

### O que é
É a etapa que responde: **“Isso funciona fora da amostra?”**

Aqui você sai do “parece bom” e vai para “passa num teste justo”.

### Conceitos mínimos (sem teoria pesada)
- **In-sample (IS):** período onde você “aprende”/otimiza.
- **Out-of-sample (OOS):** período que você não viu antes.
- **Walk-Forward (WFA):** repetir IS→OOS várias vezes em janelas no tempo, simulando uso no mundo real.
- **PBO (Probability of Backtest Overfitting):** chance de você ter “escolhido a estratégia sortuda” entre muitas.
- **Stress test:** piorar custos/slippage/delay e ver se a estratégia ainda sobrevive.

### Se falhar, o que significa
A estratégia é provavelmente:
- overfitted (decorou o passado)
- frágil (morre com custo real)
- baseada em erro de cálculo

---

## Marco 4 — Promotion Gates

### O que é
Regras automáticas do tipo: **“só sobe de fase se passar”**.

Ex.: “só promover se OOS Sharpe NET ≥ 0.5, PBO ≤ 0.15 e passar 4/5 stresses”.

### Se falhar, o que significa
Você está colocando em “produção/paper” coisas que não deveriam passar.

---

## Marco 5 — Artefatos finais (prova e replay)

### O que é
É o pacote de “prova”: tudo o que permite reproduzir o resultado no futuro.

Inclui:
- config da estratégia
- config de execução (custos etc.)
- hashes (dataset/config)
- commit do git
- script de replay

### Se falhar, o que significa
Mesmo que você ache a estratégia boa, **você não consegue provar** o que aconteceu.

---

## O que você disse que está faltando (“melhor papel / pior papel”)
Isso entra principalmente no **Marco 3 (Validação)** como **Attribution / Diagnóstico**:

- Quais ativos fizeram mais dinheiro (top winners)
- Quais ativos destruíram o resultado (top losers)
- Quantas trades por ativo
- PnL líquido por ativo
- Contribuição percentual no retorno total
