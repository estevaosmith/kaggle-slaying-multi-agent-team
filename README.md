# Kaggle-Slaying Multi-Agent Team

MVP local e de baixo custo para automatizar o ciclo de uma competicao Kaggle:
coleta de dados, validacao, treinamento, geracao de submissao e monitoramento.

O primeiro marco usa uma competicao tabular simples e mantem a submissao sob
aprovacao humana.

## Arquitetura do MVP

O nucleo e orientado por um contrato de competicao, e nao pelo Titanic. Cada
adaptador informa os arquivos, alvo, identificador, metrica e, quando houver,
colunas de grupo ou tempo. O perfilador inspeciona um novo conjunto de dados e
o planejador recomenda a estrategia de validacao antes de liberar treinamento.

O Titanic funciona como teste de integracao conhecido. Heuristicas especificas
dele ficam isoladas dos agentes genericos.

```powershell
.\.venv\Scripts\kaggle-slaying.exe profile --competition titanic
```

A politica inicial do futuro Competition Scout esta em
`config/scout_policy.yaml`. Ela prioriza competicoes tabulares, com submissao
CSV via API, custo compativel com CPU ou GPU de 6 GB e risco de validacao baixo.

Depois de revisar o plano de validacao, a fabrica generica compara um benchmark
ingenuo, um modelo linear e Extra Trees. Ela suporta classificacao binaria,
multiclasse e regressao para as metricas declaradas no codigo. Uma candidata so
e gerada quando o melhor modelo supera o benchmark e o plano de validacao foi
aprovado.

```powershell
.\.venv\Scripts\kaggle-slaying.exe model-factory --competition titanic `
  --approve-validation-review
```
