# Kaggle-Slaying Multi-Agent Team

MVP local e de baixo custo para executar o ciclo de uma competicao Kaggle:
obter dados, validar, treinar modelos, preparar uma submissao com aprovacao
humana e acompanhar o leaderboard.

## Estado da versao 0.1

O fluxo completo ja foi executado em uma competicao real e testado em mais de
um formato tabular. O projeto evita chamadas pagas: modelos e validacao rodam
localmente; Ollama e opcional e ainda nao participa do caminho principal.

O envio ao Kaggle nunca ocorre dentro de `run`. Ele exige um segundo comando
com o hash exato apresentado pelo gate. Um recibo local impede que o mesmo hash
seja enviado novamente.

## Requisitos

- Windows com Python 3.11 ou 3.12;
- conta Kaggle;
- CPU e memoria suficientes para o conjunto de dados;
- GPU NVIDIA e Ollama sao opcionais.

## Instalacao

No PowerShell, dentro da pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\kaggle-slaying.exe doctor
```

Para autenticar a conta Kaggle:

```powershell
.\.venv\Scripts\python.exe -m kaggle_slaying.kaggle_cli auth login
.\.venv\Scripts\kaggle-slaying.exe doctor --online
```

Credenciais, dados, modelos e submissoes ficam em pastas ignoradas pelo Git.

## Uso principal

Cada competicao possui um contrato YAML em `config/competitions`. Depois de
ler e aceitar manualmente suas regras no Kaggle, execute:

```powershell
.\.venv\Scripts\kaggle-slaying.exe run `
  --competition playground-series-s6e9
```

O comando baixa dados ausentes, cria o perfil, escolhe a fabrica compativel,
treina ou reutiliza modelos, valida o CSV e salva o estado. Possiveis fases:

- `awaiting_approval`: candidata pronta, sem envio;
- `submitted`: recibo encontrado e leaderboard atualizado;
- `blocked`: alguma verificacao falhou.

Consulte o ultimo estado sem acessar a internet ou treinar novamente:

```powershell
.\.venv\Scripts\kaggle-slaying.exe status `
  --competition playground-series-s6e9
```

## Envio com aprovacao humana

Somente depois de revisar o estado e autorizar explicitamente o hash:

```powershell
.\.venv\Scripts\kaggle-slaying.exe submit-approved `
  --competition playground-series-s6e9 `
  --sha256 HASH_APROVADO
```

Repetir esse comando com um hash que ja possui recibo nao cria outra
submissao.

## Nova competicao

Crie `config/competitions/SLUG.yaml` informando pelo menos:

```yaml
name: Nome da competicao
slug: slug-kaggle
modality: tabular
problem_type: binary_classification
target_column: target
id_column: id
metric: roc_auc
data_directory: data/raw/slug-kaggle
submission_file: submission.csv
group_column: null
time_column: null
requires_rule_acceptance: true
```

O MVP escolhe o experimento v2 para classificacao binaria com AUC e usa a
fabrica generica v1 nas demais combinacoes suportadas. Se houver possiveis
grupos ou ordem temporal, o treinamento e bloqueado para revisao.

## Comandos auxiliares

```powershell
# Descobrir competicoes sem entrar ou baixar dados
.\.venv\Scripts\kaggle-slaying.exe scout --limit 20 --top 5

# Gerar apenas o perfil
.\.venv\Scripts\kaggle-slaying.exe profile --competition titanic

# Mostrar todos os comandos
.\.venv\Scripts\kaggle-slaying.exe --help
```
