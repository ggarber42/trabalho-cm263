# Previsão de absenteísmo — GERCON Porto Alegre (CMP263)

Trabalho da disciplina CMP263 – Aprendizagem de Máquina: previsão de faltas em consultas ambulatoriais de
atenção secundária, comparando algoritmos sob validação temporal. Regras do projeto em `CLAUDE.md`;
passo a passo de reprodução em `README_REPRODUCAO.md`; pipeline semanal em `pipeline_v2/README.md`.

## Base de dados utilizada

O pipeline (`src/`, `etapas/`, `Makefile`) lê a base indicada em `config.yaml` (chave `base`):

| Arquivo | Conteúdo | SHA-256 |
|---|---|---|
| `gercon_poa_censo_distancias_com_natureza_juridica.parquet` | Uma linha por consulta agendada no GERCON (Porto Alegre, 2018–2026) | `8e78fa959fc7406a1bfb1e888b441e81a0f95fd0fa9d17b78101d50162a617c6` |
| `gercon_poa_censo_estabelecimentos.parquet` | Cadastro auxiliar das CNES | `0f3a1333877ffcf5d781615adbf95b1415146ade7bdad2b8c492924f6e489a95` |

- **Os parquets não estão no git** (dados por consulta/paciente, LGPD, e tamanho ~42 MB). Coloque-os na raiz
  do projeto, ao lado do `config.yaml`.
- Os hashes esperados ficam em `dados/hashes_esperados.json`; confira com
  `python3 etapas/S0_confere_hash.py` (ou `make confere-hash`). Se algum der `DIFERENTE`, os resultados não
  são comparáveis.
- Origem: dados do GERCON (sistema de regulação de consultas de Porto Alegre), reestruturados em R
  (`reestruturacao.qmd`) e enriquecidos com distâncias (`distancias.r`, Google Routes API) e natureza jurídica
  do cadastro CNES/DATASUS (competência 09/2026).

## Como rodar

```bash
make venv          # cria .venv na pasta do projeto (Python 3.12) com as versões do requirements.lock
make smoke         # teste rápido, todos os anos e esquemas (usa .venv automaticamente)
./run_all.sh 7     # perfil completo com 7 processos (horas; retomável)
```
O perfil completo (`make full`) leva horas; ver `README_REPRODUCAO.md`.
