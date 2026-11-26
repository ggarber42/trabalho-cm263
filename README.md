# Previsão de absenteísmo — GERCON Porto Alegre (CMP263)

O objetivo do trabalho é fazer a  previsão de faltas em consultas ambulatoriais de
atenção secundária, comparando algoritmos sob validação temporal. 

Regras do projeto em `CLAUDE.md`, passo a passo de reprodução em `README_REPRODUCAO.md` e pipeline semanal em `pipeline_v2/README.md`.

## Base de dados utilizada

O pipeline (`src/`, `etapas/`, `Makefile`) lê a base indicada em `config.yaml` (chave `base`):


`gercon_poa_censo_distancias_com_natureza_juridica.parquet` - Uma linha por consulta agendada no GERCON (Porto Alegre, 2018–2026)

`gercon_poa_censo_estabelecimentos.parquet` - Cadastro auxiliar das CNES


## Como rodar

```bash
make venv          # cria .venv na pasta do projeto (Python 3.12) com as versões do requirements.lock
make smoke         # teste rápido, todos os anos e esquemas (usa .venv automaticamente)
./run_all.sh 7     # perfil completo com 7 processos (horas; retomável)
```
O perfil completo (`make full`) leva horas; ver `README_REPRODUCAO.md`.
