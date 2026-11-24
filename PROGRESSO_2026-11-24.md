# PROGRESSO 2026-11-24

## Feito
- **Ambiente:** venv criado em `~/.venvs/absenteismo` (fora da pasta do Drive) com Python 3.12 e as versões exatas de
  `requirements.lock`. O `python3` padrão desta máquina é 3.10 sem sklearn; por isso usar `python3.12 -m venv`
  (o alvo `make venv` usa `python3` e, aqui, criaria um venv 3.10).
- **Hashes:** `make confere-hash` → entradas conferem.
- **Smoke executado e verificado:** `make smoke PYTHON=~/.venvs/absenteismo/bin/python` terminou sem erro em 6 min 37 s;
  115 testes passaram; `[S5] REPRODUZ o legado dentro da tolerância`; S13 gerou 29 arquivos em `figuras/smoke` e
  `tabelas/md/smoke`. Log em `logs/smoke_2026-11-24.log` (local, fora do git).
- **Git:** criado `.gitignore` (parquets e demais dados por consulta, `*.pkl`/`*.joblib`, zips, ambientes, caches,
  segredos, `pipeline_v2/config_local.yaml`, `resultados/`, `logs/`, `tabelas/`, `figuras/`). Commit inicial fica com ~220 arquivos.
  Com `resultados/`, `tabelas/` e `figuras/` fora do git, as tabelas e figuras da rodada completa de 20/11 (fonte dos números do texto)
  só existem localmente; guardar cópia fora do git.
- **README.md** criado na raiz, com a referência à base usada (`gercon_poa_censo_distancias_com_natureza_juridica.parquet`,
  SHA-256 de `dados/hashes_esperados.json`) e como conferir.
- **`saida_*.txt` movidos para `antigos/saidas/`:** são logs de terminal dos scripts legados 06–21. Referências em
  `PROGRESSO_2026-11-11.md` e `historico_sessao_2026-11-10.md` citam o nome antigo (sem pasta); não foram editadas por
  serem registros históricos.

- **`resultados_municipal/` movida para `antigos/resultados_municipal/`:** saídas dos scripts legados 21–24
  (seleção de variáveis, poda 32 → 6); o pipeline atual não lê nem grava nessa pasta. `config.yaml`
  (`saidas.legado_municipal`) atualizado para o novo caminho. Os scripts legados (congelados, não editados) gravam em
  `resultados_municipal/` relativo à pasta de execução; se forem rodados de novo, recriam a pasta na raiz.

- **Raiz limpa:** 45 saídas dos scripts legados (CSV agregados, PNG, `relatorio_decisao.md`, `calibracao.json`,
  `*.joblib`, `*.pkl`, painéis `*.parquet`) movidas para `antigos/saidas/`; `historico_sessao_2026-11-10.md` e
  `PROMPT_COLEGA_IA.md` para `antigos/notas/`; `06_modelo_paciente.py` da raiz (cópia de 20/11 que só comenta o
  `to_csv`) para `antigos/06_modelo_paciente_sem_csv.py`. Ficaram na raiz `comparacao_series.csv` e
  `coorte_2019_2026.csv`, lidos por S0/S5. Verificado depois da mudança: hashes conferem, 115 testes passam,
  S5 reproduz o legado. Os scripts legados leem esses arquivos na raiz; para rodá-los de novo, executar a partir de
  `antigos/saidas/` ou copiar os arquivos de volta.

- **venv movido para a pasta do projeto (pedido do usuário):** `.venv/` (Python 3.12.16, versões do lock), ignorado
  pelo git; `~/.venvs/absenteismo` removido. `Makefile` (`PYTHON` padrão e alvo `venv`, que agora usa `python3.12`)
  e `run_all.sh` usam `.venv` por padrão. README, README_REPRODUCAO e CLAUDE.md (seção 6) atualizados.
  Verificado: `make env` mostra o .venv sem pacotes faltando; `make confere-hash` ok.

## Pendente
- (Resolvido, com aprovação do usuário) CLAUDE.md seções 3.1 (regra 3), 6 e 7 atualizadas para os caminhos atuais:
  `resultados/<escopo>_<perfil>/experimentos.csv`, números do texto a partir de `tabelas/` (S9), saídas conforme
  `saidas` do `config.yaml`.
- `reestruturacao.qmd` (gera a base) não está no repositório.
- `.git` tem ~334 MB de objetos soltos não comitados; `git gc --prune=now` libera o espaço.
- `resultados/` fora do git: guardar cópia (ex.: Drive) para não depender de rodar o completo de novo.
- Perfil completo (~10 h) só com aprovação.
