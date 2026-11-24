# Prompt para o assistente de IA (colega que vai rodar os modelos)

> Cole o texto abaixo no início da sessão com o assistente (Claude Code, Codex, etc.) aberto **dentro desta pasta**.

---

## Contexto

Você vai me ajudar a **executar, em uma máquina potente, o pipeline de modelagem** desta pasta. O projeto é um trabalho da disciplina **CMP263 – Aprendizagem de Máquina**: previsão de **absenteísmo (faltas) em consultas ambulatoriais de atenção secundária** (fila GERCON, Porto Alegre), tratada como problema de série temporal com validação temporal, comparando vários algoritmos com o mesmo protocolo. O autor original é outra pessoa; eu estou rodando a parte pesada.

**Primeiro passo obrigatório:** leia inteiro, nesta ordem, `CLAUDE.md` (regras do projeto, valem para você), `README_REPRODUCAO.md` (como rodar em outra máquina), `PIPELINE.md`, `entregaveis/plano_analise.md` (decisões congeladas) e `HANDOFF_COWORK.md` / `PROGRESSO_2026-11-16.md` (estado atual). Em caso de conflito entre um pedido meu e o `CLAUDE.md`, avise-me em vez de contornar.

## O que é o problema

- Desfecho por consulta: `y = (situacaoagenda == "FALTANTE")`.
- Unidades de análise: (a) consulta (probabilidade de falta, modelos por paciente); (b) **CNES-dia** (número de faltas), horizonte previsto **L = 3 dias** (operacional) e **L = 14** (sensibilidade).
- Base: `gercon_poa_censo_distancias_com_natureza_juridica.parquet` (~984 mil consultas, 221 CNES, 2018-01 a 2026-03). Escopo atual: recorte **municipal** (111 CNES sob gestão da Prefeitura, 360.878 consultas), parâmetro `escopo` do `config.yaml` (`municipal | geral | criterio_x`). Não assuma o municipal como definitivo.
- Modelos (em `src/models`, lista em `config.yaml`): naive sazonal, baseline Poisson, ETS, SARIMA/SARIMAX, GLM-NB, regressão Poisson, HGB Poisson (com e sem recalibração k), floresta aleatória, LightGBM, XGBoost, CatBoost, MLP, HGB por paciente somado por CNES-dia.
- Esquemas de divisão temporal (comparados como fatores): holdout, janela expansiva (**principal**), deslizante, purged + embargo. Anos de teste 2023–2026; teste comum 2024-01 a 2026-03.
- Tuning por **validação temporal aninhada** (folds só com dados anteriores ao teste).

## Estrutura da pasta

- `config.yaml` — **única fonte de parâmetros** (caminhos relativos, sementes, modelos, esquemas, L). Decisões G2 estão **congeladas** (17/11/2026): não altere sem registrar e justificar.
- `Makefile` / `run_all.sh` — ponto de entrada. `make smoke` (teste rápido, 4 CNES), `make full JOBS=N` (perfil completo, retomável), `make estimativa` (tempo/RAM do completo), `make venv`, `make lock`, `make env`, `make test`.
- `src/` — código compartilhado (dados, coorte, splits, painel, métricas, avaliação, teste DM, modelos).
- `etapas/S*.py` — etapas numeradas do pipeline (S0 ambiente/hash, S1 auditoria, S2 coorte, S3.1 horizonte, S4 divisões, S5 reprodução do baseline, S7 experimentos [a etapa pesada], S9 avaliação, S10 importância, S11 overbooking, S12 sensibilidade, S13 tabelas/figuras, S15 estimativa de recursos).
- `tests/` — pytest (~115 testes).
- `resultados_municipal/` — saídas (CSV, PNG, MD, JSON); `experimentos.csv` registra cada combinação (esquema, L, modelo, escopo) com sementes, P, E, janela e hash dos dados.
- `logs/`, `manifestos/` — log por etapa e hashes SHA-256 de entradas/saídas.
- `entregaveis/` — textos finais (Metodologia, Resultados, checklist TRIPOD+AI) e planos; ainda **não** escritos com o perfil completo.
- `antigos/` — scripts legados 01–24, **congelados**; o pipeline novo não os importa. Não altere.
- Raiz: parquets de dados, CSVs de coorte/níveis/flag municipal, `requirements.lock`, `requirements-extra.txt`.

## Regras que você deve seguir (resumo; o `CLAUDE.md` manda)

1. **Anti-vazamento é regra dura.** Nenhuma escolha de modelo, hiperparâmetro, variável, coorte ou lag pode usar dados do período de teste ou posteriores. O teste externo não serve para escolher variantes. Se um resultado parecer bom demais, procure vazamento antes de comemorar.
2. **Dados pessoais:** os parquets têm dados de pacientes (LGPD). Nunca imprima, logue, comite ou coloque em figura identificadores de paciente; trabalhe com agregados. Não envie os dados a serviços externos.
3. **Reprodutibilidade:** sementes fixas (nada de `random_state=None`), venv **fora** da pasta sincronizada (`make venv`), caminhos relativos, sem número digitado à mão nos textos (tudo vem de `resultados_municipal/`).
4. **Todos os modelos nas mesmas linhas** e com a mesma família de distribuição preditiva; registre linhas perdidas por modelo. Orçamento de tuning equivalente entre modelos.
5. Código e comentários em **português do Brasil**; docstrings, `# ANTI-VAZAMENTO:` e `# DECISÃO:` como no `CLAUDE.md`. Scripts novos recebem novo número/sufixo; não edite os originais legados.
6. Qualquer filtro que remove linhas imprime quantas removeu.
7. Diga sempre o que foi **executado e verificado** versus apenas lido/escrito. Não afirme resultado que não saiu de arquivo ou execução.

## O que eu quero que você faça agora

Siga esta ordem e **pare para me confirmar antes do passo 5**:

1. Conferir o ambiente: `make venv`, depois `make lock PYTHON=~/.venvs/absenteismo/bin/python` e `make env` (versões, núcleos, RAM). Instalar também `lightgbm`, `xgboost`, `catboost` (`requirements-extra.txt`); sem elas o modelo é pulado.
2. Conferir os dados: `python3 etapas/S0_confere_hash.py` deve dizer que as entradas conferem. Se divergir, **pare** e me avise.
3. Rodar `make smoke PYTHON=...` e confirmar que todos os testes passam e que `[S5]` reproduz o baseline legado.
4. Rodar `make estimativa JOBS=N` e me dizer tempo e RAM previstos do perfil completo, e sugerir `N` adequado a esta máquina.
5. **Somente com meu OK**, rodar `make full JOBS=N` (longo, retomável; rode em background com log e relate o progresso). Se cair, retome em vez de reiniciar.
6. Ao final: conferir `resultados_municipal/experimentos.csv` (todas as combinações presentes, sem NaN inesperado, alinhamento por chave), atualizar `PROGRESSO_AAAA-MM-DD.md` e `HANDOFF_COWORK.md`, e resumir em linguagem simples o que rodou, o que foi validado e o que ficou pendente.

## O que NÃO fazer sem me perguntar

- Alterar `config.yaml` (principalmente decisões congeladas: `lag_principal`, `divisao.principal`, `exclusoes_cnes`, escopo), grade de hiperparâmetros ou lista de modelos.
- Alterar scripts de `antigos/`, o `reestruturacao.qmd` ou a regra de `t0` (`data_solicitacao + max(diasfilaespera, 0)`).
- Escrever introdução, discussão geral ou artigo; as entregas são só **Metodologia** e **Resultados** (`entregaveis/`), e só depois de rodar o perfil completo.
- Copiar os dados para fora desta máquina ou da pasta do projeto.

## Pendências conhecidas (para você saber o que ainda está aberto)

Poda de variáveis do legado (hipótese, não usada no pipeline novo); documentar `status`; confirmar com o serviço atraso de registro do desfecho, `possuiunidadeindicada` e `idade`; SHAP/ALE e sensibilidades (escopo geral, coorte global, remoção de variáveis suspeitas); comparação sequencial × paralela com L = 14; textos finais e checklist TRIPOD+AI.
