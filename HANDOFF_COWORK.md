# Handoff — tese de absenteísmo (GERCON Porto Alegre), CMP263 — atualizado em 17/11/2026

Pasta: `serie_temporal/` (git inicializado). **Leia primeiro:** `CLAUDE.md`, `PIPELINE.md`, `README_REPRODUCAO.md`,
`PROGRESSO_2026-11-16.md` (M0–M4 + empacotamento) e `AUDITORIA_2026-11-16.md`.

## Estado
- M0–M4 concluídos e verificados no **smoke** (4 CNES). **Nada foi treinado no conjunto completo.** `make smoke` passa (96 testes).
- Pipeline novo: `config.yaml` (perfis `smoke` e `completo`) → `etapas/S*.py` → `src/` (config, coorte, splits, panels, dados, models, experimento,
  metrics, dm_test, avaliacao). Scripts legados 01–24 intactos (não fazem parte do pipeline novo).
- Para rodar o completo em outra máquina: `README_REPRODUCAO.md` (`make venv`, `make smoke`, `make estimativa`, `./run_all.sh JOBS`).

## Decisões do usuário registradas
- 2018 **mantido** (histórico do paciente); nenhuma exclusão de CNES por inatividade/volume/placeholder.
- Coorte e níveis por corte de ano (só dados anteriores); a versão global é só comparação.

## Pendências (ver PROGRESSO)
1. G2: esquema principal, L principal (S3.1 mostra 92,4% da agenda conhecida a 3 dias, 79,2% a 7, 56% a 14; 55,8% das consultas com antecedência
   exatamente 14 dias), modo de agenda, período comum, métrica primária. **Não congelados.**
2. Confirmar com o serviço: significado de `t0`, atraso de registro do desfecho, `possuiunidadeindicada`, `idade`, remarcações.
3. lightgbm (wrapper nunca executado); XGBoost/CatBoost/MLP, ETS/SARIMA/GLM-NB e recalibração k não portados; S10–S12 não feitos.
4. Arquivo bruto anterior ao `reestruturacao.qmd` e script das distâncias ausentes.

## Cuidados
- Rodar da raiz, com `python3 script.py` ou `make` (sem `-I`). O `python3` atual é o venv compartilhado em `~/Google_drive/skill_academico/.venv`
  (dentro do Drive); na máquina nova use `make venv` (venv fora do Drive).
- Dados pessoais nunca em saídas. Parquets não vão no git. Conferir hashes com `etapas/S0_confere_hash.py`.


## Atualização 17/11/2026
Estado: pipeline pronto e testado (115 testes; smoke de ponta a ponta com L=3 e 14). G2 congelado (ver `entregaveis/plano_analise.md`). Nada foi treinado nos dados completos. Próximo passo: o usuário aprova e roda o completo na máquina potente (`README_REPRODUCAO.md`, `PROGRESSO_2026-11-16.md` para o histórico e as pendências). LSTM descartada.
