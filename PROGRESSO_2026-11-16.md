# Progresso — M0 (ambiente e auditoria dos dados), 16/11/2026

## Decisões do usuário nesta sessão
- 2018 **mantido** (alimenta o histórico do paciente). Nenhuma exclusão de CNES por inatividade, volume mínimo ou placeholder; a classificação de atividade é só descritiva. CLAUDE.md corrigido (período real: 2018-01-03 a 2026-03-31).

## O que foi feito (executado e verificado)
- Estrutura de pastas (`src/`, `etapas/`, `tests/`, `tabelas/`, `figuras/`, `logs/`, `manifestos/`, `entregaveis/`, `dados/`, `resultados/`, `scripts_legados/`). **Nenhum script legado foi movido ou alterado** (mover quebraria os caminhos relativos de 19–24).
- `config.yaml` (semente 20261116, caminhos relativos, escopo, L, esquemas, smoke), `src/config.py`, `src/utils_io.py` (SHA-256, manifesto, log).
- `etapas/S0_lock_ambiente.py` → `requirements.lock` e `manifestos/ambiente.json`.
- `etapas/S1_auditoria.py` (com `--smoke`) → `tabelas/auditoria_dados.csv`, `ausentes_por_variavel_ano.csv`, `atividade_cnes.csv`, `fluxograma_selecao.csv`, `dicionario_variaveis.md`, manifesto e log.
- `Makefile` (`make smoke`, `make test`, `make auditoria`, `make lock`, `make env`) e `tests/test_m0.py` (8 testes passando).
- Execução: auditoria completa em ~9 s; `make smoke` em poucos segundos (amostra de 5% dos pacientes, cobre 2018–2026).

## Principais resultados da auditoria (base completa, 983.875 linhas, 41 colunas, 221 CNES)
- `situacaoagenda`: REALIZADA 820.090 (83,35%), FALTANTE 163.785 (16,65%). O parquet já está filtrado; a contagem das demais categorias depende do arquivo bruto (ausente).
- `status`: ok 932.632; sem_rota 22.485; mesmo_local 17.711; ausente 11.047. Ausência de `status` coincide em 100% das linhas com a ausência das 6 variáveis de distância (descreve o cálculo de rota, não o desfecho). 83% dos ausentes estão em 5 CNES, entre elas os códigos 1111111 e 1111112.
- Duplicatas: 0 por `idsolicitacao`, `rid` e linha inteira; 5 por chave composta (paciente+solicitação+consulta+especialidade), mantidas.
- Datas: 0 violações (solicitação ≤ t0 ≤ consulta; `dias_ate_consulta` = consulta − solicitação; todas dentro de 2018-01..2026-03).
- **Histórico:** n_consultas_previas e n_faltas_previas recalculados com `data_ev < t0` batem em 100% das linhas (0 divergências, 0 casos em que o original tem mais eventos que o recálculo).
- Antecedência do agendamento (consulta − t0): mediana 14 dias, p75 a p99 = 14, máximo 90. Fração com antecedência ≥ h: h=1 99,9%; 3 92,8%; 5 87,5%; 7 80,9%; 10 70,3%; 14 59,0%. (relevante para S3.1)
- Fluxograma: base 983.875 → escopo municipal 360.878 (111 CNES) → coorte municipal A+B 262.488 (10 CNES).
- Atividade das CNES: 117 ativas; 53 inativas ou paradas desde 2024; 23 sem 2026; 28 novas (2025+).

## Pendências / cuidados
1. **Arquivo bruto anterior ao qmd e script das distâncias** continuam ausentes: a contagem de categorias antes dos filtros e a definição de `status` ficam como limitação.
2. `possuiunidadeindicada`: taxa de falta 11,9% (SIM) vs 17,7% (NÃO). Momento de preenchimento desconhecido; marcada "suspeita" no dicionário até confirmar com o serviço. Idem `datahora_consulta` (remarcação) e `idade`.
3. **Antecedência = 14 dias para a maioria** (p50 a p99). Precisa de confirmação: pode ser regra real de abertura da agenda ou artefato de como `t0` é construído (`data_solicitacao + diasfilaespera`). Decidir em S3.1 (M2).
4. A coorte municipal A+B soma 262.488 consultas (72,7% das municipais) nesta contagem com 2018; PROGRESSO anterior cita 71,5%. Provável diferença de 2018 (script 19 usava 2019+); a conferir no M1.
5. Ambiente: o `python3` ainda é o venv compartilhado dentro do Drive (`~/Google_drive/skill_academico/.venv`). Faltam lightgbm, xgboost, catboost. **Aguardo autorização para criar o venv novo fora do Drive e instalar** (necessário antes do M3).
6. `requirements.lock` reflete o venv atual (sem lightgbm/xgboost/catboost) até o novo venv existir.
7. Scripts legados continuam na raiz; migrar para `scripts_legados/` só com autorização.
8. Prazo/formato da entrega da disciplina: `config.yaml: prazo_entrega: null`.

---

# M1 — Amostra de teste, divisões temporais e coorte por corte (16/11/2026)

## O que foi feito (executado e verificado)
- `src/coorte.py` + `etapas/S2_coorte_por_corte.py`: coorte e níveis A/B/C com dados **até o corte** de cada ano de teste (2023–2026), mesmos limiares dos scripts 10/19/20, agora em `config.yaml: coorte`. Escopo vem do config (`criterio_x` levanta erro em vez de ser assumido).
- **Conferência com o legado:** a versão "global" (série inteira) reproduz exatamente a coorte e os níveis de `niveis_municipal.csv` (10 CNES, 6 A + 4 B). O 71,5% do PROGRESSO anterior bate quando 2018 é excluído do denominador (245.749 de 343.523 consultas municipais de 2019+).
- `src/splits.py`: holdout, expansiva, deslizante (12/24/36 meses), purged + embargo (P ∈ {0,28,90}, E ∈ {0,7}), folds internos (para 2026: validações em 2024 e 2025), `valida_divisao`, `restringe_teste_comum`, `purga_recomendada`.
- `etapas/S2_amostra_smoke.py`: 4 CNES (8006881, 2237334, 7210280, 2237822), 101.548 linhas (10,3%), todas com consultas em 2018–2026. Lista no `config.yaml: smoke.cnes`.
- `etapas/S4_divisoes.py` (com `--smoke`): 41 divisões geradas e validadas na base municipal completa e na amostra.
- Testes: 38 passando (`tests/test_m0.py`, `test_splits.py`, `test_coorte.py`). `make smoke` completo em ~21 s.

## Resultados
- Coorte por corte (CNES / consultas até o corte / níveis): 2023: 12 / 114.648 / 11A+1B; 2024: 12 / 163.465 / 9A+3B; 2025: 11 / 202.471 / 8A+3B; 2026: 10 / 236.123 / 7A+3B; global: 10 / 245.749 / 6A+4B.
- Diferenças vs. global: 7095708 e 7214219 entram nas coortes de 2023–2025 e saem depois (pararam); 2237385, 9132562 e 7031092 são A nos cortes antigos e viram B no global (a quebra de regime só aparece depois). A versão global teria usado informação futura para classificá-las.
- Purga recomendada (municipal): antecedência consulta−t0 p99 = 14 dias e máximo = 64 dias. P = antecedência + L; com L=3: 17 (p99) ou 67 (máximo). O grid P ∈ {0, 28, 90} cobre o p99 (28) e o máximo (90).

## Pendências / cuidados
1. **Aprovar a lista de CNES do smoke** (regra objetiva descrita no config.yaml; inclui o Materno Infantil 2237822 como estrato).
2. O esquema principal, L principal e período comum continuam **rascunho** (G2). Nada foi escolhido olhando o teste.
3. `usar_futuro=True` (embargo com efeito) só existe como opção testada; nenhum esquema do estudo a usa por padrão.
4. A etapa S3.1 (M2) continua a decidir L; a antecedência de 14 dias precisa de confirmação.
5. Pendências do M0 seguem abertas (venv novo, arquivo bruto, script das distâncias, mover legados).

---

# M2 → M4, auditoria e empacotamento para a máquina potente (16/11/2026)

> Autorizado pelo usuário: executar os testes de todos os marcos, auditar código e resultados e preparar a
> execução completa em outra máquina. **Nenhum treino no conjunto inteiro foi executado** (só smoke, em 4 CNES).

## M2 — Variáveis e painéis (S3, S3.1) [executado e verificado]
- `src/panels.py`: painel CNES-dia com L parametrizável (1, 3, 7, 14) e atraso d; duas versões de agenda:
  final (`ag`, `m_*`) e **conhecida em t − L** (`ag_h`, `m_*_h`, antecedência ≥ L). Defasagens semanais respeitam L (para L=14 não usa t−7).
  Sem `fillna` com média global (o legado usava a média da série inteira).
- Testes "embaralhar o futuro" para L = 1, 3, 7, 14 (`tests/test_panels.py`): alterar desfechos/consultas após T não muda nenhuma
  feature do passado dos dias ≤ T+L; alterar o que foi agendado depois de T não muda `ag_h`/`m_*_h`; fração conhecida monotônica em h.
- **S3.1** (`etapas/S3_1_horizonte.py`, escopo municipal, 360.878 consultas): fração da agenda final já conhecida em t−h:
  h=1 99,8%; 2 96,4%; 3 92,4%; 5 87,6%; 7 79,2%; 10 67,4%; 14 56,0%. **55,8% das consultas têm antecedência exatamente 14 dias**
  (47–63% por ano): sugere regra de abertura de agenda (ou `t0` nominal). Atraso de registro (d): a base não tem o campo (premissa 0).
  ACF média da taxa diária: lag 1 = 0,79; 3 = 0,54; 7 = 0,33; 14 = 0,21. Saída: `entregaveis/justificativa_horizonte.md`.
- **O L principal NÃO foi decidido** (critério operacional do usuário, G2). O código roda todos os L; `lag_principal: 3` segue RASCUNHO.
- Imputação/padronização: só no treino, dentro dos Pipelines de cada modelo.

## M3 — Modelos de referência e candidatos (S5) [executado e verificado]
- Interface comum `src/models/base.py` (`fit`, `predict_mean`, `predict_dist`, espaço de busca). Modelos: `naive`, `baseline_poisson`,
  `hgb_poisson`, `rf`, `poisson_reg` (Poisson L2), `paciente_hgb` (HGB por paciente somado por CNES-dia) e `lightgbm` (**opcional; wrapper
  escrito mas NÃO executado: biblioteca não instalada aqui**).
- **Reprodução do legado** (`etapas/S5_reproduz_baseline.py`, coorte A+B antiga, L=3, agenda final): baseline WAPE A 0,4511 / B 0,3131 e
  naive de 4 semanas A 0,4861 / B 0,3377 — **idênticos ao script 17** (n idêntico após replicar a regra de linhas do legado: as 17 primeiras
  dias de 2023 de cada CNES ficavam fora porque o fator k só via dados de 2023). Baseline A+B = 0,3826 (legado 0,383).

## M4 — Tuning, treino, avaliação e saídas (S6–S9, S13) [executado e verificado no smoke]
- `src/experimento.py` + `etapas/S7_experimentos.py`: tuning aninhado (folds internos 2021–2025 conforme o ano de teste; para 2026, 2024 e
  2025), mesmo orçamento (`n_configs`) para todos, janela do deslizante como hiperparâmetro, k_h e dispersão estimados fora da amostra,
  sementes derivadas, **execução retomável**. Corrigidos nos novos módulos: `max_bins` removido da grade; folds até 2025.
- `src/metrics.py`, `src/dm_test.py`, `src/avaliacao.py`, `etapas/S9_avaliacao.py`: mesmas linhas (interseção) e mesma distribuição
  preditiva (Binomial Negativa, φ por divisão) para todos; WAPE, viés, CRPS, cobertura 80/50, largura, ECE de P(≥k), AUC por célula;
  Diebold-Mariano (HAC semanal) vs baseline e vs melhor, Holm por família (n de comparações informado), bootstrap semanal do ganho de WAPE;
  subgrupos (nível, ano, sem Materno Infantil); esquemas no mesmo conjunto de dias de teste; defasagens; modos de agenda.
- `etapas/S13_tabelas_figuras.py`: tabelas Markdown e figuras (PNG/PDF) só a partir dos CSVs.
- **Teste anti-vazamento do tuning** (`tests/test_experimento.py`): corromper desfechos e variáveis a partir de 1/jan/2025 não muda
  hiperparâmetros, φ, critério de validação nem k_h escolhidos para o teste de 2025 (hgb_poisson, poisson_reg, paciente_hgb).
- Resultado do smoke (4 CNES, teste 2023–2026, 3.078 CNES-dias, L=3, agenda conhecida) — **só valida o código, não é resultado científico**:
  WAPE baseline 0,416; naive 0,439; Poisson reg. 0,410; RF 0,408; HGB Poisson 0,408; HGB por paciente 0,392 (ganho 5,8%, IC 95% 3,6–7,7%).
  Mesma ordem dos resultados antigos (por paciente < séries < baseline).
- `make smoke` completo: 96 testes passam; 15,6 min em série (n_jobs=1); com 4 processos (config atual) a etapa S7 levou 4 min e o resultado é idêntico (111/111 previsões iguais).

## Auditoria (ver `AUDITORIA_2026-11-16.md`) e empacotamento
- `README_REPRODUCAO.md`, `Makefile` (`make smoke`, `make full`, `make estimativa`, `make venv`), `run_all.sh`, `etapas/S0_confere_hash.py`
  + `dados/hashes_esperados.json`, `requirements-extra.txt`, `etapas/S15_estimativa_recursos.py`.
- Estimativa do completo (extrapolada, ±2x): ~33 CPU-horas ≈ 3–4 h com 11 processos; ~0,7 GB de RAM por processo.

## Pendências
1. **G2** (esquema principal, L, modo de agenda, período comum, métrica primária) e confirmação de `t0`/regra de 14 dias com o serviço.
2. Instalar lightgbm na máquina nova e conferir o wrapper (nunca rodou). XGBoost/CatBoost/MLP, ETS/SARIMA/GLM-NB e recalibração k: não portados.
3. S10 (importância de variáveis), S11 (overbooking) e S12 completo: não implementados.
4. Arquivo bruto anterior ao qmd e script das distâncias (limitações de dados).
5. Scripts legados continuam na raiz (não movidos); `scripts_legados/` vazio.


## 17/11/2026 — Limitações resolvidas, L ∈ {3, 14}, pasta `antigos/`
**Feito e verificado [V]**
- Ambiente: venv `~/.venvs/absenteismo` (fora do Drive) com lock + lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10 (`requirements.lock` regravado).
- Scripts fora do pipeline (01–24, grafico_*, overbooking_*, curva_*, prever_dia, utils_municipal) movidos para `antigos/` (35 arquivos); `scripts_legados/` vazia removida.
- Novos modelos (`src/models/extras.py`, `estatisticos.py`): XGBoost, CatBoost, MLP, GLM-NB, HGB com recalibração k, ETS, SARIMA, SARIMAX. LightGBM executado pela 1ª vez.
- Novas etapas: S10 (importância por permutação), S11 (overbooking por quantil), S12 (sensibilidade, estabilidade do ranking, 5 sementes); S13 gera tabelas/figuras delas.
- Cenários de L: só 3 e 14 (config). Smoke completo rodou: 115 testes, S5 reproduz o legado, S7 266 divisões (7,5 min, 6 processos), S9–S13 ok.
- Testes novos: previsões de ets/hgb_poisson_k/sarima só usam desfechos ≤ t−L (L=3 e 14); filtro de Kalman = `forecast` nativo; teste de vazamento do tuning estendido a 10 modelos.
**Estimativa do completo:** ~88 CPU-h ≈ 10 h com 11 processos.
**Ressalvas:** MLP é instável (WAPE smoke 0,45–0,56, dp entre sementes 0,013); `hgb_poisson_k` usa previsões dentro da amostra no início do teste; GLM-NB com matriz de desenho singular (pinv); SARIMA com ajuste nos últimos 730 dias e 12 configs; SHAP/ALE não feitos; escopo geral, coorte global e remoção de variáveis suspeitas não rodados. Resultados do smoke validam código, não são científicos.


## 17/11/2026 (tarde) — G2 congelado, auditoria atualizada, smoke final
**Decisões do usuário**
- **G2 congelado** (`entregaveis/plano_analise.md`; comentários no `config.yaml`): L principal 3 (L=14 = sensibilidade), esquema expansiva, agenda `conhecida`, teste comum 2024-01-01 a 2026-03-31, métrica primária WAPE + CRPS (DM com HAC semanal, Holm). Demais variantes = sensibilidade.
- LSTM **não** entra no estudo (instável, cara, sem ganho esperado; o mínimo da disciplina já está coberto).
- Lista do smoke mantida: 8006881, 2237334, 7210280, 2237822 (todas com consultas em todos os anos 2018–2026); serve só para validar o código. O perfil `completo` usa todas as CNES do escopo municipal (111).
**Feito e verificado [V]**
- `AUDITORIA_2026-11-16.md` atualizada para o estado de 17/11 (115 testes, modelos e etapas novos, caveats, estimativa).
- `CLAUDE.md`: L ∈ {3, 14} (seção 3.1) e seção 9 de pendências reescrita (resolvidas × abertas).
- Smoke final do zero (`make limpa-smoke` + `make smoke`, código de saída 0): 115 testes, S5 reproduz o legado, S7 266 divisões (634 s), S9–S13 ok (29 saídas).
**Ideias da aula de séries temporais (discussão, NADA implementado [E])**
1. ACF/PACF e teste de estacionariedade só no treino anterior ao 1º teste, para justificar L, janela de 28 dias e ordens do SARIMA.
2. Taxa com decaimento exponencial (meia-vida escolhida nos folds internos) em vez da média plana de 28 dias.
3. MLP como média de várias sementes, `max_iter` escolhido nos folds internos.
4. Tabela de ganho × custo (WAPE, ganho sobre o baseline, tempo de treino).
5. Sensibilidade com reajuste trimestral do modelo (≈4x CPU; só após o completo e com aprovação).
**Falta**
- Aprovar e rodar o completo na outra máquina (`make venv`, `make smoke`, `make estimativa JOBS=11`, `./run_all.sh 11`).
- Premissas de dados (`t0`, atraso do desfecho, `possuiunidadeindicada`, `idade`); arquivo bruto e script das distâncias; prazo/formato.
- SHAP/ALE; sensibilidades de escopo geral, coorte global e remoção de variáveis suspeitas; comparação sequencial × paralela refeita.
- Só depois do completo: `entregaveis/metodologia.md`, `resultados.md`, `tripod_ai_checklist.md`.

## 17/11/2026 (noite) — Confirmações do usuário sobre os dados
- `t0 = data_solicitacao + max(diasfilaespera, 0)` vem do arquivo bruto; `diasfilaespera` é o tempo da solicitação até o **agendamento** no sistema, não até a consulta. Logo `t0` é a data de agendamento, o que sustenta o modo `conhecida`. Segue como observação a regra de abertura de agenda (55,8% das consultas com antecedência exatamente 14 dias).
- Arquivo bruto e script das distâncias: usuário confirmou que está tudo certo; saem das pendências e das conferências.
- Melhorias da aula ficam para depois.
- Ainda abertos: atraso de registro do desfecho, `possuiunidadeindicada`, `idade`, definição de `status`.
