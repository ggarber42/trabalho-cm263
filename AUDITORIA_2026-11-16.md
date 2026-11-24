# Auditoria de código e resultados — M0 a M4 (16/11/2026), atualizada em 17/11/2026

Escopo: tudo o que foi escrito nestas sessões (`src/`, `etapas/`, `tests/`, `config.yaml`, `Makefile`). Os scripts anteriores ao pipeline (01–24 e afins) foram movidos para `antigos/`, **não foram auditados nem alterados**, e o pipeline não os importa. Legenda: **[V]** executado e verificado; **[E]** só escrito/lido, não executado.

Atualização de 17/11/2026 (commit 9a7ebf3): L restrito a {3, 14}; novos modelos (XGBoost, CatBoost, MLP, GLM-NB, ETS, SARIMA, SARIMAX, HGB com recalibração k); novas etapas S10 (importância), S11 (overbooking) e S12 (sensibilidade); libs instaladas; `make smoke` completo executado. Itens atualizados estão marcados com "(17/11)".

## 1. O que foi verificado

| # | Verificação | Resultado |
|---|---|---|
| 1 | Suíte de testes automáticos (`make test`) | **[V]** (17/11) **115 passam** (eram 96 em 16/11). O teste de contrato cobre todos os modelos registrados, inclusive sarima |
| 2 | **Recálculo independente do baseline** (só pandas, sem importar `src/`) nas 4 CNES do smoke, agenda final, L=3 (16/11) | **[V]** idêntico: n = 3.078, WAPE 0,4059, max\|Δμ\| = 0 |
| 3 | **Reprodução do legado** (script 17) na coorte antiga A+B | **[V]** baseline A 0,4511 / B 0,3131 e naive A 0,4861 / B 0,3377, n idênticos (10.333 e 4.696); reconferido no smoke de 17/11 ("REPRODUZ o legado dentro da tolerância") |
| 4 | Painel: "embaralhar o futuro" (L = 1, 3, 7, 14), `ag_h` ignora o agendado depois, `rd1`/`rdow4` com L=14 não usam t−7 | **[V]** |
| 5 | Coorte/níveis: não mudam ao truncar ou adulterar dados após o corte (sintético e base real) | **[V]** |
| 6 | Divisões: nenhum treino ≥ início do teste, nenhuma linha em dois conjuntos, purga exata (P = 0, 28, 90), folds para todo ano (2026 → 2024/2025), embargo, teste comum; teste que injeta vazamento é detectado | **[V]** |
| 7 | **Tuning não vê o teste**: corromper desfechos/variáveis a partir de 1/jan/2025 não muda params, φ, critério nem k_h | **[V]** (17/11) agora parametrizado em **10 modelos**: hgb_poisson, poisson_reg, paciente_hgb, ets, glm_nb, xgboost, catboost, mlp, hgb_poisson_k, sarima |
| 8 | Métricas: CRPS da NB confere com Monte Carlo; φ recuperado por momentos; cobertura ≥ nominal (discreta); DM com HAC maior que o ingênuo sob autocorrelação; Holm; bootstrap | **[V]** |
| 9 | Determinismo: mesma unidade duas vezes = mesmas previsões; sequencial × paralelo | **[V]** (16/11, sem os modelos novos; ver seção 4) |
| 10 | Convenções do CLAUDE.md (docstrings, cabeçalhos, sem caminho absoluto, sem `random_state=None`) por análise estática | **[V]** (17/11) o teste estático pegou `md` sem docstring em S13 e foi corrigido |
| 11 | LGPD: nenhuma tabela/previsão tem `idpaciente`/`idsolicitacao`; parquets com linhas de paciente no `.gitignore` | **[V]** (16/11) |
| 12 | Hashes das entradas gravados e conferidos (`dados/hashes_esperados.json`) | **[V]** |
| 13 | (17/11) Modelos de série (ets, hgb_poisson_k, sarima) com L = 3 e 14: alterar desfechos posteriores a t−L não muda a previsão de t | **[V]** teste novo |
| 14 | (17/11) Truque do filtro de Kalman (`apply` + Z·T^L·estado filtrado) × `forecast` nativo do statsmodels | **[V]** teste novo, resultados iguais |
| 15 | (17/11) **Smoke completo de ponta a ponta** (4 CNES, 3 configurações de tuning, todos os esquemas, L ∈ {3,14}, todos os modelos): S5, S7 (266 divisões, 445 s com 6 processos), S9 (227.772 linhas de previsão), S10, S11, S12, S13 (29 saídas) | **[V]** sem erro |
| 16 | (17/11) Libs instaladas e exercitadas: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10 (numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, statsmodels 0.15.0) | **[V]** |

## 2. Problemas encontrados e o que foi feito

| Problema | Origem | Ação |
|---|---|---|
| Lista de colunas do painel pedia `taxa_falta_prev`, que não existe no parquet | código novo | corrigido |
| Rótulo do fluxograma trocava vírgula por ponto; rótulos de p sobrepostos | S13 | corrigido |
| Tabela de conferência com o legado reaproveitava colunas de nível | S2 | arquivo próprio `coorte_conferencia_legado_<escopo>.csv` |
| Funções internas sem docstring | S1, S3.1, S9, S13, `series.py` | corrigido, coberto por teste estático |
| (17/11) SARIMA levava 44–70 s para 4 CNES (inviável em escala) | `estatisticos.py` | ajuste só nos últimos 730 dias + grade de 12 configurações (DECISÃO declarada) |
| (17/11) S11: φ por linha e expressão de k confusa | S11 | simplificado; q=0 vira a linha "sem_previsao" |
| (17/11) GLM-NB: `SingularMatrixWarning` (desenho colinear, IRLS com pseudo-inversa) | `extras.py` | aceito e documentado; cai em r28 se falhar |
| **Legado:** coorte e níveis usavam a série inteira | scripts 10/19/20 | substituído por `src/coorte.py` |
| **Legado:** `painel_series` imputava `m_*` com a média global (regra 7) | `utils_municipal.py` | não replicado |
| **Legado:** baseline do script 17 só avaliava linhas com k finito | script 17 | replicado só na reprodução (S5) |
| Legado: `FOLD_YEARS` até 2024 e `max_bins=128` com CNES categórica | scripts 22/23 | resolvido nos módulos novos |

## 3. Limitações que a auditoria NÃO elimina

1. **Nada foi treinado no conjunto completo.** O smoke (4 CNES, 3 configurações) valida o código, **não** é resultado científico. Estimativa do completo (17/11): **≈ 88 CPU-h ≈ 9,6 h com 11 processos, ≈ 0,7 GB por processo**, ±2x.
2. ~~LightGBM nunca rodou~~ **Resolvido (17/11):** LightGBM, XGBoost e CatBoost instalados e rodaram no smoke.
3. **Ainda não feitos:** SHAP/ALE (a S10 usa importância por permutação); sensibilidades de escopo geral, coorte global e remoção de variáveis suspeitas (exigem novas execuções). ETS, SARIMA/X, GLM-NB, k, MLP, XGBoost, CatBoost, S10, S11 e S12 **foram implementados em 17/11**.
4. **Caveats dos modelos novos (17/11):**
   - **MLP instável:** WAPE de 0,45 a 0,56 no smoke, desvio de ~0,013 entre sementes.
   - **`hgb_poisson_k`:** no início do teste o k usa previsões dentro da amostra (declarado); no smoke ficou pior que o HGB simples (0,419 × 0,408).
   - **SARIMA/SARIMAX:** ajuste só nos últimos 730 dias e grade de 12 configurações; só no modo `conhecida`; SARIMAX com 3 exógenas da agenda.
   - **GLM-NB:** matriz de desenho singular, resolvida por pseudo-inversa.
5. **Premissas de dados:** `t0` **confirmado pelo usuário em 17/11/2026** como data de agendamento (`data_solicitacao + max(diasfilaespera, 0)`, `diasfilaespera` = tempo até o agendamento); permanece a observação de que 55,8% das consultas têm antecedência exatamente 14 dias (regra de abertura de agenda a confirmar). Não confirmadas: atraso de registro do desfecho, `possuiunidadeindicada`, `idade`, remarcações. O modo `conhecida` depende de `t0` ser a data real de agendamento. Com L = 14 só ~56% da agenda final é conhecida.
6. **Distribuição preditiva única (NB):** coberturas de 80% acima do nominal (~0,91–0,93); igual para todos os modelos, mas o texto deve dizer.
7. **Comparações múltiplas:** Holm dentro de cada família, com `n_comparacoes` informado; sem correção entre famílias (esquemas, L, modos, sensibilidades: descritivas). "Melhor modelo" identificado no mesmo teste: descritivo.
8. **Janela deslizante e `usar_futuro=True`:** embargo com efeito só sob `usar_futuro=True` (testado; nenhum esquema do estudo o usa).
9. **Categorias das variáveis categóricas** vêm do vocabulário da base inteira (só nomes); categorias novas no teste entram como ausentes no HGB.
10. **2026** tem só 13 semanas: ICs largos.
11. **G2 congelado em 17/11/2026** (`entregaveis/plano_analise.md`): L = 3, esquema expansiva, agenda conhecida, teste comum 2024-01 a 2026-03, métrica primária WAPE + CRPS. Demais variantes = sensibilidade.
12. **Arquivo bruto e script das distâncias:** confirmados pelo usuário (17/11/2026); fora das conferências.

## 4. Determinismo em paralelo

Em 16/11: execução sequencial (n_jobs = 1) × repetição com 4 processos (`--refazer`): **111 de 111 arquivos de previsão idênticos**, max|Δμ| = 0, params, φ, critério, k_h, janela e sementes iguais. [V]
(17/11) Esta comparação **não foi refeita** com os modelos novos nem com L = 14 [E]; o teste de que a mesma unidade duas vezes dá as mesmas previsões vale para os modelos registrados, mas a comparação sequencial × paralelo completa fica como pendência.

## 5. Resultados do smoke de 17/11 (apenas validação do código; 4 CNES, 3.078 CNES-dias, L=3, `conhecida`)

WAPE: baseline 0,4159; naive 0,4390; ets 0,4118; glm_nb 0,4106; poisson_reg 0,4103; hgb_poisson 0,4077; hgb_poisson_k 0,4190; rf 0,4078; lightgbm 0,4021; xgboost 0,4012; catboost 0,4045; mlp 0,4629; paciente_hgb 0,3917. Com L = 14 o baseline sobe para 0,4946.
Repetição com 5 sementes: catboost dp 0,001; rf 0,0008; mlp 0,013–0,015.
S10: com L = 3 domina o grupo `taxas_defasadas` (ΔWAPE ≈ 0,134); com L = 14 domina `agenda` (≈ 0,45 contra ≈ 0,18); calendário é desprezível.
S11: o benefício do overbooking atinge o máximo perto de q ≈ 0,3 e depende da calibração (descritivo).
**Não interpretar como achado:** amostra de 4 CNES e tuning reduzido.
