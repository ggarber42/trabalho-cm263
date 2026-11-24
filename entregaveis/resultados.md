# Resultados

*Versão 1 (rascunho para revisão), 20/11/2026. Todos os valores vêm de `tabelas/` e `resultados/municipal_completo/` (execução do perfil completo). Protocolo em `metodologia.md`. Escopo municipal, desfecho FALTANTE, unidade CNES-dia salvo indicação.*

## 1. Seleção dos dados

A Figura 1 (`figuras/completo/fluxograma_selecao.png`) resume a seleção. A base analítica tem 983.875 consultas agendadas em 221 CNES, e todas têm desfecho observável (REALIZADA ou FALTANTE). O escopo municipal retém 360.878 consultas (36,7%) em 111 CNES; as 622.997 consultas restantes são de estabelecimentos estaduais, federais ou filantrópicos. Nenhum estabelecimento foi excluído por inatividade, volume ou código placeholder.

Os filtros de elegibilidade anteriores ao parquet (solicitante de centro de saúde/UBS, população do setor, data de agendamento presente) são descritos na Metodologia, mas suas contagens não puderam ser recalculadas porque o arquivo bruto não está disponível neste repositório.

## 2. Descrição da coorte

A população de teste de cada ano é a coorte estável do corte, definida com dados anteriores ao início do ano (Tabela 1). Não houve nível C em nenhum corte.

**Tabela 1.** Coorte e níveis por corte (`tabelas/coorte_por_corte_municipal.csv`).

| Corte | Consultas da coorte até o corte | CNES na coorte | Nível A | Nível B |
|---|---|---|---|---|
| 01/01/2023 | 114.648 | 12 | 11 | 1 |
| 01/01/2024 | 163.465 | 12 | 9 | 3 |
| 01/01/2025 | 202.471 | 11 | 8 | 3 |
| 01/01/2026 | 236.123 | 10 | 7 | 3 |

Em relação à coorte global (com a série inteira), três estabelecimentos mudam de nível no corte de 2023 (todos de A para B) e um estabelecimento muda de A para B nos cortes de 2024 a 2026 (`tabelas/coorte_comparacao_global_municipal.csv`). Essas diferenças são descritivas; a análise usa a coorte de cada corte.

## 3. Desempenho dos modelos (principal)

Expansiva, L = 3, modo de agenda conhecida, teste de 2023 a 2026. A comparação usa **8.073 CNES-dias** (interseção das linhas de todos os modelos; ver Metodologia, seção 10). A Tabela 2 reúne os 15 modelos. O ganho é a redução relativa do WAPE em relação ao baseline Poisson.

**Tabela 2.** Desempenho no teste principal (`tabelas/res_municipal_completo_principal_metricas.csv`).

| Modelo | WAPE | Ganho vs baseline | Viés | CRPS | Cobertura 80% | Cobertura 50% | ECE | AUC (P≥1) |
|---|---|---|---|---|---|---|---|---|
| `paciente_hgb` | **0,348** | +4,3% | 0,979 | **0,968** | 0,914 | 0,730 | 0,019 | 0,918 |
| `catboost` | 0,351 | +3,5% | 1,001 | 0,968 | 0,909 | 0,726 | 0,010 | 0,910 |
| `lightgbm` | 0,352 | +3,2% | 1,003 | 0,970 | 0,908 | 0,727 | 0,012 | 0,909 |
| `xgboost` | 0,353 | +3,1% | 1,001 | 0,971 | 0,909 | 0,723 | 0,012 | 0,909 |
| `hgb_poisson` | 0,354 | +2,8% | 1,004 | 0,974 | 0,910 | 0,727 | 0,011 | 0,909 |
| `rf` | 0,357 | +1,8% | 1,007 | 0,984 | 0,908 | 0,720 | 0,011 | 0,908 |
| `mlp` | 0,360 | +1,1% | 1,000 | 0,994 | 0,914 | 0,723 | 0,012 | 0,906 |
| `poisson_reg` | 0,361 | +0,8% | 0,972 | 1,001 | 0,913 | 0,719 | 0,019 | 0,901 |
| `glm_nb` | 0,361 | +0,8% | 0,972 | 1,002 | 0,913 | 0,718 | 0,020 | 0,901 |
| `ets` | 0,361 | +0,7% | 1,000 | 0,997 | 0,910 | 0,715 | 0,012 | 0,908 |
| **`baseline_poisson`** | **0,364** | — | 0,979 | 1,009 | 0,903 | 0,711 | 0,018 | 0,905 |
| `hgb_poisson_k` | 0,364 | −0,0% | 1,006 | 1,004 | 0,902 | 0,709 | 0,012 | 0,907 |
| `sarima` | 0,367 | −0,9% | 0,991 | 1,021 | 0,903 | 0,705 | 0,017 | 0,904 |
| `sarimax` | 0,372 | −2,2% | 1,005 | 1,034 | 0,902 | 0,706 | 0,015 | 0,910 |
| `naive` | 0,394 | −8,2% | 0,981 | 1,101 | 0,881 | 0,678 | 0,028 | 0,896 |

**Significância do ganho (Diebold-Mariano, erro absoluto, vs baseline).** Ganho relativo com IC 95% (bootstrap em blocos semanais) e p ajustado por Holm dentro das 14 comparações com o baseline (`tabelas/res_municipal_completo_principal_dm.csv`):

- `paciente_hgb`: +4,3% (IC 3,2 a 5,5); p_Holm = 1,1 × 10⁻⁶.
- `catboost`: +3,5% (IC 2,7 a 4,4); p_Holm = 2,6 × 10⁻¹⁰.
- `lightgbm`: +3,2% (IC 2,3 a 4,1); p_Holm = 3,7 × 10⁻⁸.
- `xgboost`: +3,1% (IC 2,2 a 4,0); p_Holm = 3,9 × 10⁻⁸.
- `hgb_poisson`: +2,8% (IC 2,0 a 3,7); p_Holm = 1,3 × 10⁻⁷.
- `rf`: +1,8% (IC 1,1 a 2,6); p_Holm = 0,003.
- `mlp`, `ets`, `glm_nb`, `poisson_reg`, `sarima`, `sarimax` e `hgb_poisson_k`: sem diferença significativa após Holm (p_Holm entre 0,49 e 0,99).
- `naive`: −8,2% (IC −9,7 a −6,8); p_Holm < 10⁻¹⁰.

**Comparação com o melhor modelo.** Tomando `paciente_hgb` como referência (14 comparações, Holm, descritivo): `catboost` (p_Holm = 0,35), `lightgbm` (0,35) e `xgboost` (0,35) não diferem dele. `hgb_poisson` (0,26) também não. `rf` (0,011) e os demais modelos (`mlp`, `ets`, `glm_nb`, `poisson_reg`, `sarima`, `sarimax`, `hgb_poisson_k` e o baseline) diferem, com p_Holm ≤ 0,001.

Para CRPS, todos os modelos de árvore diferem do baseline com p_Holm < 10⁻⁶; `paciente_hgb` tem o menor CRPS entre os 15 (0,968), praticamente igual ao de `catboost` (0,968).

## 4. Calibração

- **Cobertura dos intervalos.** Os intervalos de 80% cobrem entre 0,90 e 0,91 dos casos para a maioria dos modelos de árvore, e os de 50% cobrem entre 0,70 e 0,73. Ou seja, os intervalos são mais largos que o nominal. O baseline tem cobertura de 0,90 a 80% e 0,71 a 50%.
- **Viés.** Σŷ/Σy fica entre 0,97 e 1,01 na maioria dos modelos. A exceção é `paciente_hgb` em 2026 (1,097; ver Tabela 4).
- **ECE** de P(faltas ≥ k) varia de 0,010 (`catboost`) a 0,028 (`naive`); o baseline tem 0,018.
- **Por consulta (`paciente_hgb`, por ano de teste):** Brier de 0,157 a 0,173, AUC de 0,665 a 0,677 e log loss de 0,485 a 0,523 (`tabelas/res_municipal_completo_paciente_metricas_consulta.csv`). Não há comparação com o baseline nessa métrica.

## 5. Subgrupos: nível, ano e sem o Materno Infantil

**Tabela 3.** WAPE por subgrupo, para o baseline e os dois melhores modelos (`tabelas/res_municipal_completo_principal_subgrupos.csv`).

| Subgrupo | n (CNES-dias) | Baseline | `paciente_hgb` | `lightgbm` |
|---|---|---|---|---|
| Geral | 8.073 | 0,364 | 0,348 | 0,352 |
| Nível A | 6.175 | 0,408 | 0,392 | 0,398 |
| Nível B | 1.898 | 0,303 | 0,288 | 0,289 |
| Teste 2023 | 2.745 | 0,348 | 0,340 | 0,336 |
| Teste 2024 | 2.521 | 0,364 | 0,346 | 0,352 |
| Teste 2025 | 2.232 | 0,391 | 0,367 | 0,378 |
| Teste 2026 (jan–mar) | 575 | 0,353 | 0,347 | 0,349 |
| Sem Materno Infantil | 7.283 | 0,351 | 0,338 | 0,339 |

Em todos os subgrupos, `paciente_hgb` e `lightgbm` têm WAPE abaixo do baseline. Entre os dois, `paciente_hgb` é menor em todos os subgrupos, exceto em 2023, em que `lightgbm` é ligeiramente menor (0,336 contra 0,340). O viés de `paciente_hgb` é 1,03 em 2025 e 1,10 em 2026. Excluindo o Hospital Materno Infantil Presidente Vargas (CNES 2237822, taxa de falta de cerca de 7%), o WAPE cai para 0,338 em `paciente_hgb` e 0,339 em `lightgbm`, com o mesmo ordenamento.

## 6. Horizonte de previsão (L = 3 contra L = 14)

**Tabela 4.** WAPE por horizonte, nas mesmas 8.068 linhas (`tabelas/res_municipal_completo_defasagens.csv`).

| Modelo | L = 3 | L = 14 | Ganho L = 14 vs baseline |
|---|---|---|---|
| `baseline_poisson` | 0,364 | 0,457 | — |
| `paciente_hgb` | 0,348 | 0,439 | +4,0% |
| `xgboost` | 0,353 | 0,394 | +13,9% |
| `catboost` | 0,351 | 0,394 | +13,8% |
| `lightgbm` | 0,352 | 0,395 | +13,6% |
| `hgb_poisson` | 0,354 | 0,394 | +13,7% |
| `ets` | 0,361 | 0,453 | +0,9% |

Em L = 14, `paciente_hgb` fica em 10º lugar entre os 15 modelos, e os modelos de árvore sobre a série CNES-dia são os de melhor WAPE. Ao passar de L = 3 para L = 14, o WAPE do baseline sobe 0,093, o de `paciente_hgb` sobe 0,090 e o de `lightgbm` sobe 0,042. O ranking entre modelos muda: a correlação de Spearman entre as ordens em L = 3 e em L = 14 é 0,77 (Kendall 0,64; `tabelas/sensibilidade_municipal_completo_ranking.csv`).

## 7. Esquemas de divisão temporal

No período comum de teste (01/01/2024 a 31/03/2026, 5.479 CNES-dias), o WAPE dos modelos selecionados por esquema é (`tabelas/res_municipal_completo_esquemas_teste_comum.csv`):

| Esquema | Baseline | `hgb_poisson` | `lightgbm` | `paciente_hgb` | Ganho do `paciente_hgb` |
|---|---|---|---|---|---|
| Expansiva (principal) | 0,374 | 0,364 | 0,362 | 0,354 | +5,1% |
| Holdout (corte 2023) | 0,373 | 0,365 | 0,365 | 0,360 | +3,5% |
| Deslizante | 0,376 | 0,365 | 0,363 | 0,359 | +4,4% |
| Purga P = 0 | 0,374 | 0,365 | 0,362 | 0,355 | +4,9% |
| Purga P = 28 | 0,374 | 0,365 | 0,363 | 0,356 | +4,8% |
| Purga P = 90 | 0,374 | 0,365 | 0,365 | 0,357 | +4,3% |

Os ganhos ficam entre +3,5% e +5,1% para `paciente_hgb` em todos os esquemas. A correlação de postos entre o ranking da expansiva e o de cada outro esquema é 0,90 (deslizante), 0,85 (holdout), 0,97 (purga P = 0), 0,90 (purga P = 28) e 0,95 (purga P = 90).

## 8. Modo da agenda (sensibilidade)

Em L = 3, o WAPE com a agenda conhecida em t − L (principal) e com a agenda final (otimista):

| Modelo | Conhecida (n = 8.224) | Final |
|---|---|---|
| Baseline | 0,364 | 0,352 |
| `lightgbm` | 0,353 | 0,340 |
| `paciente_hgb` | 0,349 | 0,333 |

Com a agenda final, o WAPE cai para todos os modelos, e o ganho do `paciente_hgb` sobe de +4,2% para +5,6%. O modo final não é o resultado principal (ver Metodologia, seção 5).

## 9. Utilidade de decisão: overbooking

Regra: agendar k encaixes extras por CNES-dia, com k = quantil 0,20 da distribuição preditiva (risco aceito P(faltas < k) ≤ 0,20). Os valores são somas no teste de L = 3 (n = 8.073 CNES-dias; 32.099 faltas observadas). Não há intervalos de confiança (`tabelas/overbooking_municipal_completo.csv`).

**Tabela 5.** Consultas recuperadas, excesso de encaixes e benefício líquido com custo relativo c = 1 (excesso custa o mesmo que uma consulta recuperada), para L = 3.

| Modelo | Recuperadas | Excesso | Dias com excesso | Benefício líquido (c = 1) |
|---|---|---|---|---|
| `lightgbm` | 18.520 | 954 | 8,0% | 17.566 |
| `catboost` | 18.501 | 960 | 8,0% | 17.541 |
| `xgboost` | 18.497 | 958 | 8,0% | 17.539 |
| `hgb_poisson` | 18.506 | 983 | 8,0% | 17.523 |
| `paciente_hgb` | 17.795 | 786 | 6,8% | 17.009 |
| `baseline_poisson` | 17.627 | 902 | 7,3% | 16.725 |
| `naive` | 17.030 | 995 | 8,1% | 16.035 |

Em L = 3, `lightgbm` lidera nos três custos (c = 1, 2 e 3). Com L = 14 (n = 8.221), `paciente_hgb` lidera nos três custos (benefício de 17.882, 16.075 e 14.268), à frente de `ets` e `sarima`, mas com excesso bem maior (1.807 encaixes e 12,9% dos dias com excesso, contra 6,8% em L = 3). A ordem entre modelos no overbooking não é a mesma que no WAPE. Os benefícios são resultados descritivos sobre o teste; os custos foram declarados, não estimados.

## 10. Importância de variáveis

Importância por permutação para os modelos de árvore de série, em L = 3, média dos anos de teste. Aumento do WAPE quando o grupo é embaralhado (`tabelas/importancia_municipal_completo_grupos.csv`):

| Grupo | `hgb_poisson` | `xgboost` |
|---|---|---|
| Taxas de falta defasadas | +0,080 | +0,078 |
| Agenda | +0,023 | +0,018 |
| Calendário | +0,003 | +0,003 |

No `hgb_poisson`, as taxas defasadas são o grupo mais importante em cada ano de teste (2023 a 2026). Como variáveis correlacionadas dividem o efeito entre si, a leitura é por grupo. Não foi calculada importância para `paciente_hgb`.

## 11. Estabilidade entre sementes

Cinco sementes por modelo (`tabelas/sensibilidade_municipal_completo_sementes.csv`). Em L = 3, o desvio-padrão do WAPE é 0,0004 para `catboost`, abaixo de 0,001 para `rf` e 0,003 para `mlp`. Em L = 14, `mlp` varia de 0,410 a 0,437 (desvio-padrão 0,011). Os modelos de árvore são estáveis; a rede neural não é.

## 12. Limitações

1. **Poucas CNES e pouco tempo de teste.** O teste reúne 10 a 12 estabelecimentos por corte (8.073 CNES-dias). O ano de 2026 cobre só janeiro a março (575 CNES-dias). As diferenças entre `paciente_hgb`, `catboost` e `lightgbm` são pequenas e não significativas.
2. **Calibração conservadora.** A cobertura de 80% fica em 0,90 a 0,91, e a de 50% em 0,70 a 0,73. Os intervalos são mais largos que o nominal. Isso afeta a regra de overbooking, cujo quantil vem da mesma distribuição.
3. **Horizonte operacional.** O critério que justifica L = 3 como horizonte principal ainda não foi declarado com fonte. Em L = 14, a vantagem do modelo por paciente desaparece no WAPE (ele fica em 10º lugar). Não se deve generalizar o resultado de L = 3 para outros horizontes.
4. **Interseção de linhas.** A comparação principal tem 8.073 linhas, 151 a menos que as 8.224 válidas para o baseline. A diferença coincide com as 151 falhas de ajuste do SARIMAX; o SARIMA também falha em 86 linhas, e a tabela não separa as duas contribuições. Essas linhas saem de todos os modelos, então o ganho mede a mesma população, mas a população é ligeiramente menor.
5. **Orçamento de busca desigual.** ETS (20 configurações), SARIMA (12) e os GLMs (6) foram avaliados com grades inteiras menores que as 24 configurações dos demais modelos.
6. **Premissas de dados não confirmadas.** O atraso de registro do desfecho (d = 0), o momento de `idade`, e a data de preenchimento de `status` e `possuiunidadeindicada` (excluídas do modelo) dependem de confirmação com o serviço. Os filtros de elegibilidade e o script das distâncias não puderam ser reexecutados.
7. **Única cidade e um único sistema de agendamento.** Não há validação externa. A generalização para outras redes depende de regras de agendamento semelhantes.
8. **Equidade limitada.** O dicionário de variáveis (`dicionario_variaveis.md`) não tem sexo nem raça/cor; a análise de subgrupos usa nível, ano e o estrato do Materno Infantil. A idade foi mantida como variável, mas não há estratificação por faixa etária.
9. **Interpretação.** A importância por permutação não é causal, e SHAP e ALE não foram calculados.
10. **Múltiplas comparações.** Holm é aplicado dentro de cada família de 14 comparações. Não há correção entre as famílias (principal, esquemas, horizonte e modo). O melhor modelo foi identificado no mesmo teste, então as comparações com ele são descritivas.
