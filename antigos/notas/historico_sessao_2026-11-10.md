# Histórico da sessão — 10/11/2026

Projeto: previsão de faltas (absenteísmo) em consultas ambulatoriais, GERCON Porto Alegre. Pasta: `serie_temporal/`.

## 1. Ponto de partida (scripts 01–16, de sessões anteriores)
- **Modelo por paciente** (`11_modelo_final.py`, `15_treinar_modelo_final.py`): `HistGradientBoostingClassifier` (19 variáveis numéricas, 13 categóricas) que estima a probabilidade de falta de cada consulta.
- Treino expansivo anual (teste 2023–2026). V0 usa todas as linhas; V1 exclui CNES de nível C e o período anterior ao último regime nos CNES de nível B.
- Probabilidades recalibradas pelo nível dos últimos 28 dias da CNES (defasagem 3), e distribuição do total diário por simulação Poisson-binomial (300 amostras).
- Coorte: 14 CNES nível A (estáveis) + 6 nível B; 15.029 CNES-dias avaliados.
- Base: `gercon_poa_censo_distancias.parquet` (983.875 consultas). Nesta sessão foi apenas lida.

## 2. Comparação com modelos de série temporal (script 17)
Mesmo protocolo (CNES-dia, h = 3 dias, mesmas métricas). Modelos: baseline, naive7, dow4, ETS, SARIMA, GLM Poisson/NB por CNES, HGB Poisson com lags, híbrido (GLM + previsão do V1) e controle sem a covariável do paciente.

Resultado (A+B, 2023–2026, WAPE): baseline 0,383 · ETS 0,385 · SARIMA 0,380 · GLM 0,402 · HGB 0,376 · **V0 0,367 · V1 0,369**.
- ETS, SARIMA e GLM não superam o baseline (Diebold-Mariano p > 0,09). O HGB tem ganho pequeno e significativo (p = 0,001).
- V1 ganha ≈ 4% de WAPE sobre o baseline (p = 0,0002). No CRPS não difere do HGB.
- Híbrido (2024+): WAPE 0,376 contra 0,389 do controle; empata com V1 no erro absoluto e é melhor no CRPS.
- Nível B: ninguém se destaca; o GLM por CNES piora.
- Intervalos do V1 mais estreitos (cob80 ≈ 0,86); os de série ficam largos (≈ 0,90–0,91).

Arquivos: `17_comparacao_series_temporais.py`, `saida_17.txt`, `comparacao_series*.csv`, `previsoes_series.parquet`, `grafico_comparacao_series.py/.png`.

## 3. Previsão por especialidade (script 18)
Soma das probabilidades das consultas por CNES-dia-especialidade (mesmo fator de recalibração). 74.062 células.

| Método | Brier | AUC | AUC dentro do dia |
|---|---|---|---|
| V1 por paciente | 0,162 | 0,832 | 0,788 |
| Taxa 28d da especialidade | 0,173 | 0,808 | 0,751 |
| Total do dia × fatia da agenda | 0,171 | 0,809 | 0,740 |

- Calibração boa até 70%; superestima levemente nas faixas altas.
- hit@1 não discrimina (a maior agenda já acerta 82,8%); recall@3: 85% contra 82,7%.
- Ganho nulo ou negativo em OCI-oftalmologia, obstetrícia, urologia e neurologia.

Arquivos: `18_especialidade.py`, `saida_18.txt`, `previsao_especialidade.csv`, `especialidade_*.csv`.

## 4. CNES 2237571, odonto endodontia
- Total no período: 677 faltas observadas, 642 previstas (V1), 647 (baseline). A soma bate porque erros se compensam; o WAPE diário é 0,42.
- Por semana (74 semanas com agenda): WAPE 0,23 (base 0,27), correlação 0,94.
- Intervalos de previsão semanais exatos (Poisson-binomial): cobertura IC80 = 79,7%, IC95 = 95,9%. Em 2024–26 a cobertura foi menor (75% e 92%).
- **Agenda:** desde 2024 abre cerca de 1 vez por mês, em bloco de 4–5 dias úteis (antes era quase semanal). Mediana de 118 dias entre agendamento e consulta; 52,5% de primeira vez.

Arquivos: `grafico_especialidade_2237571.*`, `especialidade_2237571.csv`, `grafico_endodontia_semanal*.png/.py`, `endodontia_2237571_semanal*.csv`.

## 5. Overbooking na endodontia (CNES 2237571)
Regra: extras = quantil α da distribuição prevista de faltas. Excesso = faltas reais < extras chamados.
- **Por dia, quantil 10%:** ≈ 96% dos dias sem excesso, aproveita 39% das faltas. Em 2026: 12 extras (1 dia com excesso).
- **Curva diária (V1 vs baseline):** com ≤ 5% de excesso, aproveitamento de 43% contra 34%; com ≤ 20%, 72% contra 65%. Um m fixo não funciona.
- **Por bloco mensal (39 blocos):** quantil 10% aproveita 69% das faltas com 90% dos blocos sem excesso; em 2026 seriam 24 extras. O V1 empata com o baseline no nível do bloco, e o IC80 por bloco cobre só 69% (consultas do mesmo mês não faltam de forma independente). Medido no dia, o excesso sobe para 16% dos dias.
- **Combinação bloco + dia (teto mensal + regra diária): não melhora.** A regra diária sozinha está na fronteira. Recomendação: regra diária, escolhendo o quantil pelo excesso tolerado (q10% ≈ 4% de dias com excesso e 39% de aproveitamento; q30% ≈ 13% e 61%).
- Custo relativo da vaga vazia contra o paciente em excesso não está disponível por CNES, então o ponto de operação fica a critério da operação.

Arquivos: `overbooking_endodontia.py`, `curva_overbooking_endodontia.*`, `overbooking_2026.py`, `overbooking_bloco_endodontia.py`, `curva_overbooking_bloco_endodontia.png`, `overbooking_combinado_endodontia.py`, `curva_overbooking_combinado_endodontia.png`.

## 6. Ressalvas gerais
- O modelo de série e o por paciente usam probabilidades/distribuições diferentes (Poisson, NB, Poisson-binomial); os intervalos mistos não são comparáveis 1 a 1.
- Os intervalos assumem independência entre consultas; eventos coletivos (greve, chuva) derrubam a cobertura.
- Amostras pequenas na análise da endodontia (278 dias, 74 semanas, 39 blocos, uma CNES e uma especialidade).
- A seleção e a avaliação usam a mesma coorte A+B.
- O overbooking aqui é só contagem: não considera horário, quem é chamado, nem a mudança de comportamento causada pela própria intervenção.
- Ferramenta: `statsmodels` foi instalado no virtualenv `skill_academico/.venv`.

## 7. Próximos passos em aberto
1. Teste Diebold-Mariano direto `hib_glm` contra `ctrl_glm` e um HGB híbrido.
2. Teto de bloco que incorpore as faltas já ocorridas no mês (ajustar o risco dos dias seguintes).
3. Repetir a análise de overbooking para outras especialidades e CNES.
