# Seleção de variáveis do HGB por paciente — recorte municipal

## Método

- **Dados e protocolo:** somente consultas anteriores a 2023 (o teste 2023–2026 não foi tocado). Dois folds temporais: treino < 2021 → validação 2021; treino < 2022 → validação 2022. Treino com a base inteira; validação nas consultas das 10 CNES da coorte municipal (`niveis_municipal.csv`).
- **Modelo de referência:** `HistGradientBoostingClassifier`, 200 iterações, learning_rate 0,1, 31 folhas, min_samples_leaf 200, l2 1,0, sem early stopping.
- **Métrica:** Brier por consulta (menor é melhor); AUC como apoio. delta = Brier(sem a variável) − Brier(com); delta > 0 significa que a variável ajuda.
- **Passo 1, permutação:** embaralha a variável (ou o grupo) na validação, 5 repetições, e mede o aumento do Brier e a queda do AUC.
- **Passo 2, ablação por grupo:** re-treina sem o grupo inteiro, o que evita subestimar variáveis correlacionadas.
- **Passo 3, eliminação retroativa:** a cada rodada re-treina sem cada variável; remove a de menor delta se o IC 95% (bootstrap por CNES-mês, 500 réplicas) incluir 0 ou valores negativos; repete até todas as restantes serem significativas.
- **Passo 4, candidatas novas** sobre o conjunto final (adotadas só se o ganho tiver IC 95% inferior > 0).

## Resultado

- Modelo completo (32 variáveis): Brier 0.16222, AUC 0.6926.
- Conjunto final (6 variáveis): Brier 0.16243, AUC 0.6917.
- **Removidas (26):** mes_consulta, ultima_previa_mesma_especialidademae, n_faltas_previas_especialidademae, primeira_vez, quant_bus, status, n_consultas_previas_motivo, possuiunidadeindicada, n_faltas_previas_motivo, coordenadoria_saude, n_consultas_previas_especialidademae, distancia_km, tempo_caminhando, renda_media_winsor, ultima_previa_mesmo_motivo, n_consultas_previas, tempo_desloc_trans_publico, favela_comunidade, baldiacoes, dia_semana, falta_recente, distrito_sanitario, cid_1dig, n_faltas_previas, corregulador, taxa_alf_geral
- **Mantidas (6):** idade, taxa_falta_prev, dias_ate_consulta, distancia_km_trans_publico, especialidademae, cnesexecutante

## Importância por permutação

| tipo | nome | d_brier | d_brier_dp | d_auc |
|---|---|---|---|---|
| grupo | servico | 0.01743 | 0.0002 | 0.12514 |
| grupo | perfil | 0.00465 | 0.0001 | 0.03326 |
| grupo | agenda | 0.0031 | 0.00016 | 0.01966 |
| grupo | historico_paciente | 0.00164 | 0.00013 | 0.00967 |
| grupo | acesso | 0.00102 | 0.0001 | 0.00723 |
| grupo | territorio | 0.00088 | 2e-05 | 0.00521 |
| grupo | administrativas | 4e-05 | 2e-05 | 0.00016 |
| variavel | cnesexecutante | 0.00958 | 0.00022 | 0.07248 |
| variavel | especialidademae | 0.0071 | 0.00016 | 0.04877 |
| variavel | idade | 0.00465 | 0.0001 | 0.03326 |
| variavel | dias_ate_consulta | 0.00306 | 0.00015 | 0.01945 |
| variavel | taxa_falta_prev | 0.00099 | 0.0001 | 0.00637 |
| variavel | tempo_desloc_trans_publico | 0.00052 | 7e-05 | 0.00401 |
| variavel | cid_1dig | 0.00037 | 4e-05 | 0.00264 |
| variavel | taxa_alf_geral | 0.00028 | 2e-05 | 0.00168 |
| variavel | distrito_sanitario | 0.00023 | 5e-05 | 0.00151 |
| variavel | n_faltas_previas | 0.00016 | 2e-05 | 0.00092 |
| variavel | n_consultas_previas | 0.00014 | 3e-05 | 0.00084 |
| variavel | dia_semana | 0.00013 | 3e-05 | 0.00079 |
| variavel | corregulador | 0.00012 | 2e-05 | 0.00062 |
| variavel | favela_comunidade | 0.00011 | 3e-05 | 0.00058 |
| variavel | falta_recente | 8e-05 | 3e-05 | 0.00023 |
| variavel | renda_media_winsor | 6e-05 | 2e-05 | 0.00069 |
| variavel | possuiunidadeindicada | 3e-05 | 2e-05 | 0.00012 |
| variavel | status | 1e-05 | 0.0 | 4e-05 |
| variavel | distancia_km_trans_publico | 1e-05 | 3e-05 | 0.00024 |
| variavel | n_consultas_previas_especialidademae | 1e-05 | 0.0 | 3e-05 |
| variavel | ultima_previa_mesmo_motivo | 1e-05 | 0.0 | 4e-05 |
| variavel | n_faltas_previas_especialidademae | 0.0 | 1e-05 | 2e-05 |
| variavel | primeira_vez | 0.0 | 0.0 | 0.0 |
| variavel | n_consultas_previas_motivo | 0.0 | 0.0 | 0.0 |
| variavel | n_faltas_previas_motivo | 0.0 | 0.0 | 0.0 |
| variavel | baldiacoes | 0.0 | 0.0 | 0.0 |
| variavel | coordenadoria_saude | -0.0 | 0.0 | -0.0 |
| variavel | quant_bus | -1e-05 | 1e-05 | -2e-05 |
| variavel | distancia_km | -1e-05 | 1e-05 | -1e-05 |
| variavel | ultima_previa_mesma_especialidademae | -2e-05 | 1e-05 | -0.00017 |
| variavel | tempo_caminhando | -5e-05 | 1e-05 | -0.0002 |
| variavel | mes_consulta | -6e-05 | 6e-05 | -0.00066 |

## Ablação por grupo

| grupo | n_vars | d_brier | ic_lo | ic_hi | d_auc |
|---|---|---|---|---|---|
| servico | 3 | 0.0078 | 0.00663 | 0.00882 | 0.06147 |
| perfil | 1 | 0.00117 | 0.0008 | 0.00154 | 0.00683 |
| agenda | 3 | 0.00091 | 0.00048 | 0.00135 | 0.00508 |
| historico_paciente | 11 | 0.00072 | 0.00043 | 0.00104 | 0.0054 |
| territorio | 6 | 0.00026 | -3e-05 | 0.00054 | 0.00187 |
| administrativas | 2 | -0.0001 | -0.00031 | 0.00013 | -0.00026 |
| acesso | 6 | -0.0001 | -0.00036 | 0.00012 | -0.00023 |

## Eliminação retroativa

| rodada | variavel | d_brier | ic_lo | ic_hi | removida |
|---|---|---|---|---|---|
| 1 | mes_consulta | -0.00021 | -0.000493 | 4e-05 | True |
| 1 | distrito_sanitario | -0.000207 | -0.000452 | 1.9e-05 | False |
| 1 | favela_comunidade | -0.000191 | -0.000406 | 2.2e-05 | False |
| 1 | distancia_km_trans_publico | -0.000169 | -0.000384 | 2.4e-05 | False |
| 1 | dia_semana | -0.000164 | -0.00037 | 5e-05 | False |
| 1 | ultima_previa_mesma_especialidademae | -0.000152 | -0.000334 | 4.3e-05 | False |
| 1 | taxa_alf_geral | -0.000134 | -0.000337 | 5.4e-05 | False |
| 1 | n_consultas_previas_especialidademae | -0.000131 | -0.000283 | 1.4e-05 | False |
| 1 | n_faltas_previas | -0.000113 | -0.000341 | 8e-05 | False |
| 1 | tempo_caminhando | -9.6e-05 | -0.000308 | 0.000124 | False |
| 1 | cid_1dig | -6.5e-05 | -0.000292 | 0.000163 | False |
| 1 | n_consultas_previas | -6.1e-05 | -0.000282 | 0.00016 | False |
| 1 | taxa_falta_prev | -5.5e-05 | -0.000266 | 0.000154 | False |
| 1 | possuiunidadeindicada | -5e-05 | -0.000265 | 0.00016 | False |
| 1 | renda_media_winsor | -4.9e-05 | -0.000249 | 0.000148 | False |
| 1 | n_faltas_previas_especialidademae | -4.7e-05 | -0.000155 | 5.6e-05 | False |
| 1 | distancia_km | -3.7e-05 | -0.000222 | 0.000144 | False |
| 1 | falta_recente | -2.9e-05 | -0.000256 | 0.000165 | False |
| 1 | tempo_desloc_trans_publico | -2.4e-05 | -0.000255 | 0.000229 | False |
| 1 | primeira_vez | 0.0 | 0.0 | 0.0 | False |
| 1 | n_consultas_previas_motivo | 0.0 | 0.0 | 0.0 | False |
| 1 | n_faltas_previas_motivo | 0.0 | 0.0 | 0.0 | False |
| 1 | quant_bus | 0.0 | 0.0 | 0.0 | False |
| 1 | baldiacoes | 0.0 | 0.0 | 0.0 | False |
| 1 | corregulador | 1e-06 | -0.000242 | 0.000233 | False |
| 1 | coordenadoria_saude | 2.9e-05 | -2.3e-05 | 7.3e-05 | False |
| 1 | status | 3.1e-05 | -7.5e-05 | 0.000132 | False |
| 1 | ultima_previa_mesmo_motivo | 5.1e-05 | -7.1e-05 | 0.000176 | False |
| 1 | dias_ate_consulta | 0.001084 | 0.000718 | 0.001514 | False |
| 1 | idade | 0.001168 | 0.000834 | 0.001539 | False |
| 1 | cnesexecutante | 0.001615 | 0.001028 | 0.002183 | False |
| 1 | especialidademae | 0.001803 | 0.00125 | 0.002473 | False |
| 2 | ultima_previa_mesma_especialidademae | -0.000302 | -0.000476 | -0.000126 | True |
| 2 | distancia_km_trans_publico | -0.00028 | -0.000476 | -0.000103 | False |
| 2 | n_faltas_previas | -0.000262 | -0.000438 | -6.3e-05 | False |
| 2 | distrito_sanitario | -0.000197 | -0.000395 | 5e-06 | False |
| 2 | n_consultas_previas | -0.00016 | -0.000349 | 9e-06 | False |
| 2 | n_consultas_previas_motivo | -0.000139 | -0.00025 | -2.9e-05 | False |
| 2 | n_consultas_previas_especialidademae | -0.000133 | -0.000302 | 1.7e-05 | False |
| 2 | coordenadoria_saude | -0.000132 | -0.000251 | -2.3e-05 | False |
| 2 | ultima_previa_mesmo_motivo | -0.000117 | -0.000252 | 3.2e-05 | False |
| 2 | possuiunidadeindicada | -0.000102 | -0.000292 | 0.000105 | False |
| 2 | distancia_km | -9.7e-05 | -0.000277 | 0.000108 | False |
| 2 | falta_recente | -9.4e-05 | -0.000309 | 0.000105 | False |
| 2 | status | -8.4e-05 | -0.000193 | 2.5e-05 | False |
| 2 | tempo_desloc_trans_publico | -8.3e-05 | -0.000288 | 0.000112 | False |
| 2 | taxa_falta_prev | -4.4e-05 | -0.000249 | 0.000154 | False |
| 2 | n_faltas_previas_especialidademae | -3.6e-05 | -0.000138 | 7e-05 | False |
| 2 | tempo_caminhando | -2e-05 | -0.000216 | 0.000171 | False |
| 2 | taxa_alf_geral | -9e-06 | -0.000224 | 0.0002 | False |
| 2 | favela_comunidade | -9e-06 | -0.000199 | 0.000181 | False |
| 2 | primeira_vez | 0.0 | 0.0 | 0.0 | False |
| 2 | n_faltas_previas_motivo | 0.0 | 0.0 | 0.0 | False |
| 2 | quant_bus | 0.0 | 0.0 | 0.0 | False |
| 2 | baldiacoes | 0.0 | 0.0 | 0.0 | False |
| 2 | renda_media_winsor | 9e-06 | -0.000202 | 0.000229 | False |
| 2 | corregulador | 4.7e-05 | -0.000152 | 0.000255 | False |
| 2 | dia_semana | 5e-05 | -0.000176 | 0.000281 | False |
| 2 | cid_1dig | 6.8e-05 | -0.000179 | 0.000288 | False |
| 2 | idade | 0.001034 | 0.000653 | 0.001418 | False |
| 2 | dias_ate_consulta | 0.001046 | 0.000663 | 0.001472 | False |
| 2 | cnesexecutante | 0.001626 | 0.001043 | 0.002213 | False |
| 2 | especialidademae | 0.00198 | 0.001465 | 0.002563 | False |
| 3 | n_faltas_previas_especialidademae | -1.3e-05 | -0.000127 | 0.0001 | True |
| 3 | primeira_vez | 0.0 | 0.0 | 0.0 | False |
| 3 | n_faltas_previas_motivo | 0.0 | 0.0 | 0.0 | False |
| 3 | quant_bus | 0.0 | 0.0 | 0.0 | False |
| 3 | baldiacoes | 0.0 | 0.0 | 0.0 | False |
| 3 | status | 0.0 | 0.0 | 0.0 | False |
| 3 | possuiunidadeindicada | 2.6e-05 | -0.000208 | 0.000271 | False |
| 3 | coordenadoria_saude | 2.6e-05 | -6e-05 | 0.000119 | False |
| 3 | n_consultas_previas_especialidademae | 3.4e-05 | -0.000129 | 0.000208 | False |
| 3 | tempo_desloc_trans_publico | 3.6e-05 | -0.000156 | 0.000237 | False |
| 3 | n_consultas_previas_motivo | 4.5e-05 | -5.2e-05 | 0.000147 | False |
| 3 | ultima_previa_mesmo_motivo | 7.2e-05 | -4e-05 | 0.00018 | False |
| 3 | corregulador | 7.8e-05 | -0.000128 | 0.000295 | False |
| 3 | distancia_km | 0.000102 | -8.8e-05 | 0.000271 | False |
| 3 | n_consultas_previas | 0.000117 | -4.7e-05 | 0.000286 | False |
| 3 | taxa_alf_geral | 0.000129 | -5.1e-05 | 0.000326 | False |
| 3 | n_faltas_previas | 0.000132 | -6.3e-05 | 0.000329 | False |
| 3 | distrito_sanitario | 0.000192 | -1.5e-05 | 0.000429 | False |
| 3 | distancia_km_trans_publico | 0.000203 | -1e-06 | 0.000401 | False |
| 3 | favela_comunidade | 0.000203 | 3e-06 | 0.000427 | False |
| 3 | dia_semana | 0.000213 | 5.5e-05 | 0.000385 | False |
| 3 | renda_media_winsor | 0.000218 | 1.8e-05 | 0.000408 | False |
| 3 | falta_recente | 0.000226 | 1.6e-05 | 0.000443 | False |
| 3 | cid_1dig | 0.000228 | -1.7e-05 | 0.000463 | False |
| 3 | tempo_caminhando | 0.000237 | 6.6e-05 | 0.000419 | False |
| 3 | taxa_falta_prev | 0.000317 | 0.000109 | 0.000549 | False |
| 3 | idade | 0.001448 | 0.0011 | 0.001825 | False |
| 3 | dias_ate_consulta | 0.001476 | 0.001116 | 0.001875 | False |
| 3 | cnesexecutante | 0.001975 | 0.001405 | 0.002555 | False |
| 3 | especialidademae | 0.002002 | 0.001481 | 0.002591 | False |
| 4 | primeira_vez | 0.0 | 0.0 | 0.0 | True |
| 4 | quant_bus | 0.0 | 0.0 | 0.0 | False |
| 4 | baldiacoes | 0.0 | 0.0 | 0.0 | False |
| 4 | status | 0.0 | 0.0 | 0.0 | False |
| 4 | n_consultas_previas_motivo | 5e-06 | -9e-05 | 9.7e-05 | False |
| 4 | ultima_previa_mesmo_motivo | 1.1e-05 | -9.1e-05 | 0.000129 | False |
| 4 | n_faltas_previas_motivo | 1.6e-05 | -2.4e-05 | 6e-05 | False |
| 4 | possuiunidadeindicada | 6.5e-05 | -0.000153 | 0.00028 | False |
| 4 | n_faltas_previas | 7e-05 | -0.000117 | 0.000276 | False |
| 4 | coordenadoria_saude | 9e-05 | -3e-06 | 0.00017 | False |
| 4 | tempo_desloc_trans_publico | 9.2e-05 | -8.9e-05 | 0.000277 | False |
| 4 | falta_recente | 0.000117 | -7.9e-05 | 0.000321 | False |
| 4 | n_consultas_previas | 0.00015 | -3.9e-05 | 0.000322 | False |
| 4 | corregulador | 0.000152 | -4.8e-05 | 0.000366 | False |
| 4 | n_consultas_previas_especialidademae | 0.000182 | -5e-06 | 0.000385 | False |
| 4 | tempo_caminhando | 0.000187 | 1.4e-05 | 0.000341 | False |
| 4 | distancia_km | 0.0002 | 2.1e-05 | 0.000391 | False |
| 4 | taxa_alf_geral | 0.0002 | -2e-06 | 0.000398 | False |
| 4 | renda_media_winsor | 0.000219 | 2.7e-05 | 0.000411 | False |
| 4 | favela_comunidade | 0.000237 | 6e-05 | 0.000424 | False |
| 4 | distrito_sanitario | 0.000281 | 5.6e-05 | 0.000478 | False |
| 4 | cid_1dig | 0.00029 | 3.8e-05 | 0.000518 | False |
| 4 | dia_semana | 0.000297 | 0.000101 | 0.000503 | False |
| 4 | taxa_falta_prev | 0.000307 | 0.000116 | 0.000528 | False |
| 4 | distancia_km_trans_publico | 0.000321 | 0.000133 | 0.000516 | False |
| 4 | idade | 0.001387 | 0.001016 | 0.001783 | False |
| 4 | dias_ate_consulta | 0.001404 | 0.001062 | 0.001845 | False |
| 4 | cnesexecutante | 0.001897 | 0.001321 | 0.00243 | False |
| 4 | especialidademae | 0.001997 | 0.001494 | 0.002529 | False |
| 5 | quant_bus | 0.0 | 0.0 | 0.0 | True |
| 5 | baldiacoes | 0.0 | 0.0 | 0.0 | False |
| 5 | status | 0.0 | 0.0 | 0.0 | False |
| 5 | n_consultas_previas_motivo | 5e-06 | -9e-05 | 9.7e-05 | False |
| 5 | ultima_previa_mesmo_motivo | 1.1e-05 | -9.1e-05 | 0.000129 | False |
| 5 | n_faltas_previas_motivo | 1.6e-05 | -2.4e-05 | 6e-05 | False |
| 5 | possuiunidadeindicada | 6.5e-05 | -0.000153 | 0.00028 | False |
| 5 | n_faltas_previas | 7e-05 | -0.000117 | 0.000276 | False |
| 5 | coordenadoria_saude | 9e-05 | -3e-06 | 0.00017 | False |
| 5 | tempo_desloc_trans_publico | 9.2e-05 | -8.9e-05 | 0.000277 | False |
| 5 | falta_recente | 0.000117 | -7.9e-05 | 0.000321 | False |
| 5 | n_consultas_previas | 0.00015 | -3.9e-05 | 0.000322 | False |
| 5 | corregulador | 0.000152 | -4.8e-05 | 0.000366 | False |
| 5 | n_consultas_previas_especialidademae | 0.000182 | -5e-06 | 0.000385 | False |
| 5 | tempo_caminhando | 0.000187 | 1.4e-05 | 0.000341 | False |
| 5 | distancia_km | 0.0002 | 2.1e-05 | 0.000391 | False |
| 5 | taxa_alf_geral | 0.0002 | -2e-06 | 0.000398 | False |
| 5 | renda_media_winsor | 0.000219 | 2.7e-05 | 0.000411 | False |
| 5 | favela_comunidade | 0.000237 | 6e-05 | 0.000424 | False |
| 5 | distrito_sanitario | 0.000281 | 5.6e-05 | 0.000478 | False |
| 5 | cid_1dig | 0.00029 | 3.8e-05 | 0.000518 | False |
| 5 | dia_semana | 0.000297 | 0.000101 | 0.000503 | False |
| 5 | taxa_falta_prev | 0.000307 | 0.000116 | 0.000528 | False |
| 5 | distancia_km_trans_publico | 0.000321 | 0.000133 | 0.000516 | False |
| 5 | idade | 0.001387 | 0.001016 | 0.001783 | False |
| 5 | dias_ate_consulta | 0.001404 | 0.001062 | 0.001845 | False |
| 5 | cnesexecutante | 0.001897 | 0.001321 | 0.00243 | False |
| 5 | especialidademae | 0.001997 | 0.001494 | 0.002529 | False |
| 6 | status | 0.0 | 0.0 | 0.0 | True |
| 6 | n_consultas_previas_motivo | 5e-06 | -9e-05 | 9.7e-05 | False |
| 6 | ultima_previa_mesmo_motivo | 1.1e-05 | -9.1e-05 | 0.000129 | False |
| 6 | n_faltas_previas_motivo | 1.6e-05 | -2.4e-05 | 6e-05 | False |
| 6 | possuiunidadeindicada | 6.5e-05 | -0.000153 | 0.00028 | False |
| 6 | n_faltas_previas | 7e-05 | -0.000117 | 0.000276 | False |
| 6 | coordenadoria_saude | 9e-05 | -3e-06 | 0.00017 | False |
| 6 | tempo_desloc_trans_publico | 9.2e-05 | -8.9e-05 | 0.000277 | False |
| 6 | falta_recente | 0.000117 | -7.9e-05 | 0.000321 | False |
| 6 | n_consultas_previas | 0.00015 | -3.9e-05 | 0.000322 | False |
| 6 | corregulador | 0.000152 | -4.8e-05 | 0.000366 | False |
| 6 | n_consultas_previas_especialidademae | 0.000182 | -5e-06 | 0.000385 | False |
| 6 | tempo_caminhando | 0.000187 | 1.4e-05 | 0.000341 | False |
| 6 | distancia_km | 0.0002 | 2.1e-05 | 0.000391 | False |
| 6 | taxa_alf_geral | 0.0002 | -2e-06 | 0.000398 | False |
| 6 | renda_media_winsor | 0.000219 | 2.7e-05 | 0.000411 | False |
| 6 | favela_comunidade | 0.000237 | 6e-05 | 0.000424 | False |
| 6 | baldiacoes | 0.000244 | 6.3e-05 | 0.00042 | False |
| 6 | distrito_sanitario | 0.000281 | 5.6e-05 | 0.000478 | False |
| 6 | cid_1dig | 0.00029 | 3.8e-05 | 0.000518 | False |
| 6 | dia_semana | 0.000297 | 0.000101 | 0.000503 | False |
| 6 | taxa_falta_prev | 0.000307 | 0.000116 | 0.000528 | False |
| 6 | distancia_km_trans_publico | 0.000321 | 0.000133 | 0.000516 | False |
| 6 | idade | 0.001387 | 0.001016 | 0.001783 | False |
| 6 | dias_ate_consulta | 0.001404 | 0.001062 | 0.001845 | False |
| 6 | cnesexecutante | 0.001897 | 0.001321 | 0.00243 | False |
| 6 | especialidademae | 0.001997 | 0.001494 | 0.002529 | False |
| 7 | n_consultas_previas_motivo | 5e-06 | -9e-05 | 9.7e-05 | True |
| 7 | tempo_desloc_trans_publico | 8e-06 | -0.000184 | 0.000194 | False |
| 7 | n_faltas_previas_motivo | 1.6e-05 | -2.4e-05 | 6e-05 | False |
| 7 | possuiunidadeindicada | 6.5e-05 | -0.000153 | 0.00028 | False |
| 7 | n_faltas_previas | 7e-05 | -0.000117 | 0.000276 | False |
| 7 | ultima_previa_mesmo_motivo | 7.6e-05 | -2.9e-05 | 0.000204 | False |
| 7 | coordenadoria_saude | 0.000105 | 1e-06 | 0.000202 | False |
| 7 | renda_media_winsor | 0.00013 | -7.7e-05 | 0.000351 | False |
| 7 | taxa_alf_geral | 0.000137 | -4.4e-05 | 0.000343 | False |
| 7 | n_consultas_previas | 0.00015 | -3.9e-05 | 0.000322 | False |
| 7 | corregulador | 0.000152 | -4.8e-05 | 0.000366 | False |
| 7 | distancia_km | 0.0002 | 2.1e-05 | 0.000391 | False |
| 7 | favela_comunidade | 0.000237 | 6e-05 | 0.000424 | False |
| 7 | baldiacoes | 0.000244 | 6.3e-05 | 0.00042 | False |
| 7 | dia_semana | 0.00026 | 7.2e-05 | 0.000458 | False |
| 7 | n_consultas_previas_especialidademae | 0.000262 | 7e-05 | 0.000467 | False |
| 7 | cid_1dig | 0.000263 | 1.9e-05 | 0.000497 | False |
| 7 | tempo_caminhando | 0.000273 | 9.2e-05 | 0.000443 | False |
| 7 | falta_recente | 0.000273 | 7.6e-05 | 0.000462 | False |
| 7 | distrito_sanitario | 0.00028 | 6.2e-05 | 0.000478 | False |
| 7 | taxa_falta_prev | 0.000307 | 0.000116 | 0.000528 | False |
| 7 | distancia_km_trans_publico | 0.000321 | 0.000133 | 0.000516 | False |
| 7 | idade | 0.001388 | 0.001035 | 0.001787 | False |
| 7 | dias_ate_consulta | 0.001407 | 0.001021 | 0.001829 | False |
| 7 | cnesexecutante | 0.001919 | 0.001366 | 0.002458 | False |
| 7 | especialidademae | 0.002075 | 0.00156 | 0.002627 | False |
| 8 | possuiunidadeindicada | -5.2e-05 | -0.000296 | 0.000175 | True |
| 8 | ultima_previa_mesmo_motivo | 1.7e-05 | -0.000148 | 0.000186 | False |
| 8 | tempo_desloc_trans_publico | 2.9e-05 | -0.000165 | 0.000213 | False |
| 8 | n_faltas_previas_motivo | 3.2e-05 | -3.7e-05 | 9.6e-05 | False |
| 8 | n_faltas_previas | 6.5e-05 | -0.000125 | 0.000262 | False |
| 8 | coordenadoria_saude | 0.0001 | -4e-06 | 0.000201 | False |
| 8 | taxa_alf_geral | 0.000116 | -8.1e-05 | 0.000332 | False |
| 8 | renda_media_winsor | 0.000125 | -0.0001 | 0.000355 | False |
| 8 | n_consultas_previas | 0.000145 | -3.6e-05 | 0.000318 | False |
| 8 | corregulador | 0.000147 | -4.1e-05 | 0.000335 | False |
| 8 | distancia_km_trans_publico | 0.000222 | 1.7e-05 | 0.000448 | False |
| 8 | cid_1dig | 0.000223 | -1.7e-05 | 0.000465 | False |
| 8 | favela_comunidade | 0.000232 | 3.5e-05 | 0.000428 | False |
| 8 | distancia_km | 0.000234 | 4.5e-05 | 0.000425 | False |
| 8 | n_consultas_previas_especialidademae | 0.000257 | 5.9e-05 | 0.000469 | False |
| 8 | tempo_caminhando | 0.000264 | 7.9e-05 | 0.000462 | False |
| 8 | falta_recente | 0.000268 | 7.4e-05 | 0.000472 | False |
| 8 | dia_semana | 0.000273 | 6.3e-05 | 0.000471 | False |
| 8 | distrito_sanitario | 0.000274 | 5e-05 | 0.000487 | False |
| 8 | baldiacoes | 0.00029 | 8.6e-05 | 0.000497 | False |
| 8 | taxa_falta_prev | 0.000302 | 0.000102 | 0.000518 | False |
| 8 | idade | 0.001381 | 0.001019 | 0.001778 | False |
| 8 | dias_ate_consulta | 0.001517 | 0.001149 | 0.001954 | False |
| 8 | cnesexecutante | 0.001924 | 0.001382 | 0.002483 | False |
| 8 | especialidademae | 0.00202 | 0.001523 | 0.002573 | False |
| 9 | n_faltas_previas_motivo | 0.0 | 0.0 | 0.0 | True |
| 9 | coordenadoria_saude | 5.9e-05 | -1.1e-05 | 0.000134 | False |
| 9 | n_consultas_previas_especialidademae | 0.000106 | -7.5e-05 | 0.000304 | False |
| 9 | baldiacoes | 0.000139 | -8.2e-05 | 0.000348 | False |
| 9 | tempo_desloc_trans_publico | 0.000149 | -7.2e-05 | 0.000342 | False |
| 9 | tempo_caminhando | 0.000152 | -3.2e-05 | 0.000333 | False |
| 9 | ultima_previa_mesmo_motivo | 0.000154 | 3.4e-05 | 0.00028 | False |
| 9 | distancia_km | 0.00019 | -2.2e-05 | 0.000423 | False |
| 9 | falta_recente | 0.000209 | -2.5e-05 | 0.000433 | False |
| 9 | corregulador | 0.000213 | -1e-06 | 0.000407 | False |
| 9 | favela_comunidade | 0.000228 | -1.3e-05 | 0.000472 | False |
| 9 | distrito_sanitario | 0.000232 | 4e-06 | 0.00046 | False |
| 9 | n_faltas_previas | 0.000242 | 6e-06 | 0.000499 | False |
| 9 | n_consultas_previas | 0.000256 | 3.2e-05 | 0.000492 | False |
| 9 | dia_semana | 0.000295 | 7.8e-05 | 0.000517 | False |
| 9 | renda_media_winsor | 0.000299 | 8.9e-05 | 0.000511 | False |
| 9 | distancia_km_trans_publico | 0.000309 | 0.000107 | 0.000525 | False |
| 9 | taxa_alf_geral | 0.000336 | 0.000113 | 0.000579 | False |
| 9 | cid_1dig | 0.000413 | 0.000151 | 0.00066 | False |
| 9 | taxa_falta_prev | 0.000452 | 0.000215 | 0.000679 | False |
| 9 | dias_ate_consulta | 0.001562 | 0.001148 | 0.001977 | False |
| 9 | idade | 0.001666 | 0.001246 | 0.002101 | False |
| 9 | cnesexecutante | 0.002126 | 0.001465 | 0.002745 | False |
| 9 | especialidademae | 0.002336 | 0.001774 | 0.002947 | False |
| 10 | coordenadoria_saude | 5.9e-05 | -1.1e-05 | 0.000134 | True |
| 10 | n_consultas_previas_especialidademae | 0.00012 | -7.1e-05 | 0.000316 | False |
| 10 | baldiacoes | 0.000123 | -9.5e-05 | 0.000332 | False |
| 10 | tempo_desloc_trans_publico | 0.000149 | -7.2e-05 | 0.000342 | False |
| 10 | tempo_caminhando | 0.000152 | -3.2e-05 | 0.000333 | False |
| 10 | distancia_km | 0.000174 | -2.7e-05 | 0.000402 | False |
| 10 | ultima_previa_mesmo_motivo | 0.00018 | 6.1e-05 | 0.000295 | False |
| 10 | corregulador | 0.000199 | -9e-06 | 0.0004 | False |
| 10 | favela_comunidade | 0.000228 | -1.3e-05 | 0.000472 | False |
| 10 | falta_recente | 0.000238 | 2e-05 | 0.000461 | False |
| 10 | n_faltas_previas | 0.00025 | 2.7e-05 | 0.00047 | False |
| 10 | distrito_sanitario | 0.000252 | 2.1e-05 | 0.000482 | False |
| 10 | n_consultas_previas | 0.000254 | 4.4e-05 | 0.000484 | False |
| 10 | renda_media_winsor | 0.000277 | 6.1e-05 | 0.000485 | False |
| 10 | dia_semana | 0.000285 | 4.3e-05 | 0.000513 | False |
| 10 | taxa_alf_geral | 0.000344 | 0.000126 | 0.000588 | False |
| 10 | distancia_km_trans_publico | 0.000353 | 0.00015 | 0.000574 | False |
| 10 | cid_1dig | 0.000436 | 0.000169 | 0.000682 | False |
| 10 | taxa_falta_prev | 0.000463 | 0.000227 | 0.000698 | False |
| 10 | dias_ate_consulta | 0.001567 | 0.001147 | 0.002032 | False |
| 10 | idade | 0.001678 | 0.001255 | 0.002115 | False |
| 10 | cnesexecutante | 0.002076 | 0.001441 | 0.002695 | False |
| 10 | especialidademae | 0.002245 | 0.001695 | 0.002838 | False |
| 11 | n_consultas_previas_especialidademae | 6.1e-05 | -0.000114 | 0.000239 | True |
| 11 | ultima_previa_mesmo_motivo | 7.8e-05 | -4.5e-05 | 0.0002 | False |
| 11 | tempo_desloc_trans_publico | 9e-05 | -0.000115 | 0.000276 | False |
| 11 | baldiacoes | 0.000115 | -8.8e-05 | 0.000301 | False |
| 11 | distancia_km | 0.000115 | -6.7e-05 | 0.000324 | False |
| 11 | tempo_caminhando | 0.000118 | -4.5e-05 | 0.000288 | False |
| 11 | corregulador | 0.000138 | -7.5e-05 | 0.000335 | False |
| 11 | falta_recente | 0.000161 | -6.2e-05 | 0.00039 | False |
| 11 | favela_comunidade | 0.000169 | -7e-05 | 0.000408 | False |
| 11 | n_faltas_previas | 0.000191 | -1.9e-05 | 0.000367 | False |
| 11 | n_consultas_previas | 0.000195 | -7e-06 | 0.000419 | False |
| 11 | renda_media_winsor | 0.000218 | 1.9e-05 | 0.000412 | False |
| 11 | distancia_km_trans_publico | 0.000218 | 3.3e-05 | 0.000414 | False |
| 11 | distrito_sanitario | 0.000237 | -1e-06 | 0.000476 | False |
| 11 | dia_semana | 0.000248 | 2e-05 | 0.000491 | False |
| 11 | taxa_falta_prev | 0.000346 | 0.000119 | 0.000563 | False |
| 11 | taxa_alf_geral | 0.000363 | 0.000176 | 0.000583 | False |
| 11 | cid_1dig | 0.000418 | 0.000154 | 0.000657 | False |
| 11 | dias_ate_consulta | 0.001457 | 0.001043 | 0.001886 | False |
| 11 | idade | 0.001619 | 0.001199 | 0.002026 | False |
| 11 | cnesexecutante | 0.002004 | 0.001384 | 0.002603 | False |
| 11 | especialidademae | 0.002186 | 0.001631 | 0.002755 | False |
| 12 | distancia_km | -8.4e-05 | -0.00028 | 0.000109 | True |
| 12 | n_consultas_previas | -2.9e-05 | -0.000204 | 0.000151 | False |
| 12 | tempo_caminhando | 3.2e-05 | -0.000135 | 0.000202 | False |
| 12 | tempo_desloc_trans_publico | 4.3e-05 | -0.000163 | 0.000251 | False |
| 12 | distancia_km_trans_publico | 6e-05 | -0.000155 | 0.000277 | False |
| 12 | falta_recente | 0.000122 | -6.9e-05 | 0.000333 | False |
| 12 | distrito_sanitario | 0.000135 | -9e-05 | 0.000386 | False |
| 12 | favela_comunidade | 0.000142 | -4.8e-05 | 0.000327 | False |
| 12 | n_faltas_previas | 0.000162 | -2e-05 | 0.000359 | False |
| 12 | baldiacoes | 0.000204 | 3.9e-05 | 0.000399 | False |
| 12 | dia_semana | 0.000209 | 4e-05 | 0.000386 | False |
| 12 | renda_media_winsor | 0.000219 | 2.3e-05 | 0.000412 | False |
| 12 | corregulador | 0.00022 | 3.2e-05 | 0.000428 | False |
| 12 | ultima_previa_mesmo_motivo | 0.000227 | 9e-05 | 0.00037 | False |
| 12 | cid_1dig | 0.000334 | 8.4e-05 | 0.000571 | False |
| 12 | taxa_alf_geral | 0.000338 | 0.000148 | 0.000529 | False |
| 12 | taxa_falta_prev | 0.000355 | 0.000157 | 0.000543 | False |
| 12 | idade | 0.001347 | 0.000999 | 0.001711 | False |
| 12 | dias_ate_consulta | 0.001379 | 0.001014 | 0.001775 | False |
| 12 | cnesexecutante | 0.001983 | 0.001409 | 0.002616 | False |
| 12 | especialidademae | 0.002239 | 0.001676 | 0.00281 | False |
| 13 | tempo_caminhando | 4.6e-05 | -0.00011 | 0.000226 | True |
| 13 | tempo_desloc_trans_publico | 4.9e-05 | -0.000144 | 0.000239 | False |
| 13 | n_faltas_previas | 0.000117 | -8.6e-05 | 0.000344 | False |
| 13 | ultima_previa_mesmo_motivo | 0.000119 | -3.4e-05 | 0.000278 | False |
| 13 | favela_comunidade | 0.000126 | -5.3e-05 | 0.000311 | False |
| 13 | falta_recente | 0.000137 | -4e-05 | 0.000305 | False |
| 13 | distrito_sanitario | 0.000167 | -5e-05 | 0.000358 | False |
| 13 | n_consultas_previas | 0.000216 | 2.6e-05 | 0.000404 | False |
| 13 | baldiacoes | 0.000216 | 5.2e-05 | 0.000397 | False |
| 13 | distancia_km_trans_publico | 0.00023 | 4e-05 | 0.000409 | False |
| 13 | dia_semana | 0.000257 | 8.6e-05 | 0.000443 | False |
| 13 | taxa_alf_geral | 0.00028 | 0.000117 | 0.00046 | False |
| 13 | corregulador | 0.000287 | 0.000106 | 0.000486 | False |
| 13 | renda_media_winsor | 0.000298 | 0.000114 | 0.000508 | False |
| 13 | taxa_falta_prev | 0.000366 | 0.000169 | 0.000563 | False |
| 13 | cid_1dig | 0.000439 | 0.000209 | 0.000627 | False |
| 13 | dias_ate_consulta | 0.001524 | 0.001118 | 0.001943 | False |
| 13 | idade | 0.001547 | 0.001219 | 0.001922 | False |
| 13 | cnesexecutante | 0.00207 | 0.001492 | 0.002652 | False |
| 13 | especialidademae | 0.002166 | 0.00165 | 0.002758 | False |
| 14 | renda_media_winsor | 5.7e-05 | -0.000155 | 0.000256 | True |
| 14 | distancia_km_trans_publico | 0.000123 | -6.9e-05 | 0.000292 | False |
| 14 | baldiacoes | 0.000133 | -6.1e-05 | 0.000302 | False |
| 14 | n_faltas_previas | 0.000158 | -4.9e-05 | 0.000372 | False |
| 14 | distrito_sanitario | 0.000159 | -9.7e-05 | 0.000388 | False |
| 14 | tempo_desloc_trans_publico | 0.000168 | -2.4e-05 | 0.000338 | False |
| 14 | favela_comunidade | 0.000209 | -8e-06 | 0.000414 | False |
| 14 | ultima_previa_mesmo_motivo | 0.000224 | 9.6e-05 | 0.000337 | False |
| 14 | n_consultas_previas | 0.000242 | 2.3e-05 | 0.000459 | False |
| 14 | corregulador | 0.000262 | 4.2e-05 | 0.000484 | False |
| 14 | taxa_alf_geral | 0.000272 | 5.4e-05 | 0.000475 | False |
| 14 | dia_semana | 0.000277 | 9.8e-05 | 0.000468 | False |
| 14 | falta_recente | 0.000285 | 6.7e-05 | 0.000478 | False |
| 14 | taxa_falta_prev | 0.000317 | 8.4e-05 | 0.00052 | False |
| 14 | cid_1dig | 0.000439 | 0.000189 | 0.000653 | False |
| 14 | dias_ate_consulta | 0.001419 | 0.001005 | 0.001818 | False |
| 14 | idade | 0.001536 | 0.001169 | 0.001932 | False |
| 14 | cnesexecutante | 0.00206 | 0.00147 | 0.002673 | False |
| 14 | especialidademae | 0.002136 | 0.001626 | 0.002698 | False |
| 15 | ultima_previa_mesmo_motivo | -4.4e-05 | -0.000178 | 9.7e-05 | True |
| 15 | tempo_desloc_trans_publico | 1e-05 | -0.000174 | 0.000208 | False |
| 15 | favela_comunidade | 9.8e-05 | -5.1e-05 | 0.000283 | False |
| 15 | n_faltas_previas | 0.000106 | -7.5e-05 | 0.000297 | False |
| 15 | n_consultas_previas | 0.000137 | -4.9e-05 | 0.000324 | False |
| 15 | taxa_alf_geral | 0.000162 | -3.4e-05 | 0.000362 | False |
| 15 | corregulador | 0.000163 | -4.7e-05 | 0.000366 | False |
| 15 | baldiacoes | 0.000191 | -2e-06 | 0.000397 | False |
| 15 | distancia_km_trans_publico | 0.000198 | 1.6e-05 | 0.000388 | False |
| 15 | cid_1dig | 0.000202 | -2e-05 | 0.000418 | False |
| 15 | distrito_sanitario | 0.000209 | 1.2e-05 | 0.000429 | False |
| 15 | dia_semana | 0.000237 | 4.3e-05 | 0.000453 | False |
| 15 | taxa_falta_prev | 0.000256 | 6.7e-05 | 0.000414 | False |
| 15 | falta_recente | 0.000263 | 5.4e-05 | 0.000457 | False |
| 15 | dias_ate_consulta | 0.001393 | 0.001037 | 0.001791 | False |
| 15 | idade | 0.001472 | 0.001148 | 0.001871 | False |
| 15 | cnesexecutante | 0.001984 | 0.001396 | 0.002588 | False |
| 15 | especialidademae | 0.002233 | 0.001718 | 0.002793 | False |
| 16 | n_consultas_previas | 7.3e-05 | -0.000102 | 0.00026 | True |
| 16 | tempo_desloc_trans_publico | 8.6e-05 | -0.000111 | 0.000294 | False |
| 16 | corregulador | 9.3e-05 | -9.8e-05 | 0.000276 | False |
| 16 | favela_comunidade | 0.000133 | -5.4e-05 | 0.000326 | False |
| 16 | distrito_sanitario | 0.000192 | -2.8e-05 | 0.000417 | False |
| 16 | falta_recente | 0.000203 | 1.3e-05 | 0.000414 | False |
| 16 | n_faltas_previas | 0.000205 | 3.4e-05 | 0.000381 | False |
| 16 | baldiacoes | 0.000224 | 2.2e-05 | 0.000431 | False |
| 16 | cid_1dig | 0.000257 | 3.9e-05 | 0.000465 | False |
| 16 | distancia_km_trans_publico | 0.00026 | 7.2e-05 | 0.000448 | False |
| 16 | taxa_falta_prev | 0.000305 | 0.000121 | 0.000476 | False |
| 16 | dia_semana | 0.000364 | 0.000189 | 0.000565 | False |
| 16 | taxa_alf_geral | 0.000437 | 0.000229 | 0.000645 | False |
| 16 | idade | 0.00141 | 0.001048 | 0.001819 | False |
| 16 | dias_ate_consulta | 0.001474 | 0.001109 | 0.001855 | False |
| 16 | cnesexecutante | 0.001863 | 0.001243 | 0.002446 | False |
| 16 | especialidademae | 0.002339 | 0.001797 | 0.002915 | False |
| 17 | tempo_desloc_trans_publico | -7e-05 | -0.000269 | 0.000126 | True |
| 17 | baldiacoes | 8.9e-05 | -6.6e-05 | 0.000262 | False |
| 17 | corregulador | 9.4e-05 | -0.000119 | 0.000312 | False |
| 17 | favela_comunidade | 0.000113 | -6e-05 | 0.000297 | False |
| 17 | distrito_sanitario | 0.000122 | -9.2e-05 | 0.000362 | False |
| 17 | n_faltas_previas | 0.00014 | -5.5e-05 | 0.000361 | False |
| 17 | dia_semana | 0.000149 | -4.9e-05 | 0.000328 | False |
| 17 | falta_recente | 0.000182 | -5e-06 | 0.000371 | False |
| 17 | cid_1dig | 0.000198 | -3.3e-05 | 0.000415 | False |
| 17 | distancia_km_trans_publico | 0.000228 | 1e-05 | 0.000435 | False |
| 17 | taxa_alf_geral | 0.000278 | 6.6e-05 | 0.000464 | False |
| 17 | taxa_falta_prev | 0.000415 | 0.000186 | 0.000646 | False |
| 17 | dias_ate_consulta | 0.001244 | 0.00088 | 0.001644 | False |
| 17 | idade | 0.001592 | 0.001245 | 0.002025 | False |
| 17 | cnesexecutante | 0.001899 | 0.001321 | 0.002498 | False |
| 17 | especialidademae | 0.002269 | 0.001743 | 0.002875 | False |
| 18 | favela_comunidade | -5.4e-05 | -0.000247 | 0.000137 | True |
| 18 | distrito_sanitario | -3.4e-05 | -0.000221 | 0.000166 | False |
| 18 | dia_semana | 1.2e-05 | -0.00021 | 0.000243 | False |
| 18 | n_faltas_previas | 3.8e-05 | -0.000125 | 0.00023 | False |
| 18 | falta_recente | 3.9e-05 | -0.000172 | 0.000239 | False |
| 18 | corregulador | 4e-05 | -0.000156 | 0.00025 | False |
| 18 | taxa_alf_geral | 6.5e-05 | -0.000154 | 0.000289 | False |
| 18 | baldiacoes | 7.6e-05 | -0.000111 | 0.000274 | False |
| 18 | cid_1dig | 0.000333 | 0.000132 | 0.000565 | False |
| 18 | distancia_km_trans_publico | 0.000399 | 0.000191 | 0.000639 | False |
| 18 | taxa_falta_prev | 0.000439 | 0.000222 | 0.000663 | False |
| 18 | dias_ate_consulta | 0.001405 | 0.001057 | 0.001819 | False |
| 18 | idade | 0.001547 | 0.0012 | 0.001915 | False |
| 18 | especialidademae | 0.002028 | 0.001508 | 0.002631 | False |
| 18 | cnesexecutante | 0.002241 | 0.001632 | 0.002853 | False |
| 19 | baldiacoes | 0.000154 | -4.3e-05 | 0.000343 | True |
| 19 | falta_recente | 0.00018 | -8e-06 | 0.000346 | False |
| 19 | corregulador | 0.000198 | -3.1e-05 | 0.000389 | False |
| 19 | taxa_alf_geral | 0.00026 | 8e-06 | 0.000497 | False |
| 19 | dia_semana | 0.000294 | 8.6e-05 | 0.000501 | False |
| 19 | distrito_sanitario | 0.000294 | 6.6e-05 | 0.00053 | False |
| 19 | n_faltas_previas | 0.000303 | 9.1e-05 | 0.000501 | False |
| 19 | cid_1dig | 0.000429 | 0.000203 | 0.000647 | False |
| 19 | taxa_falta_prev | 0.000581 | 0.000374 | 0.00079 | False |
| 19 | distancia_km_trans_publico | 0.000612 | 0.000374 | 0.000852 | False |
| 19 | dias_ate_consulta | 0.001515 | 0.001112 | 0.001908 | False |
| 19 | idade | 0.001591 | 0.001252 | 0.002006 | False |
| 19 | especialidademae | 0.002415 | 0.001854 | 0.003041 | False |
| 19 | cnesexecutante | 0.002423 | 0.001893 | 0.003018 | False |
| 20 | dia_semana | -6.3e-05 | -0.000256 | 0.000131 | True |
| 20 | falta_recente | 4.2e-05 | -0.000147 | 0.000213 | False |
| 20 | distrito_sanitario | 0.000101 | -0.000127 | 0.000349 | False |
| 20 | corregulador | 0.000106 | -8.8e-05 | 0.000288 | False |
| 20 | n_faltas_previas | 0.000163 | -1e-05 | 0.000364 | False |
| 20 | distancia_km_trans_publico | 0.000207 | -3.4e-05 | 0.000431 | False |
| 20 | taxa_alf_geral | 0.000241 | 4.8e-05 | 0.000449 | False |
| 20 | taxa_falta_prev | 0.000358 | 0.000121 | 0.000588 | False |
| 20 | cid_1dig | 0.000394 | 0.000209 | 0.000623 | False |
| 20 | dias_ate_consulta | 0.001089 | 0.000723 | 0.001512 | False |
| 20 | idade | 0.001465 | 0.001115 | 0.001856 | False |
| 20 | especialidademae | 0.002097 | 0.001563 | 0.002712 | False |
| 20 | cnesexecutante | 0.00222 | 0.001634 | 0.002818 | False |
| 21 | falta_recente | 2e-05 | -0.000156 | 0.00019 | True |
| 21 | distrito_sanitario | 9.1e-05 | -9.8e-05 | 0.000306 | False |
| 21 | n_faltas_previas | 0.000176 | 6e-06 | 0.000348 | False |
| 21 | corregulador | 0.000203 | 1.9e-05 | 0.000376 | False |
| 21 | taxa_alf_geral | 0.000305 | 4.6e-05 | 0.000534 | False |
| 21 | taxa_falta_prev | 0.000356 | 0.000135 | 0.000587 | False |
| 21 | distancia_km_trans_publico | 0.00036 | 0.000131 | 0.000588 | False |
| 21 | cid_1dig | 0.000495 | 0.000268 | 0.000725 | False |
| 21 | dias_ate_consulta | 0.00147 | 0.0011 | 0.001853 | False |
| 21 | idade | 0.001544 | 0.001153 | 0.001956 | False |
| 21 | especialidademae | 0.002129 | 0.001597 | 0.002686 | False |
| 21 | cnesexecutante | 0.002258 | 0.001615 | 0.002897 | False |
| 22 | distrito_sanitario | 4.6e-05 | -0.000179 | 0.000261 | True |
| 22 | n_faltas_previas | 0.000223 | 2.6e-05 | 0.000425 | False |
| 22 | cid_1dig | 0.000268 | 2e-05 | 0.000514 | False |
| 22 | distancia_km_trans_publico | 0.000281 | 4.8e-05 | 0.000513 | False |
| 22 | corregulador | 0.000327 | 0.000146 | 0.000521 | False |
| 22 | taxa_alf_geral | 0.000343 | 0.000144 | 0.000542 | False |
| 22 | taxa_falta_prev | 0.000516 | 0.000263 | 0.000773 | False |
| 22 | dias_ate_consulta | 0.001399 | 0.001011 | 0.001761 | False |
| 22 | idade | 0.001508 | 0.001126 | 0.001892 | False |
| 22 | cnesexecutante | 0.002178 | 0.001582 | 0.002822 | False |
| 22 | especialidademae | 0.002315 | 0.001732 | 0.002904 | False |
| 23 | cid_1dig | 0.000137 | -0.000123 | 0.000373 | True |
| 23 | n_faltas_previas | 0.000196 | 2.2e-05 | 0.000373 | False |
| 23 | corregulador | 0.00029 | 0.000104 | 0.000475 | False |
| 23 | distancia_km_trans_publico | 0.000321 | 7e-05 | 0.000577 | False |
| 23 | taxa_falta_prev | 0.000466 | 0.000255 | 0.000686 | False |
| 23 | taxa_alf_geral | 0.000578 | 0.000333 | 0.000834 | False |
| 23 | idade | 0.001183 | 0.000859 | 0.001548 | False |
| 23 | dias_ate_consulta | 0.001417 | 0.000992 | 0.001842 | False |
| 23 | especialidademae | 0.001852 | 0.0013 | 0.002422 | False |
| 23 | cnesexecutante | 0.002035 | 0.001434 | 0.002682 | False |
| 24 | n_faltas_previas | 0.000178 | -2.9e-05 | 0.000377 | True |
| 24 | corregulador | 0.00021 | 1.4e-05 | 0.000406 | False |
| 24 | taxa_falta_prev | 0.000273 | 6e-05 | 0.000479 | False |
| 24 | distancia_km_trans_publico | 0.000321 | 8e-05 | 0.000554 | False |
| 24 | taxa_alf_geral | 0.000528 | 0.000327 | 0.000727 | False |
| 24 | dias_ate_consulta | 0.001198 | 0.000811 | 0.001603 | False |
| 24 | idade | 0.001561 | 0.001221 | 0.001959 | False |
| 24 | cnesexecutante | 0.002429 | 0.001755 | 0.003025 | False |
| 24 | especialidademae | 0.003149 | 0.002389 | 0.003978 | False |
| 25 | corregulador | -1e-05 | -0.000194 | 0.000187 | True |
| 25 | taxa_alf_geral | 0.000342 | 0.000108 | 0.000555 | False |
| 25 | distancia_km_trans_publico | 0.000463 | 0.000225 | 0.000714 | False |
| 25 | taxa_falta_prev | 0.000857 | 0.000562 | 0.00113 | False |
| 25 | dias_ate_consulta | 0.001146 | 0.000728 | 0.001587 | False |
| 25 | idade | 0.001383 | 0.00103 | 0.001776 | False |
| 25 | cnesexecutante | 0.00213 | 0.001515 | 0.002729 | False |
| 25 | especialidademae | 0.003059 | 0.002263 | 0.003934 | False |
| 26 | taxa_alf_geral | 0.00027 | -2.9e-05 | 0.000513 | True |
| 26 | distancia_km_trans_publico | 0.000373 | 0.000128 | 0.000604 | False |
| 26 | taxa_falta_prev | 0.000859 | 0.000596 | 0.001116 | False |
| 26 | dias_ate_consulta | 0.001402 | 0.000994 | 0.001852 | False |
| 26 | idade | 0.001603 | 0.001257 | 0.002009 | False |
| 26 | cnesexecutante | 0.002072 | 0.001483 | 0.002634 | False |
| 26 | especialidademae | 0.003185 | 0.002355 | 0.004131 | False |
| 27 | taxa_falta_prev | 0.000833 | 0.000594 | 0.001103 | False |
| 27 | distancia_km_trans_publico | 0.000915 | 0.000628 | 0.001203 | False |
| 27 | dias_ate_consulta | 0.00141 | 0.000997 | 0.001839 | False |
| 27 | idade | 0.001598 | 0.00123 | 0.002006 | False |
| 27 | cnesexecutante | 0.002385 | 0.001707 | 0.003113 | False |
| 27 | especialidademae | 0.003299 | 0.002483 | 0.004269 | False |

## Candidatas novas

| conjunto | brier | auc | ganho_vs_final | ic_lo | ic_hi |
|---|---|---|---|---|---|
| final | 0.16243 | 0.69169 | 0.0 | 0.0 | 0.0 |
| +taxa_cnes_28d | 0.1626 | 0.68902 | -0.00017 | -0.00055 | 0.00016 |
| +taxa_prev_suav | 0.16236 | 0.69196 | 6e-05 | -0.00015 | 0.00025 |
| troca cnes -> taxa_cnes_28d | 0.16316 | 0.68524 | -0.00074 | -0.0013 | -0.00021 |
