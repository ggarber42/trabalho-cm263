# PROGRESSO 23/11/2026 — pipeline_v2, rodada completa com D19

## Feito (executado e verificado)
- Auditoria da referência taxa × agenda em h=2: taxa_ag4 chegava a 55 faltas/consulta conhecida
  (agenda incompleta na origem). Correção D19: taxas cortadas em 1 (escolhido só no treino 2022–23).
- IC95% e margem de erro (metade do IC, bootstrap em blocos de 4 semanas, 2000 reamostragens) para
  WAPE, viés, MAE e ganhos; coluna `significativo` (p_holm < 0,05 e IC95 sem 0); `tabela_principal.csv`.
- Teste novo `tests/test_d19_ic.py`; 21/21 testes passando.
- Rodada completa e01–e06 (23/11/2026, 11h27–12h37, ≈ 70 min). Cópia congelada em
  `pipeline_v2/saidas/resultados_2026-11-23_D19/`.

## Resultado principal (h=2, nível CNES, consistentes; de tabela_principal.csv)
HGB 0,219 ± 0,030 (ganho 15,5% ± 3,2% sobre a referência, significativo); Floresta, MLP, GLM também
significativos; SARIMAX, ETS, Croston não. HGB significativamente melhor que todos os demais.

## Pendências
1. (RESOLVIDO, D20) "outros" h=2: referência mantida (melhor que mm4 no treino); comparação secundária
   contra mm4 em dm_vs_mm4.csv; piora da referência em 2024 vira limitação.
2. (RESOLVIDO, D20) Cobertura 98–99%: mesmo Poisson cobre 96,9%; efeito da discreção e de contagens
   limitadas pela agenda; declarado como limitação (intervalos conservadores).
3. (RESOLVIDO) `etapas/d01_auditoria.py` → auditoria/metricas_sem_fallback.csv: sem fallback, ordem igual;
   HGB continua o melhor (h=2 WAPE 0,417); ETS/Croston pouco abaixo da referência em h=1, acima em h=2.
4. (RESOLVIDO) Teste de vazamento (desfecho embaralhado e invertido a partir de 03/06/2024, CNES
   consistente de maior volume): nenhuma preditora mudou antes de corte + h (h=1 e h=2). d01 entra no executar.sh.
5. Tempo do e04 não registrado separadamente no log.
6. (RESOLVIDO) Metodologia reescrita com D15–D20 em `entregaveis/metodologia_v2_modelo.md`; números
   preenchidos por `pipeline_v2/etapas/r01_metodologia.py` → `metodologia_v2.md` (rascunho antigo em
   `metodologia_v2_rascunho_2026-11-22.md`). README atualizado. Faltam 2 itens do usuário: repositório e parecer de ética.
7. (RESOLVIDO) `etapas/e07_figuras.py` → saidas/<perfil>/figuras/: observado_previsto, wape_ic95,
   ganho_ic95 (h=1 e h=2), com os dados em dados_observado_previsto.csv.
8. Commit (só quando o usuário pedir).
