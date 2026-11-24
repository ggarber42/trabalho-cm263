# Progresso — recorte municipal, seleção de variáveis e tuning (atualizado em 11/11/2026)

Pasta: `serie_temporal/`. Scripts originais (01–18) não foram alterados. Plano aprovado: `~/.claude/plans/fa-a-tudo-isso-e-proud-clarke.md`.

## 1. Recorte municipal (feito)
- Fonte da natureza jurídica: `gercon_poa_censo_distancias_com_natureza_juridica.parquet` (coluna `DS_NATUREZA_JUR`) e cadastro `gercon_poa_censo_estabelecimentos.parquet` (razão social, gestor, mantenedora). `TP_GESTAO` não discrimina (99,5% = M), não usar.
- Critério: natureza Município/Fundação municipal + gestor 431490 + razão social da Prefeitura/Município de Porto Alegre, Hospital Materno Infantil Presidente Vargas ou PA Cruzeiro do Sul. Resultado: **111 CNES, 360.878 consultas (36,7%)**, em `flag_municipal.csv`. Excluídos: 8 CNES "Hospital N. S. Conceição" com natureza Município (provável erro de cadastro), 3 CNES de outros municípios e 3 códigos placeholder (1111111, 1111112, 9912345).
- Cruzamento natureza × razão social: `cruzamento_natureza_razao_social.csv`.
- **CNES 2237571 (endodontia) é do GHC (federal) e sai do recorte.**
- Coorte A/B/C municipal (scripts 19 e 20): 10 CNES (6 A + 4 B, nenhuma C), 71,5% das consultas municipais. Arquivos: `coorte_municipal_2019_2026.csv`, `niveis_municipal.csv`, `saida_19.txt`, `saida_20.txt`.
- A regra B/C foi reconstruída (B: 1–2 quebras e taxa do último regime ≥ 5%; C: ≥ 3 quebras ou taxa < 5%) e reproduz 26/26 níveis da coorte original, mas foi ajustada olhando esses mesmos 26.

## 2. Seleção de variáveis (script 21, concluído)
Método: permutação (variável e grupo) + ablação por grupo + eliminação retroativa (remove a de menor ΔBrier se o IC 95% por bootstrap CNES-mês incluir ≤ 0). Só dados < 2023; folds treino<2021→val 2021 e treino<2022→val 2022; validação nas 10 CNES da coorte.
- **32 → 6 variáveis:** `idade`, `taxa_falta_prev`, `dias_ate_consulta`, `distancia_km_trans_publico`, `especialidademae`, `cnesexecutante`.
- Brier CV 0,16222 → 0,16243 (+0,0002); AUC por consulta 0,6926 → 0,6917.
- Maior importância: `cnesexecutante` (ΔBrier 0,0096), `especialidademae` (0,0071), `idade` (0,0047), `dias_ate_consulta` (0,0031).
- Candidatas novas (`taxa_cnes_28d`, `taxa_prev_suav`, trocar CNES por taxa recente): nenhuma ganho significativo; trocar a CNES piora (−0,0007).
- Relatório: `resultados_municipal/relatorio_selecao_variaveis_municipal.md`; demais: `importancia_permutacao.csv/.png`, `ablacao_grupos.csv`, `eliminacao_retroativa.csv`, `candidatas_novas.csv`, `variaveis_finais.json`, `saida_21.txt`.
- **Ressalvas:** a poda é agressiva (26 de 32 removidas) e usa só 2 folds e ~63 mil consultas de validação; várias removidas têm ΔBrier positivo, só não significativo. Antes de fixar para a tese, rodar o 22 também com as 32 variáveis (`ref_default`, já previsto) e comparar no teste externo.

## 3. Scripts escritos mas ainda NÃO executados
| Script | Função | Estado |
|---|---|---|
| `22_hgb_paciente_tuning.py` | Busca aleatória (24 configs) com validação aninhada; regimes base toda e só municipal; variantes ref_default / sel_default / sel_tuned; saída `p_modelos_municipal.parquet` | escrito, **sem teste de fumaça** (usar `--smoke`) |
| `23_series_tuning.py` | HGB Poisson e SARIMA/SARIMAX (puro e +agenda), tuning aninhado; saída `mu_series_municipal.parquet` | smoke test parcial OK (SARIMAX h=3 bate com o statsmodels); execução completa pendente, SARIMA = 2.880 ajustes |
| `24_comparacao_municipal.py` | Comparação final (WAPE, CRPS, cobertura, AUC por consulta e célula, DM, cortes) | só sintaxe verificada |
- `utils_municipal.py` é importado pelos scripts 21–24; rodar com `python3 script.py` (não usar `-I`, que bloqueia o import local).
- Nota: fits instáveis de SARIMAX geram previsões absurdas (ex.: −407); o código recorta em [0,1] e o WAPE descarta essas configurações.

## 4. Próximos passos
1. `python3 22_hgb_paciente_tuning.py --smoke --ncfg 3`, depois completo (estimativa de ~1 h; 192 ajustes de tuning + 24 finais). Rodar em background.
2. `python3 23_series_tuning.py --smoke --ncfg 3` já passou parcialmente; rodar completo (`--jobs 6`, ~1–2 h) em background.
3. `python3 24_comparacao_municipal.py` e conferir: V1 original reproduzido na coorte, sem NaN, mesmas linhas para todos os modelos.
4. Decidir sobre a poda de variáveis olhando `ref_default` × `sel_default` × `sel_tuned` no teste externo (e com/sem Materno Infantil).
5. Atualizar o relatório de seleção com os resultados do 24 e rever as ressalvas acima.
6. Escolher novo caso de overbooking (o 2237571 saiu): candidatos municipais odontológicos, Santa Marta (7031076) ou IAPI (7214448).
