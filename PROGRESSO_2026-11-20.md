# PROGRESSO — 2026-11-20

## Executado nesta data

- **Perfil completo rodado na máquina draco3** (job 838448, 2026-11-17 16:57 → 20:39, 187,5 min, 16 processos). Pacote entregue como `pcad-resultados.zip`.
- **Conferido no log:** S0 a S13 concluídos sem exceção; entradas com hash OK; reprodução do legado dentro da tolerância (A+B WAPE 0,3826 vs 0,383); 666 divisões (224 expansiva, 104 deslizante, 104 em cada purga P0/P28/P90, 26 holdout); 186 tarefas de experimento (`experimentos.csv` com 186 registros).
- **Integrado ao repositório** (sem commit): `resultados/municipal_completo/`, `figuras/completo/`, `tabelas/md/completo/`, `tabelas/*_municipal_completo*.csv`, `logs/draco3_838448.{out,err}`.
- **Sobrescritos** (versões antigas continuam no git):
  - `tabelas/purga_recomendada_municipal.csv`: a versão local tinha L=7 (descartado em 17/11); a nova usa L ∈ {3,14}, P = 17/67 (L=3) e 28/78 (L=14).
  - `tabelas/estimativa_recursos.csv`: tempos de execução da máquina nova.
  - `tabelas/acf_taxa_diaria_municipal.csv`: diferença só de precisão numérica.
- **Manifesto:** `manifestos/ingestao_perfil_completo.json` (SHA-256 do zip, das tabelas principais e dos logs).

## Resultados lidos (perfil completo, escopo municipal, L=3, expansiva)

- Melhor WAPE: `paciente_hgb` 0,348; LightGBM 0,352; XGBoost 0,353; CatBoost 0,351. Baseline Poisson 0,364.
- Ganho sobre o baseline (DM, erro absoluto, Holm): `paciente_hgb` +4,3% (p<0,001); LightGBM +3,2%; CatBoost +3,5%; XGBoost +3,1%; HGB +2,8%. ETS, GLM-NB e MLP não se distinguem do baseline.
- `paciente_hgb` vs. LightGBM/XGBoost/CatBoost: diferença não significativa (p_holm ≥ 0,35). O empate no topo é real; não escolher um pelo número.
- Naive é pior que o baseline (WAPE 0,394).

## Cuidados e pendências

1. **Checagem S1 com 5 chaves repetidas** (`paciente+solicitacao+consulta+especialidade`): o log marca "valor 0 esperado" e a auditoria diz "mantidas na base; só descritivo". Decidir se entra como limitação ou se é removida (decisão do usuário).
2. **`status` ausente em 1,1%** (11.047 consultas, categoria `<NA>`). Documentar no dicionário.
3. **Atraso de registro do desfecho:** tabela `atraso_registro.csv` diz "inexistente na base; d = 0 a confirmar com o serviço". Premissa pendente.
4. **Commit do código que gerou a execução** não está no zip. Confirmar na draco3 (`git rev-parse HEAD`) antes de citar a versão no artigo.
5. Os manifestos `*.json` por etapa e `ambiente.json` da execução remota não vieram no zip; só logs de texto.
6. ~~Entregáveis ainda não escritos~~ (escritos nesta data, ver abaixo).

## Entregáveis escritos nesta data (rascunho v1, para revisão)

- `entregaveis/metodologia.md`: fonte, recorte municipal, desfecho, variáveis com momento de disponibilidade, coorte por corte, esquemas de divisão, modelos com grades, distribuição preditiva NB, DM/Holm, reprodutibilidade.
- `entregaveis/resultados.md`: seleção, coorte por corte, tabela principal (15 modelos), DM com IC, calibração, subgrupos, horizonte, esquemas, modo da agenda, overbooking, importância, sementes, limitações (subseção final).
- `entregaveis/tripod_ai_checklist.md`: TRIPOD+AI, 39 itens avaliados (9 fora de escopo). PRESENT 13, PARTIAL 15, MISSING 6, N/A 5. Compliance 38,2% (82,4% presente ou parcial). Headline: item 15-AI (disponibilidade do modelo) ausente.
- Verificações: nomenclatura de frameworks (Step 4e) OK; hashes SHA-256 nos cabeçalhos do checklist; `entregaveis/qc/framework_naming_*.json`.

## Achados da redação que exigem decisão do usuário

1. **Vantagem do modelo por paciente depende do horizonte.** Em L = 3, `paciente_hgb` tem o menor WAPE (0,348). Em L = 14 ele cai para o 10º lugar (0,439), abaixo dos modelos de árvore sobre a série (≈0,394). O overbooking tem a ordem oposta (lidera em L = 14 com c = 1, 2 e 3).
2. **Calibração conservadora.** A cobertura de 80% fica em 0,90–0,91 e a de 50% em 0,70–0,73. Os intervalos são mais largos que o nominal; isso afeta a regra de overbooking. Vale investigar a dispersão φ (estimada por momentos nos folds internos).
3. **Interseção de linhas.** A comparação principal tem 8.073 CNES-dias, 151 a menos que as 8.224 do baseline, por falhas de ajuste do SARIMAX. Decidir se o SARIMAX entra na comparação principal ou se a comparação é refeita sem ele.
4. **Orçamento de busca desigual** (ETS 20, SARIMA 12, GLMs 6 configurações contra 24 nos demais). Declarado na Metodologia; decidir se é aceitável para o artigo.
5. **Horizonte L = 3 sem critério operacional declarado** (`entregaveis/justificativa_horizonte.md`, seção 5). Preencher com fonte antes de o artigo sair.
6. **Premissas de dados** (d = 0, momento de `idade`, `status`, `possuiunidadeindicada`) seguem sem confirmação com o serviço.

## Pendências do checklist (ver `tripod_ai_checklist.md`, Parte C)

Tabela descritiva por ano e nível (13b), comparação treino × teste (13c), EPV (8-AI), IC e inclinação/intercepto de calibração (16), disponibilidade de código e modelo (21-AI, 15-AI).
