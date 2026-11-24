# PIPELINE.md — Escopo completo do estudo de previsão de absenteísmo (GERCON Porto Alegre)

Complementa o `CLAUDE.md` (regras). Este arquivo define **o que** o pipeline faz, em que ordem, com quais entradas e saídas, e como cada etapa é verificada. Cada etapa tem um identificador (S0–S15) usado nos nomes de script, logs e no `experimentos.csv`.

Legenda do estado atual: **[reuso]** já existe e serve; **[corrigir]** existe, mas precisa de ajuste; **[novo]** a criar.

---

## 0. Visão geral

```
S0 Governança/ambiente
 └─ S1 Ingestão + auditoria dos dados
     └─ S2 Escopo e coorte (por corte temporal)
         └─ S3 Variáveis e painéis (lag L) [S3.1: estimar horizonte h e atraso d]
             └─ S4 Esquemas de divisão (holdout | expansiva | deslizante | purged+embargo)
                 ├─ S5 Baselines e modelos candidatos
                 │   └─ S6 Seleção de variáveis + tuning aninhado
                 │       └─ S7 Treino final e previsão do teste
                 │           └─ S8 Distribuição preditiva e calibração
                 │               ├─ S9  Avaliação (métricas, testes, subgrupos)
                 │               ├─ S10 Interpretabilidade
                 │               ├─ S11 Utilidade de decisão (overbooking)
                 │               └─ S12 Robustez / sensibilidade
                 └──────────────────────────────┬──┘
                                                 S13 Tabelas e figuras
                                                  └─ S14 Manuscrito (metodologia, resultados, artigo, TRIPOD+AI)
                                                      └─ S15 Empacotamento de reprodutibilidade
```

Portões de aprovação humana (o pipeline para e aguarda o usuário):
- **G1** após S2: escopo e critérios de coorte aprovados.
- **G2** após S4: esquema principal, defasagem principal, período de teste comum e métrica primária **declarados e congelados** (plano de análise estatística, S0).
- **G3** antes de S7: tuning concluído, nenhum teste externo foi visto.
- **G4** após S9: resultados revisados antes de escrever o texto.
- **G5** antes de S15: texto e números conferidos.

---

## 0.1 Escopo mínimo para a disciplina CMP263

Trabalho da disciplina de Aprendizagem de Máquina, com entrega de **Metodologia e Resultados** (a introdução é escrita à parte). O pipeline completo abaixo é o mapa; o mínimo viável é:

| Prioridade | Etapas | Conteúdo mínimo |
|---|---|---|
| **Essencial** | S0, S1, S2, S3, S4, S5, S6, S7, S9, S13, S14 | auditoria dos dados; um escopo (municipal); painel CNES-dia com L = 3; divisão temporal (janela expansiva **e** pelo menos um outro esquema); baseline + pelo menos 4 algoritmos de ML com a mesma validação; tuning aninhado; métricas e teste de diferença contra o baseline; tabelas e figuras geradas por script; as duas seções |
| **Desejável** | S8, S10, S12 parcial | calibração e dispersão iguais para todos; importância por permutação/SHAP; sensibilidade ao lag e ao esquema |
| **Opcional** | S11, S12 completo, S15 completo, escopo geral, deep learning | overbooking, grade fatorial completa, container, depósito do código |

Prazo e formato da entrega da disciplina (número de páginas, apresentação, data) devem ser registrados no `config.yaml` ou no `PROGRESSO` assim que forem conhecidos.

---

## 1. Perguntas e desfechos do estudo

- **Pergunta principal:** algoritmos de machine learning preveem o absenteísmo em consultas de atenção secundária melhor do que um baseline simples de série temporal, e isso se mantém entre esquemas de validação temporal e defasagens?
- **Desfecho (variável original):** `situacaoagenda` → `y = (situacaoagenda == "FALTANTE")` (a lista completa de categorias e o tratamento de cada uma são definidos em S1).
- **Alvos de previsão (todos derivados do mesmo desfecho):**
  1. P(falta) por consulta (modelos por paciente).
  2. Número de faltas por CNES-dia (soma das probabilidades ou previsão direta da série), horizonte L dias.
  3. P(≥k faltas) por CNES-dia, para decisão de overbooking.
- **Métrica primária (a congelar em G2):** proposta = WAPE por CNES-dia (ponto) e CRPS (probabilístico); diferença contra o baseline pelo teste de Diebold-Mariano. Demais métricas são secundárias.
- **Hipóteses a pré-declarar:** H1 pelo menos um modelo de ML supera o baseline no WAPE com p ajustado < 0,05; H2 a ordem dos modelos é estável entre esquemas de divisão; H3 o ganho cai com L maior; H4 o desempenho do modelo geral difere do modelo municipal.

---

## 2. Estrutura de pastas

```
serie_temporal/
├── CLAUDE.md                 regras para o assistente
├── PIPELINE.md               este arquivo
├── config.yaml               única fonte de parâmetros
├── Makefile / run_all.sh     ponto de entrada
├── requirements.lock         versões exatas
├── dados/
│   ├── brutos/               parquets (somente leitura, com hash)
│   └── derivados/            painéis, flags, coortes
├── src/
│   ├── io.py  config.py  splits.py  features.py  panels.py
│   ├── models/               um arquivo por família de modelo
│   ├── metrics.py  dm_test.py  calibration.py  explain.py
│   └── utils_municipal.py    [reuso, migrar]
├── etapas/                   scripts numerados por etapa (S1_..., S2_...)
├── scripts_legados/          01–18 originais, sem alterar
├── resultados/<escopo>/<esquema>/L<lag>/<modelo>/   saídas por experimento
├── resultados/experimentos.csv                      registro de todas as execuções
├── logs/  manifestos/        logs e hashes
├── figuras/  tabelas/        saídas finais geradas por script
├── entregaveis/
│   ├── plano_analise.md      SAP congelado em G2
│   ├── metodologia.md  resultados.md
│   ├── tripod_ai_checklist.md  suplemento.md  referencias.bib
└── tests/                    testes automáticos (vazamento, splits, métricas)
```

---

## 3. Etapas

### S0 — Governança, ambiente e plano de análise
- **Entradas:** `CLAUDE.md`, `PIPELINE.md`.
- **Ações:** criar `config.yaml`; congelar ambiente (`requirements.lock`, versão do Python, venv fora do Drive); iniciar git; definir sementes; escrever o **plano de análise estatística** (hipóteses, métrica primária, esquema principal, L principal, período de teste comum, correção para múltiplas comparações); verificar aprovação ética e LGPD (dados já anonimizados? CEP?) e registrar no manuscrito; considerar pré-registro (OSF).
- **Saídas:** `config.yaml`, `plano_analise.md` (rascunho), `manifestos/ambiente.json`.
- **Verificação:** `make env` reproduz o ambiente do zero.
- **Estado:** [novo].

### S1 — Ingestão e auditoria dos dados
- **Entradas:** `gercon_poa_censo_distancias_com_natureza_juridica.parquet`, `gercon_poa_censo_estabelecimentos.parquet`, `reestruturacao.qmd`, script das distâncias (a obter).
- **Ações:** versionar os scripts que geram a base (R e distâncias); hash SHA-256 dos parquets; auditoria: contagem de **todas** as categorias de `situacaoagenda` e `status` antes de qualquer filtro; duplicatas; datas inválidas; valores ausentes por variável e por ano; distribuição de `dias_ate_consulta`; checagem de que `t0` e as variáveis de histórico seguem `data_ev < t0`; checagem de possível informação pós-desfecho (`status`, `possuiunidadeindicada`).
- **Saídas:** `tabelas/auditoria_dados.csv`, `dicionario_variaveis.md` (nome, tipo, momento de disponibilidade, fonte), fluxograma de seleção (n por filtro).
- **Verificação:** soma de consultas por categoria bate com o total; fluxograma gerado por código.
- **Estado:** [corrigir] (falta o script das distâncias, a definição de `status` e as contagens anteriores aos filtros).

### S2 — Escopo e coorte
- **Entradas:** saída de S1, `flag_municipal.csv`.
- **Ações:** aplicar o escopo de `config.yaml` (`municipal | geral | criterio_x`); definir a coorte de CNES **com informação até o corte de cada ano de teste** (presença, volume mínimo, atividade semanal); classificar níveis A/B/C pelo mesmo corte (quebra de regime por segmentação binária na série mensal, G > 100, diferença ≥ 5 p.p., segmentos ≥ 6 meses); gerar a coorte "sem corte" apenas como análise de sensibilidade; tratar o Hospital Materno Infantil (2237822) como estrato separado.
- **Saídas:** `dados/derivados/coorte_<escopo>_<corte>.csv`, `niveis_<escopo>_<corte>.csv`, relatório de comparação coorte por corte × coorte global.
- **Verificação:** a coorte de um ano não muda ao acrescentar dados posteriores (teste automático).
- **Gate:** G1.
- **Estado:** [corrigir] scripts 10, 19, 20 usam 2019–2026 inteiros.

### S3 — Variáveis e painéis
- **Entradas:** S1, S2.
- **Ações:**
  - Nível consulta: variáveis do `utils_municipal.py` (32 candidatas em 7 grupos), com dicionário de disponibilidade temporal; candidatas novas já testadas (`taxa_cnes_28d`, `taxa_prev_suav`).
  - Nível CNES-dia: painel completo com `f`, `ag`, taxas defasadas `r7/r14/r28/r91` com lag L, `rdow4`, `rd1`, lags de 7 e 14 dias, calendário (dia da semana, mês, feriados nacionais/locais, pós-feriado), agregados da agenda do dia (`m_taxaprev`, `m_primeira`, `m_idade`, `m_dias_ate`).
  - L parametrizável (1, 3, 7, 14); nenhuma feature usa dado posterior a *t − L*, exceto o conhecido com a agenda.
  - Estatísticas de imputação e padronização calculadas só no treino de cada divisão.
- **Saídas:** `dados/derivados/painel_<escopo>_L<lag>.parquet`, `features_consulta.parquet`.
- **Verificação:** teste automático "embaralhar o futuro": alterar os dados posteriores a *t − L* não pode mudar nenhuma feature do dia *t*.
- **Estado:** [corrigir] (L fixo em 3; `fillna` pela média global em `painel_series`).

### S3.1 — Estimativa e justificativa do horizonte (h) e do atraso de disponibilidade (d)
O "3 dias" usado até aqui não tem justificativa empírica. Esta etapa substitui a escolha arbitrária por evidência e por critério operacional. Separar dois conceitos:
- **h (horizonte):** quantos dias antes da consulta a previsão é feita.
- **d (atraso de disponibilidade):** quantos dias levam para o desfecho aparecer no sistema; define quais dados existem nas taxas móveis no dia da previsão.

- **Entradas:** S1 (auditoria), S3 (painéis), `t0`, `dias_ate_consulta`, datas de registro do desfecho (se existirem).
- **Ações:**
  1. **Antecedência do agendamento:** distribuição de `dias_ate_consulta` (percentis, por CNES e especialidade); proporção de consultas marcadas com antecedência ≥ h, para h ∈ {1, 2, 3, 5, 7, 10, 14}. Confirmar o significado de `t0` no `reestruturacao.qmd` antes de interpretar.
  2. **Fração da agenda final conhecida em t − h:** para cada h, razão entre o número de consultas do dia com `t0 ≤ t − h` e o total final `ag`. Se for muito menor que 1, a avaliação com `ag` final é otimista; nesse caso reconstruir `ag` e os agregados da agenda (`m_*`) **como existiam em t − h**, e comparar os resultados das duas versões.
  3. **Atraso de registro (d):** distribuição do intervalo entre a data da consulta e a data em que o desfecho fica registrado, se a base tiver esse campo; caso contrário, registrar d como premissa a confirmar com o serviço.
  4. **Curva de desempenho × horizonte:** rodar o protocolo principal para h ∈ {1, 2, 3, 5, 7, 10, 14} (mantendo d fixo, e depois variando d) e traçar WAPE, CRPS e ganho sobre o baseline; identificar até que h o modelo ainda supera o baseline com significância.
  5. **Memória da série:** ACF/PACF da taxa diária de falta por CNES, para orientar os lags das variáveis (não o horizonte).
  6. **Critério operacional:** registrar, com fonte (regulação, gestores, rotina documentada), os prazos de confirmação, lembrete, fechamento da agenda e encaixe de lista de espera.
- **Saídas:** `tabelas/antecedencia_agendamento.csv`, `tabelas/agenda_conhecida_por_h.csv`, `tabelas/atraso_registro.csv`, `figuras/desempenho_vs_horizonte.png`, `entregaveis/justificativa_horizonte.md`.
- **Regra de decisão:** h principal é escolhido por **critério operacional**, ancorado nas evidências 1 a 3, e declarado em G2; a curva do item 4 entra como análise de sensibilidade, nunca como critério de "menor erro no teste".
- **Verificação:** teste automático de que `ag(t − h)` nunca inclui consultas com `t0 > t − h`; a fração conhecida é monotônica em h.
- **Estado:** [novo].

### S4 — Esquemas de divisão temporal
- **Entradas:** `config.yaml` (`divisao`, `corte`, `janela`, `purga`, `embargo`).
- **Ações:** implementar em `src/splits.py` um gerador único com as quatro opções:
  1. **Holdout temporal:** treino até C, teste após C; validação dentro do treino.
  2. **Janela expansiva:** treino cresce; teste por bloco (ano).
  3. **Janela deslizante:** tamanho fixo (12, 24, 36 meses), escolhido só na validação.
  4. **Purged + embargo:** purga de P dias antes do teste (P ≥ maior intervalo entre `t0` e consulta + L), embargo E depois do teste (efeito só em validação cruzada em blocos).
  - Folds internos de tuning construídos só com dados anteriores ao teste, com o mesmo esquema.
  - Definir o **conjunto comum de dias de teste** (ex.: 2024-01 a 2026-03) para comparar esquemas.
- **Saídas:** `splits/<esquema>.parquet` (índices treino/validação/teste), relatório com n por divisão.
- **Verificação (testes automáticos):** nenhum índice de treino ≥ início do teste; purga remove exatamente as linhas esperadas; folds existem para todo Y (incluindo 2026); nenhuma linha em dois conjuntos.
- **Gate:** G2 (congelar esquema principal, L principal, período comum, métrica primária).
- **Estado:** [novo] (hoje só existe a expansiva por ano, embutida nos scripts).

### S5 — Baselines e modelos candidatos
Todos usam os mesmos splits, as mesmas linhas e orçamento de tuning equivalente.
- **Referências:** naive sazonal (média do mesmo dia da semana nas 4 semanas anteriores); baseline Poisson (taxa 28 dias com lag L × agenda).
- **Estatísticos:** ETS, SARIMA/SARIMAX (d=1, D=1, s=7; exógenas da agenda), regressão Poisson/binomial negativa (GLM) com calendário.
- **ML tabular com lags:** HGB Poisson (pura e +agenda), LightGBM, XGBoost, CatBoost, floresta aleatória, regressão regularizada (elastic net), MLP.
- **Por paciente, somado por CNES-dia:** regressão logística, HGB, LightGBM/XGBoost/CatBoost; variantes V0 (todas as linhas) e V1 (exclusões definidas só com dados do treino).
- **Opcional (se houver tempo e dados):** modelos de séries profundas (N-BEATS, TFT, LSTM) como análise exploratória; incluir só com o mesmo protocolo.
- **Saídas:** `src/models/*.py` com interface comum (`fit`, `predict_mean`, `predict_dist`), tabela de hiperparâmetros e espaço de busca por modelo.
- **Verificação:** teste de contrato da interface; mesmo conjunto de linhas avaliado por todos.
- **Estado:** [reuso/corrigir] (scripts 04–09, 11, 17, 22–23).

### S6 — Seleção de variáveis e tuning aninhado
- **Entradas:** S3, S4, S5.
- **Ações:** seleção de variáveis com dados anteriores ao teste (permutação, ablação por grupo, eliminação retroativa com IC por bootstrap por CNES-mês e correção para múltiplas comparações) em **todos** os folds disponíveis; busca aleatória de hiperparâmetros por modelo (mesmo número de configurações; sementes fixas); critério de tuning alinhado ao objetivo (Brier por consulta para os modelos por paciente e também WAPE por CNES-dia depois da recalibração); número de iterações via `staged_predict_proba` (sem early stopping aleatório); grade livre de `max_bins` incompatível com variáveis categóricas.
- **Saídas:** `resultados/.../tuning_<modelo>.csv`, `melhores_hiperparametros.json`, `variaveis_finais.json`.
- **Verificação:** log mostra que nenhum fold usa dado do período de teste.
- **Gate:** G3.
- **Estado:** [corrigir] (scripts 21–23: folds até 2025, `max_bins`).

### S7 — Treino final e previsão do teste
- **Ações:** para cada combinação (escopo, esquema, L, modelo, variante) treinar com os parâmetros escolhidos e gravar as previsões brutas (probabilidade por consulta e média por CNES-dia) com chave (CNES, dia, paciente quando existir); paralelização com `joblib`; sementes registradas.
- **Saídas:** `resultados/<escopo>/<esquema>/L<lag>/<modelo>/previsoes.parquet`, linha em `experimentos.csv`.
- **Verificação:** alinhamento por chave; sem NaN; contagem de linhas igual entre modelos.
- **Estado:** [corrigir] (scripts 11, 15, 22, 23).

### S8 — Distribuição preditiva e calibração
- **Ações:** recalibração k (faltas ÷ previstas nos 28 dias anteriores, lag L); **mesma família de distribuição preditiva para todos** (ex.: binomial negativa com dispersão estimada fora da amostra, ou Poisson-binomial com mistura Gama, ou intervalos conformais); dispersão estimada em previsões **fora da amostra**, não dentro; avaliar calibração (cobertura 80/50, ECE de P(≥k), diagramas de confiabilidade).
- **Saídas:** `distribuicoes.parquet`, `calibracao.csv`.
- **Estado:** [corrigir] (script 24 mistura NB nas séries e Poisson-binomial nos modelos por paciente).

### S9 — Avaliação
- **Métricas:** WAPE, viés, CRPS, cobertura, largura do IC80, ECE; por consulta: Brier, AUC, logloss; por célula CNES-dia-especialidade: AUC de P(≥1 falta).
- **Inferência:** Diebold-Mariano (HAC semanal) contra o baseline e contra o melhor modelo; IC por bootstrap em blocos; correção para múltiplas comparações (Holm ou *Model Confidence Set*); tamanho do efeito.
- **Subgrupos (TRIPOD+AI):** por nível (A/B), ano, especialidade, CNES, faixa etária, distrito sanitário; sem o Materno Infantil; período pré e pós-pandemia.
- **Saídas:** `comparacao_*.csv`, `dm_*.csv`, `subgrupos.csv`.
- **Gate:** G4.
- **Estado:** [corrigir] (script 24 só cobre a expansiva).

### S10 — Interpretabilidade
- **Ações:** importância por permutação (variável e grupo), SHAP para os melhores modelos de árvore, dependência parcial/ALE das principais variáveis, análise de estabilidade da importância entre anos e esquemas.
- **Saídas:** `importancia_*.csv/.png`, `shap_*.png`.
- **Estado:** [parcial] (script 21).

### S11 — Utilidade de decisão
- **Ações:** simular a regra de overbooking por quantil da distribuição preditiva (risco aceito q10, q30 etc.): consultas recuperadas, sobreoferta, dias com excesso, taxa de uso; análise de decisão (*decision curve*/benefício líquido) com custos declarados; escolher casos municipais (candidatos: Santa Marta 7031076, IAPI 7214448); comparar com a regra sem previsão e com a regra do baseline.
- **Saídas:** `overbooking_*.csv/.png`, `relatorio_decisao.md`.
- **Estado:** [reuso/corrigir] (scripts 16, overbooking_*; o caso 2237571 saiu do recorte).

### S12 — Robustez e sensibilidade
- **Eixos (um fator por vez, a partir da configuração principal):** escopo (municipal × geral × critério); esquema de divisão; L; tamanho da janela deslizante; P e E da purga; coorte por corte × coorte global; com e sem Materno Infantil; V0 × V1; semente (≥ 5 repetições para modelos estocásticos); remoção das variáveis suspeitas de informação pós-desfecho.
- **Saídas:** `sensibilidade.csv`, figura de estabilidade do ranking dos modelos.
- **Experimento fatorial completo** só para a configuração principal; as demais por varredura de um fator para limitar o custo.

### S13 — Tabelas e figuras
- **Ações:** um único script (`etapas/S13_tabelas_figuras.py`) lê os CSVs e gera todas as tabelas e figuras do manuscrito (fluxograma, características da coorte, tabela principal, ganhos com IC, calibração, subgrupos, importância, utilidade de decisão, sensibilidade). Figuras vetoriais (SVG/PDF) com paleta acessível. Nenhum número digitado à mão.
- **Saídas:** `tabelas/`, `figuras/`.

### S14 — Seções do trabalho (Metodologia e Resultados)
- **Ações:** gerar `metodologia.md` e `resultados.md` com números inseridos por script a partir dos CSVs; a introdução é escrita à parte pelo usuário (não faz parte deste pipeline); preencher `tripod_ai_checklist.md` apenas nos itens de Métodos e Resultados; suplemento opcional (hiperparâmetros e resultados completos); referências em `referencias.bib` só com DOI verificado; exportar para .docx e .pdf; revisão cruzada: cada número do texto rastreável até um arquivo.
- **Verificação:** `make manuscrito` falha se um número do texto não existir no CSV de origem.
- **Gate:** G5.

### S15 — Empacotamento de reprodutibilidade
- **Ações:** `run_all.sh`/`Makefile` que executa S1→S14; `make smoke` (amostra pequena, todos os anos e todos os folds) em minutos; `make full` (completo); container opcional (Dockerfile); manifesto com hashes de entradas e saídas; README de reprodução; licença; depósito do código (GitHub + Zenodo/OSF); declaração de disponibilidade de dados (dados sensíveis não públicos; fornecer dicionário e dados sintéticos para teste do pipeline).
- **Verificação:** execução do zero em máquina limpa reproduz as tabelas dentro de tolerância definida.

---

## 4. Grade de experimentos

| Fator | Níveis |
|---|---|
| Escopo | municipal; geral; critério a definir |
| Esquema de divisão | holdout; expansiva; deslizante (12/24/36 meses); purged + embargo (P ∈ {0, 28, 90}) |
| Defasagem L | 1; 3; 7; 14 |
| Modelo | 2 referências + ~10 candidatos (S5) |
| Variante por paciente | V0; V1 |
| Variáveis | 32; selecionadas |
| Semente | ≥ 5 (modelos estocásticos) |

Estratégia: **configuração principal** (escopo municipal, janela expansiva, L = 3) com todos os modelos e todas as semente; **varredura de um fator** para os demais eixos; combinações cruzadas só para os 3 a 4 melhores modelos. Estimar o custo de computação no smoke antes de liberar o completo.

---

## 5. Mapa dos scripts existentes

| Script atual | Etapa | Estado |
|---|---|---|
| `reestruturacao.qmd` | S1 | [reuso] versionar |
| 01, 02, 03 (painéis, top 20) | S3 | [legado] top 20 usa todos os dados |
| 04, 05 (baseline) | S5 | [reuso] |
| 06, 07, 08, 09 | S5, S8, S9 | [reuso] |
| 10 (coorte), 19, 20 | S2 | [corrigir] corte temporal |
| 11 (V1), 15 | S5, S7 | [corrigir] |
| 12, 14, 16 | S8, S11 | [reuso] |
| 17, 18 | S5, S9 | [corrigir] |
| 21 | S6 | [corrigir] |
| 22, 23 | S6, S7 | [corrigir] folds até 2025, `max_bins` |
| 24 | S8, S9 | [corrigir] linhas e dispersão iguais |
| `utils_municipal.py` | S3 | [reuso] migrar para `src/` |
| `prever_dia.py` | S11 | [reuso] |

---

## 6. Marcos de implementação (ordem sugerida)

1. **M0:** S0 e S1 (config, ambiente, git, auditoria). Entrega: `auditoria_dados.csv`, plano de análise em rascunho.
2. **M1:** `splits.py` com testes (S4) e coorte por corte (S2). Gates G1 e G2.
3. **M2:** painéis com L parametrizável e testes de vazamento (S3).
4. **M3:** interface de modelos e referências (S5), com o baseline reproduzindo os números dos scripts antigos.
5. **M4:** correções dos scripts 22–24 portadas para `src/` (S6–S8); smoke completo.
6. **M5:** execução da configuração principal e G3, depois S9 e G4.
7. **M6:** S10 e S12 parcial (S11 opcional).
8. **M7:** S13 e S14 (Metodologia e Resultados), G5.
9. **M8:** S15 (execução do zero em máquina limpa; versão leve basta para a disciplina).

---

## 7. Riscos e como tratá-los

| Risco | Tratamento |
|---|---|
| Vazamento temporal em coorte, variáveis ou lags | testes automáticos (S3, S4); coorte por corte |
| Reuso do teste para escolher esquema ou variáveis | esquema principal congelado em G2; variantes descritivas |
| Comparação injusta entre modelos | mesmas linhas, mesma dispersão, mesmo orçamento de tuning |
| Poucos dados em 2026 (até março) | período comum de teste; IC amplos; não superinterpretar |
| Custo computacional da grade completa | configuração principal + varredura de um fator |
| Dados sensíveis e Drive | venv e cache fora da pasta sincronizada; sem identificadores nas saídas |
| Divergência entre notebook e app | `CLAUDE.md`, `PIPELINE.md`, `PROGRESSO` e `HANDOFF` sempre atualizados |
| Ganho pequeno sobre o baseline | relatar honestamente; discutir utilidade prática, não só significância |

---

## 8. `config.yaml` (esqueleto)

```yaml
projeto: absenteismo_gercon_poa
sementes: [0, 1, 2, 3, 4]
dados:
  base: dados/brutos/gercon_poa_censo_distancias_com_natureza_juridica.parquet
  estabelecimentos: dados/brutos/gercon_poa_censo_estabelecimentos.parquet
  desfecho: {coluna: situacaoagenda, positivo: FALTANTE}
escopo: municipal            # municipal | geral | criterio_x
coorte: {por_corte: true, min_consultas_ano: 100, min_semanas_ativas: 0.90}
lag: [1, 3, 7, 14]
lag_principal: 3
divisao:
  principal: expansiva       # holdout | expansiva | deslizante | purged_embargo
  esquemas: [holdout, expansiva, deslizante, purged_embargo]
  holdout: {corte: "2023-01-01"}
  deslizante: {janelas_meses: [12, 24, 36]}
  purged_embargo: {purga_dias: [0, 28, 90], embargo_dias: [0, 7]}
  teste_comum: ["2024-01-01", "2026-03-31"]
  anos_teste: [2023, 2024, 2025, 2026]
modelos: [naive, baseline_poisson, ets, sarima, sarimax, glm_nb, hgb_poisson, lightgbm, xgboost, catboost, rf, elasticnet, mlp, paciente_hgb]
tuning: {n_configs: 24, folds_internos: [-2, -1], criterio: [brier, wape]}
avaliacao: {primaria: [wape, crps], correcao: holm, n_simulacoes: 300, ks: [1, 2, 3, 5, 8]}
saidas: {raiz: resultados, experimentos: resultados/experimentos.csv}
```

---

## 9. Critérios de pronto

O estudo está pronto quando: (i) `make full` reproduz do zero todas as tabelas e figuras; (ii) os testes de vazamento passam; (iii) o checklist TRIPOD+AI está completo; (iv) todo número do texto é rastreável a um arquivo; (v) as limitações estão declaradas em Resultados; (vi) o código e o plano de análise estão organizados na pasta para entrega.
