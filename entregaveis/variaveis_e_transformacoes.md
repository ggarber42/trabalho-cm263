# Variáveis por modelo e transformações aplicadas

Documento de apoio à Metodologia (CMP263). Descreve o que cada modelo usa e todas as transformações
aplicadas do arquivo bruto até a matriz de cada modelo. Referências a código são `arquivo:linha`.
Escrito a partir da leitura do código em 2026-11-21; nada foi reexecutado para escrever este texto.

## 1. Arquivo de entrada

- **Arquivo lido pelo pipeline:** `gercon_poa_censo_distancias_com_natureza_juridica.parquet`
  (uma linha por consulta agendada). Lido em `src/dados.py:101-124` (`carrega_contexto`), só com as colunas de
  `colunas_necessarias` (`src/dados.py:26-31`).
- **Origem (parcialmente verificada):**
  - `reestruturacao.qmd` (repositório `gercon/andre/`, fora desta pasta) faz o pré-processamento descrito na seção 2.
    Ele grava `gercon_poa_censo_reestrurado.parquet`, mas a gravação está comentada (linha 498), e o arquivo de entrada do
    pipeline tem outro nome.
  - As variáveis de censo (`favela_comunidade`, `taxa_alf_geral`, `renda_media_winsor`), de distância/rota
    (`distancia_km`, `distancia_km_trans_publico`, `tempo_caminhando`, `tempo_desloc_trans_publico`, `quant_bus`, `baldiacoes`)
    e `DS_NATUREZA_JUR` vêm de etapas anteriores que **não estão no repositório** (pendência já registrada no `dicionario_variaveis.md`).
    Não foi possível verificar essas transformações.

## 2. Transformações no pré-processamento (`reestruturacao.qmd`)

Ordem conforme o script:

1. **Filtro de complexidade:** mantém `complexidade == "MEDIA"` (linha 103).
2. **Filtro de agendamento:** remove linhas com `datahora_consulta` nula (linhas 119–120).
3. **Filtro de desfecho:** mantém `situacaoagenda %in% {REALIZADA, FALTANTE}` (linha 137).
4. **Filtro de corregulador:** remove linhas com `corregulador` nulo (linha 165).
5. **Correção de valor:** `distrito_sanitario` do CNES 2264374 é corrigido para "HUMAITA/NAVEGANTES", porque o valor original era a
   mensagem de "fechada para reunião" (linhas 285–304).
6. **Datas:** `data_solicitacao` e `datahora_consulta` são cortadas nos 10 primeiros caracteres e convertidas para data (linhas 209–239).
7. **CNES:** `cnes_solicitante` e `cnesexecutante` são preenchidos com zeros à esquerda até 7 dígitos (linhas 254–263).
8. **CID:** `cid_3dig` = 3 primeiros caracteres; `cid_1dig` = 1º caractere (linhas 322–324).
9. **Variáveis auxiliares:**
   - `y = (situacaoagenda == "FALTANTE")` (linha 356);
   - `t0 = data_solicitacao + max(diasfilaespera, 0)`, isto é, a data do agendamento (linha 357);
   - `motivo = especialidademae | cid_3dig` (linha 358).
10. **Histórico do paciente (anti-vazamento):** para cada consulta, são usados só eventos anteriores com `data_ev < t0` do mesmo
    `idpaciente` (linhas 373–376). A partir deles:
    - `n_consultas_previas`, `n_faltas_previas` (geral);
    - `n_consultas_previas_especialidademae`, `n_faltas_previas_especialidademae`;
    - `n_consultas_previas_motivo`, `n_faltas_previas_motivo`;
    - `ultima_previa_mesma_especialidademae` e `ultima_previa_mesmo_motivo`: "sem_previa", "realizada" ou "faltante", conforme o desfecho da
      consulta anterior mais recente (linhas 386–391, 420–429);
    - `primeira_vez = (n_consultas_previas == 0)` (linha 412);
    - `falta_recente`: dias desde a última falta anterior a t0, em "nunca_faltou", "ate_90d", "91_365d" e "mais_de_365d"
      (linhas 413–419).
    - Contagens sem histórico recebem 0 (linhas 408–411).
11. **Tempo e calendário:**
    - `dias_ate_consulta = datahora_consulta − data_solicitacao`, em dias (linha 438). Inclui a espera;
    - `antecedencia_agendamento = dias_ate_consulta − diasfilaespera` (linha 439), criada mas não usada pelo pipeline;
    - `dia_semana` = fator com níveis seg…dom (`wday` com semana começando na segunda, linha 440);
    - `mes_consulta` = fator 1…12 (linha 443).
12. **Seleção de colunas:** `var_interesse` (linhas 450–473). Distâncias, `status`, `possuiunidadeindicada` (que aparece como fator, mas
    o pipeline não o usa) e as colunas de identificação ficam de fora ou são só auditoria.
13. **Tipos:** `situacaoagenda` vira fator com níveis FALTANTE/REALIZADA; variáveis categóricas viram fator; contagens viram inteiro;
    `idade`, `favela_comunidade`, `taxa_alf_geral` e `renda_media_winsor` viram numérico (linhas 477–490).

## 3. Transformações no pipeline (`src/`)

### 3.1 Recorte de escopo e linhas

- **Escopo** (`src/dados.py:125-129`): filtra por `cnesexecutante` conforme `escopo` do `config.yaml` (ex.: municipal). Imprime quantas linhas removeu.
- **Amostra do smoke** (`usa_amostra`), só para teste de fumaça (`src/dados.py:119-122`).
- **Coorte por corte** (`src/coorte.py`, `src/dados.py:74-88`): em cada ano de teste, só entram as CNES da coorte definida com dados
  anteriores ao corte. Isso restringe as linhas do treino e do teste.

### 3.2 Base por consulta (`src/panels.py:47-83`, `prepara_consultas_painel`)

- `dia` = data da consulta; `t0` = data de agendamento.
- `y` = `situacaoagenda == "FALTANTE"`, em int8.
- `antec = dia − t0` em dias. Deve ser ≥ 0 (há `assert`).
- `tp` = `taxa_falta_prev` = `n_faltas_previas / n_consultas_previas`, com NaN quando não há consultas prévias.
  Se a coluna não vier pronta, é calculada aqui.
- `primeira_vez`, `idade` e `dias_ate` (= `dias_ate_consulta`) são copiados como float.
- Não há imputação nesta etapa.

### 3.3 Painel CNES-dia (`src/panels.py:86-154`, `constroi_painel`)

**Agregação por CNES-dia de consultas (`_agrega_agenda`):**

- `f` = soma de `y` no dia (faltas observadas);
- `ag` = número de consultas do dia com a agenda final;
- `ag_h` = número de consultas com `antec ≥ L` (já agendadas em t − L), que é a agenda conhecida na hora da previsão;
- `m_taxaprev`, `m_primeira`, `m_idade`, `m_dias_ate` = médias por CNES-dia das variáveis individuais (`tp`, `primeira_vez`, `idade`,
  `dias_ate`), na versão final (sem sufixo) e na versão conhecida (sufixo `_h`).

**Calendário:** cada CNES vai da primeira data observada até a última data da base. Dias sem consulta recebem `f`, `ag` e `ag_h` = 0
(`src/panels.py:126-130`).

**Taxas móveis defasadas** (`L_ef = L + d`, com `d = atraso_d`, padrão 0):

- `r7`, `r14`, `r28`, `r91` = Σf / Σag nos últimos 7, 14, 28 e 91 dias, usando `shift(L_ef)`. Exigem 3, 7, 14 e 45 observações mínimas;
  onde o denominador é 0, o resultado é NaN (`src/panels.py:136-139`).
- `rdow4` = mesma razão Σf / Σag, mas só nos dias da mesma semana (defasagens múltiplas de 7 a partir de `k0 = ceil(L_ef/7)`), em 4 semanas.
  Quando não existe, recebe `r28` (`src/panels.py:140-142`).
- `rd1` = razão do dia de mesma semana em `7·k0`. Quando não existe, recebe `r28` (`src/panels.py:143-144`).
- `y_rate = f / ag` é calculada e **não entra em nenhum modelo**.

**Calendário:**

- `dow` (0 = segunda), `mes`, `ano` (não usado como feature);
- `fer` = 1 se o dia é feriado (`src/panels.py:37-44`: fixos nacionais e de Porto Alegre, mais Carnaval, Sexta-feira Santa e Corpus
  Christi; 20/nov a partir de 2024);
- `posfer` = 1 se o dia anterior é feriado.

**Filtro final:** `P = P_all[ag > 0]`. Só entram no treino e na previsão os CNES-dias com pelo menos uma consulta agendada
(`src/dados.py:133`). Os dias sem agenda são descartados aqui.

### 3.4 Matriz de cada modelo

**Matriz de série** (`src/dados.py:91-98`, `_features_serie`). Todos os modelos de taxa usam esta matriz, exceto o modelo por paciente:

- `TAXAS` = r7, r14, r28, r91, rdow4, rd1;
- `CALENDARIO` = dow, mes, fer, posfer;
- `AGENDA_VARS` = m_taxaprev, m_primeira, m_idade, m_dias_ate, com o sufixo `_h` no modo "conhecida" e sem sufixo no modo "final";
- `expo_bruta` = `ag` (modo final) ou `ag_h` (modo conhecida).

**Alvo e exposição** (`src/models/series.py:38-47`):

- exposição `expo` = `ag` (modo final) ou `ag_h × k_h` (modo conhecida), com `k_h = Σag / Σag_h` calculado só no treino
  (`src/dados.py:57-67`);
- alvo = `f / expo` (taxa de falta), usando só linhas com `expo > 0`;
- peso = `expo`, o que equivale a um Poisson com offset log(exposição);
- previsão de contagem: `mu = taxa_prevista × expo`.

**Modelos e transformações específicas:**

| Modelo | Variáveis usadas | Transformações e tratamento de ausentes |
|---|---|---|
| **naive** (`series.py:54`) | rdow4 | Sem treino. Taxa = rdow4 × expo. |
| **baseline_poisson** (`series.py:66`) | r28 | Sem treino. Taxa = r28 × expo. |
| **hgb_poisson** (`series.py:78`) | FEATS_NUM + FEATS_CAT | `HistGradientBoostingRegressor(loss="poisson")` com as variáveis numéricas brutas. `dow` e `mes` entram como números inteiros, sem one-hot. NaN tratado pelo próprio modelo. |
| **hgb_poisson_k** (`extras.py:158`) | igual ao hgb_poisson | Igual ao hgb_poisson, multiplicado por k de recalibração: razão faltas observadas ÷ previstas nos 28 dias que terminam em t − L, com mínimo de 14 dias, limitada a [0,5; 2] (`extras.py:168-193`). |
| **rf** (`series.py:99`) | FEATS_NUM + FEATS_CAT | Imputação por mediana (ajustada no treino) e `RandomForestRegressor(criterion="poisson")` com `dow` e `mes` como números. Peso = expo. |
| **poisson_reg** (`series.py:121`) | FEATS_NUM + FEATS_CAT | Numéricas: imputação por mediana e `StandardScaler`, ajustados no treino. `dow` e `mes`: one-hot (`handle_unknown="ignore"`). `PoissonRegressor` com penalização L2 (alpha). |
| **lightgbm** (`series.py:143`, opcional) | FEATS_NUM + FEATS_CAT | `LGBMRegressor(objective="poisson")`, variáveis brutas, NaN nativo. |
| **xgboost** (`extras.py:27`, opcional) | FEATS_NUM + FEATS_CAT | `count:poisson`, variáveis brutas, NaN nativo. |
| **catboost** (`extras.py:55`, opcional) | FEATS_NUM + FEATS_CAT | Perda Poisson, variáveis brutas, NaN nativo. |
| **mlp** (`extras.py:82`) | FEATS_NUM + FEATS_CAT | `dow` e `mes` viram dummies (`get_dummies`). Imputação por mediana e padronização ajustadas no treino. `MLPRegressor` sem early stopping. Taxa prevista limitada a [0, 1]. |
| **glm_nb** (`extras.py:108`) | FEATS_NUM + FEATS_CAT | Dummies de `dow` e `mes` (sem a primeira categoria de cada um). Imputação por mediana e padronização ajustadas no treino. Constante incluída. Modela **contagens** `f` com offset log(expo) e família Binomial Negativa (NB2, alpha da grade). Se o ajuste falha, cai para r28. |
| **ets** (`estatisticos.py:134`) | série diária da taxa de cada CNES | Série `y = f / ag` por CNES (com agenda final, em dias passados), NaN onde não há agenda. Holt-Winters aditivo com nível e sazonalidade semanal (alpha e gamma da grade). Nível inicial = média dos 28 primeiros dias. Previsão a L passos, limitada a [0, 1]. |
| **sarima** (`estatisticos.py:150`) | série diária da taxa de cada CNES | SARIMA(p,1,q)(P,1,Q,7) sobre os últimos 730 dias do treino. Dias fora do treino são NaN. Filtro de Kalman aplicado com os parâmetros do treino. Fallback para r28 se não convergir. |
| **sarimax** (`estatisticos.py:193`) | igual ao sarima + exógenas | Exógenas: `m_taxaprev`, `m_primeira`, `m_idade` (`EXOG = AGENDA_VARS[:3]`, sem `m_dias_ate`), padronizadas com média e desvio do treino, NaN → 0. |

**Modelo por paciente** (`src/models/paciente.py`, `src/dados.py:142-147`):

- Não usa a matriz de série. Usa uma linha por consulta, com as 30 variáveis de `config.yaml` (bloco `paciente`):
  - 19 numéricas: idade, n_consultas_previas, n_faltas_previas, taxa_falta_prev, primeira_vez,
    n_consultas_previas_especialidademae, n_faltas_previas_especialidademae, n_consultas_previas_motivo,
    n_faltas_previas_motivo, dias_ate_consulta, favela_comunidade, taxa_alf_geral, renda_media_winsor, distancia_km,
    tempo_caminhando, distancia_km_trans_publico, tempo_desloc_trans_publico, quant_bus, baldiacoes;
  - 11 categóricas: falta_recente, ultima_previa_mesma_especialidademae, ultima_previa_mesmo_motivo, especialidademae,
    cid_1dig, dia_semana, mes_consulta, corregulador, coordenadoria_saude, distrito_sanitario, cnesexecutante.
- `taxa_falta_prev` é calculada aqui como `n_faltas_previas / n_consultas_previas` (NaN se não há consultas prévias).
- Categóricas viram dtype `category`. Não há imputação nem padronização: NaN vai direto para o HGB.
- `HistGradientBoostingClassifier(early_stopping=False, categorical_features="from_dtype")`, com alvo `y` por consulta.
- Previsão do CNES-dia: `mu = k_h × Σ p_consulta`, somando as probabilidades das consultas conhecidas em t − L.
  No modo final, `k_h = 1`.

**Variáveis fora de todos os modelos:** `status`, `possuiunidadeindicada` (excluídas por momento de disponibilidade não confirmado),
`cnes_solicitante`, `unidadesolicitante`, `unidadeexecutante`, `idsolicitacao`, `idpaciente`, `rid`, `t0` (só como referência
temporal), `dia_semana` nos modelos de série (que usam `dow` do painel), `ano`, `y_rate`, `antecedencia_agendamento`.

## 4. Distribuição preditiva e métricas (todos os modelos)

- Todos os modelos produzem uma média `mu` por CNES-dia. Esta é transformada em uma distribuição Binomial Negativa comum,
  com `Var = mu + phi·mu²` (`src/models/base.py:96-102`, `src/metrics.py`).
- `phi` é estimado por momentos nas previsões dos folds internos anteriores ao teste (`src/experimento.py:111`, `src/metrics.py:31-45`).
  Nunca usa o teste.
- `mu` tem piso numérico (`MU_MIN`) para evitar zeros.

## 5. Pontos de atenção para a Metodologia

1. O painel descarta CNES-dias sem agenda (`ag > 0`). As métricas valem para dias com pelo menos uma consulta agendada.
2. Os modelos ETS e SARIMA usam a taxa `f / ag` com a **agenda final** do dia passado como série histórica. Isso é observado só depois do dia,
   então não vaza para o futuro. Mas a exposição de previsão, no modo conhecida, é `ag_h × k_h`.
3. Os modelos de série usam médias de variáveis individuais por CNES-dia (`m_*`), não variáveis de cada paciente.
   Só o modelo por paciente usa as variáveis individuais.
4. `dow` e `mes` são numéricos nos modelos de árvore e em HGB/RF, e viram dummies nos modelos lineares e MLP.
5. A origem das variáveis de censo, distância e natureza jurídica não está versionada. Não foi possível verificar essas transformações.
6. Não foi inspecionado o arquivo `06_modelo_paciente.py` (não versionado), que pode conter outra lista de variáveis.
