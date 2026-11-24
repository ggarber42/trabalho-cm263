# Métodos

*Versão 2 (rascunho para revisão), 22/11/2026. Estrutura pelos itens do TRIPOD+AI (Collins et al., BMJ 2024;385:e078378). Itens 1–4 (título, resumo, introdução, objetivos) e 20–27 (resultados e discussão) não estão nesta seção. Campos marcados com [PREENCHER] dependem de informação que não está nos arquivos do projeto.*

*Números extraídos de `tabelas/` e `resultados/municipal_completo/`, execução do perfil completo (draco3, job 838448). Manifesto em `manifestos/ingestao_perfil_completo.json`.*

## 1. Objetivo e desenho do estudo

**Objetivo (item 4).** Comparar algoritmos de previsão do número de faltas em consultas ambulatoriais agendadas de atenção secundária, em Porto Alegre, sob o mesmo protocolo de validação temporal, e estimar o desempenho para dois horizontes operacionais (L = 3 e L = 14 dias).

**Desenho (itens 3a, 4).** Estudo de desenvolvimento e avaliação de modelos de previsão com validação temporal interna (divisão por tempo, sem dados de teste no ajuste). Não há validação externa em outra cidade ou serviço.

**Unidades de análise.**
- *Consulta:* probabilidade de falta de cada consulta agendada. Usada pelo modelo por paciente (`paciente_hgb`).
- *CNES-dia:* número de faltas por estabelecimento executante e dia. Previsto diretamente pelos modelos de série, ou obtido pela soma das probabilidades das consultas já agendadas (`paciente_hgb`).

**Horizonte (L).** Dias de antecedência da previsão, em relação ao dia da consulta. L = 3 é o caso operacional, quando a agenda e a decisão de overbooking acontecem. L = 14 é sensibilidade. A justificativa do L principal está em `entregaveis/justificativa_horizonte.md`: 92,4% das consultas finais já estavam agendadas com 3 dias de antecedência e 56,0% com 14 dias.

## 2. Fontes de dados (item 5a)

Os dados vêm de quatro fontes, todas de registros administrativos. Nenhuma delas é ensaio clínico.

| Fonte | Conteúdo | Uso | Origem no projeto |
|---|---|---|---|
| Sistema GERCON (Central de Regulação, Porto Alegre) | Solicitações e agendamentos ambulatoriais, uma linha por consulta agendada | Desfecho, variáveis do paciente e da solicitação, histórico do paciente | Arquivos anuais `gercon_solicitacoes_*.csv`, unificados por `Unificacao_GERCON_Porto_Alegre.ipynb` |
| CNES (Cadastro Nacional de Estabelecimentos de Saúde), base de 09/2026 | Estabelecimentos (`tbEstabelecimento202609.csv`) e natureza jurídica (`tbNaturezaJuridica202609.csv`) | Identificação do executante, natureza jurídica, razão social | `preprocessar_solicitacoes.ipynb` |
| Censo territorial de APS | Favela/comunidade, taxa de alfabetização, renda média (winsorizada), coordenadoria e distrito sanitário, por unidade | Variáveis de território do **solicitante** | Repositório `abueno78/censo-territorios-aps`, aba `Indicadores_Completos_v2` |
| Rotas (Google Routes API) | Distância e tempo a pé e de ônibus entre o endereço do solicitante e o do executante, quantidade de ônibus | Variáveis de distância e transporte | `distancias.r`; base `dados/distancias/base_distancias_onibus.parquet` |

**Datas (item 5b).** Solicitações e consultas de 03/01/2018 a 31/03/2026. O período de 2018 tem cobertura parcial (36.958 consultas na base completa) e foi mantido porque alimenta o histórico do paciente. Os testes usam 2023 a 2026. A data de referência das rotas é fixa, 03/11/2026, com partida às 09:00 para o trajeto de ônibus.

**Representatividade (item 5a).** Os dados representam as solicitações de unidades básicas de Porto Alegre encaminhadas para atenção secundária da rede. Não representam a demanda de outros municípios nem a de serviços de atenção primária.

## 3. Ambiente de cuidado e elegibilidade (itens 6a, 6b)

**Ambiente (item 6a).** Atenção ambulatorial secundária em Porto Alegre. O solicitante é uma unidade de atenção básica (centro de saúde/UBS). O executante é um estabelecimento da rede de atenção secundária. A base completa tem 221 estabelecimentos executantes.

**Critérios de elegibilidade (item 6b).** Aplicados em duas etapas:

1. *Filtros de registro (`preprocessar_solicitacoes.ipynb`, células 9 e 31):*
   - solicitante com município Porto Alegre (`municipiosolicitante`);
   - tipo de unidade solicitante = CENTRO DE SAUDE/UNIDADE BASICA;
   - complexidade = MEDIA;
   - data de agendamento preenchida (`datahora_consulta`);
   - desfecho REALIZADA ou FALTANTE;
   - solicitante com dados do censo (`taxa_alf_geral` preenchida). O notebook remove primeiro as consultas sem censo e fora do padrão de nome APS, e depois todas as que ainda estão sem censo; o resultado final exige dado do censo.
2. *Recorte municipal (escopo, `flag_municipal.csv`):* executante com natureza jurídica Município ou Fundação municipal, ou gestor 431490, ou razão social da Prefeitura. Esse critério resulta em 111 CNES. A lista de nomes usada no notebook (seção 5, passo 4) seleciona exatamente os mesmos 111 CNES.

**Fluxo de seleção.** As etapas da tabela `tabelas/fluxograma_selecao.csv` estão registradas. Os filtros do passo 1 não puderam ser recontados, porque o arquivo bruto de solicitações não está no repositório.

| Etapa | Consultas | CNES | Removidas |
|---|---|---|---|
| Base analítica (parquet, com 2018) | 983.875 | 221 | — |
| Escopo municipal | 360.878 | 111 | 622.997 |

**Tratamento do recorte.** O escopo é um parâmetro de `config.yaml` (`escopo: municipal`), não uma exclusão fixa. Não houve exclusão por inatividade, volume mínimo ou código placeholder. A CNES 2237571 (GHC, federal) está fora do recorte municipal.

**Tratamento de tratamentos (item 6c).** Não há informação sobre tratamentos recebidos nos registros. Esse item não se aplica.

## 4. Pré-processamento e controle de qualidade (item 7)

O pré-processamento segue três etapas, executadas em ordem.

**4.1 Unificação (`Unificacao_GERCON_Porto_Alegre.ipynb`).** Lê os CSVs anuais (separador `;` ou `,`; codificação UTF-8 ou latin-1), filtra o município solicitante por Porto Alegre (nome normalizado ou código IBGE 4314902/431490) e concatena os anos. *Observação:* o título do notebook diz "Município Executante", mas o filtro é sobre `municipiosolicitante`. A Metodologia descreve o que o código faz.

**4.2 Pré-processamento e variáveis (`preprocessar_solicitacoes.ipynb`).**
1. Remoção de espaços e conversão de textos vazios para nulo.
2. Datas em formato misto convertidas com `errors="coerce"`; `diasfilaespera` e `idade` convertidos para número.
3. Identificadores e códigos (CNES, id de solicitação, id de paciente, código IBGE) mantidos como texto, com zeros à esquerda até 7 dígitos.
4. Filtros de registro listados na seção 3.
5. Junção com o CNES: natureza jurídica e razão social do executante (junção pela esquerda, `validate="m:1"`).
6. Junção com o censo pelo CNES do **solicitante** (`cnes_solicitante`). A verificação `validate="m:1"` foi aplicada. A correção de um valor de distrito sanitário da CNES 2264374, cujo conteúdo era uma mensagem administrativa ("Fechada para reunião"), foi feita no censo com o valor HUMAITA/NAVEGANTES.
7. Junção com as distâncias: pelo par de endereços (solicitante, executante). Quando o status é `sem_rota` e a distância está vazia, os valores de distância e tempo são zerados, conforme a regra do script R (mesmo endereço).
8. Variáveis derivadas (seção 5).
9. Seleção de 41 colunas finais.

**4.3 Script de distâncias (`distancias.r`).** Para cada par de endereços, consulta a API de rotas a pé e de ônibus. Pares com endereço igual recebem zero sem consulta (`mesmo_local`). Falhas temporárias são repetidas na execução seguinte. A base é salva a cada bloco, em arquivo temporário trocado no final.

**Controles de qualidade registrados.**
- Chaves e alinhamento: `assert` de unicidade do mapa de CNES e da base de distâncias no notebook.
- Pares de endereço: status de cada par é gravado em `status`, com a contagem por status.
- Distribuição de faltas por paciente: análise descritiva no notebook (não gera variável).
- Consultas com data de agendamento nula e com desfecho fora de REALIZADA/FALTANTE: removidas antes da análise.
- Consistência entre `t0` e a consulta: `antec = dia − t0` é verificado como não negativo no pipeline.

**Ausência de avaliação por grupos sociodemográficos (item 7).** Não foi testado se a qualidade dos dados difere entre grupos sociodemográficos. Esse é um ponto a relatar como limitação.

## 5. Desfecho (itens 8a, 8b, 8c)

**Desfecho (8a).** Variável binária por consulta: y = 1 se `situacaoagenda` = FALTANTE, e y = 0 se REALIZADA. Esse desfecho é o registro administrativo do status da consulta na data agendada. O horizonte de previsão é o dia da consulta, com previsão feita L dias antes.

No recorte municipal, a distribuição é: REALIZADA 281.622 e FALTANTE 79.256 (21,96%). *Observação:* a contagem de todas as categorias de `situacaoagenda` antes dos filtros não pôde ser refeita, porque o arquivo bruto não está no repositório. O parquet tem apenas REALIZADA e FALTANTE.

**Avaliação do desfecho (8b).** O desfecho é registro administrativo. Não há avaliação subjetiva. Esse item não se aplica.

**Cegamento (8c).** Não se aplica a registros administrativos.

**Premissa de atraso de registro.** A base não tem a data de registro do desfecho. Assume-se d = 0, ou seja, o desfecho do dia t − L já está registrado em t − L. Essa premissa está pendente de confirmação com o serviço e é tratada como limitação.

## 6. Preditores (itens 9a, 9b, 9c)

**Escolha dos preditores (9a).** Os candidatos foram escolhidos *a priori*, por (i) disponibilidade no momento do agendamento e (ii) justificativa clínica e de gestão [PREENCHER referências da literatura sobre histórico de falta; o notebook cita a literatura sem referência]. Não houve seleção por dados. Das 32 candidatas iniciais, `status` e `possuiunidadeindicada` foram removidas porque podem conter informação posterior ao agendamento e o momento de preenchimento não foi confirmado. As 30 restantes entram em todos os modelos por paciente. A poda para 6 variáveis de um script legado foi tratada como hipótese e não foi usada.

**Momento de disponibilidade (9b).** O ponto de referência é `t0`, a data de agendamento (`data_solicitacao + max(diasfilaespera, 0)`). Só entram variáveis conhecidas até t − L.

| Grupo | Variáveis | Momento de disponibilidade |
|---|---|---|
| Histórico do paciente (8) | `n_consultas_previas`, `n_faltas_previas`, `taxa_falta_prev`, `primeira_vez`, `n_consultas_previas_especialidademae`, `n_faltas_previas_especialidademae`, `n_consultas_previas_motivo`, `n_faltas_previas_motivo` | Eventos com `data_ev < t0` |
| Última consulta anterior (2 categóricas) | `ultima_previa_mesma_especialidademae`, `ultima_previa_mesmo_motivo` | Eventos com `data_ev < t0` |
| Histórico recente (1 categórica) | `falta_recente` | Eventos com `data_ev < t0` |
| Solicitação e clínica (categóricas) | `especialidademae`, `cid_1dig`, `corregulador` | Conhecidas na solicitação |
| Agendamento (2) | `idade`, `dias_ate_consulta` | Idade na solicitação; `dias_ate_consulta` é o intervalo entre solicitação e consulta, conhecido no agendamento |
| Calendário (2 categóricas) | `dia_semana`, `mes_consulta` | Conhecidos com antecedência |
| Executante (1 categórica) | `cnesexecutante` | Conhecido |
| Território do solicitante (3) | `favela_comunidade`, `taxa_alf_geral`, `renda_media_winsor` | Dados do censo, estáticos |
| Território do solicitante (2 categóricas) | `coordenadoria_saude`, `distrito_sanitario` | Dados do censo, estáticos |
| Distância e transporte (6) | `distancia_km`, `tempo_caminhando`, `distancia_km_trans_publico`, `tempo_desloc_trans_publico`, `quant_bus`, `baldiacoes` | Calculadas entre os endereços de solicitante e executante, estáticas |

*Observação:* a variável `dias_ate_consulta` é a soma de `diasfilaespera` e `antecedencia_agendamento`. Usa-se apenas uma das duas no modelo, para evitar colinearidade perfeita.

*Observação:* `taxa_falta_prev` é calculada no pipeline como `n_faltas_previas / n_consultas_previas`, e fica NaN quando não há consultas anteriores.

**Preditores dos modelos de série.** Os modelos de série não usam as variáveis do paciente. Usam taxas de falta defasadas, calendário, variáveis médias de agenda do CNES-dia e a exposição (seção 7.2). As variáveis de censo e de distância entram **apenas** no modelo por paciente.

**Medição por especialista (9c).** Não se aplica. As variáveis são registros administrativos.

## 7. Tamanho amostral (item 10)

Não houve cálculo formal de tamanho amostral. Usaram-se todas as consultas do recorte, porque o objetivo é comparar algoritmos com o mesmo conjunto de dados, não estimar um parâmetro com precisão pré-especificada. A precisão do WAPE é reportada por intervalo de confiança (bootstrap em blocos semanais, 2.000 reamostragens).

Tamanhos de referência: 360.878 consultas e 79.256 faltas no recorte municipal (2018–2026). Os testes de 2023 a 2026 usam 8.073 CNES-dias na comparação principal (seção 9).

## 8. Dados ausentes (item 11)

| Situação | Quantidade (recorte municipal) | Tratamento |
|---|---|---|
| Distâncias e tempos sem status de rota (`status` ausente) | 762 linhas (0,21%) | Os seis campos de distância são nulos juntos. Os modelos de gradient boosting aceitam nulos. Os demais modelos usam mediana do treino (seção 9.3). |
| Pares com `sem_rota` sem distância | zerados | Conforme a regra do script R (mesmo endereço). |
| Pares `mesmo_local` | 17.381 linhas com distância zero | São o mesmo endereço, por construção. Os zeros de `distancia_km` e `tempo_caminhando` são desse grupo ou dos `sem_rota` zerados. |
| Histórico do paciente sem consultas anteriores | `taxa_falta_prev` NaN | Mantido como NaN nos modelos de gradient boosting. |
| Solicitante sem dado do censo e fora do padrão APS | removido na elegibilidade | Ver seção 3. |
| Corregulador ausente | removido na elegibilidade | Ver seção 3. |

*Não há imputação múltipla.* Os valores faltantes não foram analisados quanto a padrão (MCAR, MAR ou MNAR). Esse é um ponto a relatar como limitação.

## 9. Análises: divisão temporal, modelos e tuning (itens 12a–12g)

### 9.1 Divisão temporal (12a)

Os modelos são avaliados por **divisão temporal**, sem embaralhamento. Quatro esquemas foram testados e comparados como fatores do estudo. Todos usam o mesmo módulo (`src/splits.py`).

- **Expansiva (principal, pré-declarada).** Para o teste do ano Y, o treino é todo o histórico anterior a Y. Testes em 2023, 2024, 2025 e 2026 (este último até 31/03).
- **Holdout.** Corte único em 01/01/2023. Treino antes, teste depois.
- **Deslizante.** Treino de tamanho fixo. A janela (12, 24 ou 36 meses) é escolhida como hiperparâmetro nos folds internos.
- **Purga e embargo.** Remove do treino as consultas cujo intervalo de informação se sobrepõe ao teste. A purga P ∈ {0, 28, 90} dias e o embargo E ∈ {0, 7} dias são sensibilidade pré-declarada. O intervalo entre agendamento e consulta tem percentil 99 de 14 dias e máximo de 64 dias (`tabelas/antecedencia_agendamento_municipal.csv`). A purga recomendada, P = percentil + L, é 17 dias (L = 3) e 28 dias (L = 14) pelo percentil 99, e 67 e 78 dias pelo máximo (`tabelas/purga_recomendada_municipal.csv`).

**Escolha do principal.** O esquema expansivo foi declarado como principal antes de ver os resultados de teste, no plano de análise congelado em 17/11/2026 (`entregaveis/plano_analise.md`). Os demais são sensibilidade.

**Período comum de teste.** A comparação entre esquemas usa o mesmo período, 01/01/2024 a 31/03/2026 (5.479 CNES-dias, conferido em `tabelas/res_municipal_completo_esquemas_teste_comum.csv`). Esquemas diferentes avaliam períodos diferentes, e essa restrição evita comparar testes de períodos distintos.

**Coorte e níveis por corte.** A coorte de CNES e os níveis A, B e C são definidos com dados anteriores ao início de cada ano de teste (`src/coorte.py`). Para o teste de 2023, a coorte usa dados até 31/12/2022.

| Corte (início do teste) | CNES na coorte | Nível A | Nível B | Nível C |
|---|---|---|---|---|
| 01/01/2023 | 12 | 11 | 1 | 0 |
| 01/01/2024 | 12 | 9 | 3 | 0 |
| 01/01/2025 | 11 | 8 | 3 | 0 |
| 01/01/2026 | 10 | 7 | 3 | 0 |

A coorte definida com a série inteira (até abril de 2026) é só descrição, e não é usada para escolher modelo. Um subgrupo à parte é o Hospital Materno Infantil Presidente Vargas (CNES 2237822), com taxa de falta de cerca de 7%. Ele é reportado separadamente.

### 9.2 Modelos (12c)

Quinze modelos foram comparados nas mesmas linhas, com o mesmo protocolo de avaliação.

| Modelo | Família | Ajuste e tuning |
|---|---|---|
| `naive` | Referência ingênua | Média das faltas do mesmo dia da semana nas 4 semanas anteriores |
| `baseline_poisson` | Referência obrigatória | Taxa de falta dos 28 dias anteriores, com defasagem L, × agenda conhecida |
| `ets` | Série temporal (por CNES) | Holt-Winters aditivo com sazonalidade semanal; α e γ em grade de 5 × 4 = 20 configurações |
| `sarima` | Série temporal (por CNES) | SARIMA(p,1,q)(P,1,Q,7) na taxa; p, q, P, Q em grade de 12 configurações; sobre os últimos 730 dias do treino |
| `sarimax` | Série temporal (por CNES) | Como SARIMA, com 3 exógenas de agenda padronizadas pelo treino |
| `glm_nb` | Regressão | Binomial negativa (NB2), com offset log da exposição; α em grade de 6 valores |
| `poisson_reg` | Regressão | Poisson com penalização L2; força em grade de 6 valores |
| `hgb_poisson` | Gradient boosting (série) | HistGradientBoosting com perda Poisson; 243 configurações na grade, 24 amostradas |
| `hgb_poisson_k` | Gradient boosting com recalibração | Como `hgb_poisson`, multiplicado por k = faltas observadas ÷ previstas nos 28 dias que terminam em t − L (limitado a [0,5; 2]; 1 se houver menos de 14 dias) |
| `rf` | Floresta aleatória | Critério Poisson; 81 configurações, 24 amostradas |
| `lightgbm` | Gradient boosting | Objetivo Poisson; 243 configurações, 24 amostradas |
| `xgboost` | Gradient boosting | 216 configurações, 24 amostradas |
| `catboost` | Gradient boosting | 108 configurações, 24 amostradas |
| `mlp` | Rede neural | Uma ou duas camadas ocultas; 48 configurações, 24 amostradas; imputação e padronização só no treino |
| `paciente_hgb` | Gradient boosting por consulta | Probabilidade de falta por consulta (HGB com perda logística); 243 configurações, 24 amostradas; μ do CNES-dia = k_h × soma das probabilidades das consultas agendadas |

**Justificativa das famílias.**
- *Referências (`naive`, `baseline_poisson`)* são obrigatórias para medir ganho. O baseline Poisson é o método operacional descrito no plano.
- *Séries (`ets`, `sarima`, `sarimax`)* são o padrão para séries temporais de contagens com sazonalidade semanal. Foram testadas em todos os CNES.
- *Regressões (`glm_nb`, `poisson_reg`)* modelam contagens com covariáveis e são interpretáveis.
- *Árvores e gradient boosting (`hgb_poisson`, `hgb_poisson_k`, `rf`, `lightgbm`, `xgboost`, `catboost`)* capturam não linearidades e interações entre taxas defasadas e agenda, sem especificação manual.
- *Rede neural (`mlp`)* é uma comparação de família não-árvore.
- *Modelo por paciente (`paciente_hgb`)* usa a informação individual de cada consulta, que as séries agregadas não têm. É a única família que usa as variáveis do censo e de distância.
- *Recalibração (`hgb_poisson_k`)* ajusta o nível de previsão pela razão recente entre observado e previsto, com defasagem L. É a única forma de atualização do modelo testada (item 12f).

Os modelos de série usam o número de consultas agendadas, mas não as variáveis do paciente. O modelo por paciente soma as probabilidades das consultas já agendadas em t − L.

### 9.3 Pré-processamento específico por família (12b)

- **Gradient boosting (HGB, LightGBM, XGBoost, CatBoost, RF e `paciente_hgb`):** aceitam nulos nativamente. Sem padronização.
- **Regressões, MLP e SARIMAX:** dummies para `dow` e `mes`; imputação pela mediana do treino (`SimpleImputer`); padronização pela média e desvio do treino. Nada é ajustado no teste.
- **Séries (ETS, SARIMA):** série diária da taxa por CNES, com NaN nos dias fora do treino.
- **Exposição:** `expo = ag` (modo final) ou `ag_h × k_h` (modo conhecida), com k_h = Σag ÷ Σag_h calculado só no treino.
- **Alvo:** `f ÷ expo` nas linhas com `expo > 0`, com peso igual à exposição (equivale a um Poisson com offset log da exposição).

### 9.4 Ajuste, tuning e validação interna (12c)

**Validação interna temporal aninhada.** Para o teste do ano Y, os folds internos são os anos Y − 2 e Y − 1, construídos só com dados anteriores ao início do teste. Para Y = 2026, os folds são 2024 e 2025. Cada configuração é ajustada no fold de treino e avaliada no fold de validação.

**Critério de tuning.** WAPE pooled nos folds internos para os modelos de série; Brier por consulta para `paciente_hgb`.

**Orçamento.** Até 24 configurações por modelo ajustável, amostradas de uma grade com semente fixa. Grades menores são percorridas inteiras (ETS com 20, SARIMA com 12, GLM NB e Poisson com 6). O orçamento não é idêntico entre todos os modelos; a diferença é pequena e está declarada.

**Parada e iterações.** Nenhum modelo usa early stopping. O número de iterações é escolhido nos folds internos.

**Seleção de variáveis.** Nenhuma. Todas as 30 variáveis candidatas entram no modelo por paciente. O teste de remoção do censo e da distância ainda não foi feito (seção 11).

**Heterogeneidade entre CNES (12d).** O modelo não modela agrupamento por CNES (sem efeitos aleatórios). A heterogeneidade é reportada por nível (A e B) e por CNES nos resultados, mas não é quantificada em um modelo multinível. TRIPOD-Cluster não foi aplicado.

**Sementes.** Semente-mestra única (20261116), derivada para cada unidade de trabalho. Nenhum modelo com `random_state` nulo. Sensibilidade a sementes: cinco sementes para `rf`, `mlp` e `catboost`, em L = 3 e L = 14.

### 9.5 Métricas e distribuição preditiva (12e)

Para comparar modelos que produzem previsão pontual e probabilística, todos geram uma **distribuição preditiva Binomial Negativa** com média μ e dispersão φ. A dispersão é estimada por momentos (φ = max(0; Σ[(y − μ)² − μ] ÷ Σμ²)) em previsões **fora da amostra dos folds internos**, nunca no teste.

- **Primárias:** WAPE = Σ|y − ŷ| ÷ Σy (ponto); CRPS da distribuição NB (probabilístico).
- **Secundárias:** viés = Σŷ ÷ Σy; cobertura dos intervalos de 80% e 50%; largura do IC80; ECE de P(faltas ≥ k) para k ∈ {1, 2, 3, 5, 8}; AUC de P(≥ 1 falta) por CNES-dia.
- **Por consulta (apenas `paciente_hgb`):** Brier, AUC e log loss, por ano de teste.

**Testes.** Diebold-Mariano com perda agregada por semana, variância de Newey-West (Bartlett, largura automática) e correção de Harvey-Leybourne-Newbold. IC 95% do ganho de WAPE por bootstrap em blocos semanais (2.000 reamostragens).

**Múltiplas comparações.** Duas famílias: (i) cada modelo contra o baseline (14 comparações); (ii) cada modelo contra o melhor modelo identificado no teste (14 comparações). A correção de Holm é aplicada dentro de cada família. Não há correção entre famílias. O melhor modelo é identificado no mesmo teste, portanto a comparação com ele é descritiva.

**Mesmas linhas.** Os modelos são comparados nas mesmas linhas (interseção dos pares CNES-dia que todos produzem). Na comparação principal (expansiva, L = 3, modo conhecida), a interseção tem **8.073 CNES-dias**; o baseline isolado tem 8.224 linhas válidas. A diferença de 151 linhas coincide com as falhas de ajuste do SARIMAX; as 86 falhas do SARIMA também entram na interseção. A tabela `tabelas/res_municipal_completo_linhas_perdidas.csv` registra as linhas perdidas por modelo.

### 9.6 Recalibração e atualização (12f)

Os modelos são ajustados uma vez por ano de teste, com dados até o início do ano. A atualização acontece em duas formas:
- *Expansão do treino:* a cada ano, o treino inclui o ano anterior (esquema expansivo).
- *Recalibração k:* `hgb_poisson_k` multiplica a previsão pela razão recente entre observado e previsto (28 dias, defasagem L, limitada a [0,5; 2]).

Nenhuma outra recalibração foi feita.

### 9.7 Como as previsões são calculadas (12g)

- *Séries:* μ = taxa prevista × exposição do CNES-dia.
- *Modelo por paciente:* μ do CNES-dia = k_h × Σ p_consulta, em que p_consulta é a probabilidade de falta de cada consulta agendada em t − L e k_h é o fator de reescala estimado no treino.
- *Distribuição:* NB(μ, φ), com φ estimado fora da amostra.

O código está no repositório, em `src/models/` e `src/dados.py`. As fórmulas acima estão em `entregaveis/variaveis_e_transformacoes.md`.

## 10. Estratégias testadas (resumo para o leitor)

| Fator | Níveis | Papel | Pré-declarado |
|---|---|---|---|
| Horizonte L | 3 (principal), 14 | Operacional e sensibilidade | Sim (plano 17/11/2026) |
| Esquema de divisão | Expansiva (principal); holdout, deslizante, purga P0/P28/P90 | Sensibilidade | Sim |
| Modo da agenda | Conhecida (principal, reescalada por k_h); final (sensibilidade, otimista) | Sensibilidade | Sim |
| Escopo | Municipal | Principal | Parâmetro; escopo geral não coberto |
| Coorte | Por corte anual, com dados anteriores | Principal | Sim |
| Exclusão do Materno Infantil | Com e sem | Sensibilidade | Sim |
| Recalibração | `hgb_poisson_k` versus `hgb_poisson` | Variante | Sim |
| Limiar do overbooking | q = 0,20 (principal), q = 0 (referência) | Utilidade de decisão | Sim |
| Custo relativo do excesso | c ∈ {1, 2, 3} | Análise de decisão | Sim |

**Não testado.** Escopo geral (todas as CNES), coorte global, remoção de variáveis suspeitas e remoção das variáveis do censo e de distância. SHAP e ALE não foram implementados.

## 11. Análises complementares

- **Calibração:** cobertura observada dos intervalos e ECE. Figura `figuras/completo/calibracao_cobertura`.
- **Importância por permutação:** para os modelos de árvore de série, o aumento do WAPE quando um grupo de variáveis é embaralhado nas linhas de teste (taxas defasadas, agenda, calendário). Não há importância para o modelo por paciente. Ver `tabelas/importancia_municipal_completo*.csv`.
- **Ranking entre fatores:** correlação de postos (Spearman e Kendall) entre as ordens de modelos em cada fator (L, esquema, modo, exclusão do Materno Infantil).
- **Utilidade de decisão (overbooking):** regra de agendar k encaixes extras por CNES-dia, com k o quantil q da distribuição preditiva. Métricas: consultas recuperadas, excesso, dias com excesso e benefício líquido com custo c. É uma simulação descritiva sobre o teste.

## 12. Software e reprodutibilidade (itens 18e, 18f)

- **Linguagens:** Python 3 (pipeline) e R (script de distâncias e pré-processamento do censo). Notebooks no Google Colab para unificação e pré-processamento.
- **Bibliotecas:** versões fixadas em `requirements.lock` (numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, statsmodels 0.15.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, scipy 1.18.0). A execução completa foi feita com Python 3.12.16, o que difere do 3.12.3 do lock. A diferença deve ser conferida antes de reproduzir.
- **Hardware:** CPU, sem GPU. 16 processos em paralelo; pico de 1,2 GB por processo. Tempo total de 187,5 minutos.
- **Caminhos e parâmetros:** todos os parâmetros estão em `config.yaml`. Não há caminho absoluto no código. Cada etapa grava log em `logs/`, e as saídas principais têm hash SHA-256 no manifesto.
- **Dados (item 18e):** os dados individuais são registros de pacientes e não são públicos. A disponibilidade está restrita pela LGPD. Os agregados e as tabelas de resultados estão no repositório. [PREENCHER: forma de acesso aos dados brutos, se houver.]
- **Código (item 18f):** repositório git em `serie_temporal/`. [PREENCHER: URL do repositório público e tag da versão do artigo.]
- **Plano (item 18c):** plano de análise interno, congelado em 17/11/2026 (`entregaveis/plano_analise.md`). Não foi registrado em repositório público.
- **Registro do estudo (item 18d):** não registrado. [PREENCHER se houver registro.]
- **Financiamento (item 18a):** [PREENCHER].
- **Conflitos de interesse (item 18b):** [PREENCHER].
- **Participação de pacientes e público (item 19):** [PREENCHER; provavelmente não houve.]
- **Aprovação ética (item 17):** [PREENCHER: número do parecer do CEP/UFRGS ou dispensa, com a condição de consentimento.]

## 13. Limitações do protocolo

As limitações de interpretação dos resultados estão na subseção "Limitações" de Resultados. Aqui ficam as do protocolo e dos dados.

1. **Atraso de registro do desfecho:** premissa d = 0, não confirmada com o serviço (item 7, seção 5).
2. **Momento de `idade`:** premissa de idade na solicitação, não confirmada.
3. **Origem dos dados:** o notebook `preprocessar_solicitacoes.ipynb` gera o arquivo com o nome `gercon_poa_censo_distancias_com_natureza_juridica.parquet`, mas o arquivo da raiz do projeto (data de 10/11/2026) tem 221 CNES e nenhum filtro de recorte, o que indica que é uma versão anterior ao notebook atual (22/11/2026). O script R lê `gercon_poa_censo_filtrado.parquet`, que nenhum notebook do repositório gera. A cadeia completa dos dados brutos até o parquet final não está reexecutável com os arquivos atuais.
4. **Histórico restrito ao filtrado:** o histórico do paciente é calculado sobre as consultas que passaram pelos filtros de registro (complexidade média, solicitante de APS, etc.). Consultas fora desses filtros não entram no histórico, o que pode subestimar as faltas anteriores.
5. **Censo do solicitante:** as variáveis de território vêm da unidade solicitante, não do executante. Interpretam-se como características da população encaminhada, não do local da consulta.
6. **Filtros fixos no código:** a lista de nomes de razão social, o padrão de APS e a correção da CNES 2264374 estão no notebook. A regra do projeto é que listas de estabelecimentos fiquem em arquivos de dados, não no código.
7. **Distâncias:** a rota de ônibus usa horário de partida fixo (09:00, 03/11/2026). Não captura variação de tráfego e frequência de linhas ao longo do ano.
8. **Municipal versus geral:** os resultados valem para o recorte municipal. O escopo geral não foi analisado.
9. **Orçamento de busca:** não idêntico entre modelos (seção 9.4).
10. **Viés de seleção:** a população é de unidades básicas de Porto Alegre encaminhando para a rede municipal. Não se sabe se o padrão vale para outros serviços.
