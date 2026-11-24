# Decisões do redesenho metodológico (iniciado em 22/11/2026)

Registro das decisões tomadas ponto a ponto com o usuário. Substitui partes do `CLAUDE.md`
(h = 3 dias, L ∈ {3, 14}, quatro esquemas de divisão, variáveis do censo e distâncias) quando
o desenho completo for aprovado. Até lá, o pipeline antigo (`src/`, `etapas/`) não é alterado.

## Objetivo (texto do usuário)
Desenvolver e validar modelos prognósticos de previsão de faltas, comparando diferentes
algoritmos de aprendizado de máquina, em consultas especializadas de pacientes encaminhados
pela APS no município de Porto Alegre - RS. Análise no nível dos estabelecimentos executantes
de gestão municipal, com abordagem de série temporal, a partir dos dados abertos do GERCON,
no período de 2018 a 2026.

## D1. Fonte de dados
Só o GERCON público (`../gercon/gercon_solicitacoes_executante_porto_alegre_2018_2026.csv`).
O cadastro CNES (`tbEstabelecimento`, `tbNaturezaJuridica`) entra só para marcar as CNES de
gestão municipal. Saem censo e distâncias.

## D2. Desfecho e elegibilidade
- Desfecho: `situacaoagenda`; y = 1 se FALTANTE, 0 se REALIZADA.
- Inclusão: município solicitante = Porto Alegre; tipo de unidade solicitante = centro de
  saúde/UBS; complexidade média; `datahora_consulta` preenchida; situação REALIZADA ou FALTANTE.
- CNES municipal: natureza jurídica Município ou Fundação municipal, gestor 431490 ou razão
  social da Prefeitura.
- O critério "dados do censo disponíveis" foi retirado; contagens do fluxograma serão recalculadas
  por script.
- Limitação: consultas passadas com AGENDA_CONFIRMADA (ex.: 52 mil em 2018) são excluídas;
  reportar contagens por ano.

## D3. Histórico do paciente (opção B)
Calculado só nas consultas que passaram em todos os filtros, com `data_ev < t0`,
`t0 = data_solicitacao + max(diasfilaespera, 0)` (data de agendamento). Limitação: faltas em
CNES não municipais não entram no histórico.

## D4. Período de análise
- Série de análise: 2019-04 a 2024-12. 2018-01 a 2019-03 só alimenta o histórico
  (volume baixo e mudança de patamar em 2019-04).
- Justificativa: os arquivos de 2018–2024 foram extraídos em 07/01/2025 e o de 2025 em
  01/04/2026; não há arquivo de pedidos de 2026. Consultas posteriores à extração ficam sem
  desfecho ou desaparecem: a partir de 2025-02 não há nenhuma consulta de pedidos anteriores
  a 2025. 2025–2026 entram como limitação.
- `data_extracao` é usada só como metadado para justificar o corte, nunca como preditora.

## D5. Formatação
- 2025 traz datas sem hora: truncar todas as datas ao dia, em todos os anos.
- CNES com 6 dígitos (zero à esquerda perdido) em anos antigos: completar até 7 dígitos antes
  do merge, imprimindo quantas linhas foram corrigidas.

## D6. Unidade de análise
- Principal: CNES × semana (contagem de faltas).
- Especialidade (`especialidademae`) como detalhamento: soma, por CNES × especialidade × semana,
  das probabilidades do modelo por consulta. Não há séries por especialidade nem reconciliação.
- Medido (recorte municipal antigo, 2019-04 a 2024-12): 274.914 consultas, 300 semanas,
  106 CNES, 276 combinações CNES × especialidade; 43% das CNES-semana ativas têm 0 falta.

## D7. Comparação justa
Para o usuário, "justa" quer dizer que os modelos tradicionais de série temporal recebem o mesmo
cuidado que os de ML: análise das características da série (tendência, sazonalidade anual,
quebras, variância), testes de estacionariedade, transformação quando necessária, escolha de
ordens e hiperparâmetros por critério de informação e validação temporal só no treino, com
orçamento comparável ao do ML, e diagnóstico de resíduos. Séries com muitos zeros pedem
modelos de contagem intermitente (Croston/TSB) entre os tradicionais.

## D8. Semana, origem e horizonte (recomendação aceita pelo usuário)
- Semana de segunda a domingo (ISO). Origem da previsão: fim do domingo da semana anterior.
- Horizonte principal h = 1 semana; h = 2 semanas como sensibilidade.
- Informação permitida: tudo conhecido até a origem, inclusive a agenda já marcada para a semana
  prevista (consultas com t0 ≤ origem) e o histórico dos pacientes (data_ev < t0).
- Substitui L = 3 e L = 14 dias.
- A verificar: fração das consultas da semana já agendadas na origem (t0 ≤ origem). Se for bem
  menor que 1, as variáveis de agenda usam só as consultas conhecidas na origem.

## D9. Blocos de informação (opção 1)
- Bloco A: todos os modelos usam só a série passada de faltas da CNES e calendário
  (naive sazonal, ETS, SARIMA, Croston/TSB, regressão de contagem, árvores, MLP).
- Bloco B: série + agenda conhecida na origem + histórico dos pacientes (SARIMAX/regressão com
  exógenas, ML, modelo por consulta agregado). O detalhamento por especialidade sai do modelo
  por consulta do bloco B.
- Ambos pré-declarados; os blocos não são escolhidos pelo teste.

## D10. Análise exploratória define o esquema de avaliação
A caracterização das séries (volume, intermitência, sazonalidade, ACF/PACF, média × variância,
estacionariedade, fração da agenda conhecida) usa só dados de treino (2019-04 a 2023-12) e decide
o esquema de divisão pela regra de Cerqueira et al. (2020), fixada antes de rodar.

## D12. Variáveis do bruto (decisão do usuário)
- `idpaciente`: só para calcular o histórico; nunca entra na modelagem nem nas saídas.
- `municipiosolicitante`, `tipounidade`, `complexidade`: excluídas da modelagem; usadas só
  como critérios de seleção (ficam constantes depois dos filtros).
- `possuiunidadeindicada`: incluída como preditor (atributo do paciente, segundo o usuário;
  revoga a exclusão do método antigo).
- `cnes_solicitante`: incluída como preditor (território/UBS de origem).
- Cardinalidade no treino (2019-04 a 2023-12, APS média, todas as executantes): CID completo
  7.459 categorias; CID3 1.653 (761 com < 30 casos); capítulo 26; cnes_solicitante 153;
  especialidade 251; especialidademae 66; corregulador 6; possuiunidadeindicada 2 (20% SIM).

## D13. Categóricas de alta cardinalidade
- CID: uma única variável, o capítulo (1º caractere, 26 categorias), como categoria direta.
  CID completo e CID3 não entram.
- `cnes_solicitante` e `especialidade`: target encoding temporal suavizado, com estatísticas
  estimadas só no treino (dentro do treino, cada linha usa só consultas anteriores à sua origem,
  para não codificar com o próprio desfecho); no teste, aplicam-se as estatísticas do treino.
  Categoria não vista no treino recebe a taxa geral do treino. Parâmetro de suavização fixado a priori.
- `especialidademae`, `corregulador`, `possuiunidadeindicada`: categoria direta.

## D14. Histórico do paciente: 5 variáveis
`n_consultas_previas`, `n_faltas_previas`, `dias_desde_ultima_falta` (em faixas `falta_recente`
para regressões e MLP), `n_faltas_previas_especialidademae`,
`ultima_previa_mesma_especialidademae`. Saem `primeira_vez` (= n_consultas_previas == 0),
`n_consultas_previas_especialidademae` e todo o grupo "motivo" (redundante com especialidade).
Agregados do bloco B: proporção com falta anterior, média de faltas anteriores, proporção cuja
última consulta na especialidade foi falta.

## D11. Baseline único
Média móvel das faltas da CNES nas 4 semanas anteriores (s−4 a s−1), igual nos blocos A e B.
Janela de 4 semanas declarada previamente por critério operacional, não escolhida pelo
desempenho. Saem o Poisson 28 dias × agenda e o naive do mesmo dia da semana.
- Análise formal de estacionariedade e escolha do esquema de avaliação (Cerqueira, Torgo e
  Mozetič 2020, doi:10.1007/s10994-020-05910-7).

## D15 — Um único conjunto de informação para todos os modelos (substitui os blocos A/B da D9 e a referência da D11)

Decisão do usuário (22/11/2026): o bloco A (sem agenda) foi removido. Em previsão operacional, o
modelo usa tudo o que é conhecido na origem: passado da série, calendário e covariáveis conhecidas
de antemão (agenda marcada até a origem e agregados dos pacientes agendados).
- **Alvo e avaliação:** continuam sendo o número de faltas por CNES-semana. A taxa é só o caminho interno.
- **Referência dos testes:** `referencia_taxa_agenda` = taxa de faltas por consulta agendada das 4
  semanas encerradas (suavizada para a de 13 semanas com peso m = 20 consultas) × agenda conhecida da
  semana-alvo. `referencia_mm4` (média móvel de faltas) fica só descritiva, fora dos testes DM.
- **ETS e Croston/TSB:** modelam a taxa suavizada por consulta agendada e voltam à contagem com a agenda.
- **SARIMAX:** agenda, agregados, Fourier e feriados como regressores; o SARIMA sem agenda saiu.
- **Globais (GLM, HGB, floresta, MLP):** alvo = taxa y/e com peso e = max(n_agenda, 1); para o GLM
  de Poisson isso equivale ao offset log(e).
- **Calendário ampliado:** dias úteis, ponte (feriado na terça ou quinta), feriado na semana anterior e
  na seguinte, recesso (24/12 a 01/01); 20/11 como feriado só a partir de 2024 (Lei 14.759/2023).
- **Horizontes:** h = 1 e h = 2 mantidos. Leitura operacional: a previsão é entregue na segunda-feira
  de manhã, com os dados até domingo; h = 1 prevê a própria semana e h = 2 a semana seguinte.
  O principal continua h = 1 (a confirmar com o usuário).

## D16 — Unidade CNES × especialidade-mãe × semana; h = 2 principal; só teste 2024 (substitui D6 e parte de D8)

Decisões do usuário (22/11/2026):
- **Unidade de modelagem:** série = CNES × especialidade-mãe, semanal (substitui o CNES × semana).
  Agenda conhecida, agregados dos pacientes e taxas são calculados por série. O total da CNES é a
  soma das especialidades e é avaliado à parte (nível "cnes", só métricas pontuais).
- **Modelos locais (ETS, Croston/TSB, SARIMAX):** só nas séries com ≥ 104 semanas ativas no treino;
  as demais ficam com referências e modelos globais, e as linhas perdidas são registradas.
- **Horizonte:** h = 2 principal (previsão entregue na segunda para a semana seguinte), h = 1 secundário.
- **Teste:** só 2024 (treino até 31/12/2023); a sensibilidade em 2023 saiu para reduzir o tempo.
  "Avaliação em um único ano" entra como limitação.
- **Validação interna:** mantido o holdout repetido com cortes sorteados (escolhido pela regra da
  exploratória, D10); o usuário preferiu cortes sorteados aos fixos.
- **Métrica adicional:** MAE, porque o WAPE fica instável em séries com poucas faltas.

## D17 — Estratos de CNES por consistência temporal; pré e pós-processamento comum

Decisões do usuário (22/11/2026):
- **Estratos:** "consistentes" = CNES ativa (≥ 1 consulta) em ≥ 94 das últimas 104 semanas do treino
  E com ≥ 2 faltas por semana em média nessa janela; as demais formam o estrato "outros".
  Calculado só com o treino (até 31/12/2023). Na estimativa preliminar: 17 de 103 CNES, 83% das faltas.
- **Consistentes:** todos os modelos (locais e globais), série = CNES × especialidade-mãe.
  **Outros:** referências e modelos globais, com tuning e ajuste próprios. Cada estrato é avaliado à parte.
- **Nenhuma CNES recebe tratamento separado** (o Materno Infantil deixa de ser relatado à parte); nenhuma
  CNES é excluída (mantém a decisão de 16/11/2026).
- **Pré e pós-processamento (faltas/preprocessamento.py), em todos os modelos globais:** log1p nas
  contagens, mediana do treino nos ausentes, min-max [0, 1] com corte; alvo = taxa por consulta conhecida;
  saída cortada em [0, p99,5 da taxa do treino] × agenda. Ajustado só nas linhas de treino de cada ajuste.

## D18 — Fallback dos modelos locais, suavização da referência, IC95% e comparações par a par

Decisões do usuário (22/11/2026), fixadas antes de rodar a modelagem completa:
- **Fallback:** no estrato "consistentes", nas séries sem modelo local (menos de 104 semanas ativas) ou
  com previsão falha, ETS/Croston/SARIMAX devolvem a previsão da referência taxa × agenda; a coluna
  `fallback` registra cada caso. Assim todas as séries das CNES consistentes entram na comparação.
- **Suavização da referência:** m = 50 consultas (era 20), para estabilizar a taxa em séries pequenas.
- **Intervalos de previsão:** cobertura e largura de 95%, além de 80% e 50%.
- **Testes de hipótese:** além de cada modelo × referência (DM + Holm, IC95% bootstrap do ganho),
  comparações par a par entre os modelos com WAPE menor que a referência (`dm_pares.csv`, Holm entre os pares).

## D19 (23/11/2026) — Correção da referência em h=2, IC95% e margem de erro
- Auditoria: em h=2 a agenda conhecida na origem é incompleta; 8–11% das faltas caem em semanas sem agenda conhecida e taxa_ag4 chegava a 55 faltas/consulta, inflando a referência (viés 1,12 no treino 2022–23; 1,16 no teste).
- Correção: taxa_ag4 e taxa_ag13 cortadas em 1 (no máximo uma falta por consulta). Escolhido só no treino 2022–23 (viés 0,98; WAPE 0,56 vs 0,68). Afeta a referência, o fallback dos locais e as variáveis dos globais → e02–e06 refeitos.
- Nível de confiança 95% (α = 0,05): diferença significativa se p_holm < 0,05 **e** o IC95% do ganho não contém 0 (p ≥ 0,05 = não significativo).
- Margem de erro = metade da largura do IC95% (bootstrap em blocos de 4 semanas, 2000 reamostragens), para WAPE, viés, MAE e ganhos. Saídas: colunas *_ic_inf/_ic_sup/_moe em metricas.csv; moe_ganho e significativo em dm.csv e dm_pares.csv; tabela_principal.csv.

## D20 (23/11/2026) — Referência em "outros" e intervalos de previsão
- Referência principal mantida (taxa × agenda) em todos os estratos: no treino 2022–23 ela foi melhor que mm4 em todos (outros h=2: WAPE 0,83 vs 0,93). Trocar pela mm4 só porque a referência piorou no teste de 2024 seria escolher pelo teste (proibido). A piora em 2024 entra como limitação.
- Comparação secundária pré-declarada contra referencia_mm4 (mesmo DM + Holm + IC95% + margem): dm_vs_mm4.csv.
- Cobertura acima de 95% dos intervalos: diagnóstico mostrou que mesmo a Poisson (φ = 0) cobre 96,9% (HGB, consistentes, h=2), e a cobertura é alta em todas as faixas de μ. Causas: contagens pequenas e discretas (o intervalo inclui as extremidades inteiras) e faltas limitadas pela agenda (menos dispersas que Poisson). φ não é recalibrado; declarado como limitação (intervalos conservadores).
