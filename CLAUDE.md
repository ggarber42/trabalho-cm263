# CLAUDE.md — Previsão de absenteísmo (GERCON Porto Alegre), trabalho da disciplina CMP263 – Aprendizagem de Máquina

Este arquivo vale para qualquer sessão (Claude Code no notebook ou Claude no app). Leia-o inteiro antes de escrever ou rodar código. Em caso de conflito com um pedido pontual, avise o usuário em vez de contornar uma regra.

## 1. Objetivo

Modelar, como problema de série temporal / previsão com validação temporal, o absenteísmo em consultas ambulatoriais de atenção secundária, e **comparar diferentes algoritmos de machine learning** com o mesmo protocolo.

Contexto: atividade da disciplina **CMP263 – Aprendizagem de Máquina** (não é a tese). A introdução é escrita à parte pelo usuário; este repositório entrega apenas as duas seções abaixo. Manter o escopo proporcional a um trabalho de disciplina: o essencial primeiro, o opcional só se sobrar tempo.

Entregas finais:
1. Seção **Metodologia** (`entregaveis/metodologia.md`).
2. Seção **Resultados** (`entregaveis/resultados.md`).
3. Código com **reprodutibilidade de ponta a ponta**: do parquet bruto às tabelas, figuras e números do texto.

Não escrever introdução, discussão geral nem artigo completo, salvo pedido explícito.

## 2. Dados

- Base: `gercon_poa_censo_distancias_com_natureza_juridica.parquet` (uma linha por consulta agendada; ~983.875 consultas, 221 CNES, 2018-01-03 a 2026-03-31; 2018 tem cobertura parcial, 36.958 consultas, e é **mantido** por alimentar o histórico do paciente; coortes e níveis A/B/C usam 2019–2026). Decisão de 16/11/2026: nenhuma exclusão de CNES por inatividade, volume mínimo ou código placeholder; a classificação de atividade (`tabelas/atividade_cnes.csv`) é só descritiva. Cadastro auxiliar: `gercon_poa_censo_estabelecimentos.parquet`.
- **Desfecho:** `situacaoagenda`. Variável binária por consulta: `y = (situacaoagenda == "FALTANTE")`. Antes de qualquer análise nova, listar e registrar a contagem de todas as categorias de `situacaoagenda` (e de `status`) antes dos filtros, e documentar o que foi excluído.
- Unidades de análise: (a) consulta (probabilidade de falta, modelos por paciente); (b) **CNES-dia** (número de faltas, soma das probabilidades ou previsão direta da série), horizonte **h = 3 dias**.
- A base é gerada por `reestruturacao.qmd` (R). Variáveis de histórico (`n_consultas_previas`, `n_faltas_previas`, etc.) usam apenas eventos com `data_ev < t0`, onde `t0 = data_solicitacao + max(diasfilaespera, 0)`. Não alterar essa regra sem registrar.
- Toda a base é de Porto Alegre. O recorte "municipal" não é geográfico: separa as CNES sob **gestão da Prefeitura** (as únicas em que ela pode intervir) das demais (estaduais, federais, filantrópicas).
- **Escopo (decisão em aberto):** hoje o trabalho usa o recorte municipal. O usuário ainda vai avaliar um **modelo geral** (todas as CNES) ou outros critérios de seleção. Por isso o escopo deve ser um parâmetro em `config.yaml` (`escopo: municipal | geral | criterio_x`), nunca fixo no código, e os resultados devem trazer o escopo no nome do arquivo. Não assumir o municipal como definitivo.
- Recorte municipal atual: 111 CNES, 360.878 consultas, definido em `flag_municipal.csv` (natureza jurídica Município/Fundação municipal + gestor 431490 + razão social da Prefeitura; `TP_GESTAO` não discrimina). **Coorte estável municipal:** 10 CNES (6 nível A + 4 nível B), em `niveis_municipal.csv`.
- A CNES 2237571 (endodontia) é do GHC (federal) e está fora do recorte.
- Dados pessoais: nunca imprimir, salvar em log ou colocar em figura identificadores de paciente. Trabalhar com agregados nas saídas.

## 3. Protocolo anti-vazamento (regras duras)

1. Previsão para o dia *t* usa só informação disponível até *t − L* (L = defasagem, ver seção 3.1) mais o que é conhecido com a agenda (ex.: `ag`, características dos pacientes agendados). Padrão atual: L = 3.
2. Nenhuma escolha de modelo, hiperparâmetro, variável, coorte, lag ou regra pode usar dados do período de teste ou posteriores. Isso vale para qualquer esquema de divisão da seção 3.1.
3. Tuning por **validação temporal aninhada**: os folds internos são construídos só com dados anteriores ao início do teste externo, com o mesmo esquema de divisão do experimento (ex.: para janela expansiva e teste no ano Y, treino `< v` → validação `v`, v ∈ {Y−2, Y−1}). Os folds precisam existir para todos os Y (para Y = 2026 são 2024 e 2025). Sem early stopping com split aleatório (usar `staged_predict_proba` ou número de iterações escolhido nos folds).
4. Recalibração k (faltas observadas ÷ previstas nos 28 dias anteriores, com lag 3) só com dados anteriores.
5. Seleção de coorte, níveis A/B/C, top-N de CNES e regras de exclusão devem ser definidas **apenas com dados anteriores ao corte** de cada ano de teste (ou, se usadas com a série inteira, declaradas como limitação e testadas em análise de sensibilidade).
6. O teste externo não serve para escolher entre variantes. Comparações de variantes no teste são descritivas, com todas as variantes pré-declaradas.
7. Estatísticas globais (médias para imputação, padronização, etc.) calculadas no treino.
8. Toda nova variável precisa de justificativa de disponibilidade temporal (quando ela é conhecida). Revisar `status` e `possuiunidadeindicada` quanto a possível informação pós-desfecho.

### 3.1 Experimento: defasagens e esquemas de divisão temporal

O usuário quer **testar diferentes defasagens e metodologias de divisão dos dados** como parte do estudo. Cada esquema é um módulo selecionável por `config.yaml` (`divisao: holdout | expansiva | deslizante | purged_embargo`; `lag: [3, 14]`; decisão de 17/11/2026: só esses dois cenários), e o artigo compara os resultados entre eles.

| Esquema | Definição | Observações |
|---|---|---|
| **Holdout temporal** | Um corte único: treino até a data C, teste depois de C. | Simples, mas uma só estimativa e sensível à data do corte. Reservar a validação dentro do treino (anterior a C). |
| **Janela expansiva** | O treino cresce a cada passo e o teste é o bloco seguinte (hoje, por ano, 2023–2026). | Esquema atual. Usa todo o histórico, inclusive regimes antigos. |
| **Janela deslizante** | Treino de tamanho fixo (ex.: 12, 24, 36 meses) que anda com o teste. | Testar o tamanho da janela como hiperparâmetro, escolhido só na validação anterior ao teste. Útil com quebras de regime. |
| **Purged + embargo** | Remove do treino as observações cujo intervalo de informação se sobrepõe ao teste (purga) e deixa uma folga adicional (embargo). | Ver abaixo. |

**Purga e embargo neste problema.** Uma consulta tem história até `t0` (data de solicitação mais a espera) e desfecho conhecido só na data da consulta. Para um teste que começa em C, um paciente cuja consulta é antes de C, mas cujo desfecho/atualização de histórico só se resolve depois, contamina as variáveis de histórico do teste. Regras:
- **Purga:** excluir do treino as consultas com `dia` no intervalo `[C − P, C)`, em que P cobre o maior intervalo entre `t0` e a consulta (usar `dias_ate_consulta` máximo ou um percentil alto, e registrar o valor) mais a defasagem L.
- **Embargo:** folga adicional E depois do fim do teste, relevante apenas quando o treino contém dados posteriores ao teste (validação cruzada com blocos, ex.: *purged k-fold* / CPCV). Em esquemas só para frente (expansiva, deslizante), o embargo não tem efeito; implementar e documentar mesmo assim, para a comparação ser rastreável.
- Registrar P e E no log de cada experimento e testar a sensibilidade (ex.: P ∈ {0, 28, 90 dias}).
- As variáveis de histórico (`n_faltas_previas` etc.) já usam `data_ev < t0`. Conferir que a purga é aplicada **depois** de construir essas variáveis e não recalcula nada com dados do teste.

**Origem do valor de L.** O valor 3 foi uma escolha inicial sem justificativa empírica. A etapa S3.1 do `PIPELINE.md` separa o horizonte *h* (quantos dias antes se prevê) do atraso de disponibilidade *d* (quanto demora o desfecho aparecer), mede a fração da agenda já conhecida em *t − h* e traça desempenho × horizonte. O *h* principal é decidido por critério operacional (com fonte) e declarado antes de olhar o teste. Se a fração da agenda conhecida em *t − h* for bem menor que 1, a agenda usada como variável (`ag`, `m_*`) deve ser reconstruída como existia em *t − h*.

**Defasagem (L).** L é o horizonte de previsão (quantos dias antes se prevê) e vale para as taxas móveis, a recalibração k e as variáveis defasadas. Testar L ∈ {3, 14} (decisão de 17/11/2026; 1 e 7 foram descartados). L = 3 é o caso operacional atual (quando a agenda e a decisão de overbooking acontecem). A escolha de L para o artigo é uma decisão de uso, não de menor erro no teste.

**Regras do experimento:**
1. Todos os modelos passam pelo **mesmo conjunto de divisões** em cada esquema, com orçamento de tuning equivalente.
2. Os esquemas são comparados como **fatores do estudo** (tabela modelo × esquema × lag), sem eleger o esquema pelo desempenho no teste. O esquema principal do artigo é declarado antes de olhar os resultados do teste; os demais entram como análise de sensibilidade.
3. Cada combinação (esquema, L, modelo, escopo) grava um registro em `resultados/<escopo>_<perfil>/experimentos.csv` com sementes, parâmetros P, E, tamanho da janela e hash dos dados.
4. Esquemas diferentes avaliam períodos de teste diferentes, então a comparação entre esquemas só vale **no mesmo conjunto de dias de teste**. Definir esse conjunto comum (ex.: 2024-01 a 2026-03) e reportá-lo.

## 4. Avaliação

- **Baseline obrigatório:** Poisson com taxa de falta dos 28 dias anteriores (lag 3) × agenda do dia. **Naive:** média do mesmo dia da semana nas 4 semanas anteriores.
- Métricas pontuais: WAPE (Σ|y − ŷ| / Σy), viés (Σŷ / Σy). Probabilísticas: CRPS, cobertura de intervalos 80% e 50%, largura do IC80, ECE de P(≥k). Por consulta: Brier, AUC, logloss. Por célula CNES-dia-especialidade: AUC de P(≥1 falta).
- **Todos os modelos devem ser comparados nas mesmas linhas** (interseção) e com o **mesmo tratamento de dispersão** (mesma família de distribuição preditiva para todos), senão CRPS e cobertura não são comparáveis. Registrar quantas linhas cada modelo perdeu.
- Testes: Diebold-Mariano (HAC com agregação semanal) contra o baseline e contra o melhor modelo. Informar o número de comparações e usar correção para múltiplas comparações ou indicar que não foi feita.
- Reportar resultados por nível (A, B), por ano e sem o Hospital Materno Infantil Presidente Vargas (2237822, taxa de falta ~7%, distinto das demais).
- Resultados de referência (coorte antiga A+B, antes do recorte municipal), para conferir reprodução: WAPE V0 0,367, V1 0,369, HGB Poisson 0,376, SARIMA 0,380, baseline 0,383, ETS 0,385, GLM 0,402.

## 5. Algoritmos a comparar

Já implementados: baseline Poisson, naive sazonal, ETS, SARIMA/SARIMAX (d=1, D=1, s=7), GLM, HGB Poisson com lags (puro e com agregados da agenda), HGB por paciente somado por CNES-dia (V0/V1).
Candidatos a incluir, sempre com o mesmo protocolo e a mesma validação aninhada: regressão logística regularizada, floresta aleatória, LightGBM/XGBoost/CatBoost, rede neural simples (MLP), modelo hierárquico/Bayesiano para contagens (opcional). Nenhum algoritmo novo entra no artigo sem tuning aninhado equivalente aos demais (orçamento de busca comparável).

## 6. Reprodutibilidade de ponta a ponta

- Um único ponto de entrada: `run_all.sh` (ou `Makefile`) que executa, em ordem, preparação → coorte → seleção → treino/tuning → comparação → tabelas e figuras → renderização dos textos. Cada etapa grava entradas, saídas e tempo em `logs/`.
- Ambiente fixo: `requirements.lock` (versões exatas) e versão do Python registrada. O venv fica em `.venv/` na pasta do projeto (decisão do usuário em 24/11/2026; criado por `make venv`, ignorado pelo git; se a pasta for sincronizada com o Drive, excluir `.venv/` da sincronização).
- Sementes fixas em todos os pontos aleatórios (numpy, sklearn, joblib, simulação). Registrar a semente no log. Nada de `random_state=None`.
- Caminhos relativos e configuráveis em um único `config.yaml` (nenhum caminho absoluto no código). Listas de CNES em arquivos de dados, nunca no código.
- Cada script grava um arquivo `manifesto` com hash (SHA-256) do parquet de entrada e das saídas principais.
- O R (`reestruturacao.qmd`) e o script que gerou `distancias` e `natureza_juridica` devem estar versionados na pasta (hoje o script das distâncias não está). Sem eles, a reprodução de ponta a ponta não fecha.
- **Números do texto vêm de arquivos.** Tabelas e valores citados em metodologia, resultados e artigo são gerados por script a partir dos CSVs de `tabelas/` (S9), que por sua vez vêm de `resultados/<escopo>_<perfil>/` (S7). Nunca digitar um número à mão no manuscrito.
- Rodar com `python3 script.py` (sem `-I`, senão `utils_municipal.py` não é importado).
- Controle de versão com git; um commit por etapa concluída; tag para a versão que gerou o artigo.

## 7. Convenções de código

- Python 3, scripts numerados (`NN_nome.py`). **Não alterar os scripts originais 01–18**; novas versões entram com novo número ou sufixo e uma nota no `PROGRESSO_AAAA-MM-DD.md`.
- Funções compartilhadas em `utils_municipal.py` (importar, não copiar). Importar sem executar nada.
- Docstring no início de cada script com: objetivo, protocolo, entradas, saídas, uso (`argparse`) e sempre uma opção `--smoke` que roda rápido **cobrindo todos os anos de teste e todos os folds** (o smoke não pode testar só 2023).
- Antes de rodar algo longo (>10 min): teste de fumaça, validação de formas e de NaN, `assert` de alinhamento por chave (CNES, dia, paciente) e não só por tamanho.
- Saídas nas pastas de `saidas` do `config.yaml`: `resultados/` (previsões e metadados), `tabelas/` (CSV e MD), `figuras/` (PNG e PDF), `logs/` e `manifestos/`. `antigos/resultados_municipal/` guarda só as saídas dos scripts legados. Nada de resultado só no terminal.
- **Código comentado em português do Brasil, pensado para o usuário entender e auditar.** O usuário quer entender o que cada etapa faz. Regras:
  - Todo script começa com um cabeçalho: etapa do pipeline (S0–S15), objetivo, entradas, saídas, como rodar.
  - **Toda função nova** tem docstring em português com: (1) objetivo em uma ou duas frases; (2) parâmetros e seus tipos; (3) **o que retorna** (tipo, formato e colunas, quando for tabela); (4) cuidados (ex.: onde há risco de vazamento temporal). Usar *type hints*.
  - Comentários por bloco explicam **o porquê**, não só o quê. Pontos que protegem contra vazamento levam o marcador `# ANTI-VAZAMENTO:` com a explicação.
  - Decisões arbitrárias (limiares, sementes, tamanhos de janela) levam um comentário `# DECISÃO:` com a justificativa e onde está no `config.yaml`.
  - Mensagens de log, erros e `assert` em português, com texto claro (ex.: "Treino contém datas posteriores ao início do teste").
  - Evitar código excessivamente compacto: preferir passos nomeados a uma linha encadeada difícil de auditar.
  - Ao terminar cada etapa, o assistente resume em linguagem simples o que o código faz, o que foi validado e o que ficou pendente.
- Nomes de variáveis como na base (`cnesexecutante`, `especialidademae`, etc.).
- Qualquer filtro que remove linhas precisa imprimir quantas linhas removeu.

## 8. Entregáveis científicos

- **Metodologia:** fonte e definição dos dados; critérios de elegibilidade e recorte municipal; definição do desfecho; variáveis e momento de disponibilidade; coorte e níveis; protocolo temporal e anti-vazamento; algoritmos, tuning e orçamento; métricas e testes; reprodutibilidade (versões, sementes, repositório).
- **Resultados:** fluxograma de seleção de dados; descrição da coorte; tabela principal de modelos; ganho sobre o baseline com IC e valor de p; calibração; análises por nível, por ano e sem o Materno Infantil; importância de variáveis; exemplo de uso (overbooking/risco aceito) com custo-benefício; limitações.
- **Formato:** só Metodologia e Resultados; a introdução e as referências gerais são do usuário. As limitações entram como subseção de Resultados. Como guia de relato, seguir (apenas nos itens que se aplicam a Métodos e Resultados) **TRIPOD+AI** (modelos de predição; escolha do usuário) e preencher o checklist em `entregaveis/tripod_ai_checklist.md`, citando a seção do manuscrito de cada item. Declarar o desenho temporal, o esquema de divisão principal, as defasagens testadas e a unidade de análise. Discutir limitações já conhecidas (coorte definida com a série inteira, seleção de variáveis em poucos folds, poucos meses em 2026, dependência de dados de uma única cidade).
- Referências só de fontes verificadas (DOI/URL). Nunca inventar citação.

## 9. Pendências conhecidas

Os scripts anteriores ao pipeline (01–24, `grafico_*`, `overbooking_*`, `prever_dia`, `utils_municipal`) estão em `antigos/`, congelados como referência; o pipeline atual (`src/`, `etapas/`, `Makefile`) não os importa. Situação em 17/11/2026 (ver `AUDITORIA_2026-11-16.md`):

**Resolvidas no pipeline novo (código executado e testado no smoke):**
1. `FOLD_YEARS` até 2025 e `max_bins=128` (scripts 22/23): folds para todo ano de teste; sem `max_bins` na grade.
2. Mesmo conjunto de linhas e mesma distribuição preditiva (NB com φ fora da amostra) para todos os modelos; alinhamento por chave; linhas perdidas registradas.
3. Coorte e níveis definidos só com dados anteriores ao corte de cada ano (`src/coorte.py`).
4. Módulo de divisões (holdout, expansiva, deslizante, purged + embargo) e parâmetro L (cenários 3 e 14).
5. Escopo como parâmetro de `config.yaml`.

**Ainda abertas:**
1. Poda de 32 → 6 variáveis (legado): tratar como hipótese; no pipeline novo o modelo por paciente usa as 30 candidatas e não há seleção pelo teste.
2. `ref_default` ≠ V1 do script 11: só relevante para o legado; documentado.
3. (Arquivo bruto e script das distâncias: usuário confirmou em 17/11/2026 que está tudo certo; fora das conferências.) Definição de `status` ainda a documentar.
4. Premissas de dados a confirmar com o serviço: atraso de registro do desfecho, `possuiunidadeindicada`, `idade`. (`t0` confirmado em 17/11/2026: `t0 = data_solicitacao + max(diasfilaespera, 0)`, em que `diasfilaespera` é o tempo da solicitação até o AGENDAMENTO no sistema, não até a consulta; logo `t0` é a data de agendamento.)
5. (G2 congelado em 17/11/2026: ver `entregaveis/plano_analise.md`; não alterar sem registro.)
6. SHAP/ALE e sensibilidades de escopo geral, coorte global e remoção de variáveis suspeitas.
7. Comparação sequencial × paralela refeita com os modelos novos e L = 14.
8. Rodar o perfil completo (só com aprovação), depois `entregaveis/metodologia.md`, `resultados.md` e `tripod_ai_checklist.md`.

## 10. Como o assistente deve trabalhar neste projeto

- **Não retreinar nem rodar jobs longos sem aprovação explícita.** Primeiro revisar, depois propor, depois executar.
- Ao concluir uma etapa, atualizar `PROGRESSO_AAAA-MM-DD.md` e, ao trocar de ambiente, `HANDOFF_COWORK.md`, com o que foi feito, o que falta e os cuidados.
- Dizer sempre o que foi **executado e verificado** e o que apenas foi escrito ou lido. Não afirmar um resultado que não saiu de um arquivo ou de uma execução.
- Se um resultado parecer bom demais, procurar vazamento antes de comemorar.
- Quando houver dúvida de interpretação clínica ou de gestão (ex.: critério de “município”, risco aceito no overbooking), perguntar ao usuário; são decisões do trabalho.
