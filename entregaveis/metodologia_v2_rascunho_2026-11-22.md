**2. Metodologia**

> Versão 2 (redesenho de 22/11/2026), baseada em `docs/superpowers/specs/2026-11-22-redesenho-metodologico-design.md`.
> Os marcadores [PREENCHER: …] são números que serão gerados por script a partir dos arquivos de resultados; nenhum deve ser digitado à mão.

Trata-se de um estudo de desenvolvimento e avaliação de modelos prognósticos com validação temporal interna, relatado segundo o TRIPOD+AI (Collins et al., 2024). O objetivo é comparar algoritmos de aprendizado de máquina e modelos tradicionais de série temporal na previsão do número semanal de faltas em consultas especializadas de pacientes encaminhados pela atenção primária à saúde (APS), nos estabelecimentos executantes de gestão municipal de Porto Alegre, sob o mesmo protocolo de validação. Não há validação externa em outro município ou serviço.

**2.1 Fonte de dados**

A fonte é o GERCON (Gerenciamento de Consultas), sistema da Central de Regulação de Porto Alegre, cujos dados abertos (https://dadosabertos.poa.br/dataset/gercon-gerenciamento-de-consultas) foram obtidos em 26 de agosto de 2026. Os arquivos CSV anuais, organizados por ano de solicitação (2018 a 2025), foram unificados em um único banco, no qual cada linha é uma solicitação de consulta. Não foram integradas bases externas de território nem de deslocamento. O Cadastro Nacional de Estabelecimentos de Saúde (CNES, competência 09/2026) foi usado apenas para identificar os estabelecimentos executantes de gestão municipal, pela natureza jurídica e pela razão social, com junção pelo código do executante e verificação de relação muitos-para-um.

Na padronização, todas as datas foram truncadas ao dia, porque o arquivo de 2025 não registra a hora, e os códigos CNES foram completados com zeros à esquerda até sete dígitos, porque parte dos arquivos antigos perdeu esse zero ([PREENCHER: número de códigos corrigidos]).

**2.2 Período de análise**

Cada arquivo anual é um retrato do sistema na data de sua extração: os arquivos de 2018 a 2024 foram extraídos em 07/01/2025 e o de 2025 em 01/04/2026, e não há arquivo de solicitações de 2026. Consultas posteriores à data de extração do arquivo de origem ficam sem desfecho registrado ou deixam de aparecer; a partir de fevereiro de 2025, por exemplo, não há nenhuma consulta proveniente de solicitações anteriores a 2025. Para evitar que esse artefato de extração fosse interpretado como mudança real do absenteísmo, a série de análise foi definida de 01/04/2019 a 29/12/2024 ([PREENCHER: número de semanas] semanas). O período de 01/01/2018 a 31/03/2019, de volume baixo e anterior a uma mudança de patamar no uso do sistema, foi usado apenas para compor o histórico dos pacientes. A data de extração foi usada somente para justificar esse corte e não entra como preditor.

**2.3 Seleção das consultas**

Foram incluídas as consultas que atendiam a todos os critérios seguintes: município do solicitante igual a Porto Alegre; tipo de unidade solicitante igual a centro de saúde/unidade básica de saúde, para restringir a análise a encaminhamentos da APS; complexidade média; data da consulta preenchida; e situação da agenda "REALIZADA" ou "FALTANTE". Em seguida, o escopo foi restrito aos estabelecimentos executantes de gestão municipal, definidos por natureza jurídica Município ou Fundação municipal, gestor 431490 ou razão social da Prefeitura. A distribuição de todas as categorias da situação da agenda foi registrada antes dos filtros, e cada filtro teve o número de linhas removidas registrado ([PREENCHER: fluxograma com contagens]). O recorte final resultou em [PREENCHER: consultas] consultas em [PREENCHER: estabelecimentos] estabelecimentos. Não houve exclusão por inatividade ou volume mínimo.

**2.4 Desfecho, unidade de análise e horizonte de previsão**

O desfecho por consulta é binário: falta (situação "FALTANTE") ou comparecimento (situação "REALIZADA"). No recorte, houve [PREENCHER: faltas (%)] faltas e [PREENCHER] consultas realizadas. A unidade principal de análise é o estabelecimento executante por semana (CNES-semana, semanas ISO de segunda a domingo), e o desfecho previsto é o número de faltas na semana.

A previsão é feita ao fim do domingo anterior à semana prevista, com horizonte de uma semana (h = 1); o horizonte de duas semanas (h = 2) foi analisado como sensibilidade. Só entram informações conhecidas até a origem da previsão. A agenda da semana prevista é composta pelas consultas cuja data de agendamento é igual ou anterior à origem; a fração das consultas finais já agendadas na origem foi de [PREENCHER: % para h = 1 e h = 2]. Como a base não registra a data de lançamento do desfecho, assumiu-se que o desfecho de uma consulta está disponível no dia seguinte à sua realização, premissa pendente de confirmação com o serviço.

Como detalhamento, as faltas previstas foram distribuídas por especialidade (especialidade-mãe) pela soma das probabilidades individuais do modelo por consulta (seção 2.8), sem modelagem de séries próprias por especialidade.

**2.5 Preditores**

Os preditores foram escolhidos *a priori*, pela disponibilidade no momento da previsão e pela justificativa clínica e de gestão, sem seleção orientada pelos dados. O identificador do paciente foi usado apenas para construir as variáveis de histórico e não entra em nenhum modelo. Município solicitante, tipo de unidade e complexidade servem apenas como critérios de seleção e não são preditores. Variáveis que podem refletir o estado posterior à consulta (situação da solicitação e situação de regionalização) foram excluídas.

Das variáveis categóricas, o CID entrou por uma única variável, o capítulo (26 categorias). A especialidade e a unidade solicitante, de alta cardinalidade, foram codificadas por *target encoding* temporal suavizado: cada categoria recebe sua taxa de falta histórica, puxada para a taxa geral quando há poucos casos, com parâmetro de suavização fixado *a priori*. As estatísticas foram estimadas apenas no treino, e, dentro dele, cada consulta usa somente consultas anteriores à sua origem, para não ser codificada com o próprio desfecho; no teste, aplicam-se as estatísticas do treino, e categorias não vistas recebem a taxa geral. Todos os modelos recebem as mesmas variáveis codificadas. A data de agendamento foi estimada como a data da solicitação somada ao tempo de fila até o agendamento (valores negativos tratados como zero).

Os preditores foram organizados em dois blocos, ambos declarados antes da análise. No **bloco A** (somente a série), os modelos usam apenas a série passada de faltas do estabelecimento e o calendário. No **bloco B** (série e informação dos pacientes), usam também a agenda conhecida na origem e as características dos pacientes agendados, de duas formas: individualmente, no modelo por consulta, cuja previsão por estabelecimento-semana é a soma das probabilidades de falta; e resumidas por estabelecimento-semana, nos demais modelos do bloco. A diferença de desempenho entre os blocos estima o quanto a informação dos pacientes acrescenta à previsão. A Tabela 1 lista os preditores, o momento em que cada um é conhecido e os modelos em que entra.

*Tabela 1. Preditores, disponibilidade temporal e uso nos modelos.* Origem = fim do domingo anterior à semana prevista; t0 = data de agendamento da consulta; s = semana prevista.

| Grupo | Preditor | Quando é conhecido | Bloco A | Bloco B: CNES-semana | Bloco B: por consulta |
|---|---|---|---|---|---|
| Série do estabelecimento | Faltas nas semanas anteriores (defasagens s−h a s−h−k) e médias móveis | Semanas encerradas até a origem | Sim | Sim | Não |
| Calendário | Semana do ano, mês, feriados na semana | Antecipadamente | Sim | Sim | Mês e dia da semana da consulta |
| Agenda (exposição) | Número de consultas agendadas na semana | Origem (consultas com t0 ≤ origem) | Não | Sim | Não |
| Histórico do paciente | Nº de consultas anteriores; nº de faltas anteriores; dias desde a última falta | t0 (consultas anteriores a t0) | Não | Agregado: proporção com alguma falta anterior, média de faltas anteriores | Individual |
| Histórico na especialidade | Nº de faltas anteriores na mesma especialidade-mãe; desfecho da última consulta na mesma especialidade-mãe (sem prévia, realizada, faltante) | t0 | Não | Agregado: proporção cuja última consulta na especialidade foi falta | Individual |
| Solicitação | Especialidade-mãe, capítulo do CID, corregulador | Solicitação | Não | Não | Individual (categoria) |
| Solicitação (alta cardinalidade) | Especialidade (subespecialidade) e unidade solicitante (`cnes_solicitante`) | Solicitação | Não | Não | Individual (*target encoding* temporal) |
| Paciente | Idade; possui unidade indicada | Solicitação | Não | Agregado: média da idade, proporção com unidade indicada | Individual |
| Tempos | Tempo de fila até o agendamento; antecedência entre agendamento e consulta | t0 | Não | Agregado: média da antecedência | Individual |
| Estabelecimento | Estabelecimento executante | Antecipadamente | Identificador da série | Identificador da série | Individual |

O histórico do paciente foi resumido *a priori* em cinco variáveis, para evitar redundância: número de consultas anteriores (zero identifica a primeira consulta), número de faltas anteriores, dias desde a última falta, número de faltas anteriores na mesma especialidade-mãe e desfecho da última consulta nessa especialidade. Nos modelos que não aceitam valores ausentes (regressões e MLP), os dias desde a última falta entram em faixas (nunca faltou, até 90, 91 a 365 e mais de 365 dias). O histórico usa apenas consultas com data anterior à data de agendamento e restritas ao próprio recorte do estudo; faltas em estabelecimentos fora do recorte não são contabilizadas. Os agregados do bloco B são calculados só sobre as consultas já agendadas na origem. Como o tempo até a consulta é a soma do tempo de fila e da antecedência, apenas essa decomposição foi usada, para evitar colinearidade perfeita. Com h = 2, todas as informações são tomadas uma semana antes.

**2.6 Dados ausentes**

[PREENCHER: proporção de ausentes por variável]. Os modelos de *gradient boosting* aceitam valores nulos e os demais usam a mediana do treino. Pacientes sem consultas anteriores receberam contagens iguais a zero e indicador de primeira consulta. Não foi usada imputação múltipla.

**2.7 Análise exploratória e escolha do esquema de avaliação**

Antes do ajuste dos modelos, as séries semanais foram caracterizadas usando apenas o período de treino (01/04/2019 a 31/12/2023). Foram descritos, por estabelecimento, o volume, a taxa de falta e as semanas sem atendimento; a intermitência das faltas, pela classificação de Syntetos, Boylan e Croston (2005), segundo o intervalo médio entre semanas com falta e o quadrado do coeficiente de variação; a tendência, a sazonalidade anual, as funções de autocorrelação e as quebras de nível, como a da pandemia de COVID-19; e a relação entre média e variância. A estacionariedade foi testada pelo teste de espectro de *wavelets* de Nason (2013), complementado pelos testes ADF (Dickey e Fuller, 1979) e KPSS (Kwiatkowski et al., 1992), com correção para múltiplos testes. Cada resultado orientou uma decisão: as faixas de busca dos modelos tradicionais, o uso de modelos para demanda intermitente, a transformação da série e a construção das variáveis de agenda.

O esquema de avaliação seguiu regra declarada antes da análise, baseada em Cerqueira, Torgo e Mozetič (2020): se a maioria das séries fosse não estacionária, a seleção de modelos usaria *holdout* repetido em múltiplos períodos de teste, sempre com o treino anterior ao teste; se a maioria fosse estacionária, usaria validação cruzada em blocos com intervalo de separação entre treino e teste. Resultado: [PREENCHER: proporção de séries não estacionárias e esquema resultante].

**2.8 Validação temporal, modelos e ajuste**

A avaliação segue divisão temporal, sem embaralhamento. O teste final é o ano de 2024 (52 semanas), com treino até 31/12/2023, avaliado uma única vez. Como sensibilidade, repetiu-se o procedimento com teste em 2023 e treino até 31/12/2022. A seleção de modelos e o ajuste de hiperparâmetros usaram exclusivamente o período de treino, com o esquema definido na seção 2.7. Semanas sem nenhuma consulta agendada entram como zero falta no ajuste das séries, mas ficam fora da avaliação ([PREENCHER: número]). O Hospital Materno Infantil Presidente Vargas, com taxa de falta distinta, é reportado também à parte.

A referência (*baseline*) é única: a média das faltas do estabelecimento nas quatro semanas anteriores, com a janela fixada *a priori* por critério operacional. Os modelos comparados foram, no bloco A: suavização exponencial (ETS), SARIMA, Croston e TSB para séries intermitentes (Croston, 1972; Teunter, Syntetos e Babai, 2011), regressão de contagem (Poisson e binomial negativa) com defasagens, *gradient boosting*, floresta aleatória e rede neural (MLP); e, no bloco B: SARIMAX, regressão de contagem com variáveis exógenas, *gradient boosting*, MLP e um modelo por consulta (*gradient boosting* com perda logística), cuja previsão por estabelecimento-semana é a soma das probabilidades de falta das consultas já agendadas na origem.

Os modelos tradicionais de série temporal receberam o mesmo cuidado de preparação que os de aprendizado de máquina: transformação quando indicada pela análise exploratória, escolha das ordens do SARIMA e do tipo de ETS por critério de informação (AICc) e validação temporal no treino, e diagnóstico dos resíduos pelo teste de Ljung-Box (Ljung e Box, 1978), registrado por estabelecimento. O orçamento de busca de hiperparâmetros foi equivalente entre as famílias, com [PREENCHER: número de configurações] configurações por modelo e semente fixa ([PREENCHER: semente]). Transformações, imputação e padronização foram estimadas só no treino.

**2.9 Métricas e comparação**

Todos os modelos geram uma distribuição preditiva binomial negativa, com dispersão estimada em previsões fora da amostra no período de treino. As métricas primárias são o WAPE (erro absoluto ponderado pelo total de faltas) e o CRPS da distribuição preditiva (Gneiting e Raftery, 2007). As secundárias são o viés e a cobertura e a largura dos intervalos de 80% e 50%. No modelo por consulta, reportam-se também Brier, AUC e *log loss*. Métricas baseadas em erro relativo (MAPE, MASE) não foram usadas, por causa das semanas com zero falta.

Os modelos foram comparados nas mesmas linhas (interseção de [PREENCHER] CNES-semanas), com registro das linhas perdidas por modelo. Cada modelo foi comparado ao *baseline* pelo teste de Diebold-Mariano (1995), com variância de Newey-West (1987) e correção de Harvey, Leybourne e Newbold (1997), e correção de Holm (1979) para múltiplas comparações. O intervalo de confiança de 95% do ganho de WAPE foi obtido por *bootstrap* em blocos ([PREENCHER: número de reamostragens]). Os resultados são apresentados por bloco de informação, por estabelecimento e com e sem o Hospital Materno Infantil Presidente Vargas.

**2.10 Implementação, ética e reprodutibilidade**

O pipeline foi escrito em Python [PREENCHER: versão] ([PREENCHER: bibliotecas e versões fixadas]), com execução de ponta a ponta a partir do arquivo bruto por um único ponto de entrada. Todos os parâmetros ficam em arquivo de configuração, as saídas principais têm *hash* registrado e os números deste texto são gerados a partir dos arquivos de resultados. Os dados individuais são registros de pacientes e estão sujeitos à Lei Geral de Proteção de Dados; os agregados e o código [PREENCHER: endereço do repositório e versão] são disponibilizados. O estudo [PREENCHER: número do parecer do comitê de ética ou condição de dispensa]. O plano de análise é interno e o estudo não foi registrado em repositório público.

---

*Referências a acrescentar à lista* (as marcadas com † precisam ter o DOI conferido antes do envio)

Cerqueira, V., Torgo, L. and Mozetič, I. (2020) "Evaluating time series forecasting models: an empirical study on performance estimation methods", Machine Learning, v. 109, p. 1997-2028. doi:10.1007/s10994-020-05910-7.

† Croston, J. D. (1972) "Forecasting and stock control for intermittent demands", Operational Research Quarterly, v. 23, n. 3, p. 289-303.

† Dickey, D. A. and Fuller, W. A. (1979) "Distribution of the estimators for autoregressive time series with a unit root", Journal of the American Statistical Association, v. 74, n. 366, p. 427-431.

Diebold, F. X. and Mariano, R. S. (1995) "Comparing predictive accuracy", Journal of Business & Economic Statistics, v. 13, n. 3, p. 253-263.

Gneiting, T. and Raftery, A. E. (2007) "Strictly proper scoring rules, prediction, and estimation", Journal of the American Statistical Association, v. 102, n. 477, p. 359-378.

Harvey, D., Leybourne, S. and Newbold, P. (1997) "Testing the equality of prediction mean squared errors", International Journal of Forecasting, v. 13, n. 2, p. 281-291.

Holm, S. (1979) "A simple sequentially rejective multiple test procedure", Scandinavian Journal of Statistics, v. 6, n. 2, p. 65-70.

† Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. and Shin, Y. (1992) "Testing the null hypothesis of stationarity against the alternative of a unit root", Journal of Econometrics, v. 54, p. 159-178.

† Ljung, G. M. and Box, G. E. P. (1978) "On a measure of lack of fit in time series models", Biometrika, v. 65, n. 2, p. 297-303.

† Nason, G. (2013) "A test for second-order stationarity and approximate confidence intervals for localized autocovariances for locally stationary time series", Journal of the Royal Statistical Society: Series B, v. 75, n. 5, p. 879-904.

Newey, W. K. and West, K. D. (1987) "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix", Econometrica, v. 55, n. 3, p. 703-708.

† Syntetos, A. A., Boylan, J. E. and Croston, J. D. (2005) "On the categorization of demand patterns", Journal of the Operational Research Society, v. 56, p. 495-503.

† Teunter, R. H., Syntetos, A. A. and Babai, M. Z. (2011) "Intermittent demand: linking forecasting to inventory obsolescence", European Journal of Operational Research, v. 214, n. 3, p. 606-615.
