**2. Metodologia**

> Arquivo gerado por `pipeline_v2/etapas/r01_metodologia.py` a partir de `entregaveis/metodologia_v2_modelo.md` e dos resultados em `pipeline_v2/saidas/{{perfil}}/`. Não editar os números à mão: editar o modelo e rodar o script. Itens [PENDENTE: …] dependem de informação que não está nos dados.

Trata-se de um estudo de desenvolvimento e avaliação de modelos prognósticos com validação temporal interna, relatado segundo o TRIPOD+AI (Collins et al., 2024). O objetivo é comparar algoritmos de aprendizado de máquina e modelos tradicionais de série temporal na previsão do número semanal de faltas em consultas especializadas de pacientes encaminhados pela atenção primária à saúde (APS), nos estabelecimentos executantes de gestão municipal de Porto Alegre, sob o mesmo protocolo de validação. Não há validação externa em outro município ou serviço.

**2.1 Fonte de dados**

A fonte é o GERCON (Gerenciamento de Consultas), sistema da Central de Regulação de Porto Alegre, cujos dados abertos (https://dadosabertos.poa.br/dataset/gercon-gerenciamento-de-consultas) foram obtidos em 26 de agosto de 2026. Os arquivos CSV anuais, organizados por ano de solicitação (2018 a 2025), foram unificados em um único banco, no qual cada linha é uma solicitação de consulta. Não foram integradas bases externas de território nem de deslocamento. O Cadastro Nacional de Estabelecimentos de Saúde (CNES, competência 09/2026) foi usado apenas para identificar os estabelecimentos executantes de gestão municipal, pela natureza jurídica e pela razão social, com junção pelo código do executante e verificação de relação muitos-para-um.

Na padronização, todas as datas foram truncadas ao dia, porque o arquivo de 2025 não registra a hora, e os códigos CNES foram completados com zeros à esquerda até sete dígitos, porque parte dos arquivos antigos perdeu esse zero ({{cnes_corrigidos}} códigos de estabelecimento executante e {{cnes_sol_corrigidos}} de unidade solicitante corrigidos).

**2.2 Período de análise**

Cada arquivo anual é um retrato do sistema na data de sua extração: os arquivos de 2018 a 2024 foram extraídos em 07/01/2025 e o de 2025 em 01/04/2026, e não há arquivo de solicitações de 2026. Consultas posteriores à data de extração do arquivo de origem ficam sem desfecho registrado ou deixam de aparecer; a partir de fevereiro de 2025, por exemplo, não há nenhuma consulta proveniente de solicitações anteriores a 2025. Para evitar que esse artefato de extração fosse interpretado como mudança real do absenteísmo, a série de análise foi definida de 01/04/2019 a 29/12/2024 ({{n_semanas}} semanas). O período de 01/01/2018 a 31/03/2019, de volume baixo e anterior a uma mudança de patamar no uso do sistema, foi usado apenas para compor o histórico dos pacientes. A data de extração foi usada somente para justificar esse corte e não entra como preditor.

**2.3 Seleção das consultas**

Foram incluídas as consultas que atendiam a todos os critérios seguintes: município do solicitante igual a Porto Alegre; tipo de unidade solicitante igual a centro de saúde/unidade básica de saúde, para restringir a análise a encaminhamentos da APS; complexidade média; data da consulta preenchida; e situação da agenda "REALIZADA" ou "FALTANTE". Em seguida, o escopo foi restrito aos estabelecimentos executantes de gestão municipal, definidos por natureza jurídica Município ou Fundação municipal, gestor 431490 e razão social da Prefeitura. A distribuição de todas as categorias da situação da agenda foi registrada antes dos filtros, e cada filtro teve o número de linhas removidas registrado (Tabela 1). O recorte final resultou em {{n_consultas}} consultas em {{n_cnes}} estabelecimentos no período de análise. Não houve exclusão por inatividade ou volume mínimo.

*Tabela 1. Fluxograma de seleção das consultas.*

{{tabela_fluxo}}

**2.4 Desfecho, unidade de análise e horizonte de previsão**

O desfecho por consulta é binário: falta (situação "FALTANTE") ou comparecimento (situação "REALIZADA"). No recorte, houve {{n_faltas}} faltas ({{pct_faltas}}) e {{n_realizadas}} consultas realizadas. A unidade de modelagem é a série estabelecimento executante × especialidade-mãe × semana (semanas ISO de segunda a domingo; {{n_series}} séries), e o desfecho previsto é o número de faltas na semana. Os resultados são reportados nesse nível e também somados por estabelecimento-semana. Embora os modelos estimem internamente a taxa de falta por consulta conhecida, a quantidade prevista e avaliada é sempre o número de faltas.

A origem da previsão é o fim do domingo, e a previsão é entregue ao gestor na segunda-feira. O horizonte principal é de duas semanas (h = 2: previsão, na segunda-feira, das faltas da semana seguinte), por ser o que permite agir sobre a agenda; o horizonte de uma semana (h = 1: a própria semana que começa) é secundário. Só entram informações conhecidas até a origem. A agenda da semana prevista é composta pelas consultas cuja data de agendamento é igual ou anterior à origem; a fração das consultas finais já agendadas na origem foi de {{frac_h1}} para h = 1 e {{frac_h2}} para h = 2. Como a base não registra a data de lançamento do desfecho, assumiu-se que o desfecho de uma consulta está disponível no dia seguinte à sua realização, premissa pendente de confirmação com o serviço.

**2.5 Preditores**

Os preditores foram escolhidos *a priori*, pela disponibilidade no momento da previsão e pela justificativa clínica e de gestão, sem seleção orientada pelos dados. Há um único conjunto de informações, comum a todos os modelos de série: a série passada de faltas, o calendário e a agenda conhecida na origem com as características resumidas dos pacientes agendados. O identificador do paciente foi usado apenas para construir as variáveis de histórico e não entra em nenhum modelo. Município solicitante, tipo de unidade e complexidade servem apenas como critérios de seleção. Variáveis que podem refletir o estado posterior à consulta (situação da solicitação e situação de regionalização) foram excluídas.

No modelo por consulta, o CID entrou pelo capítulo (26 categorias), e a especialidade e a unidade solicitante, de alta cardinalidade, foram codificadas por *target encoding* temporal suavizado (peso {{te_m}} consultas para a taxa geral): as estatísticas foram estimadas apenas no treino, cada consulta usa somente consultas anteriores à sua origem, e categorias não vistas recebem a taxa geral. A data de agendamento foi estimada como a data da solicitação somada ao tempo de fila até o agendamento (valores negativos tratados como zero). A Tabela 2 lista os preditores.

*Tabela 2. Preditores, disponibilidade temporal e uso nos modelos.* Origem = fim do domingo; t0 = data de agendamento; s = semana prevista.

| Grupo | Preditor | Quando é conhecido | Modelos de série | Modelo por consulta |
|---|---|---|---|---|
| Série | Faltas nas {{lags}} semanas anteriores à origem (defasagens s−h a s−h−{{lags_menos1}}), médias móveis de 4 e 13 semanas, faltas 52 semanas antes | Semanas encerradas até a origem | Sim | Não |
| Taxa histórica | Faltas ÷ agenda conhecida nas últimas 4 e 13 semanas encerradas | Origem | Sim | Não |
| Calendário | Mês, nº de feriados, dias úteis, ponte, feriado na semana anterior ou seguinte, recesso de fim de ano, termos de Fourier ({{fourier}} pares) | Antecipadamente | Sim | Mês e dia da semana |
| Agenda (exposição) | Nº de consultas agendadas na semana prevista | Origem (t0 ≤ origem) | Sim | Não |
| Histórico do paciente | Nº de consultas e de faltas anteriores; dias desde a última falta | t0 | Agregado: proporção com falta anterior, média de faltas anteriores | Individual |
| Histórico na especialidade | Nº de faltas anteriores e desfecho da última consulta na mesma especialidade-mãe | t0 | Agregado: proporção cuja última consulta na especialidade foi falta | Individual |
| Solicitação | Especialidade-mãe, capítulo do CID, corregulador, especialidade e unidade solicitante | Solicitação | Identifica a série | Individual |
| Paciente | Idade; possui unidade indicada | Solicitação | Agregado: média da idade, proporção com unidade indicada | Individual |
| Tempos | Tempo de fila; antecedência entre agendamento e consulta | t0 | Agregado: média da antecedência | Individual |

O histórico do paciente usa apenas consultas com data anterior à data de agendamento e restritas ao próprio recorte do estudo. Os agregados são calculados só sobre as consultas já agendadas na origem. Em h = 2, todas as informações são tomadas uma semana antes.

**2.6 Pré-processamento e dados ausentes**

A única variável com valores ausentes é a de dias desde a última falta ({{pct_ausente_dias}} das consultas), ausente por construção nos pacientes que nunca faltaram; esses pacientes têm o número de faltas anteriores igual a zero, o que identifica a situação. Não foi usada imputação múltipla.

Nos modelos globais (seção 2.8), um mesmo pré-processamento foi aplicado a todos, estimado apenas nas linhas de treino de cada ajuste: transformação log1p das variáveis de contagem, imputação pela mediana do treino e escala mínimo-máximo para [0, 1], com valores novos fora da faixa cortados nos limites, para evitar entradas extremas na rede neural. O alvo desses modelos é a taxa de faltas por consulta conhecida (faltas ÷ exposição, com exposição igual ao número de consultas agendadas, mínimo 1), com peso igual à exposição; a previsão é cortada entre zero e o percentil 99,5 da taxa do treino e multiplicada pela exposição, voltando ao número de faltas.

**2.7 Análise exploratória, estratos e escolha do esquema de validação**

Antes do ajuste dos modelos, as séries semanais foram caracterizadas usando apenas o período de treino (até 31/12/2023): volume, taxa de falta, semanas sem atendimento, intermitência pela classificação de Syntetos, Boylan e Croston (2005), tendência, sazonalidade anual, autocorrelação e relação entre média e variância. A estacionariedade foi testada pelos testes ADF (Dickey e Fuller, 1979) e KPSS (Kwiatkowski et al., 1992), com correção de Benjamini-Hochberg entre séries; o teste de *wavelets* de Nason (2013), previsto inicialmente, não pôde ser executado e não entrou na regra.

O esquema de validação seguiu regra declarada antes da análise, baseada em Cerqueira, Torgo e Mozetič (2020): se a maioria das séries fosse não estacionária, a seleção de modelos usaria *holdout* repetido; caso contrário, validação cruzada em blocos com intervalo de separação. Das {{n_series_expl}} séries com histórico suficiente, {{pct_nao_est}} foram classificadas como não estacionárias, e foi adotado o *holdout* repetido: {{n_dobras}} cortes sorteados (semente fixa) dentro do treino, cada um com {{semanas_val}} semanas de validação e todo o passado anterior para ajuste.

Os estabelecimentos foram divididos, só com o treino, em dois estratos: **consistentes**, com pelo menos uma consulta em ao menos {{min_ativas}} das últimas {{janela}} semanas do treino e média de pelo menos {{min_faltas}} faltas por semana nessa janela; e **outros**. Resultaram {{n_consistentes}} estabelecimentos consistentes, responsáveis por {{pct_faltas_consistentes}} das faltas da janela, e {{n_outros}} no estrato outros. Nenhum estabelecimento recebeu tratamento separado. Os modelos de cada estrato foram ajustados, tiveram os hiperparâmetros escolhidos e foram avaliados separadamente.

**2.8 Validação temporal, modelos e ajuste**

A avaliação segue divisão temporal, sem embaralhamento. O teste é o ano de 2024 (52 semanas), com treino até 31/12/2023, avaliado uma única vez, com origem móvel semanal e parâmetros fixados no fim do treino. A seleção de modelos e o ajuste de hiperparâmetros usaram exclusivamente o treino. Semanas sem nenhuma consulta agendada e sem falta não são avaliadas ({{nao_avaliaveis_h2}} das série-semanas do painel de h = 2, em todo o período).

A referência principal (*baseline*), fixada antes da análise, é a taxa de faltas por consulta conhecida nas quatro semanas encerradas até a origem, suavizada em direção à taxa de 13 semanas (peso de {{m_taxa}} consultas) e limitada a uma falta por consulta, multiplicada pela agenda conhecida da semana prevista. O limite foi incluído porque, em h = 2, a agenda conhecida na origem é incompleta e a razão sem limite produzia taxas impossíveis; ele foi definido com o treino de 2022–2023. Como comparação secundária, também declarada antes, usou-se a média das faltas nas quatro semanas anteriores. No treino de 2022–2023 a referência principal teve erro menor que a média móvel em ambos os estratos e horizontes, e por isso foi mantida.

Foram comparados: (i) **modelos locais**, ajustados série a série — suavização exponencial (ETS), Croston-TSB para demanda intermitente (Croston, 1972; Teunter, Syntetos e Babai, 2011) e SARIMAX com a agenda e o calendário como exógenas —, aplicados à taxa de faltas suavizada e convertidos em faltas pela agenda, apenas no estrato consistente e nas séries com pelo menos {{min_local}} semanas ativas; nas demais séries desse estrato o modelo local recebe a previsão da referência ({{fb_h2}} das linhas de teste em h = 2), e o desempenho também foi calculado só nas linhas sem essa substituição; (ii) **modelos globais**, ajustados em todas as séries do estrato de uma vez — regressão de Poisson penalizada, *gradient boosting* com perda de Poisson, floresta aleatória e rede neural (MLP); e (iii) um **modelo por consulta** (*gradient boosting* com perda logística), cuja previsão é a soma das probabilidades de falta das consultas já agendadas na origem.

As ordens do SARIMAX e o tipo de ETS foram escolhidos por critério de informação e validação no treino, com diagnóstico dos resíduos pelo teste de Ljung-Box (Ljung e Box, 1978). O orçamento de busca de hiperparâmetros foi equivalente entre os modelos ajustáveis, com até {{n_config}} configurações por modelo, sorteadas com semente fixa ({{semente}}), escolhidas pelo menor WAPE médio nas dobras de validação.

**2.9 Métricas e comparação**

Todos os modelos geram uma distribuição preditiva binomial negativa comum, com dispersão estimada por momentos em previsões fora da amostra (validação), por modelo e estrato. As métricas são o WAPE (soma dos erros absolutos ÷ soma das faltas), o MAE, o viés (faltas previstas ÷ observadas), o CRPS (Gneiting e Raftery, 2007) e a cobertura e a largura dos intervalos de previsão de 95%, 80% e 50%. No modelo por consulta, reportam-se também Brier, AUC e *log loss*. Métricas de erro relativo (MAPE, MASE) não foram usadas por causa das semanas com zero falta.

Os modelos foram comparados nas mesmas linhas (interseção; {{inter_h2_cons}} série-semanas no estrato consistente e {{inter_h2_out}} no estrato outros em h = 2), com registro das linhas perdidas por modelo. Cada modelo foi comparado às referências pelo teste de Diebold-Mariano (1995), com variância de Newey-West (1987), correção de Harvey, Leybourne e Newbold (1997) e perdas absolutas somadas por semana; os modelos melhores que a referência principal foram também comparados dois a dois. Usou-se a correção de Holm (1979) dentro de cada família de comparações (estrato × horizonte), com o número de comparações informado. O nível de confiança é de 95%: a diferença é considerada estatisticamente significativa quando o valor de p corrigido é menor que 0,05 e o intervalo de confiança de 95% do ganho não contém zero. Os intervalos de confiança de 95% do WAPE, do MAE, do viés e do ganho relativo de WAPE foram obtidos por *bootstrap* em blocos de {{bloco}} semanas ({{reamostragens}} reamostragens), e a margem de erro foi definida como a metade da largura desse intervalo.

Como verificação contra vazamento, o desfecho de todas as consultas a partir de 03/06/2024 foi alterado artificialmente e o painel reconstruído: nenhuma variável preditora das semanas anteriores a essa data mais o horizonte se modificou.

**2.10 Implementação, ética e reprodutibilidade**

O pipeline foi escrito em Python {{python}} ({{bibliotecas}}), com execução de ponta a ponta a partir do arquivo bruto por um único ponto de entrada (`executar.sh`) e testes automáticos executados antes de cada rodada. Todos os parâmetros ficam em arquivo de configuração, as saídas principais têm *hash* SHA-256 registrado e os números deste texto são gerados a partir dos arquivos de resultados. Os dados individuais são registros de pacientes e estão sujeitos à Lei Geral de Proteção de Dados; os agregados e o código [PENDENTE: endereço do repositório e versão] são disponibilizados. O estudo [PENDENTE: número do parecer do comitê de ética ou condição de dispensa]. O plano de análise é interno e o estudo não foi registrado em repositório público.

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
