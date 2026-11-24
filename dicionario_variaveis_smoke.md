# Dicionário de variáveis

Gerado por `etapas/S1_auditoria.py` (esquema do parquet + `dados/dicionario_manual.csv`).
Coluna **Momento** = quando a variável passa a ser conhecida (regra 8 do CLAUDE.md).
Coluna **Risco pós-desfecho**: `nao`, `suspeita`, `suspeita_baixa` ou `a_confirmar` (pendente de confirmação com o serviço).

| Variável | Tipo | Ausentes (%) | Distintos | Grupo | Momento | Risco pós-desfecho | Observação |
|---|---|---|---|---|---|---|---|
| `idsolicitacao` | str | 0.00 | (oculto) | identificacao | solicitacao | nao | Identificador da solicitacao. Nao usar como variavel. |
| `idpaciente` | str | 0.00 | (oculto) | identificacao | solicitacao | nao | Identificador pessoal. Nunca imprimir nem salvar em saida. |
| `data_solicitacao` | object | 0.00 | 2.187 | tempo | solicitacao | nao | Data em que o encaminhamento foi solicitado. |
| `datahora_consulta` | object | 0.00 | 2.257 | tempo | agendamento (t0) | a_confirmar | Data da consulta agendada. Se a consulta foi remarcada depois de t0 a data final pode ser posterior a informacao de t0 (confirmar com o servico). |
| `rid` | int32 | 0.00 | 48.962 | identificacao | construida | nao | Numero da linha criado no reestruturacao.qmd. Nao usar como variavel. |
| `t0` | object | 0.00 | 2.197 | tempo | agendamento (t0) | nao | t0 = data_solicitacao + max(diasfilaespera, 0). Data de corte do historico do paciente. |
| `situacaoagenda` | category | 0.00 | 2 | desfecho | apos a data da consulta | desfecho | Desfecho. y = (situacaoagenda == FALTANTE). Categorias restantes: REALIZADA e FALTANTE (as demais foram filtradas no qmd). |
| `corregulador` | category | 0.00 | 6 | encaminhamento | solicitacao | nao | Conferir no servico se pode mudar depois da solicitacao. |
| `especialidademae` | category | 0.00 | 73 | encaminhamento | solicitacao | nao | Especialidade solicitada. |
| `cid_3dig` | str | 0.00 | 1.233 | encaminhamento | solicitacao | nao | CID com 3 digitos informado na solicitacao. |
| `cid_1dig` | category | 0.00 | 26 | encaminhamento | solicitacao | nao | Capitulo do CID (1 digito). |
| `possuiunidadeindicada` | category | 0.00 | 2 | encaminhamento | a_confirmar | suspeita | Indica se a solicitacao trazia unidade executante indicada. Se for preenchida apos a regulacao ou o agendamento pode carregar informacao posterior. Confirmar com o servico. |
| `dias_ate_consulta` | int32 | 0.00 | 973 | tempo | agendamento (t0) | nao | datahora_consulta - data_solicitacao (inclui a espera). Conhecida quando a consulta e agendada. |
| `dia_semana` | category | 0.00 | 7 | calendario | agendamento (t0) | nao | Calculada da data da consulta. |
| `mes_consulta` | category | 0.00 | 12 | calendario | agendamento (t0) | nao | Calculada da data da consulta. |
| `idade` | float64 | 0.00 | 106 | paciente | solicitacao | a_confirmar | Confirmar se e a idade na solicitacao ou na consulta (diferenca pequena). |
| `n_consultas_previas` | int32 | 0.00 | 18 | historico | t0 | nao | Contagem de consultas do paciente com data_ev < t0 (regra do qmd). Conferida pela auditoria. |
| `n_faltas_previas` | int32 | 0.00 | 8 | historico | t0 | nao | Idem, so as faltantes. |
| `falta_recente` | category | 0.00 | 4 | historico | t0 | nao | Categoria de tempo desde a ultima falta anterior a t0. |
| `primeira_vez` | int32 | 0.00 | 2 | historico | t0 | nao | 1 se n_consultas_previas == 0. |
| `n_consultas_previas_especialidademae` | int32 | 0.00 | 13 | historico | t0 | nao | Historico na mesma especialidade. |
| `n_faltas_previas_especialidademae` | int32 | 0.00 | 6 | historico | t0 | nao | Idem. |
| `ultima_previa_mesma_especialidademae` | category | 0.00 | 3 | historico | t0 | nao | Desfecho da ultima consulta previa na mesma especialidade. |
| `n_consultas_previas_motivo` | int32 | 0.00 | 6 | historico | t0 | nao | Historico no mesmo motivo (especialidade + CID3). |
| `n_faltas_previas_motivo` | int32 | 0.00 | 5 | historico | t0 | nao | Idem. |
| `ultima_previa_mesmo_motivo` | category | 0.00 | 3 | historico | t0 | nao | Desfecho da ultima consulta previa no mesmo motivo. |
| `cnes_solicitante` | str | 0.00 | 132 | unidade | solicitacao | nao | CNES da unidade solicitante. |
| `cnesexecutante` | str | 0.00 | 173 | unidade | agendamento (t0) | nao | CNES onde a consulta ocorre. |
| `favela_comunidade` | float64 | 0.00 | 102 | territorio | estatica (censo) | nao | Variavel de setor censitario, fixa no tempo. |
| `taxa_alf_geral` | float64 | 0.00 | 132 | territorio | estatica (censo) | nao | Idem. |
| `renda_media_winsor` | float64 | 0.00 | 132 | territorio | estatica (censo) | nao | Idem. |
| `coordenadoria_saude` | category | 0.00 | 4 | territorio | estatica | nao | Territorio da unidade. |
| `distrito_sanitario` | category | 0.00 | 17 | territorio | estatica | nao | Territorio da unidade. |
| `status` | str | 1.09 | 3 | acesso | estatica (calculo de rota) | suspeita_baixa | Status do calculo de distancia (ok/sem_rota/mesmo_local, NaN coincide com as 7 variaveis de distancia). Pelo padrao de ausencia nao descreve o desfecho. Definicao exata depende do script das distancias (ausente). |
| `distancia_km` | float64 | 1.09 | 4.021 | acesso | estatica (calculo de rota) | nao | Calculada entre unidades, script gerador ausente. |
| `tempo_caminhando` | float64 | 1.09 | 3.962 | acesso | estatica (calculo de rota) | nao | Idem. |
| `distancia_km_trans_publico` | float64 | 1.09 | 4.080 | acesso | estatica (calculo de rota) | nao | Idem. |
| `tempo_desloc_trans_publico` | float64 | 1.09 | 2.766 | acesso | estatica (calculo de rota) | nao | Idem. |
| `quant_bus` | float64 | 1.09 | 5 | acesso | estatica (calculo de rota) | nao | Idem. |
| `baldiacoes` | float64 | 1.09 | 4 | acesso | estatica (calculo de rota) | nao | Idem. |
| `DS_NATUREZA_JUR` | str | 0.44 | 10 | unidade | estatica (cadastro CNES) | nao | Natureza juridica do estabelecimento, usada para o recorte municipal. 4.491 ausentes. |

## Notas

- O parquet contém só `situacaoagenda` ∈ {REALIZADA, FALTANTE}. As demais categorias foram removidas no
  `reestruturacao.qmd`; a contagem delas está no arquivo bruto, que não está disponível (pendência).
- O script que gerou as distâncias (`status`, `distancia_km`, ...) não está versionado (pendência).
- Variáveis de histórico seguem `data_ev < t0`; a conferência está em `tabelas/auditoria_dados.csv` (bloco `historico`).
