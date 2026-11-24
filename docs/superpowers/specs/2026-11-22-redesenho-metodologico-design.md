# Redesenho metodológico — previsão semanal de faltas por CNES (GERCON, CMP263)

Data: 22/11/2026. Decisões de origem: `DECISOES_REDESENHO_2026-11-22.md` (D1–D11).
Este documento substitui, quando implementado, as partes correspondentes do `CLAUDE.md`
(h = 3 dias, L ∈ {3, 14}, quatro esquemas de divisão, censo e distâncias, baseline Poisson e naive).

## 1. Intenção

Objetivo do estudo (texto do usuário): desenvolver e validar modelos prognósticos de previsão de
faltas, comparando diferentes algoritmos de aprendizado de máquina, em consultas especializadas
de pacientes encaminhados pela APS em Porto Alegre - RS, no nível dos estabelecimentos
executantes de gestão municipal, com abordagem de série temporal, a partir dos dados abertos do
GERCON.

Por que redesenhar: o pipeline anterior ficou complexo demais (série diária, dois lags, quatro
esquemas, três fontes de dados), e os modelos tradicionais de série temporal foram usados sem
análise das características da série nem escolha adequada de hiperparâmetros, o que torna a
comparação com o ML injusta.

## 2. Metas, não-metas, antimetas

Metas
- Um pipeline curto, reprodutível do CSV bruto às tabelas, com cada decisão justificada.
- Comparação justa: todos os modelos com o mesmo cuidado de preparação, tuning de orçamento
  comparável, as mesmas divisões, as mesmas linhas e a mesma distribuição preditiva.
- Esquema de avaliação escolhido por critério explícito a partir da análise exploratória do treino.
- Detalhamento das faltas previstas por especialidade.

Não-metas
- Comparar esquemas de divisão como fator do estudo.
- Séries próprias por especialidade e reconciliação hierárquica.
- Variáveis do censo, distâncias, escopo "geral" (todas as CNES).

Antimetas (falharia mesmo rodando)
- Qualquer decisão (modelo, ordem, hiperparâmetro, janela, esquema, regra de séries) usando 2024.
- Tratar artefatos de extração (2025–2026) como mudança real de nível.
- Modelos tradicionais com ordens fixas sem diagnóstico.

## 3. Dados (D1–D5)

- Fonte: `../gercon/gercon_solicitacoes_executante_porto_alegre_2018_2026.csv` (2.147.900 linhas,
  arquivos anuais por ano de solicitação, 2018–2025). Cadastro CNES
  (`tbEstabelecimento202609.csv`, `tbNaturezaJuridica202609.csv`) só para o filtro municipal.
- Inclusão: `municipiosolicitante` = PORTO ALEGRE; `tipounidade` = CENTRO DE SAUDE/UNIDADE BASICA;
  `complexidade` = MEDIA; `datahora_consulta` preenchida; `situacaoagenda` ∈ {REALIZADA, FALTANTE}.
- CNES municipal: natureza jurídica Município ou Fundação municipal, gestor 431490 ou razão social
  da Prefeitura. Sem exclusão por volume ou inatividade.
- Desfecho por consulta: y = 1 se FALTANTE. Contagem semanal = Σy por CNES-semana.
- Formatação: truncar todas as datas ao dia (2025 não tem hora); `cnesexecutante` e
  `cnes_solicitante` com `zfill(7)` antes do merge (anos antigos perderam o zero à esquerda).
- Período de análise: 2019-04-01 a 2024-12-29 (≈ 300 semanas). 2018-01 a 2019-03 só alimenta o
  histórico. Justificativa: os arquivos de 2018–2024 foram extraídos em 07/01/2025 e o de 2025 em
  01/04/2026 (sem arquivo de 2026); consultas posteriores à extração ficam sem desfecho ou somem
  (a partir de 2025-02 não há consultas de pedidos anteriores a 2025). `data_extracao` é usada só
  como metadado para justificar o corte.
- Histórico do paciente (opção B): calculado só nas consultas que passaram em todos os filtros,
  com `data_ev < t0`, `t0 = data_solicitacao + max(diasfilaespera, 0)` (data de agendamento).
- Todo filtro imprime linhas removidas; antes dos filtros, contagem de `situacaoagenda` por ano.
- Limitações a declarar: AGENDA_CONFIRMADA passadas excluídas (contagens por ano); histórico
  sem consultas em CNES não municipais; 2025–2026 fora da análise.

## 4. Unidade, origem e horizonte (D6, D8)

- Unidade principal: CNES × semana ISO (segunda a domingo).
- Origem da previsão: fim do domingo anterior à semana prevista. Horizonte h = 1 semana;
  h = 2 como sensibilidade.
- Informação permitida: tudo conhecido até a origem. Agenda da semana prevista = consultas com
  `t0` ≤ origem; histórico com `data_ev < t0`.
- Detalhamento: soma das probabilidades do modelo por consulta por CNES × `especialidademae` ×
  semana (bloco B).

## 5. Análise exploratória do treino (D10)

Só com 2019-04 a 2023-12. Cada item alimenta uma decisão:

| Análise | Decide |
|---|---|
| Volume, taxa de falta, semanas sem atendimento por CNES | Quais séries recebem modelos locais (ETS, SARIMA) e quais ficam só com modelos globais |
| Intermitência (ADI e CV², Syntetos-Boylan) | Uso de Croston/TSB por série |
| Tendência, sazonalidade anual, ACF/PACF, quebras (pandemia) | Faixas de busca de SARIMA/ETS, defasagens das variáveis de ML |
| Média × variância | Transformação e família de distribuição (binomial negativa) |
| Estacionariedade: wavelets (Nason 2013), ADF, KPSS, correção para múltiplos testes | Esquema de avaliação (regra abaixo) |
| Fração da agenda conhecida na origem | Construção das variáveis de agenda |

Regra pré-declarada (Cerqueira, Torgo e Mozetič 2020, doi:10.1007/s10994-020-05910-7): se a
maioria das séries for não estacionária, a seleção de modelos usa *holdout* repetido em múltiplos
períodos de teste, sempre para frente; se a maioria for estacionária, validação cruzada em blocos
com folga (hv-blocked). A mesma análise em 2024 é só descritiva.

## 6. Avaliação

- Seleção de modelos e hiperparâmetros: esquema definido na seção 5, só dentro do treino.
- Teste final: 2024 (52 semanas), visto uma vez; treino até 2023-12. Sensibilidade: teste 2023,
  treino até 2022-12.
- Baseline único (D11): média móvel das 4 semanas anteriores da CNES.
- Métricas: WAPE e viés (pontuais); CRPS, cobertura 50%/80% e largura do IC80 (probabilísticas),
  com a mesma família preditiva (binomial negativa, dispersão estimada fora da amostra) para todos;
  por consulta (bloco B): Brier, AUC, logloss. Sem MASE/MAPE (zeros).
- Diebold-Mariano (HAC) de cada modelo contra o baseline, com correção para múltiplas comparações.
- Recortes: por CNES, por ano de teste, sem o Hospital Materno Infantil Presidente Vargas.
- Todos os modelos nas mesmas linhas (interseção), com linhas perdidas registradas.

## 7. Modelos (D7, D9)

Bloco A (só série + calendário): baseline, ETS, SARIMA, Croston/TSB, regressão de contagem
(Poisson/binomial negativa com defasagens), árvores com gradiente, floresta aleatória, MLP.
Bloco B (série + agenda conhecida + histórico dos pacientes): SARIMAX, regressão de contagem com
exógenas, árvores com gradiente, MLP, modelo por consulta (probabilidade por consulta somada).

Cuidado igual para os tradicionais: transformação quando indicada, ordens e tipo de ETS por
AICc dentro do treino e validação temporal, Ljung-Box e diagnóstico de resíduos registrados por
série. Orçamento de busca comparável entre famílias (mesmo número de configurações avaliadas
ou tempo equivalente, registrado).

## 8. Abordagens consideradas

- Unidade: CNES × semana (escolhida) vs CNES × especialidade × semana (276 séries, mais zeros,
  exige reconciliação; ≈ 30–40% mais esforço) vs diária (pipeline anterior, complexa).
- Informação: dois blocos A/B (escolhida) vs só bloco B (deixaria ETS/Croston em desvantagem).
- Divisão: escolhida pela análise exploratória com regra prévia (escolhida) vs comparar quatro
  esquemas (pipeline anterior).
- Baseline: média móvel 4 semanas (escolhida) vs média de todo o treino (fraca demais) vs 52
  semanas (lenta a mudanças).
- Histórico: só consultas filtradas (B, escolhida) vs base inteira (A, mais completa).

## 9. Reprodutibilidade

Um ponto de entrada (`Makefile`), `config.yaml` com caminhos e parâmetros, sementes fixas,
manifesto SHA-256 do CSV bruto e das saídas, números do texto lidos dos CSVs de resultados.
O pipeline antigo (`src/`, `etapas/`) fica congelado como referência.

## 10. Deixado para quem implementa

- Organização dos scripts e módulos novos, reaproveitando o que servir de `src/` (métricas, DM).
- Bibliotecas (ex.: statsforecast/statsmodels para ETS/SARIMA/Croston, LightGBM).
- Limiar exato de "dados suficientes" para modelos locais, definido no treino e registrado.
- Grades de hiperparâmetros, desde que com orçamento comparável e registrado.

## 11. Escolhas feitas pelo assistente (para conferência)

- Semana ISO e origem no domingo; teste final 2024 e sensibilidade 2023.
- Binomial negativa como distribuição preditiva comum.
- Faixa de modelos dos blocos A e B listada na seção 7.
- Critério de maioria (> 50% das séries) na regra de escolha do esquema.
- Semanas sem nenhuma consulta agendada na CNES: entram como 0 falta no ajuste das séries, mas
  ficam fora da avaliação (todos os modelos preveriam 0 ou quase 0), e o número é registrado.
- Correção para múltiplas comparações no Diebold-Mariano: Holm.
- Variáveis do modelo por consulta: as do `preprocessar_gercon.ipynb` (histórico, CID, especialidade,
  idade, tempos de fila e antecedência, calendário), sem censo e sem distâncias; usar só uma
  parametrização de tempo (`dias_ate_consulta` ou `diasfilaespera` + `antecedencia_agendamento`).
