# Relatório de decisão — faltas e chamada da fila de espera
*Simulação retrospectiva com o histórico de 18/01/2023 a 31/03/2026 (167 semanas, 15029 CNES-dias). Nenhuma regra foi aplicada na prática.*

## Como ler
Cada CNES recebe, todo dia, uma previsão de quantas faltas deve ter. Como as faltas variam por acaso, a previsão é uma **distribuição de probabilidade**, não um número só. O gestor escolhe o **risco que aceita**: a chance de chamar da fila **mais pessoas do que as faltas que de fato ocorrerão**, deixando alguém sem vaga naquele dia. Quanto maior o risco aceito, mais pessoas se chamam e mais vagas se preenchem, mas mais dias terminam com pacientes sem vaga. **Este relatório só descreve essas consequências; a escolha do nível de risco é do gestor de cada unidade.**

**Atenção à leitura do "risco aceito":** ele é um **teto** para a chance de sobreoferta prevista pelo modelo. Como o número de faltas é inteiro e pequeno, e em muitos dias a regra manda chamar zero pessoas, a frequência real de dias com sobreoferta (coluna "Dias com sobreoferta") fica abaixo do risco escolhido. Para decidir, use essa coluna, que mostra o que de fato aconteceu no histórico.

Definições: *vaga preenchida* = pessoa chamada que ocupou uma falta; *sobreoferta* = dia em que se chamou mais gente do que houve faltas; *paciente sem vaga* = quem foi chamado e não encontrou vaga.

## Nível A (14 CNES estáveis, principal) — modelo por paciente
Faltas reais: 175 por semana neste conjunto.

| Risco aceito | Chamados/semana | Vagas preenchidas/semana | Faltas recuperadas | Dias com sobreoferta | Pacientes sem vaga/semana | Chamados que acharam vaga |
|---|---|---|---|---|---|---|
| 1% | 18.6 | 18.2 | 10% | 0% | 0.4 | 98% |
| 5% | 45.4 | 44.3 | 25% | 1% | 1.1 | 98% |
| 10% | 65.0 | 62.7 | 36% | 2% | 2.3 | 96% |
| 20% | 94.5 | 88.4 | 50% | 7% | 6.2 | 93% |
| 30% | 118.8 | 107.0 | 61% | 13% | 11.9 | 90% |
| 40% | 141.5 | 121.7 | 69% | 21% | 19.8 | 86% |
| 50% | 164.8 | 134.5 | 77% | 30% | 30.4 | 82% |

## Nível B (6 CNES com mudança moderada, secundário) — baseline (taxa recente × agenda)
Faltas reais: 173 por semana neste conjunto.

| Risco aceito | Chamados/semana | Vagas preenchidas/semana | Faltas recuperadas | Dias com sobreoferta | Pacientes sem vaga/semana | Chamados que acharam vaga |
|---|---|---|---|---|---|---|
| 1% | 40.3 | 40.1 | 23% | 0% | 0.3 | 99% |
| 5% | 69.2 | 68.1 | 39% | 2% | 1.1 | 98% |
| 10% | 87.4 | 85.1 | 49% | 4% | 2.2 | 97% |
| 20% | 112.1 | 106.5 | 62% | 11% | 5.6 | 95% |
| 30% | 131.4 | 121.2 | 70% | 18% | 10.2 | 92% |
| 40% | 149.4 | 133.2 | 77% | 27% | 16.2 | 89% |
| 50% | 166.8 | 143.1 | 83% | 36% | 23.7 | 86% |

## O modelo ajuda na decisão?
Para o mesmo percentual de dias com sobreoferta (nível A), % das faltas recuperadas por cada método:

| Dias com sobreoferta aceitos | Modelo | Baseline | Regra ingênua |
|---|---|---|---|
| 10% | 56% | 53% | 38% |
| 20% | 69% | 66% | 54% |
| 30% | 77% | 74% | n/d |

Detalhe e curvas completas: `relatorio_decisao_fronteira.png` e `relatorio_decisao_niveis.csv`. A regra ingênua chama uma fração da média das faltas do mesmo dia da semana nas 4 semanas anteriores.

## Exemplo de leitura
CNES 2237571 (nível A, 6.6 faltas por dia): com risco aceito de 20%, a regra chamaria em média 4.1 pessoas por dia, recuperaria 60% das faltas e deixaria alguém sem vaga em 9% dos dias.
CNES 2237385 (nível B, 9.7 faltas por dia): no mesmo risco, chamaria 6.8 por dia, recuperaria 68% e teria sobreoferta em 10% dos dias.

## Resultado por CNES (riscos de 10% e 30%)
| CNES | Nível | Faltas/dia | Risco 10%: chamar/dia | recupera | sobreoferta | Risco 30%: chamar/dia | recupera  | sobreoferta  |
|---|---|---|---|---|---|---|---|---|
| 2237571 | A | 6.6 | 3.1 | 46% | 4% | 4.9 | 70% | 18% |
| 2237881 | A | 5.3 | 2.5 | 46% | 3% | 4.0 | 68% | 18% |
| 8006881 | A | 4.8 | 2.0 | 41% | 4% | 3.5 | 67% | 17% |
| 7031076 | A | 3.3 | 1.3 | 40% | 2% | 2.3 | 64% | 12% |
| 2237253 | A | 3.3 | 1.1 | 32% | 3% | 2.2 | 60% | 17% |
| 7214448 | A | 3.0 | 1.0 | 33% | 3% | 2.1 | 62% | 17% |
| 6883206 | A | 2.9 | 1.0 | 33% | 2% | 2.0 | 62% | 16% |
| 2237849 | A | 2.3 | 0.7 | 28% | 4% | 1.4 | 53% | 14% |
| 2265052 | A | 2.0 | 0.4 | 20% | 2% | 1.2 | 50% | 15% |
| 7092571 | A | 1.8 | 0.4 | 23% | 1% | 1.0 | 50% | 12% |
| 2237822 | A | 0.9 | 0.1 | 9% | 0% | 0.3 | 27% | 6% |
| 7210280 | A | 0.7 | 0.0 | 3% | 0% | 0.2 | 25% | 5% |
| 2262568 | A | 0.6 | 0.1 | 10% | 3% | 0.3 | 24% | 7% |
| 2265060 | A | 0.4 | 0.0 | 1% | 0% | 0.1 | 11% | 2% |
| 2237385 | B | 9.7 | 5.5 | 56% | 4% | 7.8 | 76% | 19% |
| 2693801 | B | 7.3 | 4.0 | 52% | 7% | 5.7 | 71% | 22% |
| 9132562 | B | 6.6 | 3.3 | 50% | 4% | 5.0 | 71% | 19% |
| 2237334 | B | 6.5 | 3.3 | 49% | 5% | 5.0 | 70% | 20% |
| 2237601 | B | 3.6 | 1.4 | 37% | 4% | 2.4 | 61% | 15% |
| 7031092 | B | 2.8 | 0.8 | 29% | 3% | 1.8 | 57% | 15% |

Todos os riscos por CNES: `relatorio_decisao_por_cnes.csv`.

## O que este relatório NÃO inclui (leia antes de usar)
- **Vagas intercambiáveis.** Os números somam as faltas do CNES inteiro. Na prática, a vaga aberta por uma falta é de uma especialidade e de um horário específicos, e só serve a quem da fila precisa daquela especialidade. Isso reduz a recuperação real e não foi simulado.
- **Aceitação pelo paciente da fila.** Assume-se que quem é chamado em 3 dias consegue comparecer. Na prática, parte recusará ou não virá, e a recuperação real será menor.
- **Mudança de comportamento.** O modelo aprendeu com dados em que não se chamava a fila. Se a prática mudar o comportamento (de quem é chamado ou de quem falta), as previsões podem deixar de valer.
- **Pequenos CNES.** Com poucas faltas por dia, os resultados oscilam muito. CNES de nível C e D não têm previsão.
- **Calibração.** A dispersão extra do dia foi estimada com o mesmo período, o que é um vazamento mínimo (um número por nível). As probabilidades nas pontas da distribuição são levemente otimistas.
- **Backtest, não piloto.** Convém rodar em modo sombra por algumas semanas antes de agir.

## Arquivos
`relatorio_decisao_niveis.csv` (resultados por nível, método e risco) · `relatorio_decisao_por_cnes.csv` · `relatorio_decisao_fronteira.png`
