# Justificativa do horizonte (h) e do atraso de disponibilidade (d) — AMOSTRA SMOKE

Gerado por `etapas/S3_1_horizonte.py` (escopo `municipal`). **Descritivo: não escolhe L.** O h principal é um critério operacional, a declarar no portão G2 (com fonte) antes de ver o teste.

## 1. Antecedência do agendamento (consulta − t0)

- Mediana 14 dias; p75 14; p95 14; p99 14; máximo 35.
- Fração com antecedência **exatamente 14 dias**: 57.1%; maior que 14: 0.1%.
- Por ano, fração exatamente 14: 2018: 50%; 2019: 46%; 2020: 52%; 2021: 65%; 2022: 56%; 2023: 59%; 2024: 58%; 2025: 62%; 2026: 63%.
- **Atenção:** a concentração em 14 dias sugere regra de abertura de agenda (ou `t0` nominal). Confirmar com o serviço antes de
  interpretar a reconstrução da agenda como 'o que existia em t − h'.

## 2. Fração da agenda final já conhecida em t − h

| h (dias) | consultas já agendadas | fração (consultas) | fração média por CNES-dia |
|---|---|---|---|
| 1 | 101.468 | 99.9% | 99.9% |
| 2 | 98.056 | 96.6% | 96.3% |
| 3 | 93.750 | 92.3% | 92.1% |
| 5 | 88.068 | 86.7% | 85.9% |
| 7 | 79.807 | 78.6% | 77.2% |
| 10 | 69.276 | 68.2% | 65.4% |
| 14 | 58.101 | 57.2% | 53.3% |

Se a fração for bem menor que 1, avaliar com a agenda FINAL (`ag`, `m_*`) é otimista; o painel traz as duas versões
(`ag`/`m_*` = final; `ag_h`/`m_*_h` = conhecida em t − L) e `config.yaml: agenda_modo` escolhe qual entra nos modelos.

## 3. Atraso de registro do desfecho (d)

A base **não tem** campo com a data em que o desfecho foi registrado (ver `tabelas/atraso_registro.csv`). d fica como premissa (`config.yaml: atraso_d`, padrão 0 = o desfecho do dia t−L já é conhecido em t−L). **A confirmar com o serviço.** Sensibilidade: o painel aceita d > 0 (defasagem efetiva L + d).

## 4. Memória da série (ACF média da taxa diária de falta)

| lag | 1 | 2 | 3 | 7 | 14 | 21 | 28 |
|---|---|---|---|---|---|---|---|
| ACF | 0.516 | 0.305 | 0.171 | 0.125 | 0.123 | 0.091 | 0.087 |

Orienta os lags das variáveis (ex.: mesmo dia da semana), não o horizonte.

## 5. Critério operacional (a preencher pelo usuário, com fonte)

- Prazo de confirmação/lembrete ao paciente: _a registrar_.
- Fechamento da agenda e encaixe de lista de espera: _a registrar_.
- **h principal declarado:** _pendente (G2)_. Hoje o `config.yaml` usa `lag_principal: 3` como RASCUNHO.

## 6. Curva desempenho × horizonte

Vem dos experimentos com L ∈ {1, 3, 7, 14} (figura `figuras/desempenho_vs_horizonte.*`, etapa S13), como análise de
sensibilidade; nunca como critério de menor erro no teste.
