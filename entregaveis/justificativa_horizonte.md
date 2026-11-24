# Justificativa do horizonte (h) e do atraso de disponibilidade (d)

Gerado por `etapas/S3_1_horizonte.py` (escopo `municipal`). **Descritivo: não escolhe L.** O h principal é um critério operacional, a declarar no portão G2 (com fonte) antes de ver o teste.

## 1. Antecedência do agendamento (consulta − t0)

- Mediana 14 dias; p75 14; p95 14; p99 14; máximo 64.
- Fração com antecedência **exatamente 14 dias**: 55.8%; maior que 14: 0.2%.
- Por ano, fração exatamente 14: 2018: 47%; 2019: 51%; 2020: 55%; 2021: 60%; 2022: 59%; 2023: 54%; 2024: 55%; 2025: 56%; 2026: 63%.
- **Atenção:** a concentração em 14 dias sugere regra de abertura de agenda (ou `t0` nominal). Confirmar com o serviço antes de
  interpretar a reconstrução da agenda como 'o que existia em t − h'.

## 2. Fração da agenda final já conhecida em t − h

| h (dias) | consultas já agendadas | fração (consultas) | fração média por CNES-dia |
|---|---|---|---|
| 1 | 360.142 | 99.8% | 99.7% |
| 2 | 347.755 | 96.4% | 96.5% |
| 3 | 333.504 | 92.4% | 93.3% |
| 5 | 316.175 | 87.6% | 88.9% |
| 7 | 285.926 | 79.2% | 77.9% |
| 10 | 243.289 | 67.4% | 61.5% |
| 14 | 202.113 | 56.0% | 48.1% |

Se a fração for bem menor que 1, avaliar com a agenda FINAL (`ag`, `m_*`) é otimista; o painel traz as duas versões
(`ag`/`m_*` = final; `ag_h`/`m_*_h` = conhecida em t − L) e `config.yaml: agenda_modo` escolhe qual entra nos modelos.

## 3. Atraso de registro do desfecho (d)

A base **não tem** campo com a data em que o desfecho foi registrado (ver `tabelas/atraso_registro.csv`). d fica como premissa (`config.yaml: atraso_d`, padrão 0 = o desfecho do dia t−L já é conhecido em t−L). **A confirmar com o serviço.** Sensibilidade: o painel aceita d > 0 (defasagem efetiva L + d).

## 4. Memória da série (ACF média da taxa diária de falta)

| lag | 1 | 2 | 3 | 7 | 14 | 21 | 28 |
|---|---|---|---|---|---|---|---|
| ACF | 0.786 | 0.648 | 0.536 | 0.330 | 0.209 | 0.169 | 0.152 |

Orienta os lags das variáveis (ex.: mesmo dia da semana), não o horizonte.

## 5. Critério operacional (a preencher pelo usuário, com fonte)

- Prazo de confirmação/lembrete ao paciente: _a registrar_.
- Fechamento da agenda e encaixe de lista de espera: _a registrar_.
- **h principal declarado:** _pendente (G2)_. Hoje o `config.yaml` usa `lag_principal: 3` como RASCUNHO.

## 6. Curva desempenho × horizonte

Vem dos experimentos com L ∈ {1, 3, 7, 14} (figura `figuras/desempenho_vs_horizonte.*`, etapa S13), como análise de
sensibilidade; nunca como critério de menor erro no teste.
