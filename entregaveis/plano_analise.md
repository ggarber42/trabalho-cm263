# Plano de análise congelado (portão G2) — 17/11/2026

Declarado pelo usuário **antes** de qualquer resultado de teste em dados completos. Valores em `config.yaml`. Alterar só com registro datado e justificativa.

| Item | Valor congelado | Papel das alternativas |
|---|---|---|
| L principal | **3** (caso operacional: agenda e decisão de overbooking) | L = 14 roda completo e é **sensibilidade** |
| Esquema de divisão principal | **expansiva** (testes por ano, 2023–2026) | holdout, deslizante e purged+embargo são **sensibilidade**, comparados só no período comum |
| Modo da agenda | **conhecida** (agenda existente em t−L, reescalada por k_h do treino) | `final` é **sensibilidade** |
| Período de teste comum | **2024-01-01 a 2026-03-31** | usado nas comparações entre esquemas |
| Métrica primária | **WAPE** (ponto) e **CRPS** (probabilístico); ganho sobre o baseline por Diebold-Mariano (HAC semanal, Holm) | demais métricas (viés, cobertura, ECE, AUC, Brier) são secundárias |
| Escopo | municipal (parâmetro; geral não coberto) | — |

Regras associadas: o teste externo não escolhe entre variantes; comparações de variantes são descritivas; o "melhor modelo" é identificado no mesmo teste (descritivo); Holm aplicado dentro de cada família de comparações, sem correção entre famílias.

`t0` confirmado em 17/11/2026: é a data de agendamento (`data_solicitacao + max(diasfilaespera, 0)`; `diasfilaespera` = tempo até o agendamento, não até a consulta). Premissas ainda não confirmadas (afetam a interpretação, não o congelamento): atraso de registro do desfecho, `possuiunidadeindicada`, `idade`.
