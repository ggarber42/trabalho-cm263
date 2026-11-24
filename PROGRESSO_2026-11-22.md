# Progresso 22/11/2026 — redesenho e pipeline v2

Feito:
- Redesenho metodológico decidido ponto a ponto: `DECISOES_REDESENHO_2026-11-22.md` (D1–D14),
  desenho em `docs/superpowers/specs/2026-11-22-redesenho-metodologico-design.md`, plano em
  `docs/superpowers/plans/2026-11-22-pipeline-v2.md`, texto `entregaveis/metodologia_v2.md`.
- Nova pasta autocontida `pipeline_v2/` (não importa nada do pipeline antigo; `src/`, `etapas/`
  e `antigos/` ficam intactos). Etapas e01–e06, 14 testes automáticos, `executar.sh`, README.
- Smoke de ponta a ponta com 1 CNES (2237385), perfil `smoke`, 4 cenários (teste 2023/2024 × h 1/2),
  10 modelos em todos: executado e verificado (≈ 20 min).

Pendente:
- Rodada completa (todas as CNES) — só com aprovação.
- Confirmar critério municipal (HNSC/GHC e HPS ficam fora pela lista de razões sociais).
- Teste de wavelets: `locits` não instalado aqui; regra usou ADF + KPSS.
- Números do `metodologia_v2.md` ([PREENCHER]) a partir das saídas do perfil completo.
