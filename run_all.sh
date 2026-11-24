#!/usr/bin/env bash
# run_all.sh — executa o pipeline completo (perfil "completo") em ordem, com log em logs/run_all_<data>.log.
# Uso:  ./run_all.sh [JOBS] [PYTHON]       ex.: ./run_all.sh 7   (PYTHON padrão: .venv/bin/python, senão python3)
# É retomável: se cair no meio, rode de novo (S7 pula o que já terminou).
set -euo pipefail
cd "$(dirname "$0")"
JOBS="${1:-}"
if [ -x .venv/bin/python ]; then PY_PADRAO=.venv/bin/python; else PY_PADRAO=python3; fi
PY="${2:-$PY_PADRAO}"
mkdir -p logs
LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "[run_all] início $(date -Is) | JOBS=${JOBS:-auto} | PYTHON=$PY" | tee -a "$LOG"
make full ${JOBS:+JOBS=$JOBS} PYTHON="$PY" 2>&1 | tee -a "$LOG"
echo "[run_all] fim $(date -Is)" | tee -a "$LOG"
