#!/usr/bin/env bash
# Executa o pipeline v2 de ponta a ponta, em ordem.
# Uso:
#   ./executar.sh                                   # perfil completo, todas as CNES, config.yaml
#   ./executar.sh --perfil smoke --cnes 2237385     # teste rápido com uma CNES
#   ./executar.sh --config config_local.yaml ...    # caminhos de dados desta máquina
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== Testes automáticos =="
"$PY" -m pytest -q tests/

for etapa in e01_preparar e02_painel e03_exploratoria e04_modelar e05_avaliar e06_especialidade e07_figuras d01_auditoria r01_metodologia; do
  echo "== $etapa =="
  "$PY" "etapas/$etapa.py" "$@"
done
echo "Pronto. Resultados em saidas/<perfil>/ (perfil padrão: completo)."
