"""S0 — Confere (ou grava) os hashes SHA-256 dos arquivos de entrada do pipeline.

Objetivo
    Garantir, na máquina onde o pipeline for rodado, que os dados de entrada são EXATAMENTE os mesmos usados
    nos testes (mesmo parquet, mesmas listas de CNES). Compara com dados/hashes_esperados.json.

Entradas   arquivos listados em dados/hashes_esperados.json (caminhos relativos à raiz do projeto)
Saídas     código de saída 0 (tudo igual) ou 1 (algum diferente/ausente); com --gravar, (re)escreve o JSON
Como rodar
    python3 etapas/S0_confere_hash.py             # confere
    python3 etapas/S0_confere_hash.py --gravar    # grava os hashes atuais (só na máquina de origem)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAIZ, carrega_config  # noqa: E402
from src.utils_io import sha256_arquivo  # noqa: E402

ARQ_HASHES = RAIZ / "dados" / "hashes_esperados.json"


def arquivos_de_entrada(cfg: dict) -> list[str]:
    """Lista (relativa à raiz) dos arquivos de entrada que definem a reprodução."""
    d = cfg["dados"]
    return [d["base"], d["estabelecimentos"], d["flag_municipal"], d["niveis_municipal"], "coorte_2019_2026.csv", "comparacao_series.csv"]


def main() -> int:
    """Confere ou grava os hashes; devolve o código de saída."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravar", action="store_true")
    args = ap.parse_args()
    cfg = carrega_config()
    arqs = arquivos_de_entrada(cfg)
    if args.gravar:
        atual = {a: sha256_arquivo(RAIZ / a) for a in arqs}
        ARQ_HASHES.write_text(json.dumps(atual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[hash] gravado {ARQ_HASHES.name} com {len(atual)} arquivos.")
        return 0
    esperado = json.loads(ARQ_HASHES.read_text(encoding="utf-8"))
    ok = True
    for a, h in esperado.items():
        p = RAIZ / a
        if not p.exists():
            print(f"[hash] AUSENTE   {a}")
            ok = False
            continue
        h2 = sha256_arquivo(p)
        print(f"[hash] {'OK       ' if h2 == h else 'DIFERENTE'} {a}")
        ok &= h2 == h
    print("[hash] entradas conferem." if ok else "[hash] ENTRADAS DIFERENTES: não rode o pipeline completo antes de entender a diferença.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
