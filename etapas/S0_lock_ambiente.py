"""S0 — Congela o ambiente (requirements.lock) e registra versões e recursos da máquina.

Objetivo
    Gravar as versões EXATAS das bibliotecas usadas pelo projeto e a versão do Python, para
    reprodução em outra máquina. Só inclui os pacotes listados em PACOTES (não o venv inteiro,
    que hoje é compartilhado com outro projeto).

Saídas
    requirements.lock, manifestos/ambiente.json

Como rodar
    python3 etapas/S0_lock_ambiente.py                 # grava requirements.lock e ambiente.json
    python3 etapas/S0_lock_ambiente.py --so-mostrar    # só imprime
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAIZ  # noqa: E402
from src.utils_io import grava_json  # noqa: E402

# DECISÃO: pacotes usados pelo pipeline. Os de modelos opcionais (lightgbm, xgboost, catboost;
# ver requirements-extra.txt) aparecem como "faltando" se não estiverem instalados.
PACOTES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "matplotlib",
           "PyYAML", "joblib", "threadpoolctl", "python-dateutil", "pytest", "lightgbm", "xgboost", "catboost"]


def versoes_instaladas() -> tuple[dict[str, str], list[str]]:
    """Consulta a versão instalada de cada pacote em PACOTES.

    Retorna:
        (dict pacote -> versão para os instalados, lista dos pacotes faltando).
    """
    ok: dict[str, str] = {}
    faltam: list[str] = []
    for p in PACOTES:
        try:
            ok[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            faltam.append(p)
    return ok, faltam


def memoria_gb() -> float | None:
    """Memória RAM total em GB lendo /proc/meminfo (Linux); None se indisponível."""
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        return round(kb / 1024 / 1024, 1)
    except OSError:
        return None


def main() -> None:
    """Imprime o ambiente e (salvo --so-mostrar) grava requirements.lock e manifestos/ambiente.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-mostrar", action="store_true")
    args = ap.parse_args()
    ok, faltam = versoes_instaladas()
    info = {"python": sys.version.split()[0], "executavel": sys.executable, "sistema": platform.platform(),
            "nucleos": os.cpu_count(), "ram_gb": memoria_gb(), "pacotes": ok, "pacotes_faltando": faltam}
    print(info)
    if args.so_mostrar:
        return
    linhas = [f"# Python {info['python']} | gerado por etapas/S0_lock_ambiente.py"]
    linhas += [f"{p}=={v}" for p, v in ok.items()]
    if faltam:
        linhas.append("# OPCIONAIS não instalados (modelos extras; ver requirements-extra.txt): " + ", ".join(faltam))
    (RAIZ / "requirements.lock").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    grava_json(info, RAIZ / "manifestos" / "ambiente.json")
    print("requirements.lock e manifestos/ambiente.json gravados.")


if __name__ == "__main__":
    main()
