"""S5 — Reprodução dos números do baseline legado na coorte ORIGINAL (marco M3).

Objetivo
    Conferir que o painel novo (src/panels.py, L = 3, agenda FINAL) e as referências (baseline Poisson e
    naive de 4 semanas) reproduzem, dentro de tolerância, os números do script 17 (comparacao_series.csv)
    na coorte antiga A+B (antes do recorte municipal): baseline WAPE A 0,4511 / B 0,3131; dow4 A 0,4861 /
    B 0,3377 (total ≈ 0,383 para o baseline).

Protocolo
    Só reprodução de referência (sem tuning, sem escolha). Usa a coorte legada coorte_2019_2026.csv
    (A = estáveis, B = lista do script 11) porque o objetivo é reproduzir o passado, não definir coorte.
    Teste de 2023-01-01 a 2026-03-31; linhas = CNES-dias com agenda > 0 e baseline definido.

Entradas   base (parquet), coorte_2019_2026.csv, comparacao_series.csv
Saídas     tabelas/reproducao_baseline.csv (+ código de saída 1 se fora da tolerância)
Como rodar python3 etapas/S5_reproduz_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.metrics import wape  # noqa: E402
from src.panels import COLS_CONSULTA, constroi_painel, prepara_consultas_painel  # noqa: E402

TOLERANCIA_WAPE = 0.005     # DECISÃO: diferença absoluta máxima aceita (o legado recortava linhas por V0/V1 válidos)
TOLERANCIA_N = 0.002        # diferença relativa máxima no nº de CNES-dias (após replicar a regra de linhas do legado)
B_LEGADO = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]   # lista B do script 11/17


def main() -> int:
    """Roda a reprodução e grava a tabela; devolve 0 se tudo estiver dentro da tolerância, 1 caso contrário."""
    cfg = carrega_config()
    co = pd.read_csv(caminho(cfg, "coorte_2019_2026.csv"), dtype={"cnes": str}).set_index("cnes")
    A = list(co.index[co["estavel"]])
    nivel = {**{c: "A" for c in A}, **{c: "B" for c in B_LEGADO}}
    d = pd.read_parquet(caminho(cfg, cfg["dados"]["base"]), columns=COLS_CONSULTA)
    cons = prepara_consultas_painel(d, set(nivel))
    P = constroi_painel(cons, lag=3)
    # O script 17 só avaliava linhas em que o fator de recalibração k dos modelos por paciente era finito; como
    # esse fator foi calculado só com dados de 2023 em diante (28 dias com lag 3, mínimo de 14), as primeiras
    # 17 dias de 2023 de cada CNES ficavam de fora. Reproduzimos essa regra de linhas SÓ aqui (reprodução).
    inicio_legado = pd.Timestamp("2023-01-01") + pd.Timedelta(days=17)
    P = P[(P["ag"] > 0) & (P["dia"] >= inicio_legado)].copy()
    P["nivel"] = P["cnes"].map(nivel)
    P["baseline"] = P["r28"] * P["ag"]
    P["dow4"] = P["rdow4"] * P["ag"]
    leg = pd.read_csv(caminho(cfg, "comparacao_series.csv")).set_index(["nivel", "modelo"])
    linhas, ok = [], True
    for modelo, col, nome_leg in (("baseline_poisson", "baseline", "baseline"), ("naive_dow4", "dow4", "dow4")):
        for niv in ("A", "B"):
            g = P[(P["nivel"] == niv) & P[col].notna()]
            w, n = wape(g["f"], g[col]), len(g)
            w_leg, n_leg = float(leg.loc[(niv, nome_leg), "WAPE"]), float(leg.loc[(niv, nome_leg), "n"])
            dentro = abs(w - w_leg) <= TOLERANCIA_WAPE and abs(n - n_leg) / n_leg <= TOLERANCIA_N
            ok &= dentro
            linhas.append({"modelo": modelo, "nivel": niv, "wape_novo": w, "wape_legado": w_leg, "dif": w - w_leg,
                           "n_novo": n, "n_legado": n_leg, "dentro_da_tolerancia": dentro})
    g = P[P["baseline"].notna()]
    linhas.append({"modelo": "baseline_poisson", "nivel": "A+B", "wape_novo": wape(g["f"], g["baseline"]),
                   "wape_legado": 0.383, "dif": wape(g["f"], g["baseline"]) - 0.383, "n_novo": len(g), "n_legado": np.nan,
                   "dentro_da_tolerancia": abs(wape(g["f"], g["baseline"]) - 0.383) <= TOLERANCIA_WAPE})
    ok &= bool(linhas[-1]["dentro_da_tolerancia"])
    t = pd.DataFrame(linhas)
    destino = caminho(cfg, cfg["saidas"]["tabelas"], "reproducao_baseline.csv")
    t.to_csv(destino, index=False)
    print(t.round(4).to_string(index=False))
    print("[S5] REPRODUZ o legado dentro da tolerância." if ok else "[S5] NÃO reproduz: veja a tabela e investigue antes de seguir.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
