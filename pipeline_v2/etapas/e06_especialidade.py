"""e06 — Resumo do erro por especialidade-mãe (D16).

Objetivo: com os modelos ajustados no nível CNES × especialidade-mãe × semana (D16), resumir,
para cada modelo, as faltas observadas e previstas e o erro em cada especialidade-mãe (somando
as CNES), nas mesmas linhas da interseção usada no e05.
Entradas: previsoes.parquet.
Saídas: especialidade_resumo.csv (teste_ano, h, modelo, especialidademae, n_series,
        faltas_observadas, faltas_previstas, wape, vies, mae).
Uso: python3 etapas/e06_especialidade.py --config config.yaml --perfil smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.metricas import vies, wape  # noqa: E402

CHAVE = ["serie", "semana"]


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e06_especialidade")
    prev = pd.read_parquet(saida / "previsoes.parquet")
    resumo = []
    for (ano, h, estrato), g in prev.groupby(["teste_ano", "h", "estrato"]):
        # Mesmas linhas do e05: interseção dos modelos ajustados no estrato (D17).
        larga = g.pivot_table(index=CHAVE, columns="modelo", values="mu", aggfunc="first")
        cols = [m for m in larga.columns if larga[m].notna().any()]
        inter = larga.index[larga[cols].notna().all(axis=1)]
        g = g[g["modelo"].isin(cols)].set_index(CHAVE).loc[lambda d: d.index.isin(inter)].reset_index()
        for (mod, esp), s in g.groupby(["modelo", "especialidademae"]):
            y, mu = s["y"].to_numpy(float), s["mu"].to_numpy(float)
            resumo.append({"teste_ano": ano, "h": h, "estrato": estrato, "modelo": mod, "especialidademae": esp,
                           "n_series": s["serie"].nunique(), "faltas_observadas": y.sum(),
                           "faltas_previstas": mu.sum(), "wape": wape(y, mu), "vies": vies(y, mu),
                           "mae": float(np.abs(y - mu).mean())})
    r = pd.DataFrame(resumo).sort_values(["teste_ano", "h", "estrato", "modelo", "faltas_observadas"],
                                         ascending=[True, True, True, True, False])
    r.to_csv(saida / "especialidade_resumo.csv", index=False)
    log.info(f"{r['especialidademae'].nunique()} especialidades resumidas")
    log.manifesto([saida / "previsoes.parquet"], [saida / "especialidade_resumo.csv"])


if __name__ == "__main__":
    main()
