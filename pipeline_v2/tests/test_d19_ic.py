"""Testes D19: taxa da referência ≤ 1 e IC/margem de erro do bootstrap."""
import numpy as np
import pandas as pd

from faltas.metricas import bootstrap_metricas


def test_bootstrap_metricas_contem_estimativa():
    rng = np.random.default_rng(0)
    tab = pd.DataFrame({"semana": np.repeat(np.arange(52), 5), "y": rng.poisson(5, 260).astype(float)})
    tab["m"] = tab["y"] + rng.normal(0, 1, 260)
    r = bootstrap_metricas(tab, "m", 500, 4, 1)
    w = (tab.y - tab.m).abs().sum() / tab.y.sum()
    assert r["wape_ic_inf"] <= w <= r["wape_ic_sup"]
    assert np.isclose(r["wape_moe"], (r["wape_ic_sup"] - r["wape_ic_inf"]) / 2)
