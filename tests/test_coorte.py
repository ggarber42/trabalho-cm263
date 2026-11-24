"""Testes de src/coorte.py (S2): coorte e níveis de um corte não dependem de dados posteriores.

Usa uma base sintética (3 CNES, 2019-2026) e, quando disponível, a base real do escopo municipal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.config import carrega_config  # noqa: E402
from src.coorte import coorte_e_niveis, cnes_do_escopo, nivel_da_cnes, prepara_consultas  # noqa: E402

CFG = carrega_config()["coorte"]


def _sintetica(seed: int = 0) -> pd.DataFrame:
    """CNES X estável (taxa 20%), Y com quebra em 2021-07 (15% -> 35%), Z presente só a partir de 2022."""
    rng = np.random.default_rng(seed)
    linhas = []
    for dia in pd.date_range("2019-01-01", "2026-03-31", freq="D"):
        if dia.weekday() >= 5:
            continue
        for cnes, taxa, inicio in (("X", 0.20, "2019-01-01"), ("Y", 0.15 if dia < pd.Timestamp("2021-07-01") else 0.35, "2019-01-01"),
                                   ("Z", 0.20, "2022-01-01")):
            if dia < pd.Timestamp(inicio):
                continue
            for _ in range(3):
                linhas.append((cnes, dia, int(rng.random() < taxa)))
    return pd.DataFrame(linhas, columns=["cnesexecutante", "dia", "falta"])


def test_nivel_regra():
    assert nivel_da_cnes(0, 0.2) == "A"
    assert nivel_da_cnes(1, 0.2) == "B"
    assert nivel_da_cnes(2, 0.2) == "B"
    assert nivel_da_cnes(3, 0.2) == "C"
    assert nivel_da_cnes(1, 0.03) == "C"


def test_coorte_sintetica_niveis_e_presenca():
    """X é A, Y tem quebra (B) e Z não entra por não ter todos os anos."""
    c = coorte_e_niveis(_sintetica(), "2024-01-01", CFG)
    assert bool(c.loc["X", "coorte"]) and c.loc["X", "nivel"] == "A"
    assert bool(c.loc["Y", "coorte"]) and c.loc["Y", "nivel"] == "B"
    assert not bool(c.loc["Z", "coorte"])


def test_coorte_nao_muda_ao_acrescentar_ou_alterar_dados_posteriores():
    """ANTI-VAZAMENTO: alterar/embaralhar tudo a partir do corte não muda a coorte nem os níveis."""
    base = _sintetica()
    corte = pd.Timestamp("2024-01-01")
    ref = coorte_e_niveis(base, corte, CFG)
    # (a) truncar os dados no corte
    trunc = coorte_e_niveis(base[base["dia"] < corte], corte, CFG)
    # (b) adulterar o futuro: inverter desfechos e apagar a CNES X depois do corte
    adult = base.copy()
    fut = adult["dia"] >= corte
    adult.loc[fut, "falta"] = 1 - adult.loc[fut, "falta"]
    adult = adult[~(fut & (adult["cnesexecutante"] == "X"))]
    modif = coorte_e_niveis(adult, corte, CFG)
    pd.testing.assert_frame_equal(ref, trunc)
    pd.testing.assert_frame_equal(ref, modif)


def test_coorte_exige_dados_antes_do_corte():
    with pytest.raises(AssertionError):
        coorte_e_niveis(_sintetica()[lambda x: x["dia"] >= "2025-01-01"], "2024-01-01", CFG)


def test_criterio_x_nao_assumido():
    cfg = carrega_config()
    cfg["escopo"] = "criterio_x"
    with pytest.raises(NotImplementedError):
        cnes_do_escopo(cfg)


@pytest.mark.skipif(not (RAIZ / carrega_config()["dados"]["base"]).exists(), reason="base real ausente")
def test_base_real_municipal_coorte_de_2023_nao_depende_de_2023_em_diante():
    """Na base real: a coorte de 2023 calculada com tudo é igual à calculada só com dados < 2023."""
    cfg = carrega_config()
    d = pd.read_parquet(RAIZ / cfg["dados"]["base"], columns=["cnesexecutante", "datahora_consulta", "situacaoagenda"])
    cons = prepara_consultas(d, cnes_do_escopo(cfg))
    corte = pd.Timestamp("2023-01-01")
    pd.testing.assert_frame_equal(coorte_e_niveis(cons, corte, cfg["coorte"]),
                                  coorte_e_niveis(cons[cons["dia"] < corte], corte, cfg["coorte"]))
