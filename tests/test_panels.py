"""Testes de src/panels.py (S3): L parametrizável, 'embaralhar o futuro' e fração da agenda conhecida.

Base sintética: 2 CNES, ~1,5 ano, antecedência de agendamento variável (1 a 30 dias).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.panels import agenda_conhecida_por_h, constroi_painel  # noqa: E402

LAGS = [1, 3, 7, 14]
FEATS_PASSADO = ["r7", "r14", "r28", "r91", "rdow4", "rd1"]
FEATS_AGENDA_H = ["ag_h", "m_taxaprev_h", "m_primeira_h", "m_idade_h", "m_dias_ate_h"]


def _cons(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    linhas = []
    for cnes in ("A", "B"):
        for dia in pd.date_range("2022-01-03", "2023-06-30", freq="D"):
            if dia.weekday() >= 5:
                continue
            for _ in range(rng.integers(2, 9)):
                antec = int(rng.integers(1, 31))
                linhas.append((cnes, dia, dia - pd.Timedelta(days=antec), int(rng.random() < 0.2), antec,
                               rng.random(), float(rng.integers(0, 2)), float(rng.integers(18, 90)), float(antec + 10)))
    return pd.DataFrame(linhas, columns=["cnes", "dia", "t0", "y", "antec", "tp", "primeira_vez", "idade", "dias_ate"])


@pytest.mark.parametrize("lag", LAGS)
def test_embaralhar_o_futuro_nao_muda_features_do_passado(lag):
    """ANTI-VAZAMENTO: alterar desfechos e acrescentar consultas depois de T não muda r*/rdow4/rd1 dos dias <= T+L."""
    base = _cons()
    T = pd.Timestamp("2022-12-15")
    P0 = constroi_painel(base, lag, dia_final=pd.Timestamp("2023-06-30"))
    fut = base["dia"] > T
    alt = base.copy()
    alt.loc[fut, "y"] = 1 - alt.loc[fut, "y"]                       # inverte todos os desfechos futuros
    extra = base[fut].sample(500, random_state=1).copy()            # e inventa consultas futuras
    extra["y"] = 1
    alt = pd.concat([alt, extra], ignore_index=True)
    P1 = constroi_painel(alt, lag, dia_final=pd.Timestamp("2023-06-30"))
    ate = P0["dia"] <= T + pd.Timedelta(days=lag)
    pd.testing.assert_frame_equal(P0.loc[ate, ["cnes", "dia"] + FEATS_PASSADO].reset_index(drop=True),
                                  P1.loc[ate, ["cnes", "dia"] + FEATS_PASSADO].reset_index(drop=True))
    # E há mudança depois (o teste não é trivial): o próprio desfecho f muda.
    assert not np.allclose(P0.loc[~ate, "f"], P1.loc[~ate, "f"])


@pytest.mark.parametrize("lag", LAGS)
def test_agenda_conhecida_ignora_o_que_foi_agendado_depois(lag):
    """ANTI-VAZAMENTO: consultas agendadas depois de T não alteram ag_h/m_*_h dos dias <= T+L."""
    base = _cons()
    T = pd.Timestamp("2022-12-15")
    P0 = constroi_painel(base, lag, dia_final=pd.Timestamp("2023-06-30"))
    depois = base["t0"] > T
    alt = base.copy()
    alt.loc[depois, ["tp", "idade", "dias_ate"]] = 999.0
    alt = pd.concat([alt, base[depois].assign(t0=lambda x: x["dia"] - pd.Timedelta(days=0), antec=0)], ignore_index=True)
    P1 = constroi_painel(alt, lag, dia_final=pd.Timestamp("2023-06-30"))
    ate = P0["dia"] <= T + pd.Timedelta(days=lag)
    pd.testing.assert_frame_equal(P0.loc[ate, ["cnes", "dia"] + FEATS_AGENDA_H].reset_index(drop=True),
                                  P1.loc[ate, ["cnes", "dia"] + FEATS_AGENDA_H].reset_index(drop=True))


def test_taxa_defasada_confere_a_mao():
    """r7 do dia t com L=3 = faltas dos dias [t−9, t−3] / consultas dos mesmos dias (conta feita à mão)."""
    base = _cons()
    P = constroi_painel(base, 3)
    cnes, dia = "A", pd.Timestamp("2023-02-14")
    janela = base[(base["cnes"] == cnes) & (base["dia"] >= dia - pd.Timedelta(days=9)) & (base["dia"] <= dia - pd.Timedelta(days=3))]
    esperado = janela["y"].sum() / len(janela)
    obtido = P[(P["cnes"] == cnes) & (P["dia"] == dia)]["r7"].iloc[0]
    assert obtido == pytest.approx(esperado)


def test_rd1_e_rdow4_respeitam_o_lag_14():
    """Com L=14 o dia t−7 não pode entrar (só t−14, t−21, ...): alterar o desfecho de t−7 não muda rd1/rdow4."""
    base = _cons()
    P0 = constroi_painel(base, 14)
    dia = pd.Timestamp("2023-03-15")
    alt = base.copy()
    m = (alt["cnes"] == "A") & (alt["dia"] == dia - pd.Timedelta(days=7))
    alt.loc[m, "y"] = 1 - alt.loc[m, "y"]
    P1 = constroi_painel(alt, 14)
    a = P0[(P0["cnes"] == "A") & (P0["dia"] == dia)][["rd1", "rdow4", "r7"]].iloc[0]
    b = P1[(P1["cnes"] == "A") & (P1["dia"] == dia)][["rd1", "rdow4", "r7"]].iloc[0]
    pd.testing.assert_series_equal(a, b)


def test_fracao_conhecida_monotonica_em_h_e_limites():
    """A fração da agenda conhecida cai (ou fica igual) quando h aumenta e fica entre 0 e 1."""
    t = agenda_conhecida_por_h(_cons(), [1, 2, 3, 5, 7, 10, 14])
    assert (t["fracao"].diff().dropna() <= 1e-12).all()
    assert ((t["fracao"] >= 0) & (t["fracao"] <= 1)).all()
    assert t["fracao"].iloc[0] > t["fracao"].iloc[-1]


def test_ag_h_nunca_inclui_consultas_com_t0_posterior():
    """ag_h(t) <= ag(t) sempre, e ag_h só conta consultas com antecedência >= L."""
    base = _cons()
    for lag in LAGS:
        P = constroi_painel(base, lag)
        assert (P["ag_h"] <= P["ag"]).all()
        esperado = base[base["antec"] >= lag].groupby(["cnes", "dia"]).size()
        obtido = P.set_index(["cnes", "dia"])["ag_h"]
        assert (obtido.loc[esperado.index] == esperado).all()


def test_painel_tem_todos_os_dias_e_zeros_sem_agenda():
    """O painel inclui fins de semana (ag = 0) e cobre o intervalo contínuo por CNES."""
    P = constroi_painel(_cons(), 3)
    for _, x in P.groupby("cnes"):
        assert x["dia"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert (P.loc[P["dia"].dt.dayofweek >= 5, "ag"] == 0).all()
