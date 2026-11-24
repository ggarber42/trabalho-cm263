"""Testes de src/metrics.py e src/dm_test.py: CRPS, cobertura, dispersão, DM, Holm, bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dm_test import bootstrap_ganho_wape, dm_hac, holm, perda_semanal  # noqa: E402
from src.metrics import (cobertura_e_largura, crps_nb, ece_prob_k, estima_dispersao, metricas_consulta,  # noqa: E402
                         resume_modelo, vies, wape)


def test_wape_e_vies_a_mao():
    y, mu = np.array([2, 0, 4]), np.array([1.0, 1.0, 5.0])
    assert wape(y, mu) == pytest.approx((1 + 1 + 1) / 6)
    assert vies(y, mu) == pytest.approx(7 / 6)


def test_crps_confere_com_monte_carlo():
    """CRPS = E|X−y| − ½E|X−X'| (estimado por simulação) para uma NB conhecida."""
    rng = np.random.default_rng(0)
    mu, phi, y = 4.0, 0.3, 6
    n = 1 / phi
    x1 = rng.negative_binomial(n, n / (n + mu), 400_000)
    x2 = rng.negative_binomial(n, n / (n + mu), 400_000)
    mc = np.abs(x1 - y).mean() - 0.5 * np.abs(x1 - x2).mean()
    assert crps_nb(np.array([y]), np.array([mu]), phi)[0] == pytest.approx(mc, abs=0.02)


def test_crps_melhor_quando_media_certa():
    rng = np.random.default_rng(1)
    y = rng.poisson(5, 3000)
    bom = crps_nb(y, np.full(3000, 5.0), 0.0).mean()
    ruim = crps_nb(y, np.full(3000, 9.0), 0.0).mean()
    assert bom < ruim


def test_dispersao_recupera_phi_e_cobertura_nominal():
    rng = np.random.default_rng(2)
    mu = rng.uniform(2, 12, 40_000)
    phi = 0.25
    n = 1 / phi
    y = rng.negative_binomial(n, n / (n + mu))
    assert estima_dispersao(y, mu) == pytest.approx(phi, abs=0.03)
    cob80, larg = cobertura_e_largura(y, mu, phi, 0.8)
    assert 0.78 <= cob80 <= 0.90 and larg > 0                # discreta: cobertura >= nominal
    assert estima_dispersao(rng.poisson(mu), mu) < 0.03     # Poisson puro → phi ≈ 0
    assert all(v < 0.02 for v in ece_prob_k(y, mu, phi).values())


def test_resume_modelo_chaves_e_sem_nan():
    rng = np.random.default_rng(3)
    mu = rng.uniform(1, 8, 2000)
    r = resume_modelo(rng.poisson(mu), mu, 0.05)
    for k in ("wape", "vies", "crps", "cob80", "cob50", "larg80", "ece", "auc_celula", "phi", "n"):
        assert k in r and np.isfinite(r[k])


def test_metricas_consulta():
    r = metricas_consulta(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3]))
    assert r["auc"] == 1.0 and r["brier"] < 0.1


def test_dm_detecta_diferenca_e_nao_inventa():
    rng = np.random.default_rng(4)
    nulo = dm_hac(rng.normal(0, 1, 150))
    real = dm_hac(rng.normal(0.5, 1, 150))
    assert nulo["p"] > 0.05 and real["p"] < 0.001 and real["media"] > 0


def test_dm_hac_corrige_autocorrelacao():
    """Com d autocorrelacionado, o erro-padrão HAC é maior que o ingênuo."""
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 300)
    d = np.zeros(300)
    for t in range(1, 300):
        d[t] = 0.8 * d[t - 1] + e[t]
    r = dm_hac(d)
    assert r["se"] > d.std() / np.sqrt(300)


def test_holm():
    aj = holm([0.01, 0.04, 0.03])
    assert aj[0] == pytest.approx(0.03) and aj[2] == pytest.approx(0.06) and aj[1] == pytest.approx(0.06)
    assert (aj >= np.array([0.01, 0.04, 0.03])).all() and (aj <= 1).all()


def test_perda_semanal_e_bootstrap():
    dias = pd.date_range("2024-01-01", periods=70, freq="D")
    df = pd.DataFrame({"dia": dias, "perda": 1.0})
    s = perda_semanal(df, "perda")
    assert len(s) == 10 and (s == 7).all()
    idx = s.index
    base = pd.DataFrame({"erro_abs": 10.0, "y": 8.0}, index=idx)
    mod = pd.DataFrame({"erro_abs": 8.0, "y": 8.0}, index=idx)
    r = bootstrap_ganho_wape(base, mod, n_boot=200)
    assert r["ganho"] == pytest.approx(0.2) and r["ic_inf"] <= r["ganho"] <= r["ic_sup"]


def test_dispersao_por_linha_equivale_ao_escalar():
    """phi como array constante dá o mesmo CRPS/cobertura que o escalar; phi diferentes por bloco funcionam."""
    rng = np.random.default_rng(6)
    mu = rng.uniform(1, 9, 500)
    y = rng.poisson(mu)
    a = crps_nb(y, mu, 0.2)
    b = crps_nb(y, mu, np.full(500, 0.2))
    np.testing.assert_allclose(a, b)
    phi = np.where(np.arange(500) < 250, 0.1, 0.5)
    c = crps_nb(y, mu, phi)
    np.testing.assert_allclose(c[:250], crps_nb(y[:250], mu[:250], 0.1))
    np.testing.assert_allclose(c[250:], crps_nb(y[250:], mu[250:], 0.5))
    assert np.isfinite(resume_modelo(y, mu, phi)["crps"])
