"""Testes do contrato dos modelos e do protocolo de tuning (S5-S7) na amostra do smoke.

* Contrato: todo modelo disponível ajusta, prevê no formato certo e devolve médias não negativas.
* ANTI-VAZAMENTO: corromper TUDO a partir do início do teste (desfechos e variáveis) não pode mudar a
  escolha de hiperparâmetros, a dispersão phi, o critério de validação nem k_h (o teste não participa).
* Reprodutibilidade: a mesma unidade rodada duas vezes dá exatamente as mesmas previsões.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "etapas"))
from src.config import carrega_config  # noqa: E402
from src.dados import carrega_contexto  # noqa: E402
from src.experimento import executa_unidade  # noqa: E402
from src.models import REGISTRO, lista_modelos_disponiveis  # noqa: E402
import S7_experimentos as s7  # noqa: E402

AMOSTRA = RAIZ / "dados" / "derivados" / "amostra_smoke_municipal.parquet"
pytestmark = pytest.mark.skipif(not AMOSTRA.exists(), reason="rode etapas/S2_amostra_smoke.py antes")


def _cfg(raiz: Path | None = None) -> dict:
    cfg = carrega_config(perfil="smoke")
    cfg["tuning"]["n_configs"] = 2
    cfg["tuning"]["janelas_meses"] = [12, 24]
    if raiz is not None:
        cfg["_raiz"] = str(raiz)
    return cfg


@pytest.fixture(scope="module")
def ctx():
    return carrega_contexto(_cfg(), 3, "conhecida")


def test_contrato_dos_modelos(ctx):
    disponiveis, _ = lista_modelos_disponiveis(list(REGISTRO))
    assert {"naive", "baseline_poisson", "hgb_poisson", "rf", "poisson_reg", "paciente_hgb"} <= set(disponiveis)
    dias = ctx.P["dia"]
    tr = np.flatnonzero(dias < "2024-01-01").astype(int)
    te = np.flatnonzero((dias >= "2024-01-01") & (dias < "2024-04-01")).astype(int)
    k_h = ctx.k_h(tr)
    assert 1.0 <= k_h < 1.6, "k_h deveria reescalar a agenda conhecida para a final (fração conhecida ~ 0,92)"
    for nome in disponiveis:
        cls = REGISTRO[nome]
        params = cls.espaco()[0]
        m = cls(params, seed=0).fit(ctx, tr, k_h)
        mu = m.predict_mean(ctx, te, k_h)
        assert mu.shape == (len(te),), nome
        assert np.nanmin(mu) >= 0, nome
        assert np.isfinite(mu).mean() > 0.95, f"{nome}: muitas previsões NaN"
        d = m.predict_dist(mu, 0.2)
        assert set(d) == {"mu", "phi"} and (d["mu"] > 0).all()


def test_mesma_unidade_duas_vezes_da_o_mesmo_resultado(ctx):
    cfg = _cfg()
    divs = s7.divisoes_da_variante(cfg, ctx, "expansiva")
    a = executa_unidade(ctx, "hgb_poisson", divs["expansiva_2025"], cfg, "expansiva")
    b = executa_unidade(ctx, "hgb_poisson", divs["expansiva_2025"], cfg, "expansiva")
    np.testing.assert_array_equal(a[0]["mu"].to_numpy(), b[0]["mu"].to_numpy())
    assert a[1]["params"] == b[1]["params"] and a[1]["phi"] == b[1]["phi"]


@pytest.mark.parametrize("modelo", ["hgb_poisson", "poisson_reg", "paciente_hgb", "ets", "glm_nb", "xgboost", "catboost",
                                    "mlp", "hgb_poisson_k", "sarima"])
def test_corromper_o_teste_nao_muda_o_tuning(tmp_path, modelo):
    """ANTI-VAZAMENTO: desfechos e variáveis embaralhados a partir de 2025-01-01 não alteram escolhas feitas antes."""
    T = pd.Timestamp("2025-01-01")
    cfg = _cfg()
    ctx0 = carrega_contexto(cfg, 3, "conhecida")
    d = pd.read_parquet(AMOSTRA)
    rng = np.random.default_rng(0)
    fut = pd.to_datetime(d["datahora_consulta"]) >= T
    ruim = d.copy()
    sit = ruim.loc[fut, "situacaoagenda"].astype(str).to_numpy().copy()
    rng.shuffle(sit)
    ruim["situacaoagenda"] = ruim["situacaoagenda"].astype(str)
    ruim.loc[fut, "situacaoagenda"] = sit
    for c in ("idade", "n_faltas_previas", "dias_ate_consulta"):
        ruim.loc[fut, c] = rng.permutation(ruim.loc[fut, c].to_numpy())
    destino = tmp_path / "dados" / "derivados"
    destino.mkdir(parents=True)
    ruim.to_parquet(destino / "amostra_smoke_municipal.parquet")
    ctx1 = carrega_contexto(_cfg(tmp_path), 3, "conhecida")
    assert (ctx0.P["f"] != ctx1.P["f"]).any(), "a corrupção deveria alterar o desfecho futuro"
    divs0 = s7.divisoes_da_variante(cfg, ctx0, "expansiva")["expansiva_2025"]
    divs1 = s7.divisoes_da_variante(cfg, ctx1, "expansiva")["expansiva_2025"]
    m0 = executa_unidade(ctx0, modelo, divs0, cfg, "expansiva")[1]
    m1 = executa_unidade(ctx1, modelo, divs1, cfg, "expansiva")[1]
    for k in ("params", "phi", "criterio_validacao", "k_h", "n_treino", "semente_derivada"):
        assert m0[k] == m1[k], f"{modelo}: '{k}' mudou quando só o teste foi corrompido (vazamento!)"


@pytest.mark.parametrize("modelo", ["ets", "hgb_poisson_k", "sarima"])
@pytest.mark.parametrize("L", [3, 14])
def test_modelos_de_serie_so_usam_desfechos_ate_t_menos_L(ctx, modelo, L):
    """ANTI-VAZAMENTO: alterar os desfechos de dias > t − L não muda a previsão de t (série e recalibração k)."""
    import dataclasses
    cfg = _cfg()
    c = carrega_contexto(cfg, L, "conhecida") if L != ctx.lag else ctx
    dias = c.P["dia"]
    tr = np.flatnonzero(dias < "2024-01-01").astype(int)
    alvo = np.flatnonzero((dias >= "2024-02-01") & (dias <= "2024-02-20")).astype(int)
    limite = pd.Timestamp("2024-02-20") - pd.Timedelta(days=L)       # última data que o dia 20/fev pode usar
    k_h = c.k_h(tr)
    cls = REGISTRO[modelo]
    par = cls.espaco()[0]
    mu0 = cls(par, 0).fit(c, tr, k_h).predict_mean(c, alvo, k_h)
    P2 = c.P.copy()
    futuro = (P2["dia"] > limite).to_numpy()
    P2.loc[futuro, "f"] = (P2.loc[futuro, "f"].to_numpy() * 3 + 5)
    c2 = dataclasses.replace(c, P=P2)
    mu1 = cls(par, 0).fit(c2, tr, k_h).predict_mean(c2, alvo, k_h)
    ate = (c.P["dia"].to_numpy()[alvo] <= limite + pd.Timedelta(days=L))   # todos os alvos usam só ≤ alvo − L
    # o alvo mais cedo (1/fev) usa dados ≤ 1/fev − L, que nunca foram alterados; o último usa ≤ limite
    np.testing.assert_allclose(mu0[ate], mu1[ate], rtol=1e-9, err_msg=f"{modelo} L={L}: usou desfecho posterior a t − L")


def test_kalman_confere_com_a_previsao_nativa_do_statsmodels(ctx):
    """O truque do filtro de Kalman (L passos à frente a cada origem) é igual a ``forecast`` do statsmodels."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from src.models.estatisticos import previsao_kalman, serie_cnes
    cnes = ctx.P["cnes"].iloc[0]
    y = serie_cnes(ctx, cnes)["y"][-500:]
    ntr, L = 350, 3
    fc = previsao_kalman(y, None, ntr, (1, 1, 1), (0, 1, 1, 7), L)
    kw = dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), trend="n", enforce_stationarity=False, enforce_invertibility=False)
    r0 = SARIMAX(y[:ntr], **kw).fit(disp=False, maxiter=60)
    for s in (ntr + 20, ntr + 60):
        nativo = r0.apply(y[: s + 1]).forecast(L)[-1]
        assert abs(fc[s + L] - nativo) < 1e-6
