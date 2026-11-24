"""Modelos locais de série temporal (uma série por CNES): ETS, SARIMA(X) com Fourier, Croston, TSB.

Todos seguem a mesma interface:
    prever_local(nome, y, X, n_ajuste, alvos, h, params) -> np.ndarray de médias previstas
- y: série semanal contígua da CNES (np.ndarray float), da 1ª semana até a última semana alvo;
- X: exógenas alinhadas a y (ou None);
- n_ajuste: número de semanas iniciais usadas para estimar os parâmetros;
- alvos: índices (em y) das semanas a prever; cada alvo t usa só y[: t − h + 1] (origem móvel,
  parâmetros fixos, estado atualizado com os dados observados até a origem).
ANTI-VAZAMENTO: assert alvo − h + 1 ≤ ... garante que nada após a origem entra.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _tr(y: np.ndarray, transf: str) -> np.ndarray:
    return np.log1p(y) if transf == "log1p" else y


def _inv(z: np.ndarray, transf: str) -> np.ndarray:
    return np.clip(np.expm1(z) if transf == "log1p" else z, 0, None)


# ----------------------------------------------------------------------------- grades
def grade_ets() -> list[dict]:
    """8 configurações: tendência (nenhuma | aditiva amortecida) × sazonalidade anual
    (nenhuma | aditiva, 52) × transformação (nenhuma | log1p)."""
    return [{"tendencia": t, "sazonal": s, "transf": tr}
            for t, s, tr in itertools.product([None, "add"], [None, "add"], ["nenhuma", "log1p"])]


def grade_sarima() -> list[dict]:
    """Ordens (p,d,q) ∈ {0,1,2}×{0,1}×{0,1,2} × transformação; sazonalidade anual por Fourier.
    A pré-seleção por AICc reduz a 8 candidatas (4 por transformação) — ver preselecionar_sarima."""
    return [{"ordem": (p, d, q), "transf": tr}
            for p, d, q, tr in itertools.product([0, 1, 2], [0, 1], [0, 1, 2], ["nenhuma", "log1p"])]


def grade_croston() -> list[dict]:
    """4 valores de alfa para Croston e 4 pares (alfa, beta) para TSB = 8 configurações."""
    return ([{"metodo": "croston", "alfa": a} for a in [0.05, 0.1, 0.2, 0.3]]
            + [{"metodo": "tsb", "alfa": a, "beta": b} for a, b in [(0.1, 0.05), (0.1, 0.2), (0.3, 0.05), (0.3, 0.2)]])


# ----------------------------------------------------------------------------- ETS
def _ajustar_ets(z: np.ndarray, p: dict, fixos: dict | None = None):
    sazonal = p["sazonal"] if len(z) >= 2 * 52 + 2 else None
    if fixos is None:
        mod = ExponentialSmoothing(z, trend=p["tendencia"], damped_trend=p["tendencia"] is not None,
                                   seasonal=sazonal, seasonal_periods=52 if sazonal else None,
                                   initialization_method="estimated")
        return mod.fit(optimized=True)
    mod = ExponentialSmoothing(z, trend=p["tendencia"], damped_trend=p["tendencia"] is not None,
                               seasonal=fixos["sazonal"], seasonal_periods=52 if fixos["sazonal"] else None,
                               initialization_method="known", initial_level=fixos["initial_level"],
                               initial_trend=fixos.get("initial_trend"), initial_seasonal=fixos.get("initial_seasons"))
    return mod.fit(smoothing_level=fixos["smoothing_level"], smoothing_trend=fixos.get("smoothing_trend"),
                   smoothing_seasonal=fixos.get("smoothing_seasonal"), damping_trend=fixos.get("damping_trend"),
                   optimized=False)


def prever_ets(y, X, n_ajuste, alvos, h, p):
    z = _tr(y, p["transf"])
    res = _ajustar_ets(z[:n_ajuste], p)
    pr = dict(res.params)
    fixos = {k: (None if (v is None or (np.ndim(v) == 0 and np.isnan(v))) else v) for k, v in pr.items()}
    fixos["sazonal"] = res.model.seasonal
    # Componentes ausentes: o statsmodels devolve vetores vazios/NaN, que não podem ser repassados.
    if not fixos["sazonal"]:
        fixos["initial_seasons"], fixos["smoothing_seasonal"] = None, None
    if p["tendencia"] is None:
        fixos["initial_trend"], fixos["smoothing_trend"], fixos["damping_trend"] = None, None, None
    out = np.full(len(alvos), np.nan)
    for i, t in enumerate(alvos):
        fim = t - h + 1                       # ANTI-VAZAMENTO: dados até a origem
        assert fim <= t
        r = _ajustar_ets(z[:fim], p, fixos)
        out[i] = r.forecast(h)[-1]
    return _inv(out, p["transf"]), res


# ----------------------------------------------------------------------------- SARIMA(X)
def _sarimax(z, X, ordem):
    return SARIMAX(z, exog=X, order=ordem, trend="c" if ordem[1] == 0 else "n",
                   enforce_stationarity=False, enforce_invertibility=False)


def preselecionar_sarima(y: np.ndarray, X: np.ndarray, n_ajuste: int, n_por_transf: int = 4) -> list[dict]:
    """Ordena a grade por AICc no período de ajuste e devolve as n_por_transf melhores por
    transformação (AICc só é comparável dentro da mesma transformação)."""
    escolhidas = []
    for tr in ["nenhuma", "log1p"]:
        z = _tr(y[:n_ajuste], tr)
        notas = []
        for p in [g for g in grade_sarima() if g["transf"] == tr]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    notas.append((_sarimax(z, X[:n_ajuste], p["ordem"]).fit(disp=False, maxiter=200).aicc, p))
            except Exception:
                continue
        notas.sort(key=lambda a: a[0])
        escolhidas += [p for _, p in notas[:n_por_transf]]
    return escolhidas


def prever_sarima(y, X, n_ajuste, alvos, h, p):
    z = _tr(y, p["transf"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _sarimax(z[:n_ajuste], X[:n_ajuste], p["ordem"]).fit(disp=False, maxiter=200)
        out = np.full(len(alvos), np.nan)
        for i, t in enumerate(alvos):
            fim = t - h + 1                   # ANTI-VAZAMENTO: dados até a origem
            r = res.apply(z[:fim], exog=X[:fim], refit=False)
            out[i] = r.forecast(h, exog=X[fim:fim + h])[-1]
    return _inv(out, p["transf"]), res


# ----------------------------------------------------------------------------- Croston / TSB
def _croston_trajetoria(y: np.ndarray, p: dict) -> np.ndarray:
    """Estimativa após observar y[0..t] (f[t]); previsão para qualquer h à frente = f[t]."""
    a = p["alfa"]
    f = np.zeros(len(y))
    z, x, q = None, None, 1
    for t, v in enumerate(y):
        if v > 0:
            z = v if z is None else z + a * (v - z)
            x = q if x is None else x + a * (q - x)
            q = 1
        else:
            q += 1
        f[t] = 0.0 if z is None else z / x
    return f


def prever_croston(y, X, n_ajuste, alvos, h, p):
    # TSB usa a média inicial; para não usar o futuro, a inicialização vem só de y[:n_ajuste].
    traj = _croston_trajetoria(y, p) if p["metodo"] == "croston" else _tsb_seguro(y, n_ajuste, p)
    out = np.array([traj[t - h] if t - h >= 0 else np.nan for t in alvos])
    return out, None


def _tsb_seguro(y, n_ajuste, p):
    nz = y[:n_ajuste][y[:n_ajuste] > 0]
    z = nz.mean() if len(nz) else 0.0
    prob = (y[:n_ajuste] > 0).mean() if n_ajuste else 0.0
    f = np.zeros(len(y))
    for t, v in enumerate(y):
        prob = prob + p["beta"] * ((v > 0) - prob)
        if v > 0:
            z = z + p["alfa"] * (v - z)
        f[t] = prob * z
    return f


PREVISORES = {"ets": prever_ets, "sarima": prever_sarima, "sarimax": prever_sarima, "croston_tsb": prever_croston}


def prever_local(nome: str, y, X, n_ajuste, alvos, h, params):
    """Despacha para o previsor local; erros numéricos viram NaN (linhas perdidas registradas)."""
    try:
        return PREVISORES[nome](y, X, n_ajuste, alvos, h, params)
    except Exception:
        return np.full(len(alvos), np.nan), None


def ljung_box(res, lags: int = 10) -> float:
    """Valor-p de Ljung-Box dos resíduos do ajuste final (diagnóstico registrado por CNES)."""
    try:
        r = np.asarray(res.resid)[13:]
        return float(acorr_ljungbox(r, lags=[lags])["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan
