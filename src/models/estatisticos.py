"""Modelos estatísticos de série temporal por CNES (S5): ETS (Holt-Winters), SARIMA e SARIMAX.

Cada CNES tem uma série diária da TAXA de falta (faltas ÷ agenda final); dias sem agenda são ausentes.
A previsão para o dia t é feita com a informação até t − L: o estado do modelo é filtrado com os dados
até t − L e projetado L passos adiante (truque do filtro de Kalman, igual ao script legado 23, aqui
conferido em teste contra a previsão nativa do statsmodels). Os parâmetros (SARIMA) são ajustados SÓ com
as linhas de treino. mu = taxa prevista (limitada a [0, 1]) × exposição do dia.

Não executa nada ao ser importado.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.dados import AGENDA_VARS
from src.models.base import registra
from src.models.series import _Serie, _grade

EXOG = AGENDA_VARS[:3]


def serie_cnes(ctx: Any, cnes: Any) -> dict[str, np.ndarray]:
    """Série diária (calendário completo) de uma CNES, com cache no contexto.

    Retorna:
        dict com `d0` (primeiro dia, int de datetime64[D]), `y` (taxa f/ag, NaN onde não há agenda), `dow`,
        `ex` (matriz n × 3 de agenda, NaN onde ausente) e `pos` (posição da linha de P em cada dia, -1 se não há).
    """
    cache = ctx.__dict__.setdefault("_series", {})
    if cnes in cache:
        return cache[cnes]
    P = ctx.P
    lin = np.where(P["cnes"].to_numpy() == cnes)[0]
    dia = P["dia"].to_numpy("datetime64[D]").astype("int64")[lin]
    d0, n = int(dia.min()), int(dia.max() - dia.min() + 1)
    y, ex, pos = np.full(n, np.nan), np.full((n, len(EXOG)), np.nan), np.full(n, -1)
    y[dia - d0] = P["f"].to_numpy()[lin] / P["ag"].to_numpy()[lin]
    ex[dia - d0] = ctx.F[EXOG].to_numpy()[lin]
    pos[dia - d0] = lin
    dow = ((np.arange(n) + d0 + 3) % 7)                      # 1970-01-01 foi quinta (3 = quinta se seg = 0)
    cache[cnes] = {"d0": d0, "y": y, "dow": dow, "ex": ex, "pos": pos}
    return cache[cnes]


def previsao_kalman(y: np.ndarray, ex: np.ndarray | None, ntr: int, order: tuple, seas: tuple, lag: int) -> np.ndarray:
    """Previsão de `lag` passos à frente para toda a série, parâmetros SARIMAX estimados em y[:ntr].

    Parâmetros:
        y: taxa diária (NaN = ausente). ex: exógenas já padronizadas (n × k) ou None. ntr: nº de dias do ajuste.
        order, seas: (p,d,q) e (P,D,Q,s). lag: horizonte em dias.

    Retorna:
        vetor `fc` do tamanho de y; fc[t] usa só y[:t − lag + 1]; os `lag` primeiros valores são NaN.

    Cuidados:
        # ANTI-VAZAMENTO: ``apply`` mantém os parâmetros do treino e só filtra (não reestima); o estado em
        # s depende de y[≤ s]. A matriz de transição é elevada a `lag` para projetar sem usar o futuro.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    kw = dict(order=order, seasonal_order=seas, trend="n", enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ex is not None:
            r0 = SARIMAX(y[:ntr], exog=ex[:ntr], **kw).fit(disp=False, maxiter=60)
            rf = r0.apply(y, exog=ex)
        else:
            r0 = SARIMAX(y[:ntr], **kw).fit(disp=False, maxiter=60)
            rf = r0.apply(y)
    g = lambda k: (lambda a: a[:, :, 0] if a.ndim == 3 else a)(np.asarray(rf.model.ssm[k]))   # noqa: E731
    Z, T = g("design"), g("transition")
    pred = (Z @ np.linalg.matrix_power(T, lag) @ rf.filtered_state)[0]
    oi = np.asarray(rf.model.ssm["obs_intercept"])
    if ex is not None:
        oi = oi.reshape(1, -1) if oi.ndim == 1 else oi
        pred = pred[: len(y) - lag] + oi[0, lag:]
    else:
        pred = pred[: len(y) - lag]
    fc = np.full(len(y), np.nan)
    fc[lag:] = pred
    return fc


def ets_previsao(y: np.ndarray, dow: np.ndarray, alpha: float, gamma: float, lvl0: float, lag: int) -> np.ndarray:
    """Holt-Winters aditivo (nível + sazonalidade semanal) com previsão de `lag` passos: fc[t+lag] = l_t + s[dow[t+lag]].

    Cuidados: # ANTI-VAZAMENTO: o estado em t só usa y[≤ t]; ausências (NaN) não atualizam o estado.
    """
    n = len(y)
    fc = np.full(n, np.nan)
    nivel, saz = lvl0, np.zeros(7)
    for t in range(n):
        yt = y[t]
        if yt == yt:
            e = yt - (nivel + saz[dow[t]])
            nivel = nivel + alpha * e
            saz[dow[t]] += gamma * (yt - nivel - saz[dow[t]])
        if t + lag < n:
            fc[t + lag] = nivel + saz[dow[t + lag]]
    return fc


class _PorCNES(_Serie):
    """Base de modelos de série por CNES: prevê a taxa de cada linha a partir do filtro da própria CNES."""

    def fit(self, ctx, idx_treino, k_h):
        self.idx_treino_ = np.asarray(idx_treino)
        self.falhas_ = 0
        return self

    def _taxa_cnes(self, ctx, c, serie, n_ate, lin_treino):  # pragma: no cover - sobrescrito
        raise NotImplementedError

    def predict_mean(self, ctx, idx, k_h):
        P = ctx.P
        cnes_all = P["cnes"].to_numpy()
        mu = np.full(len(idx), np.nan)
        expo = ctx.expo(idx, k_h)
        for c in np.unique(cnes_all[idx]):
            s = serie_cnes(ctx, c)
            sel = np.where(cnes_all[idx] == c)[0]
            dias_i = P["dia"].to_numpy("datetime64[D]").astype("int64")[idx][sel] - s["d0"]
            n_ate = int(dias_i.max()) + 1                            # a série para no último dia previsto
            lin_tr = self.idx_treino_[cnes_all[self.idx_treino_] == c]
            fc = self._taxa_cnes(ctx, c, s, n_ate, lin_tr)
            mu[sel] = np.clip(fc[dias_i], 0.0, 1.0) * expo[sel]
        return mu


@registra
class ETS(_PorCNES):
    """ETS/Holt-Winters aditivo semanal na taxa; alpha (nível) e gamma (sazonal) são hiperparâmetros."""
    nome = "ets"

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        return _grade(alpha=[0.01, 0.02, 0.05, 0.1, 0.2], gamma=[0.0, 0.02, 0.05, 0.1])

    def _taxa_cnes(self, ctx, c, s, n_ate, lin_tr):
        y = s["y"][:n_ate]
        primeiros = y[np.isfinite(y)][:28]                          # nível inicial: 28 primeiros dias da série
        lvl0 = float(primeiros.mean()) if len(primeiros) else 0.2
        return ets_previsao(y, s["dow"][:n_ate], float(self.params["alpha"]), float(self.params["gamma"]), lvl0, ctx.lag)


@registra
class SARIMA(_PorCNES):
    """SARIMA (p,1,q)(P,1,Q,7) na taxa; (p, q, P, Q) são hiperparâmetros; parâmetros estimados só no treino."""
    nome = "sarima"
    usa_exog = False
    JANELA_AJUSTE = 730

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        g = _grade(p=[0, 1, 2], q=[0, 1, 2], P=[0, 1], Q=[0, 1])
        return [c for c in g if c["Q"] == 1 and c["p"] < 2]   # DECISÃO: 12 configs (p<2, Q=1): o custo do SARIMA domina; grade menor que n_configs é percorrida inteira

    def _taxa_cnes(self, ctx, c, s, n_ate, lin_tr):
        dias_tr = ctx.P["dia"].to_numpy("datetime64[D]").astype("int64")[lin_tr] - s["d0"]
        ntr = int(dias_tr.max()) + 1
        # DECISÃO: o ajuste usa só os últimos JANELA_AJUSTE dias do treino (o custo do SARIMA cresce com n);
        # o filtro percorre a série a partir do mesmo início. Fixo para todos os esquemas.
        ini = max(int(dias_tr.min()), ntr - self.JANELA_AJUSTE)
        # ANTI-VAZAMENTO: dias fora do treino (purga, fora da janela) entram como ausentes no AJUSTE.
        y_in = s["y"][ini:n_ate].copy()
        fora = np.ones(ntr - ini, bool)
        fora[dias_tr[dias_tr >= ini] - ini] = False
        y_in[: ntr - ini][fora] = np.nan
        ex = None
        if self.usa_exog:
            exo = s["ex"]
            mu_, sd_ = np.nanmean(exo[dias_tr], 0), np.nanstd(exo[dias_tr], 0) + 1e-9    # só treino
            ex = np.nan_to_num((exo[ini:n_ate] - mu_) / sd_, nan=0.0)
        p = self.params
        try:
            fc_ = previsao_kalman(y_in, ex, ntr - ini, (p["p"], 1, p["q"]), (p["P"], 1, p["Q"], 7), ctx.lag)
            fc = np.full(n_ate, np.nan)
            fc[ini:] = fc_
            return fc
        except Exception:                                           # não convergiu: cai para a taxa r28
            self.falhas_ += 1
            r28 = np.full(n_ate, np.nan)
            lin = s["pos"][:n_ate]
            ok = lin >= 0
            r28[ok] = ctx.P["r28"].to_numpy()[lin[ok]]
            return r28


@registra
class SARIMAX(SARIMA):
    """SARIMA com 3 variáveis exógenas de agenda (padronizadas com o treino): regressão com erros SARIMA."""
    nome = "sarimax"
    usa_exog = True
