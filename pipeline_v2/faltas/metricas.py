"""Métricas pontuais e probabilísticas (binomial negativa comum a todos) e testes de comparação."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wape(y: np.ndarray, mu: np.ndarray) -> float:
    """Σ|y − ŷ| / Σy."""
    return float(np.abs(y - mu).sum() / max(y.sum(), 1e-12))


def vies(y: np.ndarray, mu: np.ndarray) -> float:
    """Σŷ / Σy (1 = sem viés)."""
    return float(mu.sum() / max(y.sum(), 1e-12))


def phi_momentos(y: np.ndarray, mu: np.ndarray) -> float:
    """Dispersão da binomial negativa (Var = μ + φμ²) por momentos, em previsões fora da amostra."""
    ok = np.isfinite(mu) & np.isfinite(y)
    num = ((y[ok] - mu[ok]) ** 2 - mu[ok]).sum()
    return float(max(num / max((mu[ok] ** 2).sum(), 1e-12), 0.0))


def _dist(mu: np.ndarray, phi: float):
    mu = np.maximum(mu, 1e-9)
    if phi <= 1e-8:
        return stats.poisson(mu)
    n = 1.0 / phi
    return stats.nbinom(n, n / (n + mu))


def crps_nb(y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    """CRPS de cada observação para a binomial negativa (soma discreta sobre o suporte)."""
    d = _dist(mu, phi)
    kmax = int(max(np.nanmax(y), np.nanmax(d.ppf(0.99999)))) + 1
    k = np.arange(kmax + 1)
    F = d.cdf(k[:, None])                         # (k, n)
    ind = (y[None, :] <= k[:, None]).astype(float)
    return ((F - ind) ** 2).sum(axis=0)


def intervalo(mu: np.ndarray, phi: float, nivel: float) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo central de previsão de nível `nivel`."""
    d = _dist(mu, phi)
    a = (1 - nivel) / 2
    return d.ppf(a), d.ppf(1 - a)


def resumo_metricas(y: np.ndarray, mu: np.ndarray, phi: float) -> dict:
    """WAPE, viés, CRPS médio, cobertura e largura dos intervalos de previsão de 95%, 80% e 50%."""
    out = {"n": len(y), "wape": wape(y, mu), "vies": vies(y, mu), "crps": float(crps_nb(y, mu, phi).mean()), "phi": phi}
    for nv in (0.95, 0.8, 0.5):
        lo, hi = intervalo(mu, phi, nv)
        out[f"cob{int(nv * 100)}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"larg{int(nv * 100)}"] = float((hi - lo).mean())
    return out


def diebold_mariano(perda_a: np.ndarray, perda_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Teste de Diebold-Mariano (1995) com variância de Newey-West e correção de Harvey,
    Leybourne e Newbold (1997). Perdas já agregadas por semana.
    DECISÃO: defasagem de Newey-West = max(h − 1, ⌊T^(1/3)⌋), por causa da dependência entre semanas.
    Retorna: (estatística, valor-p bilateral, t de Student com T − 1 g.l.). d = a − b; negativo = a melhor.
    """
    d = np.asarray(perda_a) - np.asarray(perda_b)
    T = len(d)
    if T < 5 or np.allclose(d, d[0]):
        return np.nan, np.nan
    L = max(h - 1, int(np.floor(T ** (1 / 3))))
    dc = d - d.mean()
    gam = [np.dot(dc[k:], dc[:T - k]) / T for k in range(L + 1)]
    var = gam[0] + 2 * sum((1 - k / (L + 1)) * gam[k] for k in range(1, L + 1))
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / T)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    est = dm * corr
    return float(est), float(2 * stats.t.sf(abs(est), df=T - 1))


def holm(pvalores: pd.Series) -> pd.Series:
    """Correção de Holm (1979). NaN é preservado."""
    p = pvalores.dropna().sort_values()
    m = len(p)
    ajust = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    return pd.Series(np.minimum(ajust, 1), index=p.index).reindex(pvalores.index)


def bootstrap_ganho_wape(tab: pd.DataFrame, col_a: str, col_b: str, n: int, bloco: int, semente: int) -> tuple[float, float]:
    """IC95% do ganho relativo de WAPE (1 − WAPE_a / WAPE_b) por bootstrap de blocos de semanas.
    tab: colunas semana, y, col_a, col_b (uma linha por CNES-semana)."""
    t = tab.assign(ea=(tab["y"] - tab[col_a]).abs(), eb=(tab["y"] - tab[col_b]).abs()).groupby("semana")[["y", "ea", "eb"]].sum()
    arr = t.to_numpy()
    T = len(arr)
    if T < bloco * 2:
        return np.nan, np.nan
    rng = np.random.default_rng(semente)
    ganhos = []
    for _ in range(n):
        inicios = rng.integers(0, T - bloco + 1, size=int(np.ceil(T / bloco)))
        idx = np.concatenate([np.arange(s, s + bloco) for s in inicios])[:T]
        a = arr[idx]
        ganhos.append(1 - a[:, 1].sum() / a[:, 2].sum())
    return float(np.percentile(ganhos, 2.5)), float(np.percentile(ganhos, 97.5))


def bootstrap_metricas(tab: pd.DataFrame, col: str, n: int, bloco: int, semente: int,
                       confianca: float = 0.95) -> dict[str, float]:
    """IC (nível `confianca`) e margem de erro de WAPE, viés e MAE de um modelo, por bootstrap de
    blocos de semanas (mesma reamostragem de `bootstrap_ganho_wape`).
    tab: colunas semana, y, col. Retorna dict {wape_ic_inf, wape_ic_sup, wape_moe, vies_..., mae_...};
    margem de erro (moe) = metade da largura do intervalo."""
    t = tab.assign(e=(tab["y"] - tab[col]).abs(), mu=tab[col], c=1.0).groupby("semana")[["y", "e", "mu", "c"]].sum()
    arr = t.to_numpy()
    T = len(arr)
    nomes = ["wape", "vies", "mae"]
    if T < bloco * 2:
        return {f"{m}_{s}": np.nan for m in nomes for s in ("ic_inf", "ic_sup", "moe")}
    rng = np.random.default_rng(semente)
    est = []
    for _ in range(n):
        inicios = rng.integers(0, T - bloco + 1, size=int(np.ceil(T / bloco)))
        a = arr[np.concatenate([np.arange(s, s + bloco) for s in inicios])[:T]]
        est.append((a[:, 1].sum() / a[:, 0].sum(), a[:, 2].sum() / a[:, 0].sum(), a[:, 1].sum() / a[:, 3].sum()))
    est = np.array(est)
    alfa = (1 - confianca) / 2 * 100
    out = {}
    for j, m in enumerate(nomes):
        lo, hi = np.percentile(est[:, j], [alfa, 100 - alfa])
        out.update({f"{m}_ic_inf": float(lo), f"{m}_ic_sup": float(hi), f"{m}_moe": float((hi - lo) / 2)})
    return out
