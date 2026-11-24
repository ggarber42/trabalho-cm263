"""Teste de Diebold-Mariano com erro-padrão HAC (Newey-West), agregação semanal, Holm e bootstrap em blocos (S9).

Etapa: S9. Não executa nada ao ser importado.

Como usar: agregue a perda de cada modelo por SEMANA (soma sobre as CNES-dias da semana) nas MESMAS
linhas, forme a diferença d_t = perda_referência − perda_modelo (d > 0 = modelo melhor) e chame ``dm_hac``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dm_hac(d: np.ndarray, bandwidth: int | None = None) -> dict[str, float]:
    """Teste DM (H0: E[d] = 0) com variância de longo prazo de Newey-West (kernel de Bartlett).

    Parâmetros:
        d: diferenças de perda por período (semana), ordenadas no tempo.
        bandwidth: nº de defasagens; padrão ⌊4 (T/100)^(2/9)⌋ (mínimo 1).

    Retorna:
        dict: media, se, estatistica, p (bilateral; t de Student com T−1 gl, correção de amostra pequena de
        Harvey-Leybourne-Newbold), T.

    Cuidados:
        Séries curtas (2026 tem ~13 semanas) dão IC largos; o chamador deve reportar T.
    """
    d = np.asarray(d, float)
    T = len(d)
    assert T >= 4, "Poucas semanas para o teste DM"
    m = d.mean()
    e = d - m
    L = bandwidth if bandwidth is not None else max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    gamma0 = float(np.dot(e, e) / T)
    lrv = gamma0
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * float(np.dot(e[k:], e[:-k]) / T)
    lrv = max(lrv, 1e-12)
    se = np.sqrt(lrv / T)
    hln = np.sqrt((T - 1) / T)     # correção de Harvey-Leybourne-Newbold para h = 1
    stat = hln * m / se
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return {"media": float(m), "se": float(se), "estatistica": float(stat), "p": float(p), "T": int(T), "bandwidth": int(L)}


def holm(p: list[float] | np.ndarray) -> np.ndarray:
    """Correção de Holm-Bonferroni para múltiplas comparações.

    Parâmetros:
        p: p-valores brutos.

    Retorna:
        np.ndarray de p-valores ajustados (mesma ordem), em [0, 1].
    """
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    maximo = 0.0
    for rank, i in enumerate(ordem):
        maximo = max(maximo, (m - rank) * p[i])
        aj[i] = min(1.0, maximo)
    return aj


def perda_semanal(df: pd.DataFrame, coluna_perda: str) -> pd.Series:
    """Soma a perda por semana (segunda-feira) sobre todas as CNES-dias.

    Parâmetros:
        df: precisa de `dia` e da coluna de perda.
        coluna_perda: nome da coluna.

    Retorna:
        pd.Series indexada pelo início da semana, ordenada no tempo.
    """
    semana = df["dia"].dt.to_period("W-SUN").dt.start_time
    return df.groupby(semana)[coluna_perda].sum().sort_index()


def bootstrap_ganho_wape(sem_base: pd.DataFrame, sem_modelo: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """IC 95% por bootstrap em blocos semanais do ganho relativo de WAPE sobre o baseline.

    Parâmetros:
        sem_base, sem_modelo: DataFrames por semana (mesmo índice) com colunas `erro_abs` e `y`.
        n_boot: repetições; seed: semente fixa.

    Retorna:
        dict: ganho (1 − WAPE_modelo/WAPE_base), ic_inf, ic_sup.
    """
    assert sem_base.index.equals(sem_modelo.index), "Semanas desalinhadas entre baseline e modelo"
    rng = np.random.default_rng(seed)
    eb, em, y = sem_base["erro_abs"].to_numpy(), sem_modelo["erro_abs"].to_numpy(), sem_base["y"].to_numpy()
    T = len(y)
    ganho = 1 - em.sum() / eb.sum()
    gs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, T, T)
        gs[b] = 1 - em[i].sum() / eb[i].sum()
    return {"ganho": float(ganho), "ic_inf": float(np.quantile(gs, 0.025)), "ic_sup": float(np.quantile(gs, 0.975))}
