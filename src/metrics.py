"""Métricas pontuais e probabilísticas para contagens de faltas por CNES-dia (S9), com a MESMA
família de distribuição preditiva para todos os modelos: Binomial Negativa (NB).

Etapa: S8/S9. Não executa nada ao ser importado.

Distribuição preditiva
    Para uma média prevista mu e dispersão phi >= 0: Var(Y) = mu + phi * mu**2. phi = 0 → Poisson.
    A dispersão de CADA modelo é estimada por momentos em previsões FORA DA AMOSTRA anteriores ao teste
    (folds internos de validação), nunca no teste nem dentro do treino (``estima_dispersao``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

MU_MIN = 1e-6      # DECISÃO: piso numérico da média prevista (evita distribuição degenerada em mu = 0)


def wape(y: np.ndarray, mu: np.ndarray) -> float:
    """WAPE = Σ|y − mu| / Σy. Parâmetros: y observado e mu previsto (mesmo tamanho). Retorna float."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(np.abs(y - mu).sum() / y.sum())


def vies(y: np.ndarray, mu: np.ndarray) -> float:
    """Viés = Σmu / Σy (1 = sem viés; < 1 subestima)."""
    return float(np.asarray(mu, float).sum() / np.asarray(y, float).sum())


def estima_dispersao(y: np.ndarray, mu: np.ndarray) -> float:
    """Estima phi por momentos: phi = max(0, Σ[(y − mu)² − mu] / Σ mu²).

    Parâmetros:
        y, mu: observado e previsto em dados FORA DA AMOSTRA do ajuste do modelo.

    Retorna:
        phi >= 0 (float).

    Cuidados:
        # ANTI-VAZAMENTO: chamar só com previsões dos folds internos de validação (anteriores ao teste).
    """
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MU_MIN)
    num = np.sum((y - mu) ** 2 - mu)
    return float(max(0.0, num / np.sum(mu ** 2)))


def _nb_params(mu: np.ndarray, phi: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converte (mu, phi) nos parâmetros (n, p) de scipy.stats.nbinom; phi muito pequeno → n grande (≈ Poisson).

    `phi` pode ser escalar ou um array do tamanho de `mu` (dispersão por linha, ex.: uma por ano de teste).
    """
    mu = np.maximum(np.asarray(mu, float), MU_MIN)
    n = 1.0 / np.maximum(np.broadcast_to(np.asarray(phi, float), mu.shape), 1e-9)
    return n, n / (n + mu)


def pmf0(mu: np.ndarray, phi: float) -> np.ndarray:
    """P(Y = 0) sob a NB(mu, phi)."""
    n, p = _nb_params(mu, phi)
    return stats.nbinom.pmf(0, n, p)


def prob_maior_igual(mu: np.ndarray, phi: float, k: int) -> np.ndarray:
    """P(Y >= k) sob a NB(mu, phi) (k >= 1)."""
    n, p = _nb_params(mu, phi)
    return stats.nbinom.sf(k - 1, n, p)


def quantis(mu: np.ndarray, phi: float, q: float) -> np.ndarray:
    """Quantil q da NB(mu, phi) (valores inteiros)."""
    n, p = _nb_params(mu, phi)
    return stats.nbinom.ppf(q, n, p)


def crps_nb(y: np.ndarray, mu: np.ndarray, phi: float | np.ndarray, bloco: int = 4000) -> np.ndarray:
    """CRPS por linha de uma NB(mu, phi) para y inteiro: CRPS = Σ_{k>=0} (F(k) − 1{y <= k})².

    Parâmetros:
        y: observado (inteiros >= 0); mu: média prevista; phi: dispersão; bloco: linhas por bloco (memória).

    Retorna:
        np.ndarray com o CRPS de cada linha (mesma ordem). Truncado onde F ≈ 1 (mu + 12 desvios).
    """
    y = np.asarray(y, float)
    n, p = _nb_params(mu, phi)
    mu_ = np.maximum(np.asarray(mu, float), MU_MIN)
    phi_ = np.broadcast_to(np.asarray(phi, float), y.shape)
    out = np.empty(len(y))
    for i in range(0, len(y), bloco):
        sl = slice(i, i + bloco)
        sd = np.sqrt(mu_[sl] + phi_[sl] * mu_[sl] ** 2)
        K = int(max(np.max(y[sl]), np.max(mu_[sl] + 12 * sd))) + 2
        k = np.arange(K)[None, :]
        F = stats.nbinom.cdf(k, n[sl, None], p[sl, None])
        ind = (y[sl, None] <= k).astype(float)
        out[sl] = ((F - ind) ** 2).sum(axis=1)
    return out


def cobertura_e_largura(y: np.ndarray, mu: np.ndarray, phi: float, nivel: float) -> tuple[float, float]:
    """Cobertura empírica e largura média do intervalo central de nível `nivel` (0,8 ou 0,5).

    Retorna:
        (cobertura, largura média = hi − lo em nº de faltas).
    """
    lo = quantis(mu, phi, (1 - nivel) / 2)
    hi = quantis(mu, phi, 1 - (1 - nivel) / 2)
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def ece_prob_k(y: np.ndarray, mu: np.ndarray, phi: float, ks: tuple[int, ...] = (1, 2, 3, 5, 8), n_bins: int = 10) -> dict[int, float]:
    """ECE de P(Y >= k): erro de calibração esperado com bins por quantil da probabilidade prevista.

    Parâmetros:
        y, mu, phi: como acima; ks: limiares; n_bins: número de bins (quantis).

    Retorna:
        dict k → ECE (0 = perfeitamente calibrado).
    """
    y = np.asarray(y, float)
    res = {}
    for k in ks:
        pr = prob_maior_igual(mu, phi, k)
        ob = (y >= k).astype(float)
        bins = np.unique(np.quantile(pr, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.searchsorted(bins, pr, side="right") - 1, 0, max(len(bins) - 2, 0))
        ece = 0.0
        for b in np.unique(idx):
            m = idx == b
            ece += m.mean() * abs(pr[m].mean() - ob[m].mean())
        res[k] = float(ece)
    return res


def auc_celula(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    """AUC de P(Y >= 1) por célula CNES-dia (NaN se só há uma classe)."""
    ob = (np.asarray(y) >= 1).astype(int)
    if ob.min() == ob.max():
        return float("nan")
    return float(roc_auc_score(ob, 1 - pmf0(mu, phi)))


def metricas_consulta(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier, AUC e logloss por consulta (modelos por paciente).

    Parâmetros:
        y: 0/1; p: probabilidade prevista.

    Retorna:
        dict com brier, auc, logloss.
    """
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "auc": float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def resume_modelo(y: np.ndarray, mu: np.ndarray, phi: float | np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5, 8)) -> dict[str, float]:
    """Todas as métricas pontuais e probabilísticas de um modelo no mesmo conjunto de linhas.

    Retorna:
        dict: wape, vies, crps (média), cob80, cob50, larg80, ece (média sobre ks), auc_celula, phi, n.
    """
    mu = np.maximum(np.asarray(mu, float), MU_MIN)
    c80, l80 = cobertura_e_largura(y, mu, phi, 0.8)
    c50, _ = cobertura_e_largura(y, mu, phi, 0.5)
    return {"n": int(len(y)), "wape": wape(y, mu), "vies": vies(y, mu), "crps": float(crps_nb(y, mu, phi).mean()),
            "cob80": c80, "cob50": c50, "larg80": l80, "ece": float(np.mean(list(ece_prob_k(y, mu, phi, ks).values()))),
            "auc_celula": auc_celula(y, mu, phi), "phi": float(np.mean(phi))}
