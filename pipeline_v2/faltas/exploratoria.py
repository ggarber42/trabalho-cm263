"""Caracterização das séries semanais no treino e regra pré-declarada de escolha do esquema (D10)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, adfuller, kpss


def classe_syntetos_boylan(adi: float, cv2: float, corte_adi: float = 1.32, corte_cv2: float = 0.49) -> str:
    """Classe de demanda (Syntetos, Boylan e Croston 2005): suave, errática, intermitente ou irregular."""
    if np.isnan(adi) or np.isnan(cv2):
        return "sem_demanda"
    if adi < corte_adi:
        return "suave" if cv2 < corte_cv2 else "erratica"
    return "intermitente" if cv2 < corte_cv2 else "irregular"


def caracterizar_serie(y: np.ndarray, n_total: np.ndarray, cfg_exp: dict) -> dict:
    """Estatísticas de uma série semanal de faltas (só treino).

    Retorna: dict com semanas, semanas_ativas, media, prop_zero_ativas, adi, cv2, classe,
    indice_dispersao (var/média), acf1, acf52, p_adf, p_kpss.
    """
    ativas = n_total > 0
    pos = y[y > 0]
    adi = len(y) / len(pos) if len(pos) else np.nan
    cv2 = (pos.std() / pos.mean()) ** 2 if len(pos) > 1 else np.nan
    out = {
        "semanas": len(y), "semanas_ativas": int(ativas.sum()), "media": float(y.mean()),
        "prop_zero_ativas": float((y[ativas] == 0).mean()) if ativas.any() else np.nan,
        "adi": adi, "cv2": cv2, "classe": classe_syntetos_boylan(adi, cv2, cfg_exp["corte_adi"], cfg_exp["corte_cv2"]),
        "indice_dispersao": float(y.var(ddof=1) / y.mean()) if y.mean() > 0 else np.nan,
        "acf1": np.nan, "acf52": np.nan, "p_adf": np.nan, "p_kpss": np.nan,
    }
    if len(y) >= 60 and y.std() > 0:
        r = acf(y, nlags=min(52, len(y) - 2), fft=True)
        out["acf1"] = float(r[1])
        out["acf52"] = float(r[52]) if len(r) > 52 else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["p_adf"] = float(adfuller(y, regression="c", autolag="AIC")[1])
            out["p_kpss"] = float(kpss(y, regression="c", nlags="auto")[1])  # p truncado em [0,01; 0,1]
    return out


def decidir_esquema(car: pd.DataFrame, cfg_exp: dict, p_wavelet: pd.Series | None = None) -> dict:
    """Aplica a regra pré-declarada (Cerqueira, Torgo e Mozetič 2020).

    Uma série é não estacionária se (após correção de Benjamini-Hochberg entre séries) o teste de
    wavelets rejeita a estacionariedade; sem o teste de wavelets (R/locits indisponível), se o ADF
    não rejeita a raiz unitária OU o KPSS rejeita a estacionariedade.
    Se a fração de séries não estacionárias > limiar → 'rep_holdout'; senão 'cv_bloqueada'.
    Só entram séries com semanas_ativas ≥ min_semanas_ativas_modelo_local.
    Retorna: dict com esquema, teste usado, n_series, fracao_nao_estacionaria e por série.
    """
    alfa = cfg_exp["alfa"]
    elegiveis = car[(car["semanas_ativas"] >= cfg_exp["min_semanas_ativas_modelo_local"]) & car["p_adf"].notna()].copy()
    if len(elegiveis) == 0:
        return {"esquema": "rep_holdout", "teste": "nenhum", "n_series": 0, "fracao_nao_estacionaria": np.nan,
                "observacao": "Sem séries elegíveis; padrão conservador para séries não estacionárias."}
    if p_wavelet is not None and p_wavelet.reindex(elegiveis.index).notna().all():
        rej = multipletests(p_wavelet.reindex(elegiveis.index), alpha=alfa, method="fdr_bh")[0]
        elegiveis["nao_estacionaria"] = rej
        teste = "wavelets (Nason 2013), BH"
    else:
        adf_rej = multipletests(elegiveis["p_adf"], alpha=alfa, method="fdr_bh")[0]
        kpss_rej = multipletests(elegiveis["p_kpss"], alpha=alfa, method="fdr_bh")[0]
        elegiveis["nao_estacionaria"] = (~adf_rej) | kpss_rej
        teste = "ADF + KPSS, BH (wavelets indisponível)"
    frac = float(elegiveis["nao_estacionaria"].mean())
    esquema = "rep_holdout" if frac > cfg_exp["limiar_maioria_nao_estacionaria"] else "cv_bloqueada"
    return {"esquema": esquema, "teste": teste, "n_series": int(len(elegiveis)),
            "fracao_nao_estacionaria": frac,
            "series_nao_estacionarias": elegiveis.index[elegiveis["nao_estacionaria"]].tolist()}
