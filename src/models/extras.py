"""Modelos adicionais de taxa por CNES-dia (S5): XGBoost, CatBoost, MLP, GLM Binomial Negativa e
HGB Poisson com recalibração k.

Seguem o mesmo contrato de ``src/models/series.py``: preveem uma TAXA de falta por consulta agendada e
multiplicam pela exposição (agenda do dia no modo escolhido). Bibliotecas opcionais (xgboost, catboost)
só são importadas dentro de ``fit``. Não executa nada ao ser importado.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import registra
from src.models.series import FEATS_CAT, FEATS_NUM, HGBPoisson, _grade, _Serie

COLS = FEATS_NUM + FEATS_CAT


@registra
class XGBoostPoisson(_Serie):
    """XGBoost com objetivo Poisson (``count:poisson``) sobre a taxa, ponderado pela exposição. OPCIONAL."""
    nome = "xgboost"

    @classmethod
    def disponivel(cls) -> bool:
        """True se `xgboost` estiver instalado."""
        return importlib.util.find_spec("xgboost") is not None

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        return _grade(learning_rate=[0.03, 0.06, 0.1], n_estimators=[100, 200, 400], max_depth=[2, 3, 4, 6],
                      min_child_weight=[5, 20, 50], reg_lambda=[1.0, 10.0])

    def fit(self, ctx, idx_treino, k_h):
        import xgboost as xgb
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        # ANTI-VAZAMENTO: sem early stopping; nº de árvores vem da busca nos folds internos.
        self.m_ = xgb.XGBRegressor(objective="count:poisson", tree_method="hist", n_jobs=1,
                                   random_state=self.seed, verbosity=0, **self.params)
        self.m_.fit(ctx.F.iloc[i][COLS], y, sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][COLS]), 0.0)


@registra
class CatBoostPoisson(_Serie):
    """CatBoost com perda Poisson sobre a taxa, ponderado pela exposição. OPCIONAL."""
    nome = "catboost"

    @classmethod
    def disponivel(cls) -> bool:
        """True se `catboost` estiver instalado."""
        return importlib.util.find_spec("catboost") is not None

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        return _grade(learning_rate=[0.03, 0.06, 0.1], iterations=[100, 200, 400], depth=[3, 4, 6, 8],
                      l2_leaf_reg=[1.0, 3.0, 10.0])

    def fit(self, ctx, idx_treino, k_h):
        from catboost import CatBoostRegressor
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        self.m_ = CatBoostRegressor(loss_function="Poisson", random_seed=self.seed, thread_count=1, verbose=0,
                                    allow_writing_files=False, **self.params)
        self.m_.fit(ctx.F.iloc[i][COLS], y, sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][COLS]), 0.0)


@registra
class MLPTaxa(_Serie):
    """Rede neural simples (MLP) sobre a taxa; imputação (mediana) e padronização só com o treino."""
    nome = "mlp"

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        return _grade(hidden_layer_sizes=[(16,), (32,), (32, 16)], alpha=[1e-4, 1e-3, 1e-2, 1e-1],
                      learning_rate_init=[1e-3, 3e-3], max_iter=[100, 250])

    def fit(self, ctx, idx_treino, k_h):
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        X = pd.get_dummies(ctx.F.iloc[i][COLS], columns=FEATS_CAT, dtype=float)
        self.cols_ = list(X.columns)
        # ANTI-VAZAMENTO: imputador e padronização ajustados só nas linhas de treino; sem early stopping
        # (o `early_stopping` do sklearn usaria um split aleatório).
        self.m_ = Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler()),
                            ("mlp", MLPRegressor(early_stopping=False, random_state=self.seed, **self.params))])
        self.m_.fit(X, y, mlp__sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        X = pd.get_dummies(ctx.F.iloc[idx][COLS], columns=FEATS_CAT, dtype=float).reindex(columns=self.cols_, fill_value=0.0)
        return np.clip(self.m_.predict(X), 0.0, 1.0)


@registra
class GLMBinomialNegativa(_Serie):
    """GLM Binomial Negativa (statsmodels) sobre a contagem de faltas com offset log(exposição).

    O parâmetro de dispersão ``alpha`` da família NB2 é hiperparâmetro (grade), escolhido nos folds internos.
    """
    nome = "glm_nb"

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        return _grade(alpha=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8])

    def _matriz(self, ctx, idx):
        """Variáveis padronizadas (parâmetros do treino) + dummies de dia da semana e mês + constante."""
        X = pd.get_dummies(ctx.F.iloc[idx][COLS], columns=FEATS_CAT, dtype=float, drop_first=False)
        return X

    def fit(self, ctx, idx_treino, k_h):
        import statsmodels.api as sm
        i, _, _ = self._linhas_ajuste(ctx, idx_treino, k_h)
        X = self._matriz(ctx, i)
        self.cols_ = [c for c in X.columns if not c.startswith(("dow_", "mes_"))] + \
                     [c for c in X.columns if c.startswith("dow_")][1:] + [c for c in X.columns if c.startswith("mes_")][1:]
        X = X[self.cols_]
        # ANTI-VAZAMENTO: mediana e desvio calculados só no treino (regra 7).
        self.med_ = X.median()
        X = X.fillna(self.med_)
        self.sd_ = X.std().replace(0, 1.0)
        self.mu_x_ = X.mean()
        Z = sm.add_constant((X - self.mu_x_) / self.sd_, has_constant="add")
        expo = ctx.expo(i, k_h)
        f = ctx.P["f"].to_numpy()[i]
        fam = sm.families.NegativeBinomial(alpha=float(self.params["alpha"]))
        try:
            self.res_ = sm.GLM(f, Z, family=fam, offset=np.log(expo)).fit(maxiter=100)
            self.falhou_ = not np.all(np.isfinite(self.res_.params))
        except Exception:                                   # colinearidade/divergência: cai para o baseline r28
            self.falhou_ = True
        return self

    def _taxa(self, ctx, idx):
        import statsmodels.api as sm
        if self.falhou_:
            return ctx.P["r28"].to_numpy()[idx]
        X = self._matriz(ctx, idx).reindex(columns=self.cols_, fill_value=0.0).fillna(self.med_)
        Z = sm.add_constant((X - self.mu_x_) / self.sd_, has_constant="add")
        eta = np.clip(np.asarray(Z) @ np.asarray(self.res_.params), -20, 5)
        return np.exp(eta)


@registra
class HGBPoissonK(HGBPoisson):
    """HGB Poisson com recalibração k: mu × (faltas observadas ÷ previstas nos 28 dias que terminam em t − L).

    # ANTI-VAZAMENTO: k de cada dia t usa só dias ≤ t − L (faltas já observadas, regra 4 do CLAUDE.md).
    # DECISÃO: nas primeiras semanas do teste a janela contém previsões dentro da amostra (dias de treino);
    # k é limitado a [0,5; 2] e vale 1 sem pelo menos 14 dias na janela. Janela de 28 dias: PIPELINE.md.
    """
    nome = "hgb_poisson_k"
    JANELA = 28

    def predict_mean(self, ctx, idx, k_h):
        mu0 = super().predict_mean(ctx, idx, k_h)
        P, L = ctx.P, ctx.lag
        cnes = P["cnes"].to_numpy()
        dia = P["dia"].to_numpy("datetime64[D]").astype("int64")
        f = P["f"].to_numpy()
        out = mu0.copy()
        for c in np.unique(cnes[idx]):
            sel_idx = np.where(cnes[idx] == c)[0]
            d_alvo = dia[idx][sel_idx]
            lin = np.where((cnes == c) & (dia >= d_alvo.min() - L - self.JANELA) & (dia <= d_alvo.max()))[0]
            mu_lin = super().predict_mean(ctx, lin, k_h)
            d0 = dia[lin].min()
            n = int(dia[lin].max() - d0 + 1)
            sf, sm_, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
            ok = np.isfinite(mu_lin)
            np.add.at(sf, dia[lin][ok] - d0, f[lin][ok])
            np.add.at(sm_, dia[lin][ok] - d0, mu_lin[ok])
            np.add.at(cnt, dia[lin][ok] - d0, 1.0)
            cf, cm, cc = (np.concatenate([[0.0], np.cumsum(a)]) for a in (sf, sm_, cnt))
            fim = np.clip(d_alvo - L - d0 + 1, 0, n)                   # dias ≤ t − L (índice exclusivo)
            ini = np.clip(fim - self.JANELA, 0, n)
            num, den, nd = cf[fim] - cf[ini], cm[fim] - cm[ini], cc[fim] - cc[ini]
            k = np.where((nd >= 14) & (den > 0), num / np.maximum(den, 1e-9), 1.0)
            out[sel_idx] = mu0[sel_idx] * np.clip(k, 0.5, 2.0)
        return out
