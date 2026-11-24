"""Modelos de série CNES-dia (S5): naive, baseline Poisson, HGB Poisson, floresta aleatória,
Poisson regularizado e LightGBM (opcional).

Todos preveem uma TAXA de falta por consulta agendada e multiplicam pela exposição (agenda do dia,
conforme o modo). Treino: linhas com exposição > 0, alvo = faltas/exposição, peso = exposição (equivale a
um Poisson com offset log(exposição)). Não executa nada ao ser importado.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dados import AGENDA_VARS, CALENDARIO, TAXAS
from src.models.base import Modelo, registra

FEATS_NUM = TAXAS + AGENDA_VARS + ["expo_bruta", "fer", "posfer"]
FEATS_CAT = ["dow", "mes"]


def _grade(**kw: list) -> list[dict[str, Any]]:
    """Produto cartesiano de listas de valores -> lista de dicionários de hiperparâmetros."""
    chaves = list(kw)
    return [dict(zip(chaves, v)) for v in itertools.product(*kw.values())]


class _Serie(Modelo):
    """Base de modelos de taxa: ``mu = taxa_prevista × exposição``."""

    def _linhas_ajuste(self, ctx: Any, idx: np.ndarray, k_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Seleciona linhas de treino com exposição > 0 e taxa observável. Retorna (idx, taxa, peso)."""
        expo = ctx.expo(idx, k_h)
        f = ctx.P["f"].to_numpy()[idx]
        ok = expo > 0
        return idx[ok], f[ok] / expo[ok], expo[ok]

    def predict_mean(self, ctx: Any, idx: np.ndarray, k_h: float) -> np.ndarray:
        taxa = self._taxa(ctx, idx)
        return taxa * ctx.expo(idx, k_h)

    def _taxa(self, ctx: Any, idx: np.ndarray) -> np.ndarray:  # pragma: no cover - sobrescrito
        raise NotImplementedError


@registra
class Naive(_Serie):
    """Naive sazonal: taxa média do mesmo dia da semana nas 4 semanas anteriores (rdow4) × agenda."""
    nome, tunavel = "naive", False

    def fit(self, ctx, idx_treino, k_h):
        return self

    def _taxa(self, ctx, idx):
        return ctx.P["rdow4"].to_numpy()[idx]


@registra
class BaselinePoisson(_Serie):
    """Baseline Poisson: taxa de falta dos 28 dias anteriores (lag L) × agenda do dia."""
    nome, tunavel = "baseline_poisson", False

    def fit(self, ctx, idx_treino, k_h):
        return self

    def _taxa(self, ctx, idx):
        return ctx.P["r28"].to_numpy()[idx]


@registra
class HGBPoisson(_Serie):
    """HistGradientBoosting com perda Poisson sobre taxas defasadas, calendário e agenda (sem early stopping)."""
    nome = "hgb_poisson"

    @classmethod
    def espaco(cls):
        return _grade(learning_rate=[0.03, 0.06, 0.1], max_iter=[100, 200, 400], max_leaf_nodes=[8, 15, 31],
                      min_samples_leaf=[20, 50, 100], l2_regularization=[0.0, 1.0, 10.0])

    def fit(self, ctx, idx_treino, k_h):
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        # ANTI-VAZAMENTO: early_stopping=False; nº de iterações vem da validação interna (config de busca).
        self.m_ = HistGradientBoostingRegressor(loss="poisson", early_stopping=False, random_state=self.seed, **self.params)
        self.m_.fit(ctx.F.iloc[i][FEATS_NUM + FEATS_CAT], y, sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][FEATS_NUM + FEATS_CAT]), 0.0)


@registra
class RandomForest(_Serie):
    """Floresta aleatória (critério Poisson) sobre as mesmas variáveis; ausentes imputados pela mediana do treino."""
    nome = "rf"

    @classmethod
    def espaco(cls):
        return _grade(n_estimators=[100, 200, 300], max_depth=[6, 10, 16], min_samples_leaf=[5, 20, 50],
                      max_features=[0.5, 0.8, 1.0])

    def fit(self, ctx, idx_treino, k_h):
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        # ANTI-VAZAMENTO: o imputador (mediana) é ajustado só com as linhas de treino, dentro do Pipeline.
        self.m_ = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("rf", RandomForestRegressor(criterion="poisson", n_jobs=1, random_state=self.seed, **self.params))])
        self.m_.fit(ctx.F.iloc[i][FEATS_NUM + FEATS_CAT], y, rf__sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][FEATS_NUM + FEATS_CAT]), 0.0)


@registra
class PoissonRegularizado(_Serie):
    """GLM Poisson com penalização L2 (regressão regularizada); padronização e imputação só com o treino."""
    nome = "poisson_reg"

    @classmethod
    def espaco(cls):
        return _grade(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])

    def fit(self, ctx, idx_treino, k_h):
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), FEATS_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATS_CAT)])
        self.m_ = Pipeline([("pre", pre), ("glm", PoissonRegressor(max_iter=2000, **self.params))])
        self.m_.fit(ctx.F.iloc[i][FEATS_NUM + FEATS_CAT], y, glm__sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][FEATS_NUM + FEATS_CAT]), 0.0)


@registra
class LightGBMPoisson(_Serie):
    """LightGBM com objetivo Poisson. OPCIONAL: só roda se `lightgbm` estiver instalado."""
    nome = "lightgbm"

    @classmethod
    def disponivel(cls) -> bool:
        """True se `lightgbm` estiver instalado (modelo opcional)."""
        import importlib.util
        return importlib.util.find_spec("lightgbm") is not None

    @classmethod
    def espaco(cls):
        return _grade(learning_rate=[0.03, 0.06, 0.1], n_estimators=[100, 200, 400], num_leaves=[8, 15, 31],
                      min_child_samples=[20, 50, 100], reg_lambda=[0.0, 1.0, 10.0])

    def fit(self, ctx, idx_treino, k_h):
        import lightgbm as lgb
        i, y, w = self._linhas_ajuste(ctx, idx_treino, k_h)
        self.m_ = lgb.LGBMRegressor(objective="poisson", random_state=self.seed, n_jobs=1, verbose=-1, **self.params)
        self.m_.fit(ctx.F.iloc[i][FEATS_NUM + FEATS_CAT], y, sample_weight=w)
        return self

    def _taxa(self, ctx, idx):
        return np.maximum(self.m_.predict(ctx.F.iloc[idx][FEATS_NUM + FEATS_CAT]), 0.0)
