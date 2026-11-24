"""Modelo por paciente (S5): HGB que estima P(falta) de cada consulta e soma por CNES-dia.

Tuning pelo Brier por consulta nos folds internos (não por WAPE). Sem early stopping. Variáveis vêm do
config.yaml (`paciente`), sem `status` nem `possuiunidadeindicada`. Não executa nada ao ser importado.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.metrics import metricas_consulta
from src.models.base import Modelo, registra


@registra
class PacienteHGB(Modelo):
    """P(falta) por consulta (HGB); mu do CNES-dia = k_h × soma das probabilidades das consultas já agendadas."""
    nome, criterio = "paciente_hgb", "brier"

    @classmethod
    def espaco(cls):
        chaves = ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
        vals = [[0.03, 0.06, 0.1], [100, 200, 300], [15, 31, 63], [50, 100, 200], [0.0, 1.0, 10.0]]
        # DECISÃO: sem `max_bins` na grade (o 128 do script 22 era incompatível com `cnesexecutante` categórica).
        return [dict(zip(chaves, v)) for v in itertools.product(*vals)]

    def fit(self, ctx, idx_treino, k_h):
        sel = np.zeros(len(ctx.P), bool)
        sel[idx_treino] = True
        linhas = sel[ctx.raw["linha_painel"].to_numpy()]
        # ANTI-VAZAMENTO: só consultas cujo CNES-dia está no treino da divisão.
        self.m_ = HistGradientBoostingClassifier(early_stopping=False, categorical_features="from_dtype",
                                                 random_state=self.seed, **self.params)
        self.m_.fit(ctx.X[linhas], ctx.raw["y"].to_numpy()[linhas])
        return self

    def _prob(self, ctx, idx):
        sel = np.zeros(len(ctx.P), bool)
        sel[idx] = True
        linhas = sel[ctx.raw["linha_painel"].to_numpy()]
        if ctx.modo == "conhecida":           # só o que já estava agendado em t − L
            linhas &= ctx.raw["conhecida"].to_numpy()
        p = np.full(len(ctx.raw), np.nan)
        p[linhas] = self.m_.predict_proba(ctx.X[linhas])[:, 1]
        return p, linhas

    def predict_mean(self, ctx, idx, k_h):
        p, linhas = self._prob(ctx, idx)
        soma = np.bincount(ctx.raw["linha_painel"].to_numpy()[linhas], weights=p[linhas], minlength=len(ctx.P))
        # registra (y, p) por consulta para Brier/AUC/logloss (só agregados saem do experimento)
        y = ctx.raw["y"].to_numpy()
        self.p_consulta_ = (y[linhas], p[linhas])
        return soma[idx] * (1.0 if ctx.modo == "final" else k_h)

    def criterio_validacao(self, ctx, idx, mu, f):
        """Brier por consulta (soma dos erros quadráticos, nº de consultas) das linhas já previstas."""
        y, p = self.p_consulta_
        return float(((p - y) ** 2).sum()), float(len(y))
