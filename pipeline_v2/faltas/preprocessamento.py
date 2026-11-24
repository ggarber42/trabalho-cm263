"""Pré e pós-processamento comum a todos os modelos globais (D17).

Pré-processamento (ajustado SÓ nas linhas de treino de cada ajuste):
1. variáveis de contagem (valores ≥ 0 e máximo > 10 no treino, ex.: n_agenda, defasagens,
   médias móveis) passam por log1p, para comprimir valores altos;
2. valores ausentes recebem a mediana do treino;
3. escala min-max para [0, 1] com o mínimo e o máximo do treino; valores novos fora da faixa
   são cortados em 0 e 1 (DECISÃO: evita entradas extremas no MLP).
Alvo: taxa = faltas ÷ exposição, com exposição e = max(n_agenda, 1).
Pós-processamento: taxa prevista cortada em [0, taxa_max_treino] e multiplicada pela exposição,
devolvendo o número de faltas previsto.

ANTI-VAZAMENTO: todos os parâmetros (colunas log, medianas, mínimos, máximos, taxa máxima) vêm
só das linhas passadas a `ajustar`; nada do período previsto é usado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Preprocessador:
    """Transformações aprendidas no treino e aplicadas igualmente a validação e teste."""
    limiar_contagem: float = 10.0
    cols_log: list[str] = field(default_factory=list)
    mediana: pd.Series | None = None
    minimo: pd.Series | None = None
    amplitude: pd.Series | None = None
    taxa_max: float = np.inf

    # ------------------------------------------------------------------ entradas
    def ajustar(self, X: pd.DataFrame, y: np.ndarray, e: np.ndarray) -> "Preprocessador":
        """Aprende as transformações em (X, y, e) de treino.
        Parâmetros: X = preditores; y = faltas; e = exposição. Retorna: o próprio objeto."""
        X = X.astype(float)
        self.cols_log = [c for c in X.columns if X[c].min(skipna=True) >= 0 and X[c].max(skipna=True) > self.limiar_contagem]
        Z = self._log(X)
        self.mediana = Z.median()
        Z = Z.fillna(self.mediana)
        self.minimo = Z.min()
        self.amplitude = (Z.max() - self.minimo).replace(0, 1.0)   # coluna constante vira 0
        taxa = y / e
        # DECISÃO: teto da taxa = percentil 99,5 do treino (evita previsões absurdas).
        self.taxa_max = float(np.quantile(taxa, 0.995)) if len(taxa) else np.inf
        return self

    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.astype(float).copy()
        for c in self.cols_log:
            Z[c] = np.log1p(Z[c].clip(lower=0))
        return Z

    def transformar(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica log1p, imputação e min-max (cortado em [0, 1]). Retorna np.ndarray float."""
        Z = self._log(X).fillna(self.mediana)
        Z = (Z - self.minimo) / self.amplitude
        return Z.clip(0, 1).to_numpy(float)

    # ------------------------------------------------------------------ alvo
    @staticmethod
    def alvo(y: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Taxa de falta por consulta conhecida: y / e."""
        return y / e

    def pos(self, taxa_prevista: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Pós-processamento: corta a taxa em [0, taxa_max] e devolve faltas = taxa × e."""
        return np.clip(taxa_prevista, 0, self.taxa_max) * e
