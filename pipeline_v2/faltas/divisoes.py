"""Dobras de validação dentro do treino, conforme o esquema escolhido na exploratória (D10)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dobra:
    """Uma dobra de validação.

    ajuste: semanas usadas para ajustar (modelos globais); validacao: semanas avaliadas;
    inicio_validacao: primeira semana validada. Modelos locais de série usam só as semanas de
    ajuste ANTERIORES a inicio_validacao (precisam de passado contíguo).
    """
    ajuste: pd.DatetimeIndex
    validacao: pd.DatetimeIndex
    inicio_validacao: pd.Timestamp


def gerar_dobras(semanas_treino: pd.DatetimeIndex, esquema: str, cfg_tuning: dict, semente: int) -> list[Dobra]:
    """Gera as dobras de validação só com semanas do treino.

    - rep_holdout (Cerqueira et al. 2020): n_repeticoes origens a sorteadas entre
      fracao_minima_treino·N e N − semanas_validacao; ajuste = semanas < a; validação = [a, a + V).
    - cv_bloqueada: n_blocos blocos contíguos; ajuste = demais semanas menos `folga_semanas`
      de cada lado do bloco (hv-blocked).
    Retorna: lista de Dobra, ordenada pela semana de início da validação.
    ANTI-VAZAMENTO: nenhuma semana posterior ao fim do treino é usada.
    """
    semanas = pd.DatetimeIndex(sorted(semanas_treino.unique()))
    n = len(semanas)
    dobras: list[Dobra] = []
    if esquema == "rep_holdout":
        c = cfg_tuning["rep_holdout"]
        v = c["semanas_validacao"]
        lo, hi = int(c["fracao_minima_treino"] * n), n - v
        assert hi > lo, "Treino curto demais para o holdout repetido"
        rng = np.random.default_rng(semente)
        origens = np.sort(rng.choice(np.arange(lo, hi + 1), size=min(c["n_repeticoes"], hi - lo + 1), replace=False))
        for a in origens:
            dobras.append(Dobra(semanas[:a], semanas[a:a + v], semanas[a]))
    elif esquema == "cv_bloqueada":
        c = cfg_tuning["cv_bloqueada"]
        blocos = np.array_split(np.arange(n), c["n_blocos"])
        g = c["folga_semanas"]
        for b in blocos[1:]:  # o 1º bloco não tem passado para os modelos locais
            fora = set(range(max(b[0] - g, 0), min(b[-1] + g + 1, n)))
            ajuste = semanas[[i for i in range(n) if i not in fora]]
            dobras.append(Dobra(ajuste, semanas[b], semanas[b[0]]))
    else:
        raise ValueError(f"Esquema desconhecido: {esquema}")
    for d in dobras:
        assert d.validacao.max() <= semanas.max(), "Validação fora do treino"
    return dobras
