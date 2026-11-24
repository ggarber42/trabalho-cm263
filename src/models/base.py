"""Interface comum dos modelos (S5): ``fit``, ``predict_mean``, ``predict_dist`` e espaço de busca.

Todo modelo prevê o NÚMERO ESPERADO DE FALTAS por CNES-dia (mu). A distribuição preditiva é a mesma para
todos (Binomial Negativa, ver src/metrics.py), com dispersão estimada fora da amostra; por isso
``predict_dist`` é comum e fica nesta classe.

Não executa nada ao ser importado.
"""
from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.metrics import MU_MIN, wape

REGISTRO: dict[str, type["Modelo"]] = {}


def registra(cls: type["Modelo"]) -> type["Modelo"]:
    """Decorador: registra a classe do modelo em REGISTRO pelo seu `nome`."""
    REGISTRO[cls.nome] = cls
    return cls


def lista_modelos_disponiveis(nomes: list[str]) -> tuple[list[str], list[str]]:
    """Separa os modelos pedidos entre disponíveis e indisponíveis (biblioteca opcional ausente).

    Parâmetros:
        nomes: nomes pedidos no config.yaml.

    Retorna:
        (disponíveis, indisponíveis). Modelos desconhecidos levantam AssertionError.
    """
    ok, falta = [], []
    for n in nomes:
        assert n in REGISTRO, f"Modelo desconhecido: {n}. Disponíveis: {sorted(REGISTRO)}"
        (ok if REGISTRO[n].disponivel() else falta).append(n)
    return ok, falta


class Modelo(ABC):
    """Contrato dos modelos.

    Atributos de classe:
        nome: identificador (igual ao do config.yaml).
        tunavel: se tem hiperparâmetros a buscar (False = sem ajuste, ex.: baseline).
        criterio: "wape" (séries) ou "brier" (por paciente) — usado na validação interna.
    """
    nome: str = ""
    tunavel: bool = True
    criterio: str = "wape"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.params = dict(params or {})
        self.seed = int(seed)
        self.p_consulta_: tuple[np.ndarray, np.ndarray] | None = None   # (y, p) por consulta, se aplicável

    @classmethod
    def disponivel(cls) -> bool:
        """True se as bibliotecas necessárias estão instaladas."""
        return True

    @classmethod
    def espaco(cls) -> list[dict[str, Any]]:
        """Lista de configurações candidatas (grade completa). Modelos sem ajuste devolvem [{}]."""
        return [{}]

    @abstractmethod
    def fit(self, ctx: Any, idx_treino: np.ndarray, k_h: float) -> "Modelo":
        """Ajusta o modelo com as linhas de treino (índices posicionais de ctx.P).

        Parâmetros:
            ctx: Contexto (src/dados.py). idx_treino: linhas de treino. k_h: fator de agenda do treino.

        Retorna:
            self. Cuidados: imputação e padronização só com estas linhas (regra 7).
        """

    @abstractmethod
    def predict_mean(self, ctx: Any, idx: np.ndarray, k_h: float) -> np.ndarray:
        """Número esperado de faltas por CNES-dia para as linhas `idx` (NaN se não houver informação)."""

    def criterio_validacao(self, ctx: Any, idx: np.ndarray, mu: np.ndarray, f: np.ndarray) -> tuple[float, int]:
        """Soma do critério e nº de unidades para pooling entre folds (padrão: erro absoluto para o WAPE).

        Retorna:
            (soma do erro absoluto sobre linhas com mu válido, soma de y nessas linhas) — o WAPE pooled é
            a razão das somas; devolvemos o numerador e o denominador codificados em (num, den).
        """
        ok = np.isfinite(mu)
        return float(np.abs(f[ok] - mu[ok]).sum()), float(f[ok].sum())

    def predict_dist(self, mu: np.ndarray, phi: float) -> dict[str, np.ndarray | float]:
        """Distribuição preditiva comum: NB com média mu e dispersão phi (Var = mu + phi·mu²).

        Retorna:
            dict com `mu` (com piso numérico) e `phi`; as funções de src/metrics.py consomem este par.
        """
        return {"mu": np.maximum(np.asarray(mu, float), MU_MIN), "phi": float(phi)}


def amostra_configs(espaco: list[dict[str, Any]], n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Sorteia até `n` configurações distintas do espaço (grade inteira se ela for menor que n).

    Parâmetros:
        espaco: lista de configurações; n: orçamento; rng: gerador com semente fixa.

    Retorna:
        Lista de configurações (ordem estável). # DECISÃO: orçamento igual para todos os modelos;
        grades menores que n são percorridas por inteiro (config.yaml: tuning.n_configs).
    """
    if len(espaco) <= n:
        return list(espaco)
    ix = rng.choice(len(espaco), size=n, replace=False)
    return [espaco[i] for i in sorted(ix)]
