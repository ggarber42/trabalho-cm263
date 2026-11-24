"""Modelos globais de aprendizado de máquina (todas as CNES juntas) e o modelo por consulta.

Informação (D15, substitui os blocos A/B da D9): todo modelo recebe o mesmo conjunto —
defasagens, médias móveis, taxas de falta por consulta agendada, calendário, agenda conhecida na
origem e agregados dos pacientes agendados.
Alvo (D15): os modelos globais estimam a TAXA y / e, com exposição e = max(n_agenda, 1) e peso e
(equivalente ao offset log(e) na regressão de Poisson); a previsão devolvida é a contagem taxa × e.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPRegressor

from .painel import AGREGADOS_B, COLS_CALENDARIO
from .preprocessamento import Preprocessador


def colunas_preditoras(painel: pd.DataFrame) -> list[str]:
    """Lista única de preditores do painel (Tabela 1 da Metodologia)."""
    base = [c for c in painel.columns if c.startswith("lag")] + ["mm4", "mm13", "taxa_ag4", "taxa_ag13"]
    base += COLS_CALENDARIO + [c for c in painel.columns if c.startswith(("sen", "cos"))]
    base += ["n_agenda", "log_agenda"] + AGREGADOS_B
    return base


def preparar_X(painel: pd.DataFrame) -> pd.DataFrame:
    """Matriz de preditores (cria log_agenda)."""
    X = painel.copy()
    X["log_agenda"] = np.log1p(X["n_agenda"])
    return X[colunas_preditoras(painel)].astype(float)


def exposicao(painel: pd.DataFrame) -> np.ndarray:
    """Exposição e = max(n_agenda, 1): consultas conhecidas na origem (mínimo 1 para semanas
    sem agenda conhecida mas com faltas)."""
    return np.maximum(painel["n_agenda"].to_numpy(float), 1.0)


GRADES = {
    "glm_poisson": {"alpha": list(np.logspace(-4, 1, 8))},
    "hgb_poisson": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31], "min_samples_leaf": [20, 50], "max_iter": [200, 400]},
    "floresta": {"max_depth": [None, 12], "min_samples_leaf": [5, 20], "max_features": [0.5, 1.0], "n_estimators": [300]},
    "mlp": {"hidden_layer_sizes": [(32,), (64,), (32, 16)], "alpha": [1e-4, 1e-2], "learning_rate_init": [1e-3, 1e-2]},
    "consulta_hgb": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31], "min_samples_leaf": [50, 200], "max_iter": [200, 400]},
}


def configuracoes(nome: str, n: int, semente: int) -> list[dict]:
    """Até n configurações sorteadas (semente fixa) da grade do modelo; grade menor é percorrida inteira."""
    total = _tamanho(GRADES[nome])
    return list(ParameterSampler(GRADES[nome], n_iter=min(n, total), random_state=semente))


def _tamanho(grade: dict) -> int:
    return int(np.prod([len(v) for v in grade.values()]))


def criar_regressor(nome: str, params: dict, semente: int):
    """Instancia o regressor. A imputação e a escala ficam no `Preprocessador` (comum a todos)."""
    if nome == "glm_poisson":
        return PoissonRegressor(max_iter=1000, **params)
    if nome == "hgb_poisson":
        return HistGradientBoostingRegressor(loss="poisson", random_state=semente, early_stopping=False, **params)
    if nome == "floresta":
        return RandomForestRegressor(random_state=semente, n_jobs=-1, **params)
    if nome == "mlp":
        # DECISÃO: sem early stopping (usaria divisão aleatória); número de épocas fixo.
        return MLPRegressor(max_iter=500, early_stopping=False, random_state=semente, **params)
    raise ValueError(nome)


def ajustar_prever_global(nome: str, params: dict, Xa: pd.DataFrame, ya: np.ndarray, ea: np.ndarray,
                          Xp: pd.DataFrame, ep: np.ndarray, semente: int) -> np.ndarray:
    """Pré-processa (ajustado só em Xa), ajusta a taxa ya/ea com peso ea e devolve a contagem
    prevista para Xp após o pós-processamento (taxa cortada × ep).

    Parâmetros: X* = preditores; ya = faltas observadas; ea, ep = exposição (ver `exposicao`).
    Retorna: np.ndarray de médias previstas para as linhas de Xp.
    ANTI-VAZAMENTO: o Preprocessador só vê as linhas de ajuste.
    """
    pre = Preprocessador().ajustar(Xa, ya, ea)
    m = criar_regressor(nome, params, semente)
    m.fit(pre.transformar(Xa), pre.alvo(ya, ea), sample_weight=ea)
    return pre.pos(m.predict(pre.transformar(Xp)), ep)


# ----------------------------------------------------------------------------- por consulta
COLS_CONSULTA_NUM = ["n_consultas_previas", "n_faltas_previas", "dias_desde_ultima_falta",
                     "n_faltas_previas_especialidademae", "te_especialidade", "te_cnes_solicitante",
                     "idade", "possui_unidade_indicada", "tempo_fila", "antecedencia", "mes", "dia_semana"]
COLS_CONSULTA_CAT = ["ultima_previa_mesma_especialidademae", "especialidademae", "cid_capitulo", "corregulador", "cnesexecutante"]


def matriz_consulta(base: pd.DataFrame, categorias: dict[str, list] | None = None) -> tuple[pd.DataFrame, dict]:
    """Preditores do modelo por consulta. Categóricas viram códigos inteiros com as categorias
    vistas no treino (`categorias`); categoria nova vira NaN.
    Retorna: (X, categorias usadas)."""
    X = base[COLS_CONSULTA_NUM].astype(float).copy()
    if categorias is None:
        categorias = {c: sorted(base[c].astype(object).fillna("NA").astype(str).unique()) for c in COLS_CONSULTA_CAT}
    for c in COLS_CONSULTA_CAT:
        mapa = {v: i for i, v in enumerate(categorias[c])}
        X[c] = base[c].astype(object).fillna("NA").astype(str).map(mapa).astype(float)
    return X, categorias


def ajustar_prever_consulta(params: dict, base_a: pd.DataFrame, base_p: pd.DataFrame, semente: int) -> np.ndarray:
    """Ajusta o classificador por consulta (perda logística) e devolve P(falta) para base_p."""
    Xa, cats = matriz_consulta(base_a)
    Xp, _ = matriz_consulta(base_p, cats)
    eh_cat = [c in COLS_CONSULTA_CAT for c in Xa.columns]
    m = HistGradientBoostingClassifier(random_state=semente, early_stopping=False, categorical_features=eh_cat, **params)
    m.fit(Xa, base_a["y"].to_numpy())
    return m.predict_proba(Xp)[:, 1]
