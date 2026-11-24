"""Testes de divisões, modelos locais, exploratória e métricas."""
import numpy as np
import pandas as pd

from faltas.divisoes import gerar_dobras
from faltas.exploratoria import classe_syntetos_boylan
from faltas.metricas import crps_nb, holm, phi_momentos, wape
from faltas.modelos_series import prever_local

TUN = {"rep_holdout": {"n_repeticoes": 5, "semanas_validacao": 26, "fracao_minima_treino": 0.5},
       "cv_bloqueada": {"n_blocos": 5, "folga_semanas": 4}}


def test_rep_holdout_so_passado():
    sem = pd.date_range("2019-04-01", periods=200, freq="7D")
    for d in gerar_dobras(sem, "rep_holdout", TUN, 1):
        assert d.ajuste.max() < d.validacao.min()
        assert d.validacao.max() <= sem.max()
        assert len(d.validacao) == 26


def test_cv_bloqueada_folga():
    sem = pd.date_range("2019-04-01", periods=200, freq="7D")
    for d in gerar_dobras(sem, "cv_bloqueada", TUN, 1):
        dist = np.min(np.abs((d.ajuste.values[:, None] - d.validacao.values[None, :]).astype("timedelta64[D]").astype(int)))
        assert dist > 4 * 7


def test_modelos_series_serie_zero():
    y = np.zeros(150)
    for nome, p in [("ets", {"tendencia": None, "sazonal": None, "transf": "nenhuma"}),
                    ("croston_tsb", {"metodo": "croston", "alfa": 0.1}),
                    ("croston_tsb", {"metodo": "tsb", "alfa": 0.1, "beta": 0.1})]:
        mu, _ = prever_local(nome, y, None, 120, list(range(120, 150)), 1, p)
        assert len(mu) == 30  # não quebra; pode ser 0 ou NaN


def test_ets_todas_configuracoes_produzem_previsao():
    from faltas.modelos_series import grade_ets
    rng = np.random.default_rng(1)
    y = rng.poisson(20, 160).astype(float)
    for p in grade_ets():
        for n in (80, 130):  # sem e com 2 ciclos anuais
            mu, _ = prever_local("ets", y, None, n, list(range(n, n + 5)), 1, p)
            assert np.isfinite(mu).all(), p


def test_croston_nao_usa_futuro():
    rng = np.random.default_rng(0)
    y = rng.poisson(2, 120).astype(float)
    p = {"metodo": "croston", "alfa": 0.2}
    mu1, _ = prever_local("croston_tsb", y, None, 100, [110], 2, p)
    y2 = y.copy(); y2[109:] = 999          # mudar a semana 109 (= origem para h=2 é 108) não altera
    mu2, _ = prever_local("croston_tsb", y2, None, 100, [110], 2, p)
    assert mu1[0] == mu2[0]


def test_metricas():
    y = np.array([0, 2, 4.0])
    assert wape(y, np.array([0, 2, 4.0])) == 0
    assert phi_momentos(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0
    # CRPS de previsão quase determinística correta ≈ 0
    assert crps_nb(np.array([0.0]), np.array([1e-8]), 0.0)[0] < 1e-6
    p = holm(pd.Series([0.01, 0.04, 0.03]))
    assert np.isclose(p.iloc[0], 0.03) and np.isclose(p.max(), 0.06)


def test_classes():
    assert classe_syntetos_boylan(1.0, 0.2) == "suave"
    assert classe_syntetos_boylan(2.0, 0.2) == "intermitente"
    assert classe_syntetos_boylan(2.0, 0.8) == "irregular"
    assert classe_syntetos_boylan(1.0, 0.8) == "erratica"


def test_taxa_local_inversa_exata():
    """Com previsão perfeita da taxa, a contagem prevista reproduz a observada."""
    import sys, importlib
    sys.path.insert(0, "etapas")
    e04 = importlib.import_module("e04_modelar")
    rng = np.random.default_rng(3)
    n_ag = 4.0 * rng.integers(5, 15, 120)          # taxa exata de 0,25 por consulta agendada
    y = n_ag * 0.25
    mu, _ = e04.prever_local_taxa("croston_tsb", y, n_ag, None, 80, list(range(90, 100)), 1,
                                  {"metodo": "croston", "alfa": 1.0}, 0)
    # taxa constante -> taxa × agenda conhecida do alvo reproduz a contagem
    assert np.allclose(mu, y[90:100])


def test_global_taxa_com_peso():
    from faltas.modelos_ml import ajustar_prever_global
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.normal(size=300)})
    e = rng.integers(10, 100, 300).astype(float)
    y = rng.poisson(0.2 * e).astype(float)
    for nome, p in [("glm_poisson", {"alpha": 1e-4}), ("hgb_poisson", {"max_iter": 50}),
                    ("floresta", {"n_estimators": 20}), ("mlp", {"hidden_layer_sizes": (8,)})]:
        mu = ajustar_prever_global(nome, p, X, y, e, X, e, 0)
        assert abs(mu.sum() / y.sum() - 1) < 0.15, nome


def test_preprocessador_escala_e_sem_vazamento():
    from faltas.preprocessamento import Preprocessador
    Xa = pd.DataFrame({"n_agenda": [0, 50, 200.0], "prop": [0.1, 0.5, np.nan]})
    pre = Preprocessador().ajustar(Xa, np.array([0, 10, 40.0]), np.array([1, 50, 200.0]))
    Za = pre.transformar(Xa)
    assert Za.min() >= 0 and Za.max() <= 1 and "n_agenda" in pre.cols_log and "prop" not in pre.cols_log
    # valor novo acima do máximo do treino é cortado em 1 (não gera entrada extrema)
    assert pre.transformar(pd.DataFrame({"n_agenda": [10_000.0], "prop": [0.3]}))[0, 0] == 1.0
    # pós: taxa negativa vira 0; taxa acima do teto é cortada
    assert pre.pos(np.array([-1.0, 99.0]), np.array([10.0, 10.0]))[0] == 0
    assert pre.pos(np.array([99.0]), np.array([10.0]))[0] <= pre.taxa_max * 10 + 1e-9
