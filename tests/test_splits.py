"""Testes de src/splits.py (S4): nenhuma linha em dois conjuntos, treino anterior ao teste,
purga exata, folds internos para todos os anos de teste (inclusive 2026) e embargo.

Usam uma base SINTÉTICA com uma linha por dia (2018-01-01 a 2026-03-31), em que a posição da linha
é a própria identidade: fica fácil prever exatamente quais linhas devem entrar/sair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.splits import (ESQUEMAS, descreve_divisoes, gera_divisoes, purga_recomendada,  # noqa: E402
                        restringe_teste_comum, valida_divisao)

DIAS = pd.date_range("2018-01-01", "2026-03-31", freq="D").to_numpy()
ANOS = (2023, 2024, 2025, 2026)


def _gera(esquema: str, **kw):
    kw.setdefault("janela_meses", 24 if esquema == "deslizante" else None)
    return gera_divisoes(DIAS, esquema, anos_teste=ANOS, **kw)


@pytest.mark.parametrize("esquema", ESQUEMAS)
def test_nenhum_treino_posterior_ao_inicio_do_teste(esquema):
    """Nenhum índice de treino cai em data >= início do teste (teste externo e folds internos)."""
    for dv in _gera(esquema, purga_dias=28):
        assert DIAS[dv.treino].max() < np.datetime64(dv.inicio_teste)
        for f in dv.folds_internos:
            assert DIAS[f.treino].max() < np.datetime64(f.inicio_teste)
            assert DIAS[f.teste].max() < np.datetime64(dv.inicio_teste)
        valida_divisao(dv, DIAS)


@pytest.mark.parametrize("esquema", ESQUEMAS)
def test_nenhuma_linha_em_dois_conjuntos(esquema):
    """Treino e teste não se sobrepõem, nem treino e validação em cada fold."""
    for dv in _gera(esquema, purga_dias=28):
        assert np.intersect1d(dv.treino, dv.teste).size == 0
        for f in dv.folds_internos:
            assert np.intersect1d(f.treino, f.teste).size == 0


@pytest.mark.parametrize("esquema", ESQUEMAS)
def test_folds_existem_para_todos_os_anos_incluindo_2026(esquema):
    """Cada ano de teste tem 2 folds internos; para 2026 as validações são 2024 e 2025."""
    divs = _gera(esquema)
    if esquema == "holdout":
        assert len(divs) == 1 and [f.inicio_teste.year for f in divs[0].folds_internos] == [2021, 2022]
        return
    assert [int(d.rotulo[-4:]) for d in divs] == list(ANOS)
    for dv in divs:
        anos_val = [f.inicio_teste.year for f in dv.folds_internos]
        assert anos_val == [dv.inicio_teste.year - 2, dv.inicio_teste.year - 1]
    d2026 = [d for d in divs if d.inicio_teste.year == 2026][0]
    assert [f.inicio_teste.year for f in d2026.folds_internos] == [2024, 2025]
    for f in d2026.folds_internos:
        assert len(f.treino) > 0 and len(f.teste) > 0


def test_expansiva_treino_cresce_e_bate_com_o_esperado():
    """Na expansiva, o treino de Y é exatamente tudo antes de 1/jan/Y."""
    for dv in _gera("expansiva"):
        esperado = np.flatnonzero(DIAS < np.datetime64(dv.inicio_teste))
        assert np.array_equal(dv.treino, esperado)
    tamanhos = [len(d.treino) for d in _gera("expansiva")]
    assert tamanhos == sorted(tamanhos) and len(set(tamanhos)) == len(tamanhos)


def test_deslizante_tem_tamanho_fixo_de_meses():
    """Na deslizante com 24 meses, o treino de 2024 vai de 1/jan/2022 a 31/dez/2023."""
    dv = [d for d in _gera("deslizante", janela_meses=24) if d.inicio_teste.year == 2024][0]
    assert pd.Timestamp(DIAS[dv.treino].min()) == pd.Timestamp("2022-01-01")
    assert pd.Timestamp(DIAS[dv.treino].max()) == pd.Timestamp("2023-12-31")


def test_deslizante_exige_janela():
    with pytest.raises(AssertionError):
        gera_divisoes(DIAS, "deslizante", anos_teste=ANOS)


@pytest.mark.parametrize("P", [0, 28, 90])
def test_purga_remove_exatamente_as_linhas_esperadas(P):
    """Com purga P, saem do treino exatamente as P linhas diárias [início − P, início)."""
    for dv in _gera("purged_embargo", purga_dias=P):
        ini = np.datetime64(dv.inicio_teste)
        sem_purga = np.flatnonzero(DIAS < ini)
        assert dv.n_purgadas == P
        removidas = np.setdiff1d(sem_purga, dv.treino)
        assert removidas.size == P
        if P:
            assert DIAS[removidas].min() == ini - np.timedelta64(P, "D")
            assert DIAS[removidas].max() == ini - np.timedelta64(1, "D")
        for f in dv.folds_internos:     # a purga vale também dentro dos folds
            assert f.n_purgadas == P


def test_embargo_sem_efeito_para_frente_e_com_efeito_com_futuro():
    """Esquema só para frente: embargo não remove nada. Com usar_futuro, remove E dias após o teste."""
    for dv in _gera("purged_embargo", purga_dias=28, embargo_dias=7):
        assert dv.n_embargadas == 0
    dv = [d for d in gera_divisoes(DIAS, "purged_embargo", anos_teste=[2023], purga_dias=28, embargo_dias=7,
                                   usar_futuro=True)][0]
    assert dv.n_embargadas == 7
    fim = np.datetime64(dv.fim_teste)
    assert not np.isin(np.flatnonzero((DIAS >= fim) & (DIAS < fim + np.timedelta64(7, "D"))), dv.treino).any()
    assert np.isin(np.flatnonzero(DIAS >= fim + np.timedelta64(7, "D")), dv.treino).all()
    assert np.intersect1d(dv.treino, dv.teste).size == 0


def test_validacao_detecta_vazamento_proposital():
    """valida_divisao deve falhar se colocarmos uma linha do teste no treino ou dentro da purga."""
    dv = _gera("purged_embargo", purga_dias=28)[0]
    ruim = _gera("purged_embargo", purga_dias=28)[0]
    ruim.treino = np.append(ruim.treino, ruim.teste[0])
    with pytest.raises(AssertionError):
        valida_divisao(ruim, DIAS)
    ruim2 = _gera("purged_embargo", purga_dias=28)[0]
    ruim2.treino = np.append(ruim2.treino, np.flatnonzero(DIAS < np.datetime64(ruim2.inicio_teste))[-1])
    with pytest.raises(AssertionError):
        valida_divisao(ruim2, DIAS)
    valida_divisao(dv, DIAS)


def test_2026_termina_na_ultima_data_da_base():
    """O teste de 2026 é parcial: vai até 2026-03-31 e tem 90 linhas diárias."""
    dv = [d for d in _gera("expansiva") if d.inicio_teste.year == 2026][0]
    assert len(dv.teste) == 31 + 28 + 31
    assert pd.Timestamp(DIAS[dv.teste].max()) == pd.Timestamp("2026-03-31")


def test_teste_comum():
    """O conjunto comum de dias de teste é o mesmo para qualquer esquema (mesmo período)."""
    conjuntos = []
    for esq in ("expansiva", "deslizante", "purged_embargo"):
        idx = np.concatenate([restringe_teste_comum(dv, DIAS, "2024-01-01", "2026-03-31") for dv in _gera(esq)])
        conjuntos.append(set(idx.tolist()))
    assert conjuntos[0] == conjuntos[1] == conjuntos[2]
    assert DIAS[min(conjuntos[0])] == np.datetime64("2024-01-01")


def test_purga_recomendada():
    """P recomendada = percentil da antecedência (consulta − t0) + L."""
    assert purga_recomendada([0, 5, 14, 14, 90], lag=3, percentil=100) == 93
    assert purga_recomendada([0, 5, 14, 14, 14], lag=3, percentil=99) == 17


def test_relatorio_n_por_divisao():
    """A tabela descritiva tem uma linha para cada teste externo e cada fold interno."""
    divs = _gera("expansiva")
    t = descreve_divisoes(divs)
    assert len(t) == len(divs) * 3
    assert set(t["papel"]) == {"teste_externo", "validacao_interna"}
