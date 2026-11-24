"""Testes da preparação: histórico sem vazamento, zfill e fluxograma."""
import numpy as np
import pandas as pd

from faltas.preparacao import calcular_historico, padronizar


def _df(linhas):
    df = pd.DataFrame(linhas, columns=["idpaciente", "data_solicitacao", "diasfilaespera", "data_consulta", "y", "especialidademae"])
    df["data_solicitacao"] = pd.to_datetime(df["data_solicitacao"])
    df["data_consulta"] = pd.to_datetime(df["data_consulta"])
    return df


def test_historico_sem_vazamento():
    df = _df([
        # paciente A: falta em 01/01, realizada em 10/01, alvo agendado em 10/01 (t0) para 20/01
        ("A", "2020-01-01", 0, "2020-01-01", 1, "CARDIO"),   # t0 = 01/01: nada antes
        ("A", "2020-01-02", 0, "2020-01-10", 0, "CARDIO"),   # t0 = 02/01: a falta de 01/01 conta
        ("A", "2020-01-05", 5, "2020-01-20", 1, "DERMATO"),  # t0 = 10/01: consulta de 10/01 NÃO conta (mesmo dia)
        ("B", "2020-01-01", 0, "2020-02-01", 0, "CARDIO"),   # outro paciente
    ])
    h = calcular_historico(df)
    assert h["n_consultas_previas"].tolist() == [0, 1, 1, 0]
    assert h["n_faltas_previas"].tolist() == [0, 1, 1, 0]
    assert h.loc[2, "dias_desde_ultima_falta"] == 9
    assert np.isnan(h.loc[0, "dias_desde_ultima_falta"])
    assert h["n_faltas_previas_especialidademae"].tolist() == [0, 1, 0, 0]
    assert h["ultima_previa_mesma_especialidademae"].astype(str).tolist() == ["sem_previa", "faltante", "sem_previa", "sem_previa"]


def test_historico_ignora_propria_consulta_com_data_errada():
    # consulta registrada antes do próprio agendamento (erro): não pode contar a si mesma
    df = _df([("A", "2020-01-10", 0, "2020-01-05", 1, "CARDIO")])
    h = calcular_historico(df)
    assert h["n_consultas_previas"].iloc[0] == 0
    assert h["n_faltas_previas"].iloc[0] == 0
    assert np.isnan(h["dias_desde_ultima_falta"].iloc[0])


def test_padronizar_zfill_e_datas():
    df = pd.DataFrame({
        "data_solicitacao": ["2024-05-10 14:30:00.000", "2025-01-02"],
        "datahora_consulta": ["2024-06-01 09:00:00.000", "2025-02-03"],
        "diasfilaespera": ["3", "0"], "idade": ["40", "50"],
        "cnesexecutante": ["123456", "2237822"], "cnes_solicitante": ["1234567", " 99999 "],
    })
    p = padronizar(df, log=lambda m: None)
    assert p["cnesexecutante"].tolist() == ["0123456", "2237822"]
    assert p["cnes_solicitante"].tolist() == ["1234567", "0099999"]
    assert p["data_consulta"].dt.hour.max() == 0
