"""Testes do painel semanal e do target encoding temporal."""
import numpy as np
import pandas as pd

from faltas.painel import montar_painel, origem_da_semana, semana_de, te_temporal

CFG = {"periodo": {"inicio_historico": "2020-01-06", "inicio_analise": "2020-01-06", "fim_analise": "2020-03-01"},
       "variaveis": {"lags_y": 2, "baseline_janela": 2, "fourier_k": 1}}


def _cons(linhas):
    df = pd.DataFrame(linhas, columns=["cnesexecutante", "t0", "data_consulta", "y"])
    df["especialidademae"] = "ESP"
    df["t0"] = pd.to_datetime(df["t0"])
    df["data_consulta"] = pd.to_datetime(df["data_consulta"])
    for c, v in [("n_faltas_previas", 0), ("ultima_previa_mesma_especialidademae", "sem_previa"),
                 ("idade", 40), ("possui_unidade_indicada", 0), ("antecedencia", 5)]:
        df[c] = v
    return df


def test_origem():
    s = pd.Timestamp("2020-01-13")  # segunda
    assert origem_da_semana(s, 1) == pd.Timestamp("2020-01-12")
    assert origem_da_semana(s, 2) == pd.Timestamp("2020-01-05")
    assert semana_de(pd.Series([pd.Timestamp("2020-01-19")])).iloc[0] == s


def test_painel_semanas_contiguas_e_lags():
    cons = _cons([("X", "2020-01-01", "2020-01-07", 1), ("X", "2020-01-01", "2020-01-08", 1),
                  ("X", "2020-01-01", "2020-01-28", 0), ("X", "2020-01-01", "2020-01-29", 1)])
    p = montar_painel(cons, 1, CFG)
    # semanas 06/01, 13/01, 20/01 (vazia), 27/01, ... até 24/02
    assert p["semana"].diff().dropna().eq(pd.Timedelta(days=7)).all()
    linha = p.set_index("semana")
    assert linha.loc["2020-01-20", "y"] == 0 and not linha.loc["2020-01-20", "avaliavel"]
    assert linha.loc["2020-01-13", "lag1"] == 2           # semana anterior
    assert linha.loc["2020-01-27", "lag2"] == 0           # duas semanas antes
    assert linha.loc["2020-01-20", "mm4"] == 1.0          # média de (2, 0)


def test_agenda_conhecida_respeita_origem():
    # semana alvo 13/01; origem h=1 = 12/01; h=2 = 05/01
    cons = _cons([("X", "2020-01-04", "2020-01-14", 0),   # conhecida em h=1 e h=2
                  ("X", "2020-01-10", "2020-01-15", 1),   # conhecida só em h=1
                  ("X", "2020-01-13", "2020-01-16", 1)])  # agendada na própria semana: nunca
    p1 = montar_painel(cons, 1, CFG).set_index("semana")
    p2 = montar_painel(cons, 2, CFG).set_index("semana")
    assert p1.loc["2020-01-13", "n_agenda"] == 2
    assert p2.loc["2020-01-13", "n_agenda"] == 1
    assert p1.loc["2020-01-13", "y"] == 2 and p1.loc["2020-01-13", "n_total"] == 3


def test_te_congelado_no_teste():
    datas = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03"])
    cons = pd.DataFrame({"semana": datas, "cat": ["a"] * 5, "y": [1, 1, 0, 1, 1]})
    te = te_temporal(cons, "cat", h=1, fim_treino=pd.Timestamp("2020-01-26"), m=0)
    assert np.isnan(te.iloc[0])                     # sem histórico nem taxa geral
    assert te.iloc[1] == 1.0                        # só a semana 06/01
    assert te.iloc[2] == 1.0                        # 06/01 e 13/01
    # semanas de teste (27/01, 03/02): congeladas no fim do treino = semanas até 20/01 -> 2/3
    assert np.isclose(te.iloc[3], 2 / 3) and np.isclose(te.iloc[4], 2 / 3)
    # mudar o desfecho do teste não muda o TE do teste
    cons2 = cons.copy(); cons2.loc[3, "y"] = 0
    assert np.allclose(te_temporal(cons2, "cat", 1, pd.Timestamp("2020-01-26"), 0).iloc[3:], te.iloc[3:])


def test_calendario_feriados():
    from faltas.painel import calendario, feriados
    fer = feriados(range(2023, 2025))
    assert pd.Timestamp("2024-11-20") in fer and pd.Timestamp("2023-11-20") not in fer
    sem = pd.Series(pd.to_datetime(["2024-11-11", "2024-11-18", "2024-11-25", "2024-12-23"]))
    c = calendario(sem, 1)
    # semana de 18/11/2024: 20/11 (quarta) -> 1 feriado, 4 dias úteis, sem ponte
    assert c.loc[1, "n_feriados"] == 1 and c.loc[1, "dias_uteis"] == 4 and c.loc[1, "ponte"] == 0
    # semana de 11/11/2024: 15/11 é sexta -> sem ponte; semana seguinte tem feriado
    assert c.loc[0, "feriado_semana_seguinte"] == 1 and c.loc[2, "feriado_semana_anterior"] == 1
    assert c.loc[3, "recesso"] == 1 and c.loc[0, "recesso"] == 0


def test_taxa_agenda_nao_usa_semana_alvo():
    linhas = [("X", "2020-01-01", f"2020-{m:02d}-{d:02d}", 1) for m, d in
              [(1, 7), (1, 14), (1, 21), (1, 28), (2, 4), (2, 11), (2, 18), (2, 25)]]
    cfg = {"periodo": {"inicio_historico": "2020-01-06", "inicio_analise": "2020-01-06", "fim_analise": "2020-03-01"},
           "variaveis": {"lags_y": 1, "baseline_janela": 2, "fourier_k": 1, "taxa_suavizacao_m": 0}}
    cfg["variaveis"]["lags_y"] = 1
    p1 = montar_painel(_cons(linhas), 1, cfg).set_index("semana")
    linhas2 = linhas[:-1] + [("X", "2020-01-01", "2020-02-25", 0)]   # muda só a última semana
    p2 = montar_painel(_cons(linhas2), 1, cfg).set_index("semana")
    assert p1["taxa_ag4"].equals(p2["taxa_ag4"])
    assert p1["ref_taxa_agenda"].dropna().eq(1.0).all()      # 1 falta por consulta agendada


def test_series_por_especialidade():
    cons = _cons([("X", "2020-01-01", "2020-01-07", 1), ("X", "2020-01-01", "2020-01-08", 0)])
    cons.loc[1, "especialidademae"] = "OUTRA"
    p = montar_painel(cons, 1, CFG)
    assert set(p["serie"]) == {"X|ESP", "X|OUTRA"}
    assert not p.duplicated(["serie", "semana"]).any()
    s = p[p["semana"] == pd.Timestamp("2020-01-06")].set_index("especialidademae")
    assert s.loc["ESP", "y"] == 1 and s.loc["OUTRA", "y"] == 0 and (s["cnesexecutante"] == "X").all()
