"""Baseline DIARIO: previsao de faltas por CNES-dia, decisao 3 dias antes, agenda do dia conhecida.

Forecast = taxa de falta dos ultimos 28 dias corridos com informacao ate D-3 (lag 3)
           x consultas agendadas no dia D.
Intervalo 80%: quantis 10/90 dos residuos das ultimas 120 CNES-dias uteis (lag 3).
Teste: a partir de 2023-01-02.
"""
import numpy as np
import pandas as pd

SRC = "gercon_poa_censo_distancias.parquet"
top = pd.read_csv("ranking_top20.csv", dtype={"cnesexecutante": str})["cnesexecutante"].tolist()

raw = pd.read_parquet(SRC)
print("t0 vs datas (amostra 5):")
print(raw[["data_solicitacao", "t0", "datahora_consulta", "dias_ate_consulta", "n_faltas_previas", "rid"]].sample(5, random_state=1).to_string())
print("t0 == data_solicitacao:", (raw.t0 == raw.data_solicitacao).mean().round(3),
      "| t0 == datahora_consulta:", (raw.t0 == raw.datahora_consulta).mean().round(3))
print("dias_ate_consulta: mediana", raw.dias_ate_consulta.median(), "p10/p90", raw.dias_ate_consulta.quantile([.1, .9]).tolist())

d = raw[raw.cnesexecutante.isin(top)][["datahora_consulta", "situacaoagenda", "cnesexecutante"]].copy()
d["dia"] = pd.to_datetime(d.datahora_consulta)
d["falta"] = (d.situacaoagenda == "FALTANTE").astype(float)
g = d.groupby(["cnesexecutante", "dia"], observed=True).agg(ag=("falta", "size"), f=("falta", "sum")).reset_index()
print("\ndia da semana (agregado, % das consultas):")
print((g.groupby(g.dia.dt.dayofweek).ag.sum() / g.ag.sum()).round(3).to_dict())

rows = []
for c, x in g.groupby("cnesexecutante"):
    x = x.set_index("dia").sort_index()
    cal = pd.date_range(x.index.min(), x.index.max(), freq="D")
    x = x.reindex(cal).fillna(0)
    num = x.f.shift(3).rolling(28, min_periods=14).sum()
    den = x.ag.shift(3).rolling(28, min_periods=14).sum()
    pr = (num / den) * x.ag
    res = (x.f - pr).where(x.ag > 0)
    lo = res.shift(3).rolling(120, min_periods=40).quantile(0.10)
    hi = res.shift(3).rolling(120, min_periods=40).quantile(0.90)
    r = pd.DataFrame({"cnes": c, "y": x.f, "pred": pr, "lo": pr + lo, "hi": pr + hi, "ag": x.ag})
    rows.append(r.reset_index().rename(columns={"index": "dia"}))
r = pd.concat(rows, ignore_index=True)
r = r[(r.dia >= "2023-01-02") & (r.ag > 0) & r.pred.notna() & r.lo.notna()]
r["err"] = (r.y - r.pred).abs()
r["cob"] = ((r.y >= r.lo) & (r.y <= r.hi)).astype(float)
r["larg"] = r.hi - r.lo
print("\nGLOBAL diario: n=%d | WAPE=%.3f | cobertura80=%.3f | largura media=%.1f | faltas/dia media=%.1f" % (
    len(r), r.err.sum() / r.y.sum(), r.cob.mean(), r.larg.mean(), r.y.mean()))
cn = r.groupby("cnes").agg(dias=("y", "size"), faltas_dia=("y", "mean"), ag_dia=("ag", "mean"),
                           MAE=("err", "mean"), cob80=("cob", "mean"), larg80=("larg", "mean")).sort_values("faltas_dia", ascending=False)
cn["WAPE"] = cn.MAE / cn.faltas_dia
print(cn.round(2).to_string())
