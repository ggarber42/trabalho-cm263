"""Baselines de previsao de faltas por CNES, 1 semana a frente, validacao em janela rolante.

A: media de faltas das ultimas 8 semanas
B: taxa das ultimas 8 semanas (faltas/agendadas) x agendadas da semana prevista
   (assume que a agenda da semana alvo ja e conhecida na hora da decisao)
Intervalo 80% empirico: quantis 10/90 dos residuos das ultimas 52 semanas do proprio CNES.
Teste: semanas a partir de 2023-01-02 (inclui o evento de 2025 de proposito).
"""
import numpy as np
import pandas as pd

p = pd.read_parquet("painel_final.parquet").sort_values(["cnesexecutante", "semana"])
TESTE = pd.Timestamp("2023-01-02")
rows = []
for c, x in p.groupby("cnesexecutante"):
    x = x.set_index("semana")
    f, a = x["n_faltas"].astype(float), x["n_agendadas"].astype(float)
    pa = f.shift(1).rolling(8, min_periods=6).mean()
    taxa = f.shift(1).rolling(8, min_periods=6).sum() / a.shift(1).rolling(8, min_periods=6).sum()
    pb = taxa * a
    for nome, pr in (("A_media8", pa), ("B_taxa8xagenda", pb)):
        r = (f - pr)
        lo = r.shift(1).rolling(52, min_periods=26).quantile(0.10)
        hi = r.shift(1).rolling(52, min_periods=26).quantile(0.90)
        d = pd.DataFrame({"cnes": c, "metodo": nome, "y": f, "pred": pr, "lo": pr + lo, "hi": pr + hi, "ag": a})
        rows.append(d.reset_index())
r = pd.concat(rows, ignore_index=True)
r = r[(r.semana >= TESTE) & r.pred.notna() & r.lo.notna() & (r.ag > 0)]
r["err"] = (r.y - r.pred).abs()
r["cob"] = ((r.y >= r.lo) & (r.y <= r.hi)).astype(float)
r["larg"] = r.hi - r.lo

glob = r.groupby("metodo").agg(n=("y", "size"), WAPE=("err", lambda s: s.sum() / r.loc[s.index, "y"].sum()),
                               cobertura80=("cob", "mean"), largura_media=("larg", "mean"))
print("GLOBAL (todas as CNES-semana de teste)")
print(glob.round(3).to_string())

b = r[r.metodo == "B_taxa8xagenda"]
cn = b.groupby("cnes").agg(media_faltas=("y", "mean"), MAE=("err", "mean"),
                           WAPE=("err", lambda s: s.sum() / b.loc[s.index, "y"].sum()),
                           cob80=("cob", "mean"), larg80=("larg", "mean")).sort_values("media_faltas", ascending=False)
cn["ruido_poisson_pct"] = 100 / np.sqrt(cn.media_faltas)   # erro relativo minimo ~ 1/sqrt(media)
print("\nPOR CNES (metodo B)")
print(cn.round(2).to_string())

# soma dos 20 CNES na semana (agregado) com o metodo B
ag = b.groupby("semana").agg(y=("y", "sum"), pred=("pred", "sum"))
print("\nAgregado 20 CNES, metodo B: WAPE = %.3f" % ((ag.y - ag.pred).abs().sum() / ag.y.sum()))
