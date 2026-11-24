"""Top 20 CNES executantes por total de faltas + diagnostico da quebra de 2020."""
import pandas as pd

p = pd.read_parquet("painel_faltas_semanal.parquet")

rank = (p.groupby("cnesexecutante")
          .agg(total_faltas=("n_faltas", "sum"), total_agendadas=("n_agendadas", "sum"),
               sem_ini=("semana", "min"), sem_fim=("semana", "max"), n_semanas=("semana", "size"))
          .sort_values("total_faltas", ascending=False))
rank["taxa_falta"] = rank.total_faltas / rank.total_agendadas
rank["pct_acum"] = rank.total_faltas.cumsum() / rank.total_faltas.sum()
top = rank.head(20)
print(top.round(3).to_string())
print("\nTop 20 concentram %.1f%% das faltas" % (100 * top.total_faltas.sum() / rank.total_faltas.sum()))

p20 = p[p.cnesexecutante.isin(top.index)].copy()
# p20.to_parquet("painel_top20.parquet", index=False)
# top.to_csv("ranking_top20.csv")

# --- diagnostico 2020 ---
pre = p20[(p20.semana >= "2019-01-01") & (p20.semana < "2020-03-16")]
pos = p20[(p20.semana >= "2020-03-16") & (p20.semana < "2021-01-01")]
tail = p20[p20.semana >= "2020-03-16"]
agg = p20.groupby("semana")[["n_faltas", "n_agendadas"]].sum()
agg["taxa"] = agg.n_faltas / agg.n_agendadas
print("\nAgregado top20 (faltas/semana): 2019-01..2020-03-15 media %.0f | 2020-03-16..2020-12 media %.0f | 2021+ media %.0f" % (
    agg.loc["2019-01-01":"2020-03-15", "n_faltas"].mean(),
    agg.loc["2020-03-16":"2020-12-31", "n_faltas"].mean(),
    agg.loc["2021-01-01":, "n_faltas"].mean()))
print("Faltas/semana por ano (agregado top20):")
print(agg.n_faltas.groupby(agg.index.year).agg(["mean", "min", "max", "count"]).round(0).to_string())

# figura 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib ausente: grafico nao gerado")
else:
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(agg.index, agg.n_faltas, lw=.8)
    ax[0].axvline(pd.Timestamp("2020-03-16"), color="r", ls="--", label="2020-03-16")
    ax[0].set_ylabel("faltas/semana"); ax[0].legend(); ax[0].set_title("Top 20 CNES - agregado")
    ax[1].plot(agg.index, agg.taxa, lw=.8, color="C1")
    ax[1].axvline(pd.Timestamp("2020-03-16"), color="r", ls="--")
    ax[1].set_ylabel("taxa de falta")
    plt.tight_layout(); plt.savefig("top20_agregado.png", dpi=110)

