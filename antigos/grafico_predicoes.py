"""Graficos das predicoes do modelo (risco_diario_coorte.csv): por CNES (ultimos 180 dias), agregado semanal e calibracao."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pd.read_csv("risco_diario_coorte.csv", parse_dates=["dia"], dtype={"cnesexecutante": str})
A = R[R.nivel == "A"]
fim = R.dia.max()
ini = fim - pd.Timedelta(days=180)
cnes = ["2237571", "8006881", "2237849", "6883206", "7214448", "2265052"]

fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
for ax, c in zip(axs.ravel(), cnes):
    x = A[(A.cnesexecutante == c) & (A.dia >= ini)].sort_values("dia")
    dentro = (x.y >= x.q10) & (x.y <= x.q90)
    ax.fill_between(x.dia, x.q10, x.q90, color="C0", alpha=.20, label="intervalo 80%")
    ax.plot(x.dia, x.esperado, color="C0", lw=1.3, label="esperado (modelo)")
    ax.scatter(x.dia[dentro], x.y[dentro], s=14, color="k", label="observado")
    ax.scatter(x.dia[~dentro], x.y[~dentro], s=22, color="C3", label="fora do intervalo")
    ax.set_title("CNES %s  |  dentro do intervalo: %.0f%%  |  %.1f faltas/dia" % (c, 100 * dentro.mean(), x.y.mean()), fontsize=10)
    ax.set_ylabel("faltas no dia")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Previsao diaria de faltas por CNES (nivel A) - ultimos 180 dias dos dados", y=.995)
fig.autofmt_xdate()
plt.tight_layout(rect=(0, .04, 1, .98))
plt.savefig("grafico_predicoes_por_cnes.png", dpi=110)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.8, 1]})
w = A.assign(semana=A.dia.dt.to_period("W-SUN").dt.start_time).groupby("semana").agg(y=("y", "sum"), e=("esperado", "sum"))
w = w.iloc[1:-1]
ax[0].plot(w.index, w.y, color="k", lw=.9, label="observado")
ax[0].plot(w.index, w.e, color="C0", lw=1.3, label="esperado (modelo)")
ax[0].set_title("Nivel A - total semanal de faltas (14 CNES)")
ax[0].set_ylabel("faltas por semana"); ax[0].legend(frameon=False)
ks = [1, 2, 3, 4, 5, 6, 8, 10]
for t, cor in (("A", "C0"), ("B", "C1")):
    x = R[R.nivel == t]
    Pk = np.concatenate([x[f"P>={k}"].values for k in ks])
    E = np.concatenate([(x.y >= k).astype(float).values for k in ks])
    b = np.clip((Pk * 10).astype(int), 0, 9)
    px = [Pk[b == i].mean() for i in range(10) if (b == i).sum() > 30]
    py = [E[b == i].mean() for i in range(10) if (b == i).sum() > 30]
    ax[1].plot(px, py, "o-", color=cor, label="nivel " + t)
ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
ax[1].set_xlabel("probabilidade prevista de >= k faltas"); ax[1].set_ylabel("frequencia observada")
ax[1].set_title("Calibracao (k = 1,2,3,4,5,6,8,10)"); ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("grafico_predicoes_agregado_calibracao.png", dpi=110)
print("ok", fim.date())
