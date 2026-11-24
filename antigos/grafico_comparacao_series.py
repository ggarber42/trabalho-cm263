"""Grafico comparativo: observado x modelos (soma A+B, 2024+) e WAPE por modelo. Le previsoes_series.parquet."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pd.read_parquet("previsoes_series.parquet")
D = D[D.ano >= 2024]
mods = [c[3:] for c in D.columns if c.startswith("mu_")]
D = D[D[["mu_" + m for m in mods]].notna().all(axis=1)]
INK, SEC, MUT, GRID, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COR = {"V1": "#2a78d6", "hgb_poisson": "#eb6834", "hib_glm": "#1baf7a", "baseline": "#898781"}
NOME = {"V1": "Modelo por paciente (V1)", "hgb_poisson": "HGB Poisson (lags)", "hib_glm": "Híbrido (GLM + paciente)", "baseline": "Baseline (taxa 28d × agenda)"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": SEC, "xtick.color": MUT, "ytick.color": MUT,
                     "axes.facecolor": BG, "figure.facecolor": BG, "text.color": INK})

dia = D.groupby("dia")[["y"] + ["mu_" + m for m in mods]].sum()
sem = dia.resample("W").sum()
sem = sem.iloc[1:-1]  # descarta semanas incompletas nas pontas
fim = dia.index.max()
zoom = dia.loc[fim - pd.Timedelta(days=83):fim]

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.0], hspace=0.42)

def linhas(ax, d, lw_obs, marker=None):
    ax.plot(d.index, d.y, color=INK, lw=lw_obs, label="Observado", marker=marker, ms=3.5, zorder=5)
    ax.plot(d.index, d.mu_baseline, color=COR["baseline"], lw=1.4, ls=(0, (4, 2)), label=NOME["baseline"], zorder=2)
    for m in ("hgb_poisson", "hib_glm", "V1"):
        ax.plot(d.index, d["mu_" + m], color=COR[m], lw=1.8, label=NOME[m], zorder=3)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

a1 = fig.add_subplot(gs[0])
linhas(a1, sem, 2.2)
a1.set_title("Faltas por semana — soma das 20 CNES (coorte A+B), 2024 a mar/2026", loc="left", fontsize=12, color=INK)
a1.set_ylabel("faltas / semana")
a1.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, labelcolor=SEC)

a2 = fig.add_subplot(gs[1])
linhas(a2, zoom, 1.6, marker="o")
a2.set_title("Últimas 12 semanas, por dia (soma das CNES)", loc="left", fontsize=12, color=INK)
a2.set_ylabel("faltas / dia")

# WAPE por modelo
def wape(m): return np.abs(D.y - D["mu_" + m]).sum() / D.y.sum()
W = pd.Series({m: wape(m) for m in mods}).sort_values()
ROT = {"baseline": "baseline", "baseline_nb": "baseline (NB)", "naive7": "naive 7 dias", "dow4": "média 4 mesmos dias",
       "ets": "ETS", "sarima": "SARIMA", "glm_nb": "GLM por CNES", "hgb_poisson": "HGB Poisson", "ctrl_glm": "GLM pooled (controle)",
       "hib_glm": "Híbrido", "V0": "Paciente V0", "V1": "Paciente V1"}
a3 = fig.add_subplot(gs[2])
cores = [COR.get(m, "#b7b5ad") for m in W.index]
b = a3.barh([ROT.get(m, m) for m in W.index], W.values, color=cores, height=0.62)
a3.invert_yaxis()
for r, v in zip(b, W.values):
    a3.text(v + 0.003, r.get_y() + r.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color=SEC)
a3.set_xlim(0, W.max() * 1.12)
a3.set_title("Erro por modelo (WAPE, menor = melhor) — 10.480 CNES-dias, 2024+", loc="left", fontsize=12, color=INK)
a3.grid(axis="x", color=GRID, lw=0.7); a3.set_axisbelow(True)
for s in ("top", "right"): a3.spines[s].set_visible(False)
fig.savefig("grafico_comparacao_series.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(W.round(3).to_string()); print(len(D), "linhas;", len(sem), "semanas")
