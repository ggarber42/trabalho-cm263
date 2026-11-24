"""Endodontia, CNES 2237571: faltas semanais observadas x previsao do V1 com intervalos de previsao 80% e 95%.
Distribuicao semanal exata = Poisson-binomial das consultas da semana (p_i = p do modelo x k do CNES-dia, como no script 11)."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
CN, ESP = "2237571", "ODONTO ENDODONTIA"
R = pd.read_csv("risco_diario_coorte.csv", dtype={"cnesexecutante": str}); R["dia"] = pd.to_datetime(R.dia)
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["cnesexecutante","datahora_consulta","situacaoagenda","especialidademae"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int); raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["p"] = pickle.load(open("p_modelos.pkl","rb"))["V1"]
c = raw[(raw.cnesexecutante == CN) & raw.p.notna()]
soma = c.groupby("dia").p.sum().rename("soma_p")
r = R[R.cnesexecutante == CN].set_index("dia").join(soma); r["k"] = r.esperado / r.soma_p
c = c.join(r.k, on="dia"); c = c[c.k.notna() & (c.especialidademae == ESP)].copy()
c["pk"] = np.clip(c.p * c.k, 0, .995); c["sem"] = c.dia.dt.to_period("W-SUN").dt.end_time.dt.normalize()

def pb_pmf(p):  # Poisson-binomial exata
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
rows = []
for s, g in c.groupby("sem"):
    f = pb_pmf(g.pk.values); cdf = np.cumsum(f); q = lambda a: int(np.searchsorted(cdf, a))
    rows.append({"sem": s, "y": g.y.sum(), "esp": g.pk.sum(), "q025": q(.025), "q10": q(.10), "q90": q(.90), "q975": q(.975), "n": len(g)})
W = pd.DataFrame(rows).set_index("sem")
cob80 = ((W.y >= W.q10) & (W.y <= W.q90)).mean(); cob95 = ((W.y >= W.q025) & (W.y <= W.q975)).mean()
print("semanas:", len(W), "| faltas", int(W.y.sum()), "| esperado %.0f" % W.esp.sum())
print("cobertura IC80: %.1f%% (alvo 80) | IC95: %.1f%% (alvo 95)" % (100*cob80, 100*cob95))
print("largura media IC80: %.1f | IC95: %.1f faltas" % ((W.q90-W.q10).mean(), (W.q975-W.q025).mean()))
for a, b, t in ((2023,2023,"2023"),(2024,2026,"2024-26")):
    m = (W.index.year >= a) & (W.index.year <= b)
    print("  %s: n=%d  cob80 %.0f%%  cob95 %.0f%%" % (t, m.sum(), 100*((W.y[m]>=W.q10[m])&(W.y[m]<=W.q90[m])).mean(), 100*((W.y[m]>=W.q025[m])&(W.y[m]<=W.q975[m])).mean()))
fora = W[(W.y < W.q025) | (W.y > W.q975)]
print("semanas fora do IC95:", len(fora)); W.round(2).to_csv("endodontia_2237571_semanal_ic.csv")

INK,SEC,MUT,GRID,BG,AZ = "#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb","#2a78d6"
plt.rcParams.update({"font.size":10,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
fig, ax = plt.subplots(figsize=(13, 6))
ax.fill_between(W.index, W.q025, W.q975, color=AZ, alpha=.14, lw=0, label="Intervalo de previsão 95%")
ax.fill_between(W.index, W.q10, W.q90, color=AZ, alpha=.28, lw=0, label="Intervalo de previsão 80%")
ax.plot(W.index, W.esp, color=AZ, lw=1.8, label="Esperado (V1)")
ax.plot(W.index, W.y, color=INK, lw=1.6, marker="o", ms=3.5, label="Observado", zorder=5)
fo = W[(W.y < W.q025) | (W.y > W.q975)]
ax.scatter(fo.index, fo.y, s=60, facecolors="none", edgecolors="#e34948", lw=1.6, zorder=6, label="Fora do IC 95%")
ax.set_ylabel("faltas / semana"); ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("Odonto endodontia, CNES 2237571 — faltas por semana com intervalos de previsão do V1\n"
             "cobertura observada: IC80 = %.0f%% (alvo 80%%) · IC95 = %.0f%% (alvo 95%%)" % (100*cob80, 100*cob95), loc="left", fontsize=11.5)
ax.legend(frameon=False, labelcolor=SEC, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.07), fontsize=9)
fig.tight_layout(); fig.savefig("grafico_endodontia_semanal_ic.png", dpi=130, facecolor=BG)
