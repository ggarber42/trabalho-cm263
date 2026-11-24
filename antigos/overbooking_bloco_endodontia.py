"""Overbooking por BLOCO MENSAL (mes civil), endodontia CNES 2237571. m = quantil alfa da distribuicao de faltas do bloco (Poisson-binomial exata).
Tambem avalia o efeito diario: os m extras sao distribuidos pelos dias do bloco proporcionalmente ao esperado de faltas do dia."""
import pickle, numpy as np, pandas as pd, matplotlib
from scipy.stats import poisson
matplotlib.use("Agg"); import matplotlib.pyplot as plt
CN, ESP = "2237571", "ODONTO ENDODONTIA"
R = pd.read_csv("risco_diario_coorte.csv", dtype={"cnesexecutante": str}); R["dia"] = pd.to_datetime(R.dia)
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["cnesexecutante","datahora_consulta","situacaoagenda","especialidademae"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int); raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["p"] = pickle.load(open("p_modelos.pkl","rb"))["V1"]
c = raw[(raw.cnesexecutante == CN) & raw.p.notna()]
r = R[R.cnesexecutante == CN].set_index("dia").join(c.groupby("dia").p.sum().rename("soma_p")); r["k"] = r.esperado / r.soma_p
c = c.join(r.k, on="dia"); c = c[c.k.notna() & (c.especialidademae == ESP)].copy()
c["pk"] = np.clip(c.p * c.k, 0, .995); c["mes"] = c.dia.dt.to_period("M")
B = pd.read_csv("previsao_especialidade.csv", dtype={"cnesexecutante": str}); B["dia"] = pd.to_datetime(B.dia)
B = B[(B.cnesexecutante == CN) & (B.especialidademae == ESP)].set_index("dia").base_esp
def pmf(p):
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
blocos = []
for mes, g in c.groupby("mes"):
    dias = g.groupby("dia").agg(f=("y", "sum"), e=("pk", "sum"), n=("y", "size")).reset_index()
    blocos.append({"mes": mes, "cdf": np.cumsum(pmf(g.pk.values)), "y": g.y.sum(), "esp": g.pk.sum(), "n": len(g), "mu_b": B.reindex(dias.dia).sum(), "dias": dias})
y = np.array([b["y"] for b in blocos]); mub = np.array([b["mu_b"] for b in blocos]); nb = len(blocos)
print("blocos (meses com agenda, 2023-2026):", nb, "| consultas/bloco mediana %d | faltas/bloco mediana %d (min %d, max %d) | faltas %d" %
      (np.median([b["n"] for b in blocos]), np.median(y), y.min(), y.max(), y.sum()))
def pt(m): return 100 * (y < m).mean(), 100 * np.minimum(m, y).sum() / y.sum(), m.mean()
ALF = np.unique(np.r_[np.linspace(.001, .999, 400), .05, .10, .20, .50])
v1 = np.array([pt(np.array([int(np.searchsorted(b["cdf"], a)) for b in blocos])) for a in ALF])
bs = np.array([pt(poisson.ppf(a, mub).astype(int)) for a in ALF])
ct = np.array([pt(np.full(nb, m)) for m in range(int(y.max()) + 1)])

def aloca(m, w):  # maior resto
    if m == 0: return np.zeros(len(w), int)
    q = m * w / w.sum(); a = np.floor(q).astype(int); resto = m - a.sum()
    a[np.argsort(-(q - a))[:resto]] += 1; return a
def diario(a):
    ex = pd = ap = ndias = nexc = 0
    for b in blocos:
        m = int(np.searchsorted(b["cdf"], a)); d = b["dias"]; md = aloca(m, d.e.values)
        ex += md.sum(); ap += np.minimum(md, d.f.values).sum(); nexc += (d.f.values < md).sum(); pd += np.clip(md - d.f.values, 0, None).sum(); ndias += len(d)
    return ex, ap, nexc, pd, ndias
print("\nRegra: extras do BLOCO = quantil alfa da previsao de faltas do bloco")
print("%-8s | %-44s | %-52s" % ("alfa", "nivel do bloco (mes)", "efeito DIARIO (extras distribuidos pelos dias)"))
print("%-8s | %8s %9s %9s %12s | %8s %10s %12s %12s" % ("", "extras", "sem exc.", "aproveit.", "extras/bloco", "extras", "aproveit.", "dias c/ exc.", "pac. exc."))
for a in (.05, .10, .20, .30, .50):
    m = np.array([int(np.searchsorted(b["cdf"], a)) for b in blocos]); s, ap_, e_ = pt(m); ex, ap, nexc, pdx, nd = diario(a)
    print("%-8s | %8d %8.0f%% %8.0f%% %12.1f | %8d %9.0f%% %6d (%2.0f%%) %12d" % ("q%d%%" % (100*a), m.sum(), 100 - s, ap_, e_, ex, 100*ap/y.sum(), nexc, 100*nexc/nd, pdx))
print("\nAproveitamento maximo com ate X%% de blocos com excesso:")
for lim in (5, 10, 20):
    t = {n: (arr[arr[:, 0] <= lim][:, 1].max() if (arr[:, 0] <= lim).any() else np.nan) for n, arr in (("V1", v1), ("baseline", bs), ("m_fixo", ct))}
    print("  excesso <= %2d%%: V1 %.0f%% | baseline %.0f%% | m fixo %.0f%%" % (lim, t["V1"], t["baseline"], t["m_fixo"]))
print("\n2026 (por bloco), regra q10%:")
for b in blocos:
    if b["mes"].year == 2026:
        m = int(np.searchsorted(b["cdf"], .10)); print("  %s: consultas %d | faltas %d | esperado %.1f | IC80 [%d, %d] | extras %d | %s" % (b["mes"], b["n"], b["y"], b["esp"], np.searchsorted(b["cdf"], .10), np.searchsorted(b["cdf"], .90), m, "EXCESSO de %d" % (m - b["y"]) if b["y"] < m else "sem excesso"))
cob = np.mean([(b["y"] >= np.searchsorted(b["cdf"], .10)) and (b["y"] <= np.searchsorted(b["cdf"], .90)) for b in blocos]); print("cobertura do IC80 por bloco: %.0f%% (%d blocos)" % (100*cob, nb))
INK,SEC,MUT,GRID,BG,AZ,LJ = "#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb","#2a78d6","#eb6834"
plt.rcParams.update({"font.size":10,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
a1.plot(ct[:, 0], ct[:, 1], color=MUT, lw=1.4, ls=(0, (4, 2)), marker=".", ms=5, label="m fixo (igual em todos os blocos)")
a1.plot(bs[:, 0], bs[:, 1], color=LJ, lw=1.8, label="Baseline Poisson (taxa 28d)")
a1.plot(v1[:, 0], v1[:, 1], color=AZ, lw=2.2, label="Modelo por paciente (V1)")
for a in (.05, .10, .20, .50):
    i = int(np.argmin(abs(ALF - a))); a1.scatter(*v1[i, :2], color=AZ, s=45, zorder=5, edgecolor=BG); a1.annotate("quantil %d%%" % (100*a), v1[i, :2], textcoords="offset points", xytext=(8, -12), fontsize=8.5, color=SEC)
a1.set_xlim(-1, 60); a1.set_ylim(0, 102); a1.set_xlabel("% dos blocos mensais com excesso"); a1.set_ylabel("% das faltas aproveitadas por extras")
a1.set_title("Curva excesso x aproveitamento (%d blocos mensais)" % nb, loc="left", fontsize=11.5); a1.legend(frameon=False, labelcolor=SEC, loc="lower right", fontsize=9)
meses = pd.PeriodIndex([b["mes"] for b in blocos]).to_timestamp()
q = lambda a: np.array([np.searchsorted(b["cdf"], a) for b in blocos])
a2.fill_between(meses, q(.025), q(.975), color=AZ, alpha=.14, lw=0, label="IC 95%"); a2.fill_between(meses, q(.10), q(.90), color=AZ, alpha=.28, lw=0, label="IC 80%")
a2.plot(meses, [b["esp"] for b in blocos], color=AZ, lw=1.6, label="Esperado (V1)"); a2.plot(meses, y, color=INK, lw=1.5, marker="o", ms=3.5, label="Observado")
a2.set_title("Faltas por bloco mensal", loc="left", fontsize=11.5); a2.set_ylabel("faltas / bloco"); a2.legend(frameon=False, labelcolor=SEC, fontsize=8.5, loc="upper right")
for ax in (a1, a2):
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Overbooking por bloco mensal — endodontia, CNES 2237571", x=.05, ha="left", fontsize=13); fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig("curva_overbooking_bloco_endodontia.png", dpi=130, facecolor=BG)
