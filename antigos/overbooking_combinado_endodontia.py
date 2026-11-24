"""Combinacao bloco + dia, endodontia CNES 2237571.
 - so_dia(ad):      extras do dia = quantil ad da previsao do dia
 - so_bloco(ab):    extras do mes = quantil ab do bloco, distribuidos pelos dias na proporcao do esperado
 - combinada(ab,ad): teto do mes = quantil ab do bloco; em cada dia (ordem cronologica) extras = min(quantil ad do dia, teto restante)
Metricas no nivel do DIA (o que o paciente sente): dias com excesso, pacientes em excesso, % de faltas aproveitadas, extras chamados."""
import pickle, numpy as np, pandas as pd, matplotlib
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
def pmf(p):
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
BL = []
for mes, g in c.groupby("mes"):
    dias = [(d, h.y.sum(), h.pk.sum(), np.cumsum(pmf(h.pk.values))) for d, h in g.groupby("dia")]
    BL.append({"mes": mes, "cdf": np.cumsum(pmf(g.pk.values)), "dias": dias})
NDIAS = sum(len(b["dias"]) for b in BL); FALTAS = sum(d[1] for b in BL for d in b["dias"])
q = lambda cdf, a: int(np.searchsorted(cdf, a))
def aloca(m, w):
    if m == 0: return np.zeros(len(w), int)
    t = m * w / w.sum(); a = np.floor(t).astype(int); a[np.argsort(-(t - a))[:m - a.sum()]] += 1; return a
def roda(pol, ab=None, ad=None, so_ano=None):
    ex = ap = nexc = pexc = nd = 0
    for b in BL:
        if so_ano and b["mes"].year != so_ano: continue
        f = np.array([d[1] for d in b["dias"]]); e = np.array([d[2] for d in b["dias"]])
        if pol == "so_dia": md = np.array([q(d[3], ad) for d in b["dias"]])
        elif pol == "so_bloco": md = aloca(q(b["cdf"], ab), e)
        else:
            rem = q(b["cdf"], ab); md = np.zeros(len(f), int)
            for i, d in enumerate(b["dias"]): md[i] = min(q(d[3], ad), rem); rem -= md[i]
        ex += md.sum(); ap += np.minimum(md, f).sum(); nexc += (f < md).sum(); pexc += np.clip(md - f, 0, None).sum(); nd += len(f)
    F = sum(d[1] for b in BL for d in b["dias"] if not so_ano or b["mes"].year == so_ano)
    return {"extras": ex, "aprov_pct": 100 * ap / F, "dias_exc_pct": 100 * nexc / nd, "dias_exc": nexc, "pac_exc": pexc, "dias": nd, "faltas": F}
AD = np.unique(np.r_[np.linspace(.01, .99, 60), .05, .10, .20, .30, .50, .70]); AB = (.05, .10, .20, .30, .50)
curvas = {"Só dia": pd.DataFrame([roda("so_dia", ad=a) for a in AD]),
          "Só bloco (distribuído)": pd.DataFrame([roda("so_bloco", ab=a) for a in np.linspace(.01, .99, 60)])}
for ab in AB: curvas["Combinada, teto bloco q%d%%" % (100 * ab)] = pd.DataFrame([roda("comb", ab=ab, ad=a) for a in AD])
print("dias com agenda: %d | faltas %d (2023-2026)\n" % (NDIAS, FALTAS))
print("%-30s %s" % ("politica", " | ".join("max aproveit. com <=%d%% dias c/ excesso" % l for l in (5, 10, 20))))
for k, d in curvas.items():
    print("%-30s %s" % (k, " | ".join("%38.0f%%" % (d[d.dias_exc_pct <= l].aprov_pct.max() if (d.dias_exc_pct <= l).any() else np.nan) for l in (5, 10, 20))))
print("\nPontos de operacao (2023-2026):")
print("%-38s %7s %9s %12s %9s" % ("politica", "extras", "aproveit.", "dias c/ exc.", "pac. exc."))
pts = [("so_dia q10%", ("so_dia", None, .10)), ("so_dia q30%", ("so_dia", None, .30)), ("so_bloco q10% (distribuido)", ("so_bloco", .10, None)),
       ("combinada: teto q10%, dia q30%", ("comb", .10, .30)), ("combinada: teto q10%, dia q50%", ("comb", .10, .50)),
       ("combinada: teto q20%, dia q30%", ("comb", .20, .30)), ("combinada: teto q30%, dia q50%", ("comb", .30, .50))]
for nm, (p, ab, ad) in pts:
    s = roda(p, ab=ab, ad=ad); print("%-38s %7d %8.0f%% %7d (%2.0f%%) %9d" % (nm, s["extras"], s["aprov_pct"], s["dias_exc"], s["dias_exc_pct"], s["pac_exc"]))
print("\n2026 (17 dias de agenda, 3 blocos):")
for nm, (p, ab, ad) in pts:
    s = roda(p, ab=ab, ad=ad, so_ano=2026); print("%-38s extras %3d | aproveit. %3.0f%% | dias c/ excesso %d de %d | pac. em excesso %d" % (nm, s["extras"], s["aprov_pct"], s["dias_exc"], s["dias"], s["pac_exc"]))
INK,SEC,MUT,GRID,BG,AZ,LJ,VD = "#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb","#2a78d6","#eb6834","#1baf7a"
plt.rcParams.update({"font.size":10,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
fig, ax = plt.subplots(figsize=(11, 6.8))
est = {"Só dia": (AZ, 2.4, "-"), "Só bloco (distribuído)": (LJ, 1.8, "-"), "Combinada, teto bloco q10%": (VD, 2.0, "-"), "Combinada, teto bloco q30%": (VD, 1.4, (0, (4, 2)))}
for k, (cor, lw, ls) in est.items():
    d = curvas[k].sort_values("dias_exc_pct"); ax.plot(d.dias_exc_pct, d.aprov_pct, color=cor, lw=lw, ls=ls, label=k)
ax.axvline(10, color=GRID, lw=1); ax.set_xlim(-1, 45); ax.set_ylim(0, 102)
ax.set_xlabel("% dos DIAS com excesso de pacientes (faltas do dia < extras chamados)"); ax.set_ylabel("% das faltas aproveitadas por extras")
ax.set_title("Overbooking na endodontia (CNES 2237571): combinar bloco mensal e regra diária\nmais perto do canto superior esquerdo = melhor", loc="left", fontsize=11.5)
ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True); ax.legend(frameon=False, labelcolor=SEC, loc="lower right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("curva_overbooking_combinado_endodontia.png", dpi=130, facecolor=BG)
