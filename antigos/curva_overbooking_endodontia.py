"""Curva excesso x aproveitamento, endodontia CNES 2237571.
Politica: chamar m pacientes extras = quantil alfa da distribuicao prevista de faltas do periodo (alfa pequeno = conservador).
Comparadores: baseline Poisson (taxa 28d da especialidade) com a mesma regra de quantil; m constante (igual em todos os periodos).
Eixo x: % de periodos COM excesso (faltas reais < m). Eixo y: % das faltas reais aproveitadas = soma(min(m, faltas)) / soma(faltas)."""
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
c["pk"] = np.clip(c.p * c.k, 0, .995)
c["sem"] = c.dia.dt.to_period("W-SUN").dt.end_time.dt.normalize()
B = pd.read_csv("previsao_especialidade.csv", dtype={"cnesexecutante": str}); B["dia"] = pd.to_datetime(B.dia)
B = B[(B.cnesexecutante == CN) & (B.especialidademae == ESP)].set_index("dia").base_esp
c["base_dia"] = c.dia.map(B)
def pmf(p):
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
ALF = np.unique(np.r_[np.linspace(0.001, 0.999, 400), .05, .10, .20, .50])

def periodos(chave):
    y, cdfs, mu_b = [], [], []
    for _, g in c.groupby(chave):
        y.append(g.y.sum()); cdfs.append(np.cumsum(pmf(g.pk.values)))
        mu_b.append(g.drop_duplicates("dia").base_dia.sum() if chave == "sem" else g.base_dia.iloc[0])
    return np.array(y), cdfs, np.array(mu_b)
def pt(y, m): return 100 * (y < m).mean(), 100 * np.minimum(m, y).sum() / y.sum(), m.mean()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
INK,SEC,MUT,GRID,BG,AZ,LJ = "#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb","#2a78d6","#eb6834"
plt.rcParams.update({"font.size":10,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
saida = []
for ax, (chave, nome) in zip(axs, (("dia", "Por dia"), ("sem", "Por semana"))):
    y, cdfs, mub = periodos(chave)
    v1 = np.array([[pt(y, np.array([int(np.searchsorted(cd, a)) for cd in cdfs]))] for a in ALF])[:, 0, :]
    bs = np.array([pt(y, poisson.ppf(a, mub).astype(int)) for a in ALF])
    ct = np.array([pt(y, np.full(len(y), m)) for m in range(0, int(y.max()) + 1)])
    for nm, arr in (("V1", v1), ("baseline", bs), ("m_constante", ct)):
        saida.append(pd.DataFrame({"periodo": nome, "politica": nm, "pct_com_excesso": arr[:, 0], "pct_faltas_aproveitadas": arr[:, 1], "extras_medios": arr[:, 2]}))
    ax.plot(ct[:, 0], ct[:, 1], color=MUT, lw=1.4, ls=(0, (4, 2)), marker=".", ms=5, label="m fixo (igual em todos os periodos)")
    ax.plot(bs[:, 0], bs[:, 1], color=LJ, lw=1.8, label="Baseline Poisson (taxa 28d), regra do quantil")
    ax.plot(v1[:, 0], v1[:, 1], color=AZ, lw=2.2, label="Modelo por paciente (V1), regra do quantil")
    for a in (.05, .10, .20, .50):
        i = int(np.argmin(abs(ALF - a))); ax.scatter(*v1[i, :2], color=AZ, s=45, zorder=5, edgecolor=BG)
        ax.annotate("quantil %d%%" % (100*a), v1[i, :2], textcoords="offset points", xytext=(8, -12), fontsize=8.5, color=SEC)
    ax.axvline(10, color=GRID, lw=1); ax.text(10.3, 3, "10% dos periodos com excesso", fontsize=8.5, color=MUT)
    ax.set_xlim(-1, 60); ax.set_ylim(0, 102)
    ax.set_xlabel("% dos periodos com excesso de pacientes (faltas reais < extras chamados)")
    ax.set_ylabel("% das faltas aproveitadas por extras")
    ax.set_title("%s (%d periodos, %d faltas)" % (nome, len(y), y.sum()), loc="left", fontsize=11.5)
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if nome == "Por dia": ax.legend(frameon=False, labelcolor=SEC, fontsize=9, loc="lower right")
fig.suptitle("Overbooking na endodontia (CNES 2237571): mais perto do canto superior esquerdo = melhor", x=.06, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig("curva_overbooking_endodontia.png", dpi=130, facecolor=BG)
S = pd.concat(saida); S.round(3).to_csv("curva_overbooking_endodontia.csv", index=False)
# comparacao em pontos de operacao: aproveitamento alcancavel com <=5% e <=10% de periodos com excesso
for nome in ("Por dia", "Por semana"):
    for lim in (5, 10, 20):
        t = {}
        for pol in ("V1", "baseline", "m_constante"):
            d = S[(S.periodo == nome) & (S.politica == pol) & (S.pct_com_excesso <= lim)]
            t[pol] = d.pct_faltas_aproveitadas.max() if len(d) else np.nan
        print("%-10s excesso <= %2d%%: aproveitamento V1 %.0f%% | baseline %.0f%% | m fixo %.0f%%" % (nome, lim, t["V1"], t["baseline"], t["m_constante"]))
