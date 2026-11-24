"""Modelo final por paciente -> distribuicao de faltas por CNES-dia, para a coorte 2019-2026.

Niveis (de coorte_2019_2026.csv): A = estaveis (14, principal), B = mudanca moderada (6, secundario), C = instaveis (fora).
V0: modelo global treinado com todas as linhas (< 1/jan/Y).
V1: modelo global 'limpo': sem as linhas dos CNES nivel C e, nos CNES nivel B, sem as linhas anteriores ao inicio do
    ultimo regime detectado SO com dados anteriores ao corte (sem olhar o futuro).
Treino expansivo anual (Y = 2023..2026). Probabilidades recalibradas pelo nivel dos ultimos 28 dias do CNES (lag 3);
distribuicao do total diario por simulacao (Poisson-binomial). Baseline = Poisson(taxa 28d x agenda).
Saidas: risco_diario_coorte.csv (todas as CNES-dias), risco_ultimo_dia.csv, p_modelos.pkl.
"""
import time, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
t0_ = time.time()
SRC = "gercon_poa_censo_distancias.parquet"
YEARS = (2023, 2024, 2025, 2026)
NS = 300
KS = (1, 2, 3, 5, 8)

CO = pd.read_csv("coorte_2019_2026.csv", dtype={"cnes": str}).set_index("cnes")
A = list(CO.index[CO.estavel])
B = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]
C = ["7513151", "2262630", "2264072", "2262355", "2262819", "2262401"]
SET = A + B
tier = {**{c: "A" for c in A}, **{c: "B" for c in B}}
print("nivel A (%d): %s\nnivel B (%d): %s\nnivel C fora (%d): %s" % (len(A), A, len(B), B, len(C), C))

raw = pd.read_parquet(SRC)
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["taxa_falta_prev"] = raw.n_faltas_previas / raw.n_consultas_previas.where(raw.n_consultas_previas > 0)
NUM = ["idade", "n_consultas_previas", "n_faltas_previas", "taxa_falta_prev", "primeira_vez",
       "n_consultas_previas_especialidademae", "n_faltas_previas_especialidademae",
       "n_consultas_previas_motivo", "n_faltas_previas_motivo", "dias_ate_consulta",
       "favela_comunidade", "taxa_alf_geral", "renda_media_winsor", "distancia_km", "tempo_caminhando",
       "distancia_km_trans_publico", "tempo_desloc_trans_publico", "quant_bus", "baldiacoes"]
CAT = ["falta_recente", "ultima_previa_mesma_especialidademae", "ultima_previa_mesmo_motivo",
       "especialidademae", "cid_1dig", "possuiunidadeindicada", "dia_semana", "mes_consulta",
       "corregulador", "coordenadoria_saude", "distrito_sanitario", "status", "cnesexecutante"]
X = raw[NUM + CAT].copy()
for c_ in CAT:
    X[c_] = X[c_].astype("category")
y = raw.y.values
dia = raw.dia.values
cnes_arr = raw.cnesexecutante.values
is_set = np.isin(cnes_arr, SET)

# ---------- deteccao de regime (so dados < corte) ----------
def ll(F, N):
    p = np.clip(np.where(N > 0, F / np.maximum(N, 1), 0), 1e-9, 1 - 1e-9)
    return F * np.log(p) + (N - F) * np.log(1 - p)
def segmenta(f, n, lo, hi, out, min_len=6):
    if hi - lo < 2 * min_len:
        return
    cf, cn = np.r_[0, np.cumsum(f)], np.r_[0, np.cumsum(n)]
    F, N = cf[hi] - cf[lo], cn[hi] - cn[lo]
    base = ll(np.array(F), np.array(N))
    best = (0, None, 0, 0)
    for t in range(lo + min_len, hi - min_len + 1):
        F1, N1, F2, N2 = cf[t] - cf[lo], cn[t] - cn[lo], cf[hi] - cf[t], cn[hi] - cn[t]
        if N1 == 0 or N2 == 0:
            continue
        g = 2 * (ll(np.array(F1), np.array(N1)) + ll(np.array(F2), np.array(N2)) - base)
        if g > best[0]:
            best = (g, t, F1 / N1, F2 / N2)
    if best[1] is not None and best[0] > 100 and abs(best[2] - best[3]) >= 0.05:
        out.append(best[1])
        segmenta(f, n, lo, best[1], out, min_len)
        segmenta(f, n, best[1], hi, out, min_len)
Mb = (raw[raw.cnesexecutante.isin(B) & (raw.dia >= "2019-01-01")].assign(mes=lambda d: d.dia.dt.to_period("M"))
      .groupby(["cnesexecutante", "mes"]).agg(f=("y", "sum"), n=("y", "size")))
def regime_inicio(c, corte):
    meses = pd.period_range("2019-01", (corte - pd.Timedelta(days=1)).strftime("%Y-%m"), freq="M")
    x = Mb.loc[c].reindex(meses, fill_value=0)
    cuts = []
    segmenta(x.f.values, x.n.values, 0, len(meses), cuts)
    return meses[max(cuts)].to_timestamp() if cuts else meses[0].to_timestamp()

def ajusta(tr):
    n = tr.sum()
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=int(np.clip(n / 300, 20, 200)),
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        categorical_features="from_dtype", random_state=0)
    return m.fit(X[tr], y[tr])

P = {"V0": np.full(len(raw), np.nan), "V1": np.full(len(raw), np.nan)}
for Y in YEARS:
    corte = pd.Timestamp(f"{Y}-01-01")
    antes = dia < np.datetime64(corte)
    te = is_set & (dia >= np.datetime64(corte)) & (dia < np.datetime64(f"{Y + 1}-01-01"))
    tr1 = antes & ~np.isin(cnes_arr, C)
    ini = {c: regime_inicio(c, corte) for c in B}
    for c in B:
        tr1 &= ~((cnes_arr == c) & (dia < np.datetime64(ini[c])))
    print(f"{Y}: regimes B (inicio do ultimo regime conhecido): " + ", ".join(f"{c}:{ini[c].strftime('%y-%m')}" for c in B))
    print(f"   treino V0 {antes.sum()} | V1 {tr1.sum()} | teste {te.sum()}", flush=True)
    P["V0"][te] = ajusta(antes).predict_proba(X[te])[:, 1]
    P["V1"][te] = ajusta(tr1).predict_proba(X[te])[:, 1]
    print(f"   ok {time.time() - t0_:.0f}s", flush=True)
with open("p_modelos.pkl", "wb") as fh:
    pickle.dump(P, fh)

# ---------- esqueleto CNES-dia ----------
tm = is_set & (dia >= np.datetime64("2023-01-01"))
S = raw.loc[tm, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gk = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gk.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
Dg = gk.agg(y=("y", "sum"), ag=("y", "size")).reset_index()
NG = len(Dg)
Dg["nivel"] = Dg.cnesexecutante.map(tier)

Bd = raw[is_set].groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size")).reset_index()
bl = []
for c_, x in Bd.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")).fillna(0)
    num = x.f.shift(3).rolling(28, min_periods=14).sum()
    den = x.ag.shift(3).rolling(28, min_periods=14).sum()
    bl.append(pd.DataFrame({"cnesexecutante": c_, "pred_base": (num / den) * x.ag}).rename_axis("dia").reset_index())
Dg = Dg.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")

def fator_k(pred):
    out = np.full(NG, np.nan)
    for c_, ix in Dg.groupby("cnesexecutante").indices.items():
        x = pd.DataFrame({"y": Dg.y.values[ix], "pred": pred[ix]}, index=Dg.dia.values[ix])
        cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
        num = cal.y.shift(3).rolling(28, min_periods=14).sum()
        den = cal.pred.shift(3).rolling(28, min_periods=14).sum()
        out[ix] = (num / den.where(den > 0)).reindex(x.index).values
    return out

res = {}
for v in ("V0", "V1"):
    p = P[v][ordr]
    pred = np.add.reduceat(p, starts)
    res[v] = {"p": p, "pred": pred, "k": fator_k(pred)}
valid = Dg.pred_base.notna().values.copy()
for r in res.values():
    valid &= np.isfinite(r["k"])
yv = Dg.y.values[valid]
nivel_v = Dg.nivel.values[valid]
print("\nCNES-dias avaliados: %d (A: %d | B: %d)" % (valid.sum(), (nivel_v == "A").sum(), (nivel_v == "B").sum()))

rng = np.random.default_rng(1)
def sim_modelo(p, k):
    pk = np.clip(p * np.where(valid, k, 0)[grp], 0, 0.995).astype(np.float32)
    out = np.empty((NG, NS), dtype=np.int16)
    for a in range(0, NG, 1500):
        b = min(a + 1500, NG)
        r0, r1 = starts[a], (starts[b] if b < NG else len(pk))
        u = rng.random((r1 - r0, NS), dtype=np.float32)
        out[a:b] = np.add.reduceat((u < pk[r0:r1, None]).astype(np.int16), starts[a:b] - r0, axis=0)
    return out[valid]

def metricas(sims, mean, yy):
    q = lambda a: np.quantile(sims, a, axis=1)
    s = np.sort(sims, axis=1).astype(np.float64)
    i = np.arange(1, NS + 1)
    crps = np.abs(sims - yy[:, None]).mean(1) - 0.5 * (2 / NS ** 2) * ((2 * i - NS - 1) * s).sum(1)
    Pk = np.concatenate([(sims >= k).mean(1) for k in KS])
    E = np.concatenate([(yy >= k).astype(float) for k in KS])
    bins = np.clip((Pk * 10).astype(int), 0, 9)
    ece = sum(abs(Pk[bins == b].mean() - E[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(Pk)
    return {"WAPE": np.abs(yy - mean).sum() / yy.sum(), "vies": mean.sum() / yy.sum(), "CRPS": crps.mean(),
            "cob80": ((yy >= q(.1)) & (yy <= q(.9))).mean(), "cob50": ((yy >= q(.25)) & (yy <= q(.75))).mean(),
            "larg80": (q(.9) - q(.1)).mean(), "ECE": ece}

SIMS = {"baseline": rng.poisson(Dg.pred_base.values[valid][:, None], (valid.sum(), NS)).astype(np.int16)}
MEAN = {"baseline": Dg.pred_base.values[valid]}
for v in ("V0", "V1"):
    SIMS[v] = sim_modelo(res[v]["p"], res[v]["k"])
    MEAN[v] = (res[v]["pred"] * res[v]["k"])[valid]

rows = {}
for t in ("A", "B", "A+B"):
    m = np.ones(len(yv), bool) if t == "A+B" else (nivel_v == t)
    for v in ("baseline", "V0", "V1"):
        rows[(t, v)] = metricas(SIMS[v][m], MEAN[v][m], yv[m])
T = pd.DataFrame(rows).T
T.index.names = ["nivel", "modelo"]
print("\nRESULTADO (alvos de cobertura: cob80=0.80, cob50=0.50)")
print(T.round(3).to_string())

print("\nCALIBRACAO DE P(faltas >= k), k em %s, modelo V1 (prob. prevista -> frequencia observada)" % (KS,))
for t in ("A", "B"):
    m = nivel_v == t
    Pk = np.concatenate([(SIMS["V1"][m] >= k).mean(1) for k in KS])
    E = np.concatenate([(yv[m] >= k).astype(float) for k in KS])
    bins = np.clip((Pk * 10).astype(int), 0, 9)
    tb = pd.DataFrame({"faixa": [f"{b * 10}-{b * 10 + 10}%" for b in range(10)],
                       "n": [(bins == b).sum() for b in range(10)],
                       "prev_media": [Pk[bins == b].mean() if (bins == b).any() else np.nan for b in range(10)],
                       "observado": [E[bins == b].mean() if (bins == b).any() else np.nan for b in range(10)]})
    print("nivel", t); print(tb.round(3).to_string(index=False))

# ---------- tabela de risco ----------
out = Dg[valid][["cnesexecutante", "dia", "nivel", "y", "ag"]].copy()
sm = SIMS["V1"]
out["esperado"] = MEAN["V1"]
for a_, n_ in ((.10, "q10"), (.25, "q25"), (.50, "q50"), (.75, "q75"), (.90, "q90")):
    out[n_] = np.quantile(sm, a_, axis=1)
for k in (1, 2, 3, 4, 5, 6, 8, 10):
    out[f"P>={k}"] = (sm >= k).mean(1)
out.round(3).to_csv("risco_diario_coorte.csv", index=False)
ult = out[(out.dia == out.dia.max()) & (out.nivel == "A")].sort_values("esperado", ascending=False)
print("\nEXEMPLO DE TABELA DE RISCO - ultimo dia com dados (%s), nivel A:" % ult.dia.max().date())
print(ult[["cnesexecutante", "ag", "esperado", "q10", "q90", "P>=1", "P>=2", "P>=3", "P>=5", "y"]].round(2).to_string(index=False))
ult.round(3).to_csv("risco_ultimo_dia.csv", index=False)
print("\ntempo total %.0fs" % (time.time() - t0_))
