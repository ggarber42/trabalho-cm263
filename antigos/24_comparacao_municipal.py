"""Comparacao final no recorte municipal: baseline, V1 original, HGB por paciente (ref/selecionado/tuned; 2 regimes de treino),
HGB Poisson e SARIMA (puro e +agenda). Mesmo protocolo do script 17: CNES-dia da coorte (niveis_municipal.csv), 2023-2026, h = 3 dias,
simulacao NS = 300; metricas WAPE/vies/CRPS/cob80/cob50/larg80/ECE; Diebold-Mariano (HAC semanal) contra baseline e V1 original.
Tambem: metricas por consulta (Brier, AUC, logloss) e por celula CNES-dia-especialidade (AUC de P(>=1)).
Entradas: resultados_municipal/p_modelos_municipal.parquet (22), mu_series_municipal.parquet (23), p_modelos.pkl (V1 original).
Saidas (resultados_municipal/): comparacao_municipal.csv, comparacao_cortes.csv, comparacao_consulta.csv, comparacao_celula.csv,
  comparacao_dm.csv, comparacao_por_cnes.csv, relatorio_comparacao_municipal.md
"""
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score
from utils_municipal import brier, carrega_pacientes, md_table, painel_series

warnings.filterwarnings("ignore")
T0 = time.time()
tic = lambda: "%.0fmin" % ((time.time() - T0) / 60)
OUT = "resultados_municipal/"
NS, KS, LAG = 300, (1, 2, 3, 5, 8), 3
MIPV = "2237822"
YEARS = (2023, 2024, 2025, 2026)
raw, _ = carrega_pacientes()
n = len(raw)
L = painel_series(raw)
yy = raw.y.values
co = raw.coorte.values
rng = np.random.default_rng(1)

# ---------- probabilidades por consulta de cada modelo (NaN fora do teste) ----------
PM = pd.read_parquet(OUT + "p_modelos_municipal.parquet")
P = {}
for (reg, var), g in PM.groupby(["regime", "variante"]):
    a = np.full(n, np.nan); a[g.idx.values] = g.p.values
    P[f"{var}|{reg}"] = a
old = pickle.load(open("p_modelos.pkl", "rb"))["V1"]
assert len(old) == n, "p_modelos.pkl nao corresponde a base atual"
P["V1_original"] = old.astype(float)
NOMES_PAC = list(P)
print("modelos por paciente:", NOMES_PAC, flush=True)

# ---------- esqueleto CNES-dia da coorte, 2023+ ----------
sel = co & (raw.dia.dt.year.isin(YEARS)).values
S = raw.loc[sel, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gk = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gk.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
Dg = gk.agg(y=("y", "sum"), ag=("y", "size")).reset_index().rename(columns={"cnesexecutante": "cnes"})
NG = len(Dg)
Dg["nivel"] = Dg.cnes.map(dict(raw[co].drop_duplicates("cnesexecutante").set_index("cnesexecutante").nivel))
Dg["ano"] = Dg.dia.dt.year
Dg = Dg.merge(L[["cnes", "dia", "r28"]], on=["cnes", "dia"], how="left")
Dg["pred_base"] = Dg.r28 * Dg.ag


def fator_k(pred):
    out = np.full(NG, np.nan)
    for c_, ix in Dg.groupby("cnes").indices.items():
        x = pd.DataFrame({"y": Dg.y.values[ix], "pred": pred[ix]}, index=Dg.dia.values[ix])
        cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
        num = cal.y.shift(LAG).rolling(28, min_periods=14).sum()
        den = cal.pred.shift(LAG).rolling(28, min_periods=14).sum()
        out[ix] = (num / den.where(den > 0)).reindex(x.index).values
    return out


RES = {}
for nm in NOMES_PAC:
    p = P[nm][ordr]
    assert np.isfinite(p).all(), nm
    pred = np.add.reduceat(p, starts)
    RES[nm] = {"p": p, "pred": pred, "k": fator_k(pred)}
valid = Dg.pred_base.notna().values.copy()
for r in RES.values():
    valid &= np.isfinite(r["k"])
E = Dg[valid].reset_index(drop=True)
NE = len(E)
yv = E.y.values.astype(float)
print("CNES-dias avaliados: %d (A %d | B %d) (%s)" % (NE, (E.nivel == "A").sum(), (E.nivel == "B").sum(), tic()), flush=True)

SIMS, MU = {}, {}


def sim_pac(p, k):
    pk = np.clip(p * np.where(valid, k, 0)[grp], 0, 0.995).astype(np.float32)
    out = np.empty((NG, NS), dtype=np.int16)
    for a in range(0, NG, 1500):
        b = min(a + 1500, NG)
        r0, r1 = starts[a], (starts[b] if b < NG else len(pk))
        u = rng.random((r1 - r0, NS), dtype=np.float32)
        out[a:b] = np.add.reduceat((u < pk[r0:r1, None]).astype(np.int16), starts[a:b] - r0, axis=0)
    return out[valid]


SIMS["baseline"] = rng.poisson(E.pred_base.values[:, None], (NE, NS)).astype(np.int16)
MU["baseline"] = E.pred_base.values
for nm in NOMES_PAC:
    SIMS[nm] = sim_pac(RES[nm]["p"], RES[nm]["k"])
    MU[nm] = (RES[nm]["pred"] * RES[nm]["k"])[valid]

# ---------- series (NB com dispersao por CNES estimada nos 730 dias anteriores ao corte) ----------
MS = pd.read_parquet(OUT + "mu_series_municipal.parquet")
fL = L.set_index(["cnes", "dia"]).f
for nome, g in MS.groupby("modelo"):
    mu = np.full(NE, np.nan); al = np.full(NE, np.nan)
    for Y, gy in g.groupby("ano"):
        corte = pd.Timestamp(f"{Y}-01-01")
        tr = gy[(gy.dia >= corte - pd.Timedelta(days=730)) & (gy.dia < corte) & np.isfinite(gy.mu)]
        f_tr = fL.reindex(pd.MultiIndex.from_frame(tr[["cnes", "dia"]])).values
        d = pd.DataFrame({"c": tr.cnes.values, "num": (f_tr - tr.mu.values) ** 2 - tr.mu.values, "den": tr.mu.values ** 2}).groupby("c").sum()
        a_c = (d.num / d.den).clip(lower=1e-3).to_dict()
        te = gy[gy.dia.dt.year == Y].set_index(["cnes", "dia"]).mu
        r = np.flatnonzero((E.ano == Y).values)
        mu[r] = te.reindex(pd.MultiIndex.from_frame(E.loc[r, ["cnes", "dia"]])).values
        al[r] = [a_c.get(c, np.nan) for c in E.cnes.values[r]]
    ok = np.isfinite(mu) & np.isfinite(al)
    sims = np.full((NE, NS), -1, dtype=np.int16)
    mu_ = np.maximum(mu[ok], 1e-6); r_ = 1.0 / al[ok]
    sims[ok] = rng.negative_binomial(r_[:, None], (r_ / (r_ + mu_))[:, None], (ok.sum(), NS)).astype(np.int16)
    SIMS[nome], MU[nome] = sims, np.where(ok, mu, np.nan)
print("series ok (%s)" % tic(), flush=True)
ORDEM = ["baseline", "sarima", "sarimax", "hgb_pura", "hgb_agenda", "V1_original"] + [m for m in NOMES_PAC if m != "V1_original"]
ORDEM = [m for m in ORDEM if m in MU]
NOME = {"sarima": "SARIMA (pura)", "sarimax": "SARIMAX (+agenda)", "hgb_pura": "HGB Poisson (pura)", "hgb_agenda": "HGB Poisson (+agenda)"}


def crps_rows(sims, yy_):
    s = np.sort(sims, axis=1).astype(np.float64)
    i = np.arange(1, NS + 1)
    return np.abs(sims - yy_[:, None]).mean(1) - 0.5 * (2 / NS ** 2) * ((2 * i - NS - 1) * s).sum(1)


def metricas(sims, mean, yy_):
    q = lambda a: np.quantile(sims, a, axis=1)
    Pk = np.concatenate([(sims >= k).mean(1) for k in KS])
    Ek = np.concatenate([(yy_ >= k).astype(float) for k in KS])
    bins = np.clip((Pk * 10).astype(int), 0, 9)
    ece = sum(abs(Pk[bins == b].mean() - Ek[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(Pk)
    return {"n": len(yy_), "WAPE": np.abs(yy_ - mean).sum() / yy_.sum(), "vies": mean.sum() / yy_.sum(), "CRPS": crps_rows(sims, yy_).mean(),
            "cob80": ((yy_ >= q(.1)) & (yy_ <= q(.9))).mean(), "cob50": ((yy_ >= q(.25)) & (yy_ <= q(.75))).mean(),
            "larg80": (q(.9) - q(.1)).mean(), "ECE": ece}


def tabela(mask):
    rows = {}
    for m in ORDEM:
        mm = mask & np.isfinite(MU[m]) & (SIMS[m][:, 0] >= 0)
        if mm.sum() < 0.9 * mask.sum():
            continue
        rows[m] = metricas(SIMS[m][mm], MU[m][mm], yv[mm])
    return pd.DataFrame(rows).T.rename_axis("modelo").reset_index()


T_all = tabela(np.ones(NE, bool))
T_all.round(4).to_csv(OUT + "comparacao_municipal.csv", index=False)
cortes = []
for nome, mk in (("A", (E.nivel == "A").values), ("B", (E.nivel == "B").values), ("A+B sem Materno Infantil", (E.cnes != MIPV).values),
                 ("so Materno Infantil", (E.cnes == MIPV).values)) + tuple((f"ano {Y}", (E.ano == Y).values) for Y in YEARS):
    t = tabela(mk); t.insert(0, "corte", nome); cortes.append(t)
CT = pd.concat(cortes)
CT.round(4).to_csv(OUT + "comparacao_cortes.csv", index=False)
print(T_all.round(3).to_string(index=False), flush=True)

# ---------- por CNES ----------
pc = {m: pd.DataFrame({"c": E.cnes[np.isfinite(MU[m])], "e": np.abs(yv - MU[m])[np.isfinite(MU[m])], "y": yv[np.isfinite(MU[m])]}).groupby("c").sum().pipe(lambda g: g.e / g.y) for m in ORDEM}
PC = pd.DataFrame(pc); PC.round(3).to_csv(OUT + "comparacao_por_cnes.csv")

# ---------- por consulta e por celula (modelos por paciente) ----------
rowsC, rowsE = [], []
d = raw.loc[ordr, ["cnesexecutante", "dia", "especialidademae", "y"]].reset_index(drop=True)
vrow = valid[grp]  # linhas que pertencem a CNES-dias avaliados
for nm in NOMES_PAC:
    p = RES[nm]["p"][vrow]; y_ = d.y.values[vrow]
    pk = np.clip(RES[nm]["p"] * np.where(valid, RES[nm]["k"], 0)[grp], 0, 0.995)[vrow]
    for corte, mk in (("coorte", np.ones(len(p), bool)), ("sem Materno Infantil", (d.cnesexecutante.values[vrow] != MIPV))):
        rowsC.append({"modelo": nm, "corte": corte, "n": int(mk.sum()), "AUC": roc_auc_score(y_[mk], p[mk]), "Brier": brier(y_[mk], p[mk]),
                      "logloss": log_loss(y_[mk], np.clip(p[mk], 1e-6, 1 - 1e-6)), "Brier_recalibrado": brier(y_[mk], pk[mk])})
    c = pd.DataFrame({"c": d.cnesexecutante.values[vrow], "dia": d.dia.values[vrow], "e": d.especialidademae.astype(str).values[vrow],
                      "y": y_, "lq": np.log1p(-pk)})
    cell = c.groupby(["c", "dia", "e"]).agg(y=("y", "max"), lq=("lq", "sum")).reset_index()
    cell["p1"] = 1 - np.exp(cell.lq)
    rowsE.append({"modelo": nm, "n_celulas": len(cell), "AUC_celula": roc_auc_score(cell.y, cell.p1), "Brier_celula": brier(cell.y, cell.p1)})
CC = pd.DataFrame(rowsC); CE = pd.DataFrame(rowsE)
CC.round(5).to_csv(OUT + "comparacao_consulta.csv", index=False); CE.round(5).to_csv(OUT + "comparacao_celula.csv", index=False)
print(CC.round(4).to_string(index=False)); print(CE.round(4).to_string(index=False), flush=True)

# ---------- Diebold-Mariano ----------
sem = E.dia.dt.to_period("W").astype(str).values
perda = {}
for m in ORDEM:
    ok = np.isfinite(MU[m]) & (SIMS[m][:, 0] >= 0)
    ae = np.full(NE, np.nan); cr = np.full(NE, np.nan)
    ae[ok] = np.abs(yv[ok] - MU[m][ok]); cr[ok] = crps_rows(SIMS[m][ok], yv[ok])
    perda[m] = {"erro_abs": ae, "CRPS": cr}
dm = []
for ref in ("baseline", "V1_original"):
    for m in ORDEM:
        if m == ref:
            continue
        for k in ("erro_abs", "CRPS"):
            ok = np.isfinite(perda[m][k]) & np.isfinite(perda[ref][k])
            dif = pd.Series(perda[ref][k][ok] - perda[m][k][ok]); g = dif.groupby(sem[ok])
            fit = sm.OLS((g.sum() / g.size()).values, np.ones(g.ngroups)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            dm.append({"ref": ref, "modelo": m, "perda": k, "ganho_medio": fit.params[0], "p": fit.pvalues[0], "n": int(ok.sum())})
DM = pd.DataFrame(dm); DM.round(4).to_csv(OUT + "comparacao_dm.csv", index=False)

# ---------- relatorio ----------
try:
    sv = json.load(open(OUT + "variaveis_finais.json"))
except Exception:
    sv = {}
md = ["# Comparação final dos modelos — recorte municipal (coorte A+B, 10 CNES)", "",
      "Protocolo: CNES-dia, h = 3 dias, teste 2023–2026 com treino expansivo; hiperparâmetros e variáveis escolhidos apenas com dados anteriores ao ano de teste (validação temporal aninhada). %d CNES-dias avaliados." % NE, "",
      "Variáveis mantidas (%d): %s" % (len(sv.get("variaveis", [])), ", ".join(sv.get("variaveis", []))),
      "Variáveis removidas (%d): %s" % (len(sv.get("removidas", [])), ", ".join(sv.get("removidas", []))), "",
      "## Previsão do total diário por CNES (WAPE: menor é melhor)", "", md_table(T_all.round(4)), "",
      "## Por consulta (modelos por paciente)", "", md_table(CC.round(4)), "",
      "## Por célula CNES-dia-especialidade (P(≥1 falta))", "", md_table(CE.round(4)), "",
      "## Diebold-Mariano (ganho > 0: o modelo é melhor que a referência)", "", md_table(DM.round(4)), "",
      "## Cortes (nível, sem Materno Infantil, ano)", "", md_table(CT.round(4)), ""]
open(OUT + "relatorio_comparacao_municipal.md", "w").write("\n".join(md))
print("salvo (%s)" % tic())
