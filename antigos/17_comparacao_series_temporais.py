"""Comparacao do modelo por paciente (V0/V1, script 11) com modelos de serie temporal, no MESMO protocolo:
faltas por CNES-dia, coorte A+B, treino expansivo anual (Y = 2023..2026), horizonte h = 3 dias (toda feature usa dados <= t-3),
mesmas metricas (WAPE, vies, CRPS, cob80, cob50, larg80, ECE) e saida probabilistica por simulacao (NS = 300).

Modelos (todos recebem a agenda do dia `ag` e o calendario, que sao conhecidos com antecedencia):
  baseline      Poisson(taxa 28d x agenda) - o do script 11
  naive7        taxa do mesmo dia da semana na semana anterior x agenda
  dow4          taxa media dos 4 mesmos dias da semana anteriores x agenda
  ets           Holt-Winters aditivo (nivel + sazonalidade semanal) na TAXA, alpha/gamma otimizados para h=3
  sarima        SARIMAX(1,1,1)(0,1,1)7 na taxa, previsao h=3 pelo filtro de Kalman com parametros fixos
  glm_nb        GLM Poisson por CNES (offset log agenda; taxas defasadas, dia da semana, mes, feriados) + dispersao NB
  hgb_poisson   HistGradientBoosting (perda Poisson) global sobre taxas defasadas + calendario
  hib_glm       GLM pooled com a previsao do modelo por paciente (V1) como covariavel   [so 2024-2026]
  ctrl_glm      o mesmo GLM pooled SEM a covariavel do paciente (controle do hibrido)    [so 2024-2026]
  V0, V1        modelo por paciente (recalculado a partir de p_modelos.pkl, sem retreinar)

Para toda media prevista mu, a distribuicao e Poisson (baseline, como no script 11) ou Binomial Negativa com dispersao
estimada por momentos nos 730 dias anteriores ao corte (todos os demais). `baseline_nb` e o controle do efeito da distribuicao.

Uso: python 17_comparacao_series_temporais.py [--anos 2023 2024 2025 2026] [--sem-sarima] [--cnes N]
Saidas: comparacao_series.csv, comparacao_series_2024mais.csv, comparacao_series_por_cnes.csv, comparacao_series_dm.csv,
        previsoes_series.parquet
"""
import argparse
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.easter import easter
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--anos", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
ap.add_argument("--sem-sarima", action="store_true")
ap.add_argument("--cnes", type=int, default=0, help="limita a N CNES (teste rapido)")
ap.add_argument("--jobs", type=int, default=6)
args = ap.parse_args()
T0 = time.time()
tic = lambda: "%.0fs" % (time.time() - T0)

SRC = "gercon_poa_censo_distancias.parquet"
NS, KS, LAG = 300, (1, 2, 3, 5, 8), 3
YEARS = tuple(args.anos)
CO = pd.read_csv("coorte_2019_2026.csv", dtype={"cnes": str}).set_index("cnes")
A = list(CO.index[CO.estavel])
B = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]
SET = A + B
tier = {**{c: "A" for c in A}, **{c: "B" for c in B}}

# =====================================================================================================================
# 1. Reconstrucao do esqueleto CNES-dia e do modelo por paciente (identico ao script 11, sem retreinar)
# =====================================================================================================================
raw = pd.read_parquet(SRC, columns=["cnesexecutante", "datahora_consulta", "situacaoagenda"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
assert (raw.dia == raw.dia.dt.normalize()).all(), "datahora_consulta tem hora; o script 11 agruparia por timestamp"
with open("p_modelos.pkl", "rb") as fh:
    P = pickle.load(fh)
assert len(P["V1"]) == len(raw), "p_modelos.pkl nao corresponde a base atual"
dia = raw.dia.values
cnes_arr = raw.cnesexecutante.values
is_set = np.isin(cnes_arr, SET)

tm = is_set & (dia >= np.datetime64("2023-01-01"))
S = raw.loc[tm, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gk = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gk.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
Dg = gk.agg(y=("y", "sum"), ag=("y", "size")).reset_index()
NG = len(Dg)
Dg["nivel"] = Dg.cnesexecutante.map(tier)

# painel diario completo (2019+) de faltas e agenda por CNES do conjunto
Bd = raw[is_set].groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size")).reset_index()
full_idx = pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")
bl = []
for c_, x in Bd.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(full_idx).fillna(0)
    num = x.f.shift(LAG).rolling(28, min_periods=14).sum()
    den = x.ag.shift(LAG).rolling(28, min_periods=14).sum()
    bl.append(pd.DataFrame({"cnesexecutante": c_, "pred_base": (num / den) * x.ag}).rename_axis("dia").reset_index())
Dg = Dg.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")


def fator_k(pred):
    out = np.full(NG, np.nan)
    for c_, ix in Dg.groupby("cnesexecutante").indices.items():
        x = pd.DataFrame({"y": Dg.y.values[ix], "pred": pred[ix]}, index=Dg.dia.values[ix])
        cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
        num = cal.y.shift(LAG).rolling(28, min_periods=14).sum()
        den = cal.pred.shift(LAG).rolling(28, min_periods=14).sum()
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
if args.cnes:
    keep = sorted(SET)[:args.cnes]
    valid &= Dg.cnesexecutante.isin(keep).values
    SET = [c for c in SET if c in keep]
valid &= (Dg.dia.dt.year.isin(YEARS)).values

E = Dg[valid][["cnesexecutante", "dia", "nivel", "y", "ag"]].rename(columns={"cnesexecutante": "cnes"}).reset_index(drop=True)
E["ano"] = E.dia.dt.year
yv = E.y.values.astype(float)
NE = len(E)
print("CNES-dias avaliados: %d (A %d | B %d) %s" % (NE, (E.nivel == "A").sum(), (E.nivel == "B").sum(), tic()), flush=True)

rng = np.random.default_rng(1)


def sim_modelo_pac(p, k):
    pk = np.clip(p * np.where(valid, k, 0)[grp], 0, 0.995).astype(np.float32)
    out = np.empty((NG, NS), dtype=np.int16)
    for a in range(0, NG, 1500):
        b = min(a + 1500, NG)
        r0, r1 = starts[a], (starts[b] if b < NG else len(pk))
        u = rng.random((r1 - r0, NS), dtype=np.float32)
        out[a:b] = np.add.reduceat((u < pk[r0:r1, None]).astype(np.int16), starts[a:b] - r0, axis=0)
    return out[valid]


# =====================================================================================================================
# 2. Painel diario L com features (so dados <= t-3) e calendario
# =====================================================================================================================
def feriados(anos):
    s = set()
    for y in anos:
        e = pd.Timestamp(easter(y))
        fixos = [(1, 1), (2, 2), (4, 21), (5, 1), (9, 7), (9, 20), (10, 12), (11, 2), (11, 15), (12, 25)]
        if y >= 2024:
            fixos.append((11, 20))
        for m, d in fixos:
            s.add(pd.Timestamp(y, m, d))
        for off in (-48, -47, -2, 60):  # carnaval (seg/ter), sexta santa, corpus christi
            s.add(e + pd.Timedelta(days=off))
    return s


FER = feriados(range(full_idx.min().year, full_idx.max().year + 1))
rows = []
for c_ in SET:
    x = Bd[Bd.cnesexecutante == c_].set_index("dia").reindex(full_idx).fillna(0)
    f, a = x.f, x.ag
    d = pd.DataFrame({"cnes": c_, "f": f, "ag": a})
    for w, mp in ((7, 3), (14, 7), (28, 14), (91, 45)):
        num = f.shift(LAG).rolling(w, min_periods=mp).sum()
        den = a.shift(LAG).rolling(w, min_periods=mp).sum()
        d[f"r{w}"] = num / den.where(den > 0)
    nd = sum(f.shift(7 * k) for k in (1, 2, 3, 4))
    dd = sum(a.shift(7 * k) for k in (1, 2, 3, 4))
    d["rdow4"] = (nd / dd.where(dd > 0)).fillna(d.r28)
    d["rd1"] = (f.shift(7) / a.shift(7).where(a.shift(7) > 0)).fillna(d.r28)
    d["y_rate"] = f / a.where(a > 0)
    rows.append(d.rename_axis("dia").reset_index())
L = pd.concat(rows, ignore_index=True)
L["dow"] = L.dia.dt.dayofweek
L["mes"] = L.dia.dt.month
L["fer"] = L.dia.isin(FER).astype(int)
L["posfer"] = (L.dia - pd.Timedelta(days=1)).isin(FER).astype(int)
for c in ("r7", "r14", "r91"):
    L[c] = L[c].fillna(L.r28)
L = L.sort_values(["cnes", "dia"]).reset_index(drop=True)
LIDX = pd.MultiIndex.from_frame(L[["cnes", "dia"]])
POS = LIDX.get_indexer(pd.MultiIndex.from_frame(E[["cnes", "dia"]]))
assert (POS >= 0).all()
Lc = L.groupby("cnes").indices  # cnes -> posicoes em L (ordenadas por dia)
print("painel L: %d linhas %s" % (len(L), tic()), flush=True)

MU = {}     # nome -> mu por linha de E (NaN onde o modelo nao se aplica)
ALPHA = {}  # nome -> dispersao NB por linha de E (nan => Poisson)
POISSON = {"baseline"}


def alpha_por_cnes(mu_all, corte):
    """dispersao NB por CNES pelos momentos, nos 730 dias antes do corte (taxas defasadas fora/dentro da amostra)."""
    m = ((L.dia >= corte - pd.Timedelta(days=730)) & (L.dia < corte) & (L.ag > 0) & np.isfinite(mu_all)).values
    d = pd.DataFrame({"c": L.cnes.values[m], "num": ((L.f.values[m] - mu_all[m]) ** 2 - mu_all[m]), "den": mu_all[m] ** 2})
    g = d.groupby("c").sum()
    return (g.num / g.den).clip(lower=1e-3).to_dict()


def registra(nome, mu_all_por_ano):
    """mu_all_por_ano: {Y: array len(L)} (previsao para todas as linhas de L usando modelo treinado antes de Y)."""
    mu = np.full(NE, np.nan)
    al = np.full(NE, np.nan)
    for Y, mu_all in mu_all_por_ano.items():
        r = np.flatnonzero((E.ano == Y).values)
        if len(r) == 0:
            continue
        mu[r] = mu_all[POS[r]]
        if nome not in POISSON:
            a_c = alpha_por_cnes(mu_all, pd.Timestamp(f"{Y}-01-01"))
            al[r] = [a_c.get(c, np.nan) for c in E.cnes.values[r]]
    MU[nome], ALPHA[nome] = mu, al


def regra(fun):
    base = fun(L)
    mu_all = np.clip(base, 0, None) * L.ag.values
    return {Y: mu_all for Y in YEARS}


# ---------- baselines por regra ----------
registra("baseline", regra(lambda d: d.r28.values))
registra("baseline_nb", regra(lambda d: d.r28.values))
registra("naive7", regra(lambda d: d.rd1.values))
registra("dow4", regra(lambda d: d.rdow4.values))
print("regras ok %s" % tic(), flush=True)


# ---------- ETS (Holt-Winters aditivo semanal na taxa, otimizado para h=3) ----------
def ets_run(y, dow, alpha, gamma, lvl0):
    n = len(y)
    fc = np.full(n, np.nan)
    l, s = lvl0, np.zeros(7)
    for t in range(n):
        yt = y[t]
        if yt == yt:
            e = yt - (l + s[dow[t]])
            l = l + alpha * e
            s[dow[t]] += gamma * (yt - l - s[dow[t]])
        if t + LAG < n:
            fc[t + LAG] = l + s[dow[t + LAG]]
    return fc


def ets_cnes(c, ix):
    y, w, dw, dias = L.y_rate.values[ix], L.ag.values[ix], L.dow.values[ix], L.dia.values[ix]
    first = y[np.isfinite(y)][:28]
    lvl0 = float(first.mean()) if len(first) else 0.2
    out = {}
    for Y in YEARS:
        ntr = int((dias < np.datetime64(f"{Y}-01-01")).sum())
        best = (np.inf, 0.05, 0.05)
        for al in (0.01, 0.02, 0.05, 0.1, 0.2):
            for ga in (0.0, 0.02, 0.05, 0.1):
                fc = ets_run(y[:ntr], dw[:ntr], al, ga, lvl0)
                m = np.isfinite(fc) & np.isfinite(y[:ntr])
                m[:90] = False
                sse = np.sum(w[:ntr][m] * (y[:ntr][m] - fc[m]) ** 2)
                if sse < best[0]:
                    best = (sse, al, ga)
        out[Y] = ets_run(y, dw, best[1], best[2], lvl0)
    return c, out


t_ = time.time()
ets_res = Parallel(n_jobs=args.jobs)(delayed(ets_cnes)(c, ix) for c, ix in Lc.items())
mu_ets = {Y: np.full(len(L), np.nan) for Y in YEARS}
for c, out in ets_res:
    for Y in YEARS:
        mu_ets[Y][Lc[c]] = np.clip(out[Y], 0, 1) * L.ag.values[Lc[c]]
registra("ets", mu_ets)
print("ets ok %s" % tic(), flush=True)


# ---------- SARIMA (previsao h=3 pelo filtro de Kalman com parametros fixos) ----------
def sarima_cnes(c, ix, anos):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = L.y_rate.values[ix].astype(float)
    dias = L.dia.values[ix]
    out = {}
    for Y in anos:
        ntr = int((dias < np.datetime64(f"{Y}-01-01")).sum())
        mod = SARIMAX(y[:ntr], order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), trend="n",
                      enforce_stationarity=False, enforce_invertibility=False)
        r0 = mod.fit(disp=False, maxiter=60)
        rf = r0.apply(y)  # mesmos parametros, filtra a serie inteira (estado em t so usa dados <= t)
        g = lambda k: (lambda a: a[:, :, 0] if a.ndim == 3 else a)(np.asarray(rf.model.ssm[k]))
        Z, Tm = g("design"), g("transition")
        assert not np.any(rf.model.ssm["state_intercept"]) and not np.any(rf.model.ssm["obs_intercept"])
        T3 = np.linalg.matrix_power(Tm, LAG)
        pred = (Z @ T3 @ rf.filtered_state)[0]  # pred[s] = E[y_{s+3} | dados <= s]
        fc = np.full(len(y), np.nan)
        fc[LAG:] = pred[:-LAG]
        out[Y] = fc
    return c, out


if not args.sem_sarima:
    sar = Parallel(n_jobs=args.jobs)(delayed(sarima_cnes)(c, ix, YEARS) for c, ix in Lc.items())
    mu_sar = {Y: np.full(len(L), np.nan) for Y in YEARS}
    for c, out in sar:
        for Y in YEARS:
            mu_sar[Y][Lc[c]] = np.clip(out[Y], 0, 1) * L.ag.values[Lc[c]]
    registra("sarima", mu_sar)
    print("sarima ok %s" % tic(), flush=True)

# ---------- GLM Poisson por CNES ----------
lg = lambda r: np.log(np.clip(r, 0.02, 0.9))


def design(d, cols=None, extra=None):
    X = pd.DataFrame({"c": 1.0, "l28": lg(d.r28.values), "l7": lg(d.r7.values), "ldow": lg(d.rdow4.values),
                      "fer": d.fer.values, "posfer": d.posfer.values}, index=d.index)
    # categorias fixas (dow 0..6, mes 1..12) + drop_first: mesma base em treino e teste, sem colinearidade com a constante
    dd = pd.get_dummies(pd.Categorical(d.dow.astype(int).values, categories=range(7)), prefix="d", drop_first=True).astype(float)
    mm = pd.get_dummies(pd.Categorical(d.mes.astype(int).values, categories=range(1, 13)), prefix="m", drop_first=True).astype(float)
    dd.index = mm.index = d.index
    X = pd.concat([X, dd, mm], axis=1)
    if extra is not None:
        for k, v in extra.items():
            X[k] = v
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def glm_ajusta(d_tr, extra_tr=None):
    X = design(d_tr, extra=extra_tr)
    keep = [c for c in X.columns if c == "c" or X[c].std() > 0]
    X = X[keep]
    fit = sm.GLM(d_tr.f.values, X, family=sm.families.Poisson(), offset=np.log(d_tr.ag.values)).fit()
    return fit, keep


mu_glm = {Y: np.full(len(L), np.nan) for Y in YEARS}
for Y in YEARS:
    corte = pd.Timestamp(f"{Y}-01-01")
    for c, ix in Lc.items():
        d = L.iloc[ix]
        ok = ((d.ag > 0) & d.r28.notna()).values
        tr = d[ok & (d.dia < corte).values]
        try:
            fit, cols = glm_ajusta(tr)
            dok = d[ok]
            mu_glm[Y][ix[ok]] = np.minimum(fit.predict(design(dok, cols=cols)) * dok.ag.values, dok.ag.values)
        except Exception as e:  # nao converge -> usa o baseline para esta CNES/ano
            print("GLM falhou", c, Y, e)
            mu_glm[Y][ix] = d.r28.values * d.ag.values
registra("glm_nb", mu_glm)
print("glm ok %s" % tic(), flush=True)

# ---------- HGB Poisson global ----------
FEAT = ["r7", "r14", "r28", "r91", "rdow4", "rd1", "ag", "dow", "mes", "fer", "posfer"]
Lh = L[FEAT + ["cnes"]].copy()
Lh["cnes"] = pd.Categorical(Lh.cnes, categories=sorted(SET))
mu_hgb = {}
for Y in YEARS:
    corte = pd.Timestamp(f"{Y}-01-01")
    tr = ((L.dia < corte) & (L.ag > 0) & L.r28.notna()).values
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=50, l2_regularization=1.0, categorical_features="from_dtype",
                                      random_state=0)
    m.fit(Lh[tr], (L.f / L.ag.where(L.ag > 0)).values[tr], sample_weight=L.ag.values[tr])
    mu_all = np.where(L.ag.values > 0, np.clip(m.predict(Lh), 0, 1) * L.ag.values, np.nan)
    mu_hgb[Y] = mu_all
registra("hgb_poisson", mu_hgb)
print("hgb ok %s" % tic(), flush=True)

# ---------- hibrido: previsao do modelo por paciente como covariavel (pooled; 2024+) ----------
E["esp_pac"] = (res["V1"]["pred"] * res["V1"]["k"])[valid]
Ldf = L.iloc[POS].reset_index(drop=True)  # features de L nas linhas de E
cn_dum = pd.get_dummies(E.cnes).astype(float)


def pooled(extra_cols):
    X = design(Ldf)
    for k, v in extra_cols.items():
        X[k] = v
    return pd.concat([X.drop(columns=["c"]), cn_dum], axis=1)


Xc = pooled({})
Xh = pooled({"lpac": lg((E.esp_pac / E.ag).values)})
for nome, X in (("ctrl_glm", Xc), ("hib_glm", Xh)):
    mu = np.full(NE, np.nan)
    al = np.full(NE, np.nan)
    for Y in YEARS:
        tr = (E.ano < Y).values
        te = (E.ano == Y).values
        if tr.sum() < 500 or te.sum() == 0:
            continue
        keep = [c for c in X.columns if X.loc[tr, c].std() > 0]
        fit = sm.GLM(E.y.values[tr], X.loc[tr, keep], family=sm.families.Poisson(), offset=np.log(E.ag.values[tr])).fit()
        mu_tr = fit.predict(X.loc[tr, keep]) * E.ag.values[tr]
        mu[te] = np.minimum(fit.predict(X.loc[te, keep]) * E.ag.values[te], E.ag.values[te])
        a_ = max(((E.y.values[tr] - mu_tr) ** 2 - mu_tr).sum() / (mu_tr ** 2).sum(), 1e-3)
        al[te] = a_
    MU[nome], ALPHA[nome] = mu, al
print("hibrido ok %s" % tic(), flush=True)

# =====================================================================================================================
# 3. Simulacao, metricas, testes
# =====================================================================================================================
SIMS = {}
for nome, mu in MU.items():
    sims = np.full((NE, NS), -1, dtype=np.int16)
    ok = np.isfinite(mu)
    if nome in POISSON:
        sims[ok] = rng.poisson(mu[ok][:, None], (ok.sum(), NS)).astype(np.int16)
    else:
        ok &= np.isfinite(ALPHA[nome])
        mu_ = np.maximum(mu[ok], 1e-6)
        r_ = 1.0 / ALPHA[nome][ok]
        p_ = r_ / (r_ + mu_)
        sims[ok] = rng.negative_binomial(r_[:, None], p_[:, None], (ok.sum(), NS)).astype(np.int16)
    sims[~ok] = -1
    SIMS[nome] = sims
for v in ("V0", "V1"):
    SIMS[v] = sim_modelo_pac(res[v]["p"], res[v]["k"])
    MU[v] = (res[v]["pred"] * res[v]["k"])[valid]


def crps_rows(sims, yy):
    s = np.sort(sims, axis=1).astype(np.float64)
    i = np.arange(1, NS + 1)
    return np.abs(sims - yy[:, None]).mean(1) - 0.5 * (2 / NS ** 2) * ((2 * i - NS - 1) * s).sum(1)


def metricas(sims, mean, yy):
    q = lambda a: np.quantile(sims, a, axis=1)
    Pk = np.concatenate([(sims >= k).mean(1) for k in KS])
    Ek = np.concatenate([(yy >= k).astype(float) for k in KS])
    bins = np.clip((Pk * 10).astype(int), 0, 9)
    ece = sum(abs(Pk[bins == b].mean() - Ek[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(Pk)
    return {"WAPE": np.abs(yy - mean).sum() / yy.sum(), "vies": mean.sum() / yy.sum(), "CRPS": crps_rows(sims, yy).mean(),
            "cob80": ((yy >= q(.1)) & (yy <= q(.9))).mean(), "cob50": ((yy >= q(.25)) & (yy <= q(.75))).mean(),
            "larg80": (q(.9) - q(.1)).mean(), "ECE": ece, "n": len(yy)}


ORDEM = ["baseline", "baseline_nb", "naive7", "dow4", "ets", "sarima", "glm_nb", "hgb_poisson", "ctrl_glm", "hib_glm", "V0", "V1"]
ORDEM = [m for m in ORDEM if m in MU]


def tabela(mask_extra, titulo):
    rows = {}
    for t in ("A", "B", "A+B"):
        mt = mask_extra & (np.ones(NE, bool) if t == "A+B" else (E.nivel == t).values)
        for m in ORDEM:
            mm = mt & np.isfinite(MU[m]) & (SIMS[m][:, 0] >= 0)
            if mm.sum() < 0.9 * mt.sum():
                continue  # modelo nao cobre este subconjunto
            rows[(t, m)] = metricas(SIMS[m][mm], MU[m][mm], yv[mm])
    T = pd.DataFrame(rows).T
    T.index.names = ["nivel", "modelo"]
    print("\n" + titulo + " (alvos: cob80=0.80, cob50=0.50)")
    print(T.round(3).to_string())
    return T


T_all = tabela(np.ones(NE, bool), "RESULTADO %d-%d, modelos que cobrem todo o periodo" % (min(YEARS), max(YEARS)))
T_all.round(4).to_csv("comparacao_series.csv")
m24 = (E.ano >= 2024).values
T24 = tabela(m24, "RESULTADO 2024+, inclui hibrido e controle (mesmas linhas para todos)")
T24.round(4).to_csv("comparacao_series_2024mais.csv")

# por CNES: WAPE de cada modelo
pc = {}
for m in ORDEM:
    ok = np.isfinite(MU[m])
    g = pd.DataFrame({"c": E.cnes[ok], "e": np.abs(yv[ok] - MU[m][ok]), "y": yv[ok]}).groupby("c").sum()
    pc[m] = g.e / g.y
PC = pd.DataFrame(pc)
PC.insert(0, "nivel", [tier[c] for c in PC.index])
PC.round(3).to_csv("comparacao_series_por_cnes.csv")
print("\nWAPE por CNES (menor = melhor)")
print(PC.round(3).to_string())

# Diebold-Mariano em blocos semanais (HAC) contra baseline e V1
sem = E.dia.dt.to_period("W").astype(str).values
perda = {}
for m in ORDEM:
    ok = np.isfinite(MU[m]) & (SIMS[m][:, 0] >= 0)
    ae, cr = np.full(NE, np.nan), np.full(NE, np.nan)
    ae[ok] = np.abs(yv[ok] - MU[m][ok])
    cr[ok] = crps_rows(SIMS[m][ok], yv[ok])
    perda[m] = {"erro_abs": ae, "CRPS": cr}
dm = []
for ref in ("baseline", "V1"):
    for m in ORDEM:
        if m == ref:
            continue
        for k in ("erro_abs", "CRPS"):
            ok = np.isfinite(perda[m][k]) & np.isfinite(perda[ref][k])
            dif = pd.Series(perda[ref][k][ok] - perda[m][k][ok])
            g = dif.groupby(sem[ok])
            d = (g.sum() / g.size()).values  # ganho medio por CNES-dia em cada semana (bloco)
            fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            dm.append({"ref": ref, "modelo": m, "perda": k, "ganho_medio_por_cnes_dia": fit.params[0], "p": fit.pvalues[0], "n": int(ok.sum())})
DM = pd.DataFrame(dm)
DM.round(4).to_csv("comparacao_series_dm.csv", index=False)
print("\nDIEBOLD-MARIANO (ganho > 0 => o modelo e melhor que a referencia; p com HAC por semana)")
print(DM.round(4).to_string(index=False))

out = E.copy()
for m in ORDEM:
    out["mu_" + m] = MU[m]
out.to_parquet("previsoes_series.parquet", index=False)
print("\ntempo total %s" % tic())
