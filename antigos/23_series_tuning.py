"""Modelos de serie temporal no recorte municipal (coorte de niveis_municipal.csv), com validacao temporal aninhada e busca de
hiperparametros: HGB Poisson com lags e SARIMA, cada um em duas versoes: PURA (so a serie) e +AGENDA (agregados diarios das
variaveis dos pacientes agendados: taxa de falta previa media, % primeira vez, idade media, dias ate a consulta; todos conhecidos
com a agenda, como `ag`).
Protocolo (igual ao 22): para o ano de teste Y, folds internos treino < v -> validacao v, v em {Y-2, Y-1}; teste externo nao entra.
  HGB Poisson: busca aleatoria (N_CFG configs + conjunto de lags), criterio = deviance de Poisson media na validacao.
  SARIMA: grade p,q in {0,1,2}, P,Q in {0,1} (d = 1, D = 1, sazonalidade 7); criterio = WAPE de h = 3 dias (global nas CNES).
          Previsao h = 3 pelo filtro de Kalman com parametros fixos (ajustados em treino < corte), como no script 17.
          Versao +AGENDA = SARIMAX com 3 exogenas padronizadas (regressao com erros SARIMA).
Uso: python 23_series_tuning.py [--ncfg 24] [--jobs 6] [--smoke]
Obs.: mu_series_municipal.parquet inclui os 2 anos anteriores a Y (ano < Y e in-sample/filtrado) so para estimar a dispersao NB; o teste e ano == Y.
Saidas (resultados_municipal/): tuning_series.csv, melhores_series.json, mu_series_municipal.parquet
"""
import argparse
import itertools
import json
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from utils_municipal import AGENDA_COLS, LAG, carrega_pacientes, painel_series

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ncfg", type=int, default=24)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T0 = time.time()
tic = lambda: "%.0fmin" % ((time.time() - T0) / 60)
OUT = "resultados_municipal/"
raw, _ = carrega_pacientes()
L = painel_series(raw)
del raw
cnes_list = sorted(L.cnes.unique())
Lc = L.groupby("cnes").indices
print("painel L: %d linhas, %d CNES (%s)" % (len(L), len(cnes_list), tic()), flush=True)
YEARS = [2023] if args.smoke else [2023, 2024, 2025, 2026]
FOLD_YEARS = [2021, 2022] if args.smoke else [2021, 2022, 2023, 2024]
ano = L.dia.dt.year.values
base_ok = ((L.ag > 0) & L.r28.notna()).values

# ============================ HGB Poisson ============================
FEATSETS = {"curto": ["r7", "r14", "r28", "rdow4", "ag", "dow", "mes", "fer", "posfer"],
            "base": ["r7", "r14", "r28", "r91", "rdow4", "rd1", "ag", "dow", "mes", "fer", "posfer"],
            "longo": ["r7", "r14", "r28", "r91", "rdow4", "rd1", "ag", "dow", "mes", "fer", "posfer", "r28"]}
FEATSETS["longo"] = FEATSETS["base"] + ["y_lag7", "y_lag14"]
L["y_lag7"] = L.groupby("cnes").y_rate.shift(7)
L["y_lag14"] = L.groupby("cnes").y_rate.shift(14)
L["y_lag7"] = L.y_lag7.fillna(L.r28); L["y_lag14"] = L.y_lag14.fillna(L.r28)
rs = np.random.default_rng(7)
CFG = [dict(learning_rate=float(np.exp(rs.uniform(np.log(0.02), np.log(0.2)))), max_iter=int(rs.choice([100, 200, 400, 800])),
            max_leaf_nodes=int(rs.choice([4, 8, 15, 31])), min_samples_leaf=int(rs.choice([20, 50, 100, 200])),
            l2_regularization=float(np.exp(rs.uniform(np.log(0.1), np.log(30)))), featset=str(rs.choice(list(FEATSETS))))
       for _ in range(args.ncfg)]
rate = (L.f / L.ag.where(L.ag > 0)).values


def hgb_cols(cfg, agenda):
    return FEATSETS[cfg["featset"]] + (AGENDA_COLS if agenda else [])


def hgb_fit(cfg, agenda, tr):
    cols = hgb_cols(cfg, agenda)
    Xh = L[cols + ["cnes"]].copy()
    Xh["cnes"] = pd.Categorical(Xh.cnes, categories=cnes_list)
    kw = {k: v for k, v in cfg.items() if k != "featset"}
    m = HistGradientBoostingRegressor(loss="poisson", categorical_features="from_dtype", random_state=0, **kw)
    m.fit(Xh[tr], rate[tr], sample_weight=L.ag.values[tr])
    return m, Xh


def hgb_inner(v, agenda, ci):
    tr = base_ok & (ano < v)
    va = base_ok & (ano == v)
    m, Xh = hgb_fit(CFG[ci], agenda, tr)
    mu = np.clip(m.predict(Xh[va]), 1e-4, 1) * L.ag.values[va]
    return dict(modelo="hgb_agenda" if agenda else "hgb_pura", fold=v, cfg=ci, dev=float(mean_poisson_deviance(L.f.values[va], mu)),
                wape=float(np.abs(L.f.values[va] - mu).sum() / L.f.values[va].sum()))

jobs = [(v, ag_, ci) for ag_ in (False, True) for v in FOLD_YEARS for ci in range(len(CFG))]
R = Parallel(n_jobs=args.jobs)(delayed(hgb_inner)(*j) for j in jobs)
TU_H = pd.DataFrame(R)
print("HGB tuning ok: %d fits (%s)" % (len(jobs), tic()), flush=True)

# ============================ SARIMA / SARIMAX ============================
GRID = list(itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]))  # p, q, P, Q
if args.smoke:
    GRID = GRID[:4]
EXOG = AGENDA_COLS[:3]


def sarima_fc(y, ex, ntr, order, seas):
    """previsao h=3 (y_{s+3} | dados <= s) para toda a serie, parametros ajustados em y[:ntr]."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    kw = dict(order=order, seasonal_order=seas, trend="n", enforce_stationarity=False, enforce_invertibility=False)
    if ex is not None:
        mu_, sd_ = ex[:ntr].mean(0), ex[:ntr].std(0) + 1e-9
        ex = (ex - mu_) / sd_
        r0 = SARIMAX(y[:ntr], exog=ex[:ntr], **kw).fit(disp=False, maxiter=60)
        rf = r0.apply(y, exog=ex)
    else:
        r0 = SARIMAX(y[:ntr], **kw).fit(disp=False, maxiter=60)
        rf = r0.apply(y)
    g = lambda k: (lambda a: a[:, :, 0] if a.ndim == 3 else a)(np.asarray(rf.model.ssm[k]))
    Z, Tm = g("design"), g("transition")
    assert not np.any(rf.model.ssm["state_intercept"])
    pred = (Z @ np.linalg.matrix_power(Tm, LAG) @ rf.filtered_state)[0]
    oi = np.asarray(rf.model.ssm["obs_intercept"])
    if ex is not None:
        oi = oi.reshape(1, -1) if oi.ndim == 1 else oi
        assert oi.shape[1] == len(y), oi.shape
        pred = pred[: len(y) - LAG] + oi[0, LAG:]
    else:
        assert not np.any(oi)
        pred = pred[: len(y) - LAG]
    fc = np.full(len(y), np.nan)
    fc[LAG:] = pred
    return fc


def sar_inner(c, v, agenda, gi):
    ix = Lc[c]
    d = L.iloc[ix]
    y = d.y_rate.values.astype(float)
    ex = d[EXOG].values.astype(float) if agenda else None
    ntr = int((d.dia < pd.Timestamp(f"{v}-01-01")).sum())
    p, q, P, Q = GRID[gi]
    try:
        fc = sarima_fc(y, ex, ntr, (p, 1, q), (P, 1, Q, 7))
    except Exception:
        return dict(c=c, fold=v, modelo="sarimax" if agenda else "sarima", g=gi, abs=np.nan, tot=np.nan)
    va = ((d.dia.dt.year == v) & (d.ag > 0)).values & np.isfinite(fc)
    mu = np.clip(fc[va], 0, 1) * d.ag.values[va]
    return dict(c=c, fold=v, modelo="sarimax" if agenda else "sarima", g=gi, abs=float(np.abs(d.f.values[va] - mu).sum()), tot=float(d.f.values[va].sum()))

if args.smoke:
    # verificacao do truque do filtro de Kalman com exogenas contra a previsao nativa do statsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    c0 = cnes_list[0]; d0 = L.iloc[Lc[c0]]; y0 = d0.y_rate.values.astype(float); e0 = d0[EXOG].values.astype(float)
    ntr0 = int((d0.dia < "2023-01-01").sum())
    fc = sarima_fc(y0, e0, ntr0, (1, 1, 1), (0, 1, 1, 7))
    mu_, sd_ = e0[:ntr0].mean(0), e0[:ntr0].std(0) + 1e-9
    ez = (e0 - mu_) / sd_
    r0 = SARIMAX(y0[:ntr0], exog=ez[:ntr0], order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), trend="n",
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
    for t in (ntr0 + 100, ntr0 + 200):
        nat = r0.apply(y0[: t + 1], exog=ez[: t + 1]).forecast(LAG, exog=ez[t + 1: t + 1 + LAG])[-1]
        print("verificacao SARIMAX h=3 em t=%d: kalman %.5f | nativo %.5f" % (t, fc[t + LAG], nat))
        assert abs(fc[t + LAG] - nat) < 1e-6

jobs = [(c, v, ag_, gi) for ag_ in (False, True) for v in FOLD_YEARS for gi in range(len(GRID)) for c in cnes_list]
print("SARIMA tuning: %d ajustes (%s)" % (len(jobs), tic()), flush=True)
R = Parallel(n_jobs=args.jobs)(delayed(sar_inner)(*j) for j in jobs)
S = pd.DataFrame(R)
TU_S = S.groupby(["modelo", "fold", "g"]).agg(abs_=("abs", "sum"), tot=("tot", "sum"), falhas=("abs", lambda s: s.isna().sum())).reset_index()
TU_S["wape"] = TU_S.abs_ / TU_S.tot
TU_S["ordem"] = [str(GRID[g]) for g in TU_S.g]
print("SARIMA tuning ok (%s)" % tic(), flush=True)
pd.concat([TU_H.assign(criterio="deviance"), TU_S.assign(criterio="wape")]).to_csv(OUT + "tuning_series.csv", index=False)

# ============================ escolha por ano e predicao do teste externo ============================
BEST, MU = {}, []
for Y in YEARS:
    fy = [Y - 2, Y - 1]
    for agenda in (False, True):
        nome = "hgb_agenda" if agenda else "hgb_pura"
        d = TU_H[(TU_H.modelo == nome) & TU_H.fold.isin(fy)].groupby("cfg").dev.mean().sort_values()
        ci = int(d.index[0])
        BEST[f"{Y}|{nome}"] = {"cfg": CFG[ci], "dev_cv": float(d.iloc[0])}
        tr = base_ok & (ano < Y)
        m, Xh = hgb_fit(CFG[ci], agenda, tr)
        te = ((L.dia >= pd.Timestamp(f"{Y - 2}-01-01")) & (ano <= Y) & (L.ag > 0)).values  # inclui 730d de treino p/ dispersao NB
        MU.append(pd.DataFrame({"cnes": L.cnes.values[te], "dia": L.dia.values[te], "ano": Y, "modelo": nome,
                                "mu": np.clip(m.predict(Xh[te]), 1e-4, 1) * L.ag.values[te]}))
        nome = "sarimax" if agenda else "sarima"
        d = TU_S[(TU_S.modelo == nome) & TU_S.fold.isin(fy) & (TU_S.falhas == 0)].groupby("g").wape.mean().sort_values()
        gi = int(d.index[0]); p, q, P, Q = GRID[gi]
        BEST[f"{Y}|{nome}"] = {"ordem": [p, 1, q], "sazonal": [P, 1, Q, 7], "wape_cv": float(d.iloc[0])}

        def final(c):
            ix = Lc[c]; dd = L.iloc[ix]
            ntr = int((dd.dia < pd.Timestamp(f"{Y}-01-01")).sum())
            fc = sarima_fc(dd.y_rate.values.astype(float), dd[EXOG].values.astype(float) if agenda else None, ntr, (p, 1, q), (P, 1, Q, 7))
            te_ = ((dd.dia >= pd.Timestamp(f"{Y - 2}-01-01")) & (dd.dia.dt.year <= Y) & (dd.ag > 0)).values
            return pd.DataFrame({"cnes": c, "dia": dd.dia.values[te_], "ano": Y, "modelo": nome,
                                 "mu": np.clip(fc[te_], 0, 1) * dd.ag.values[te_]})
        MU += Parallel(n_jobs=args.jobs)(delayed(final)(c) for c in cnes_list)
    print("ano %d ok (%s)" % (Y, tic()), flush=True)
pd.concat(MU, ignore_index=True).to_parquet(OUT + "mu_series_municipal.parquet", index=False)
json.dump(BEST, open(OUT + "melhores_series.json", "w"), indent=1, default=str)
print(pd.DataFrame({k: {kk: (str(vv) if not isinstance(vv, float) else round(vv, 5)) for kk, vv in v.items()} for k, v in BEST.items()}).T.to_string())
print("salvo (%s)" % tic())
