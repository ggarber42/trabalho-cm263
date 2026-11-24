"""HGB por paciente no recorte municipal: validacao temporal aninhada + busca aleatoria de hiperparametros.

Para cada ano de teste externo Y (2023..2026) e cada regime de treino (base_toda | so_municipal):
  folds internos (rolling-origin): treino < v -> validacao v, com v em {Y-2, Y-1}; metrica = Brier nas consultas da coorte municipal.
  Os 4 folds unicos (v = 2021..2024) sao compartilhados entre os Y (cada um so usa dados anteriores a Y).
  Busca aleatoria (N_CFG configuracoes, semente fixa); numero de iteracoes escolhido por staged_predict_proba no fold
  (sem early stopping aleatorio, que vazaria o tempo). Escolhe (config, n_iter) de menor Brier medio nos 2 folds.
  Depois re-treina com treino < Y e prediz o ano Y (o teste externo nunca entra na escolha).
Variantes preditas (todas por regime): ref_default (32 variaveis, hiperparametros do script 11, 300 iter),
  sel_default (variaveis selecionadas, mesmos hiperparametros), sel_tuned (variaveis selecionadas + hiperparametros do tuning).
Uso: python 22_hgb_paciente_tuning.py [--ncfg 24] [--smoke]
Saidas (resultados_municipal/): tuning_hgb_paciente.csv, melhores_hiperparametros.json, p_modelos_municipal.parquet
"""
import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils_municipal import ALL_VARS, brier, carrega_pacientes, make_hgb

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ncfg", type=int, default=24)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--jobs", type=int, default=3)
args = ap.parse_args()
T0 = time.time()
tic = lambda: "%.0fmin" % ((time.time() - T0) / 60)
OUT = "resultados_municipal/"
VARS = json.load(open(OUT + "variaveis_finais.json"))["variaveis"]
raw, X = carrega_pacientes()
yy = raw.y.values
ano = raw.dia.dt.year.values
mun = raw.municipal.values
co = raw.coorte.values
print("variaveis selecionadas (%d): %s" % (len(VARS), VARS), flush=True)
EXTRA = [v for v in VARS if v not in X.columns]
if EXTRA:  # candidatas novas (taxa_cnes_28d, taxa_prev_suav)
    from utils_municipal import feature_extras
    ex = feature_extras(raw)
    X = pd.concat([X, ex[EXTRA]], axis=1)

rs = np.random.default_rng(42)
CFG = []
for _ in range(args.ncfg):
    CFG.append(dict(learning_rate=float(np.exp(rs.uniform(np.log(0.02), np.log(0.15)))),
                    max_leaf_nodes=int(rs.choice([15, 31, 63, 127])),
                    max_depth=[None, 6, 10][rs.integers(3)],
                    min_samples_leaf=int(rs.choice([50, 100, 200, 400, 800])),
                    l2_regularization=float(np.exp(rs.uniform(np.log(0.1), np.log(30)))),
                    max_features=float(rs.uniform(0.5, 1.0)),
                    max_bins=int(rs.choice([128, 255]))))
MAXIT, STEP = (60 if args.smoke else 600), 10
REGIMES = {"base_toda": np.ones(len(raw), bool), "so_municipal": mun}
FOLD_YEARS = [2021, 2022] if args.smoke else [2021, 2022, 2023, 2024]
YEARS = [2023] if args.smoke else [2023, 2024, 2025, 2026]


def inner(v, reg, ci):
    """Brier da validacao v (coorte) por numero de iteracoes, treino < v do regime reg."""
    tr = (ano < v) & REGIMES[reg]
    va = (ano == v) & co
    m = make_hgb(max_iter=MAXIT, **CFG[ci]).fit(X.loc[tr, VARS], yy[tr])
    its, bs = [], []
    for i, p in enumerate(m.staged_predict_proba(X.loc[va, VARS]), 1):
        if i % STEP == 0 or i == MAXIT:
            its.append(i); bs.append(brier(yy[va], p[:, 1]))
    return dict(fold=v, regime=reg, cfg=ci, its=its, brier=bs)

jobs = [(v, reg, ci) for reg in REGIMES for v in FOLD_YEARS for ci in range(len(CFG))]
print("tuning: %d fits (%d configs x %d folds x %d regimes) (%s)" % (len(jobs), len(CFG), len(FOLD_YEARS), len(REGIMES), tic()), flush=True)
R = Parallel(n_jobs=args.jobs, prefer="threads", verbose=0)(delayed(inner)(*j) for j in jobs)
rows = []
for r in R:
    for it, b in zip(r["its"], r["brier"]):
        rows.append({"fold": r["fold"], "regime": r["regime"], "cfg": r["cfg"], "n_iter": it, "brier": b})
TU = pd.DataFrame(rows)
TU.to_csv(OUT + "tuning_hgb_paciente.csv", index=False)
print("tuning ok (%s)" % tic(), flush=True)

BEST = {}
for Y in YEARS:
    for reg in REGIMES:
        fy = [Y - 2, Y - 1]
        d = TU[(TU.regime == reg) & TU.fold.isin(fy)].groupby(["cfg", "n_iter"]).brier.agg(["mean", "size"]).reset_index()
        d = d[d["size"] == len(fy)].sort_values("mean")
        b = d.iloc[0]
        BEST[f"{Y}|{reg}"] = {"cfg": int(b.cfg), "n_iter": int(b.n_iter), "brier_cv": float(b["mean"]), "params": CFG[int(b.cfg)]}
json.dump(BEST, open(OUT + "melhores_hiperparametros.json", "w"), indent=1, default=str)
print(pd.DataFrame(BEST).T[["cfg", "n_iter", "brier_cv"]].to_string(), flush=True)

# ---------------- predicao do teste externo ----------------
DEF = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0)
out = []
def final(Y, reg, var):
    tr = (ano < Y) & REGIMES[reg]
    te = (ano == Y) & mun
    if var == "ref_default":
        vs, kw = ALL_VARS, DEF
    elif var == "sel_default":
        vs, kw = VARS, DEF
    else:
        b = BEST[f"{Y}|{reg}"]
        vs, kw = VARS, dict(b["params"], max_iter=b["n_iter"])
    m = make_hgb(**kw).fit(X.loc[tr, vs], yy[tr])
    return pd.DataFrame({"idx": np.flatnonzero(te).astype(np.int32), "ano": Y, "regime": reg, "variante": var,
                         "p": m.predict_proba(X.loc[te, vs])[:, 1].astype(np.float32)})
fj = [(Y, reg, var) for Y in YEARS for reg in REGIMES for var in ("ref_default", "sel_default", "sel_tuned")]
out = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(final)(*j) for j in fj)
PM = pd.concat(out, ignore_index=True)
PM.to_parquet(OUT + "p_modelos_municipal.parquet", index=False)
t = PM.merge(pd.DataFrame({"idx": np.arange(len(raw)), "y": yy, "coorte": co}), on="idx")
t = t[t.coorte]
print("\nTESTE EXTERNO (coorte municipal): Brier / AUC por variante e regime")
from sklearn.metrics import roc_auc_score
print(t.groupby(["regime", "variante"]).apply(lambda g: pd.Series({"n": len(g), "brier": brier(g.y, g.p), "auc": roc_auc_score(g.y, g.p)})).round(5).to_string())
print("salvo p_modelos_municipal.parquet (%s)" % tic())
