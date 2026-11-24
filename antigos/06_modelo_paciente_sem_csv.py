"""Modelo por paciente: P(falta) de cada consulta agendada -> soma por CNES-dia + intervalos.

Treino expansivo, refeito a cada ano (treino < 1/jan/Y, teste = ano Y, Y = 2023..2026).
Comparacao com o baseline diario (taxa 28d com lag 3 x agenda do dia) nos MESMOS CNES-dias.
Decisao em D-3: so variaveis conhecidas no agendamento; historico do paciente ate t0.
"""
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

t_ini = time.time()
SRC = "gercon_poa_censo_distancias.parquet"
top = pd.read_csv("ranking_top20.csv", dtype={"cnesexecutante": str})["cnesexecutante"].tolist()

raw = pd.read_parquet(SRC)
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["gap_t0"] = (raw.dia - pd.to_datetime(raw.t0)).dt.days
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
for c in CAT:
    X[c] = X[c].astype("category")
print("sklearn", sklearn.__version__, "| linhas", len(X), "| features", X.shape[1])

raw["p"] = np.nan
modelos = {}
for Y in (2023, 2024, 2025, 2026):
    tr = (raw.dia < f"{Y}-01-01").values
    te = ((raw.dia >= f"{Y}-01-01") & (raw.dia < f"{Y + 1}-01-01")).values
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        categorical_features="from_dtype", random_state=0)
    m.fit(X[tr], raw.y[tr])
    raw.loc[te, "p"] = m.predict_proba(X[te])[:, 1]
    raw.loc[te, "p_const"] = raw.y[tr].mean()
    modelos[Y] = (m, te)
    print(f"  {Y}: treino {tr.sum()} | teste {te.sum()} | iter {m.n_iter_} | {time.time() - t_ini:.0f}s", flush=True)

# ---------- 1) metricas por paciente ----------
T = raw[raw.p.notna()]
def met(df, nome):
    print("%-34s n=%7d  AUC=%.3f  logloss=%.4f (const %.4f)  Brier=%.4f  y_medio=%.3f  p_medio=%.3f" % (
        nome, len(df), roc_auc_score(df.y, df.p), log_loss(df.y, df.p), log_loss(df.y, df.p_const),
        brier_score_loss(df.y, df.p), df.y.mean(), df.p.mean()))
print("\nPOR PACIENTE (teste 2023-2026)")
met(T, "todas as CNES")
met(T[T.cnesexecutante.isin(top)], "top 20 CNES")
met(T[T.cnesexecutante.isin(top) & (T.gap_t0 == 14)], "top 20, so t0 = 14 dias (realista)")
met(T[T.cnesexecutante.isin(top) & (T.gap_t0 < 14)], "top 20, t0 < 14 dias")

# ---------- 2) importancia (modelo 2025, amostra) ----------
m25, te25 = modelos[2025]
idx = np.flatnonzero(te25)
rs = np.random.default_rng(0)
amostra = rs.choice(idx, size=min(25000, len(idx)), replace=False)
pi = permutation_importance(m25, X.iloc[amostra], raw.y.iloc[amostra], scoring="roc_auc",
                            n_repeats=2, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nIMPORTANCIA (queda de AUC ao embaralhar), modelo 2025:")
print(imp.head(10).round(4).to_string())

# ---------- 3) agregacao CNES-dia ----------
S = T[T.cnesexecutante.isin(top)][["cnesexecutante", "dia", "y", "p"]].copy()
S = S.sort_values(["cnesexecutante", "dia"]).reset_index(drop=True)
grp = S.groupby(["cnesexecutante", "dia"], sort=True).ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
NS = 300
rng = np.random.default_rng(1)
sims = np.add.reduceat((rng.random((len(S), NS)) < S.p.values[:, None]).astype(np.int16), starts, axis=0)
D = S.groupby(["cnesexecutante", "dia"], sort=True).agg(y=("y", "sum"), ag=("y", "size"), pred=("p", "sum")).reset_index()
D["lo_b"] = np.quantile(sims, 0.10, axis=1)
D["hi_b"] = np.quantile(sims, 0.90, axis=1)
D["q20_b"] = np.quantile(sims, 0.20, axis=1)
del sims

# baseline diario (taxa 28d, lag 3, x agenda do dia), historico completo
B = raw[raw.cnesexecutante.isin(top)][["cnesexecutante", "dia", "y"]].groupby(["cnesexecutante", "dia"]).agg(
    f=("y", "sum"), ag=("y", "size")).reset_index()
bl = []
for c, x in B.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(pd.date_range(B.dia.min(), B.dia.max(), freq="D")).fillna(0)
    num = x.f.shift(3).rolling(28, min_periods=14).sum()
    den = x.ag.shift(3).rolling(28, min_periods=14).sum()
    bl.append(pd.DataFrame({"cnesexecutante": c, "pred_base": (num / den) * x.ag}).rename_axis("dia").reset_index())
D = D.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")

def emp(df, col, lag=3, win=120):
    r = (df.y - df[col])
    out = {}
    for q, nome in ((0.10, "lo"), (0.90, "hi"), (0.20, "q20")):
        out[nome] = df.groupby("cnesexecutante").apply(
            lambda g: (g.y - g[col]).shift(lag).rolling(win, min_periods=40).quantile(q)).reset_index(level=0, drop=True)
    return out
D = D.sort_values(["cnesexecutante", "dia"]).reset_index(drop=True)
for col, suf in (("pred", "m"), ("pred_base", "b")):
    e = emp(D, col)
    D[f"lo_{suf}e"] = D[col] + e["lo"]
    D[f"hi_{suf}e"] = D[col] + e["hi"]
    D[f"q20_{suf}e"] = D[col] + e["q20"]
D = D.dropna(subset=["pred", "pred_base", "lo_me", "lo_be"]).reset_index(drop=True)
D["ano"] = D.dia.dt.year

# D.to_csv("pred_diaria_modelo_vs_base.csv", index=False)

def wape(c): return (D.y - D[c]).abs().sum() / D.y.sum()
print("\nCNES-DIA (top 20): n=%d | faltas/dia media=%.2f" % (len(D), D.y.mean()))
print("WAPE: baseline %.3f | modelo por paciente %.3f" % (wape("pred_base"), wape("pred")))
print("WAPE por ano  ->  " + " | ".join("%d: base %.3f / modelo %.3f" % (
    a, (g.y - g.pred_base).abs().sum() / g.y.sum(), (g.y - g.pred).abs().sum() / g.y.sum()) for a, g in D.groupby("ano")))
print("vies (soma prevista / soma real): baseline %.3f | modelo %.3f" % (D.pred_base.sum() / D.y.sum(), D.pred.sum() / D.y.sum()))

print("\nINTERVALO 80% (cobertura nominal 0.80)")
for nome, lo, hi in (("baseline + residuo empirico", "lo_be", "hi_be"), ("modelo + residuo empirico", "lo_me", "hi_me"),
                     ("modelo, Bernoulli simulado", "lo_b", "hi_b")):
    cob = ((D.y >= D[lo]) & (D.y <= D[hi])).mean()
    print("  %-30s cobertura=%.3f  largura media=%.2f" % (nome, cob, (D[hi] - D[lo]).mean()))

print("\nLIMITE INFERIOR UNILATERAL (percentil 20): quanto da pra contar como falta 'segura'")
for nome, col in (("baseline + residuo empirico", "q20_be"), ("modelo + residuo empirico", "q20_me"), ("modelo, Bernoulli", "q20_b")):
    b = np.floor(D[col].clip(lower=0))
    print("  %-30s P(faltas >= limite)=%.3f | limite medio=%.2f faltas/dia (real medio %.2f) | vagas 'seguras' = %.1f%% das faltas" % (
        nome, (D.y >= b).mean(), b.mean(), D.y.mean(), 100 * b.sum() / D.y.sum()))

print("\nPOR CNES (modelo): WAPE base -> modelo")
pc = D.groupby("cnesexecutante").apply(lambda g: pd.Series({
    "faltas_dia": g.y.mean(), "WAPE_base": (g.y - g.pred_base).abs().sum() / g.y.sum(),
    "WAPE_modelo": (g.y - g.pred).abs().sum() / g.y.sum()})).sort_values("faltas_dia", ascending=False)
print(pc.round(3).to_string())
print("\ntempo total %.0fs" % (time.time() - t_ini))
