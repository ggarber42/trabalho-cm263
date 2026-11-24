"""Comparacao de estrategias de agrupamento para o modelo por paciente (top 20 CNES).

Estrategias: global (todas as 221 CNES) | quadrantes (volume x taxa de falta) | volume (2 grupos) | por CNES.
Grupos definidos so com dados de 2019-2022 (antes do teste). Treino expansivo anual (treino < 1/jan/Y).
Saida = distribuicao de faltas por CNES-dia: p_i recalibrados pelo nivel dos ultimos 28 dias do CNES
(lag 3) e somados por simulacao (Poisson-binomial). Baseline = Poisson(taxa 28d x agenda do dia).
Metricas: WAPE da media, CRPS (distribuicao inteira), cobertura 80%/50%, calibracao P(faltas>=k).
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
t0_ = time.time()
SRC = "gercon_poa_censo_distancias.parquet"
top = pd.read_csv("ranking_top20.csv", dtype={"cnesexecutante": str})["cnesexecutante"].tolist()
YEARS = (2023, 2024, 2025, 2026)
NS = 300
KS = (1, 2, 3, 5, 8)

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
for c in CAT:
    X[c] = X[c].astype("category")
y = raw.y.values
dia = raw.dia.values
cnes_arr = raw.cnesexecutante.values
is_top = raw.cnesexecutante.isin(top).values

# ---------- grupos (so dados 2019-2022) ----------
pre = raw[is_top & (raw.dia >= "2019-01-01") & (raw.dia < "2023-01-01")]
dd = pre.groupby(["cnesexecutante", "dia"]).agg(ag=("y", "size"), f=("y", "sum")).reset_index()
cn = dd.groupby("cnesexecutante").agg(vol=("ag", "mean"), ag=("ag", "sum"), f=("f", "sum"))
cn["taxa"] = cn.f / cn.ag
vol_alto = cn.vol >= cn.vol.median()
taxa_alta = cn.taxa >= cn.taxa.median()
cn["quadrante"] = np.where(vol_alto, "volAlto", "volBaixo") + "_" + np.where(taxa_alta, "taxaAlta", "taxaBaixa")
cn["volume"] = np.where(vol_alto, "volAlto", "volBaixo")
print("GRUPOS (definidos com 2019-2022):")
print(cn.sort_values(["quadrante", "vol"], ascending=[True, False])[["vol", "taxa", "quadrante"]].round(3).to_string())

def grupos(col):
    return {g: list(s.index) for g, s in cn.groupby(col)}
ESTRATEGIAS = {
    "global": (None, {"todas": top}),
    "quadrantes": (None, grupos("quadrante")),
    "volume": (None, grupos("volume")),
    "por_cnes": (None, {c: [c] for c in top}),
}

def ajusta_preve(train_all, grp):
    p = np.full(len(raw), np.nan)
    for Y in YEARS:
        anual = (dia >= np.datetime64(f"{Y}-01-01")) & (dia < np.datetime64(f"{Y + 1}-01-01"))
        for gid, mem in grp.items():
            m_mem = np.isin(cnes_arr, mem)
            te = anual & m_mem & is_top
            tr = (dia < np.datetime64(f"{Y}-01-01")) & (np.ones(len(raw), bool) if train_all else m_mem)
            if te.sum() == 0:
                continue
            if len(np.unique(y[tr])) < 2:
                p[te] = y[tr].mean()
                continue
            n = tr.sum()
            m = HistGradientBoostingClassifier(
                max_iter=300 if train_all else 200, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=int(np.clip(n / 300, 20, 200)), l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                categorical_features="from_dtype", random_state=0)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
    return p

# ---------- esqueleto CNES-dia ----------
test_mask = is_top & (dia >= np.datetime64("2023-01-01"))
S = raw.loc[test_mask, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gkey = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gkey.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
Dg = gkey.agg(y=("y", "sum"), ag=("y", "size")).reset_index()
NG = len(Dg)

# baseline Poisson (taxa 28d, lag 3, x agenda), historico completo
Bd = raw[is_top].groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size")).reset_index()
bl = []
for c, x in Bd.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")).fillna(0)
    num = x.f.shift(3).rolling(28, min_periods=14).sum()
    den = x.ag.shift(3).rolling(28, min_periods=14).sum()
    bl.append(pd.DataFrame({"cnesexecutante": c, "pred_base": (num / den) * x.ag}).rename_axis("dia").reset_index())
Dg = Dg.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")

def fator_k(pred):
    out = np.full(NG, np.nan)
    for c, ix in Dg.groupby("cnesexecutante").indices.items():
        x = pd.DataFrame({"y": Dg.y.values[ix], "pred": pred[ix]}, index=Dg.dia.values[ix])
        cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
        num = cal.y.shift(3).rolling(28, min_periods=14).sum()
        den = cal.pred.shift(3).rolling(28, min_periods=14).sum()
        out[ix] = (num / den.where(den > 0)).reindex(x.index).values
    return out

# ---------- treino das estrategias ----------
res = {}
for nome, (_, g) in ESTRATEGIAS.items():
    p = ajusta_preve(nome == "global", g)[ordr]
    pred = np.add.reduceat(p, starts)
    res[nome] = {"p": p, "pred": pred, "k": fator_k(pred)}
    print(f"  treinado: {nome} ({len(g)} modelo(s) por ano) | {time.time() - t0_:.0f}s", flush=True)

import pickle
with open("res_08.pkl", "wb") as fh:
    pickle.dump({"res": res, "ordr": ordr}, fh)
valid = Dg.pred_base.notna().values.copy()
for r in res.values():
    valid &= np.isfinite(r["k"])
yv = Dg.y.values[valid]
print("\nCNES-dias avaliados (comuns a todas as estrategias): %d | faltas/dia media %.2f" % (valid.sum(), yv.mean()))

rng = np.random.default_rng(1)

def sim_modelo(p, k):
    pk = np.clip(p * np.where(valid, k, 0)[grp], 0, 0.995).astype(np.float32)
    out = np.empty((NG, NS), dtype=np.int16)
    for a in range(0, NG, 1500):
        b = min(a + 1500, NG)
        r0, r1 = starts[a], (starts[b] if b < NG else len(pk))
        u = rng.random((r1 - r0, NS), dtype=np.float32)
        s = np.add.reduceat((u < pk[r0:r1, None]).astype(np.int16), starts[a:b] - r0, axis=0)
        out[a:b] = s
    return out[valid]

def metricas(sims, mean, yv):
    q = lambda a: np.quantile(sims, a, axis=1)
    s = np.sort(sims, axis=1).astype(np.float64)
    i = np.arange(1, NS + 1)
    e_xx = (2 / NS ** 2) * ((2 * i - NS - 1) * s).sum(1)
    crps = np.abs(sims - yv[:, None]).mean(1) - 0.5 * e_xx
    P = np.concatenate([(sims >= k).mean(1) for k in KS])
    E = np.concatenate([(yv >= k).astype(float) for k in KS])
    bins = np.clip((P * 10).astype(int), 0, 9)
    ece = sum(abs(P[bins == b].mean() - E[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(P)
    return {"WAPE": np.abs(yv - mean).sum() / yv.sum(), "vies": mean.sum() / yv.sum(), "CRPS": crps.mean(),
            "cob80": ((yv >= q(.1)) & (yv <= q(.9))).mean(), "larg80": (q(.9) - q(.1)).mean(),
            "cob50": ((yv >= q(.25)) & (yv <= q(.75))).mean(), "ECE_P>=k": ece, "Brier_P>=k": ((P - E) ** 2).mean(), "_crps_dia": crps}

linhas, crps_dia, wape_cnes = {}, {}, {}
base_sims = rng.poisson(Dg.pred_base.values[valid][:, None], (valid.sum(), NS)).astype(np.int16)
m = metricas(base_sims, Dg.pred_base.values[valid], yv)
crps_dia["baseline_poisson"] = m.pop("_crps_dia"); linhas["baseline_poisson"] = m
wape_cnes["baseline_poisson"] = Dg.pred_base.values[valid]
del base_sims
for nome, r in res.items():
    sims = sim_modelo(r["p"], r["k"])
    mean = (r["pred"] * r["k"])[valid]
    m = metricas(sims, mean, yv)
    crps_dia[nome] = m.pop("_crps_dia"); linhas[nome] = m
    wape_cnes[nome] = mean
    del sims

T = pd.DataFrame(linhas).T
print("\nRESULTADO GERAL (menor = melhor, exceto cobertura: alvo 0.80 e 0.50)")
print(T.round(3).to_string())

cv = Dg.cnesexecutante.values[valid]
pc = pd.DataFrame({n: pd.Series(np.abs(yv - m_)).groupby(cv).sum() / pd.Series(yv).groupby(cv).sum()
                   for n, m_ in wape_cnes.items()})
pc.insert(0, "faltas_dia", pd.Series(yv).groupby(cv).mean())
pc["grupo_quadrante"] = cn.quadrante.reindex(pc.index).values
pc["melhor"] = pc[[c for c in pc.columns if c not in ("faltas_dia", "grupo_quadrante")]].idxmin(axis=1)
print("\nWAPE POR CNES")
print(pc.sort_values("faltas_dia", ascending=False).round(3).to_string())
print("\nQuantos CNES cada estrategia vence (WAPE):")
print(pc.melhor.value_counts().to_string())

cr = pd.DataFrame(crps_dia)
cr["cnes"] = cv
pcc = cr.groupby("cnes").mean()
pcc["melhor_CRPS"] = pcc.idxmin(axis=1)
print("\nCRPS medio por CNES - estrategia vencedora (quantidade):")
print(pcc.melhor_CRPS.value_counts().to_string())

out = Dg[valid][["cnesexecutante", "dia", "y", "ag", "pred_base"]].copy()
for n, m_ in wape_cnes.items():
    out["media_" + n] = m_
out.to_csv("pred_diaria_estrategias.csv", index=False)
print("\ntempo total %.0fs" % (time.time() - t0_))
