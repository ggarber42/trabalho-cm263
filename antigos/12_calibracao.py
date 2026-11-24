"""Calibracao das pontas: sobredispersao no dia (fator gama comum a todas as consultas do CNES-dia).

Mistura: p_i' = clip(p_i * k * g), g ~ Gamma(1/phi, phi), media 1. phi=0 => modelo atual (Bernoulli independente).
phi estimado por momentos com dados do ANO ANTERIOR (sem olhar o futuro): phi = max(0, sum[(y-mu)^2 - var]/sum(mu^2)).
Avaliacao em 2024-2026 (2023 serve de historico para estimar phi de 2024).
Baseline: Poisson(mu) (phi=0) vs binomial negativa (phi>0).
Saida: calibracao.json com phi para producao (ultimos 365 dias dos dados).
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
NS, KS = 400, (1, 2, 3, 5, 8)
CO = pd.read_csv("coorte_2019_2026.csv", dtype={"cnes": str}).set_index("cnes")
A = list(CO.index[CO.estavel])
B = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]
tier = {**{c: "A" for c in A}, **{c: "B" for c in B}}
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["situacaoagenda", "datahora_consulta", "cnesexecutante"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
P = pickle.load(open("p_modelos.pkl", "rb"))["V0"]
is_set = raw.cnesexecutante.isin(tier).values & (raw.dia >= "2023-01-01").values

S = raw.loc[is_set, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gk = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gk.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
D = gk.agg(y=("y", "sum"), ag=("y", "size")).reset_index()
NG = len(D)
D["nivel"] = D.cnesexecutante.map(tier)
D["ano"] = D.dia.dt.year
p = P[ordr]
D["pred"] = np.add.reduceat(p, starts)

# baseline (taxa 28d, lag 3, x agenda) e fator k do modelo
Bd = raw[raw.cnesexecutante.isin(tier)].groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size")).reset_index()
bl = []
for c_, x in Bd.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")).fillna(0)
    bl.append(pd.DataFrame({"cnesexecutante": c_, "pred_base": (x.f.shift(3).rolling(28, min_periods=14).sum() /
              x.ag.shift(3).rolling(28, min_periods=14).sum()) * x.ag}).rename_axis("dia").reset_index())
D = D.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")
k = np.full(NG, np.nan)
for c_, ix in D.groupby("cnesexecutante").indices.items():
    x = pd.DataFrame({"y": D.y.values[ix], "pred": D.pred.values[ix]}, index=D.dia.values[ix])
    cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
    num = cal.y.shift(3).rolling(28, min_periods=14).sum()
    den = cal.pred.shift(3).rolling(28, min_periods=14).sum()
    k[ix] = (num / den.where(den > 0)).reindex(x.index).values
D["k"] = k
D["mu_m"] = D.pred * D.k
pk_rows = np.clip(p * D.k.values[grp], 0, 0.995)
D["var_m"] = np.add.reduceat(pk_rows * (1 - pk_rows), starts)
D["mu_b"] = D.pred_base
D["var_b"] = D.mu_b
D["valid"] = D.k.notna() & D.pred_base.notna()
D = D[D.valid | True]

def phi_mom(df, mu, var):
    d = df[df.valid]
    return max(0.0, ((d.y - d[mu]) ** 2 - d[var]).sum() / (d[mu] ** 2).sum())

rng = np.random.default_rng(7)
def sim_modelo(idx_days, phi_day):
    """idx_days: indices de D (posicoes) a simular; phi_day: phi por dia."""
    out = np.empty((len(idx_days), NS), dtype=np.int16)
    for j, d_ in enumerate(idx_days):
        r0 = starts[d_]; r1 = starts[d_ + 1] if d_ + 1 < NG else len(p)
        ph = max(phi_day[j], 1e-6)
        g = rng.gamma(1 / ph, ph, size=NS).astype(np.float32)
        pr = np.clip(p[r0:r1, None].astype(np.float32) * np.float32(D.k.values[d_]) * g[None, :], 0, 0.995)
        out[j] = (rng.random(pr.shape, dtype=np.float32) < pr).sum(0)
    return out
def sim_base(mu, phi_day):
    ph = np.maximum(phi_day, 1e-6)[:, None]
    g = rng.gamma(1 / ph, ph, size=(len(mu), NS))
    return rng.poisson(mu[:, None] * g).astype(np.int16)

def metricas(sims, mean, yy):
    q = lambda a: np.quantile(sims, a, axis=1)
    s = np.sort(sims, axis=1).astype(np.float64)
    i = np.arange(1, NS + 1)
    crps = np.abs(sims - yy[:, None]).mean(1) - 0.5 * (2 / NS ** 2) * ((2 * i - NS - 1) * s).sum(1)
    Pk = np.concatenate([(sims >= kk).mean(1) for kk in KS]); E = np.concatenate([(yy >= kk).astype(float) for kk in KS])
    b = np.clip((Pk * 10).astype(int), 0, 9)
    ece = sum(abs(Pk[b == i_].mean() - E[b == i_].mean()) * (b == i_).sum() for i_ in range(10) if (b == i_).any()) / len(Pk)
    lo, hi = (b == 0), (b == 9)
    return {"CRPS": crps.mean(), "cob80": ((yy >= q(.1)) & (yy <= q(.9))).mean(), "cob50": ((yy >= q(.25)) & (yy <= q(.75))).mean(),
            "ECE": ece, "ponta_baixa(prev>obs)": "%.3f>%.3f" % (Pk[lo].mean(), E[lo].mean()),
            "ponta_alta(prev>obs)": "%.3f>%.3f" % (Pk[hi].mean(), E[hi].mean())}

rows, phis = {}, {}
for t in ("A", "B"):
    for var, mu, vr in (("modelo", "mu_m", "var_m"), ("baseline", "mu_b", "var_b")):
        for Y in (2024, 2025, 2026):
            phis[(t, var, Y)] = phi_mom(D[(D.nivel == t) & (D.ano == Y - 1)], mu, vr)
print("PHI estimado (ano anterior):")
print(pd.Series(phis).unstack().round(4).to_string())

for t in ("A", "B"):
    for var, mu in (("modelo", "mu_m"), ("baseline", "mu_b")):
        m = (D.nivel == t) & D.valid & (D.ano >= 2024)
        idx = np.flatnonzero(m.values)
        yy = D.y.values[idx]
        phi_d = np.array([phis[(t, var, a)] for a in D.ano.values[idx]])
        for nome, ph in (("phi=0", np.zeros(len(idx))), ("phi estimado", phi_d)):
            sims = sim_modelo(idx, ph) if var == "modelo" else sim_base(D[mu].values[idx], ph)
            r = metricas(sims, D[mu].values[idx], yy)
            r["WAPE"] = np.abs(yy - D[mu].values[idx]).sum() / yy.sum()
            rows[(t, var, nome)] = r
T = pd.DataFrame(rows).T
T.index.names = ["nivel", "base", "dispersao"]
print("\nAVALIACAO 2024-2026 (cob80 alvo 0.80, cob50 alvo 0.50; ponta = prob. prevista > frequencia observada)")
print(T[["WAPE", "CRPS", "cob80", "cob50", "ECE", "ponta_baixa(prev>obs)", "ponta_alta(prev>obs)"]].astype({"WAPE": float, "CRPS": float, "cob80": float, "cob50": float, "ECE": float}).round(3).to_string())

# phi de producao: estimativa agrupada de todo o periodo 2023-2026 (mais estavel que 1 ano: o ultimo ano
# inclui efeitos de calendario, ex. semana do Carnaval 2026, que inflam o phi)
ult = D[D.dia > D.dia.max() - pd.Timedelta(days=365)]
prod, ref = {}, {}
for t in ("A", "B"):
    for var, mu, vr in (("modelo", "mu_m", "var_m"), ("baseline", "mu_b", "var_b")):
        prod[f"{t}_{var}"] = round(float(phi_mom(D[D.nivel == t], mu, vr)), 4)
        ref[f"{t}_{var}"] = round(float(phi_mom(ult[ult.nivel == t], mu, vr)), 4)
json.dump(prod, open("calibracao.json", "w"), indent=1)
print("\nphi de producao (agrupado 2023-2026):", prod)
print("phi dos ultimos 365 dias (referencia):", ref)
