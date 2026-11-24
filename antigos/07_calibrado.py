"""Modelo por paciente recalibrado pelo nivel recente do CNES.

pred_cal = soma(p_i do dia) x k, com k = faltas reais / faltas previstas pelo modelo nos 28 dias
anteriores (lag 3) do proprio CNES. Isola o ganho de saber QUEM esta agendado (composicao do dia),
com o nivel de faltas do CNES vindo dos dados recentes (como no baseline).
"""
import numpy as np
import pandas as pd

D = pd.read_csv("pred_diaria_modelo_vs_base.csv", parse_dates=["dia"])
D = D[["cnesexecutante", "dia", "y", "ag", "pred", "pred_base"]].copy()
D["cnesexecutante"] = D["cnesexecutante"].astype(str)

out = []
for c, x in D.groupby("cnesexecutante"):
    x = x.set_index("dia")
    cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna({"y": 0, "pred": 0, "ag": 0})
    num = cal.y.shift(3).rolling(28, min_periods=14).sum()
    den = cal.pred.shift(3).rolling(28, min_periods=14).sum()
    k = (num / den).reindex(x.index)
    r = x.assign(k=k.values)
    r["pred_cal"] = r.pred * r.k
    r["cnesexecutante"] = c
    out.append(r.reset_index())
D = pd.concat(out, ignore_index=True).dropna(subset=["pred_cal"]).sort_values(["cnesexecutante", "dia"]).reset_index(drop=True)
D["ano"] = D.dia.dt.year

def wape(df, c): return (df.y - df[c]).abs().sum() / df.y.sum()
print("CNES-dias: %d | faltas/dia media %.2f" % (len(D), D.y.mean()))
print("WAPE: baseline %.3f | modelo bruto %.3f | modelo recalibrado %.3f" % (wape(D, "pred_base"), wape(D, "pred"), wape(D, "pred_cal")))
print("por ano: " + " | ".join("%d base %.3f / cal %.3f" % (a, wape(g, "pred_base"), wape(g, "pred_cal")) for a, g in D.groupby("ano")))
print("vies (prev/real): base %.3f | cal %.3f" % (D.pred_base.sum() / D.y.sum(), D.pred_cal.sum() / D.y.sum()))

def emp(col, q):
    return D.groupby("cnesexecutante").apply(
        lambda g: (g.y - g[col]).shift(3).rolling(120, min_periods=40).quantile(q)).reset_index(level=0, drop=True)
for col in ("pred_base", "pred_cal"):
    D["lo_" + col] = D[col] + emp(col, 0.10)
    D["hi_" + col] = D[col] + emp(col, 0.90)
    D["q20_" + col] = D[col] + emp(col, 0.20)
E = D.dropna(subset=["lo_pred_base", "lo_pred_cal"])
print("\nusando %d CNES-dias com janela de residuos completa" % len(E))
print("WAPE nesse conjunto: base %.3f | cal %.3f" % (wape(E, "pred_base"), wape(E, "pred_cal")))
for col, nome in (("pred_base", "baseline"), ("pred_cal", "modelo recalibrado")):
    cob = ((E.y >= E["lo_" + col]) & (E.y <= E["hi_" + col])).mean()
    lar = (E["hi_" + col] - E["lo_" + col]).mean()
    b = np.floor(E["q20_" + col].clip(lower=0))
    print("%-20s IC80: cobertura %.3f largura %.2f | limite p20: P(y>=lim)=%.3f, limite medio %.2f, seguro=%.1f%% das faltas" % (
        nome, cob, lar, (E.y >= b).mean(), b.mean(), 100 * b.sum() / E.y.sum()))

pc = E.groupby("cnesexecutante").apply(lambda g: pd.Series({
    "faltas_dia": g.y.mean(), "WAPE_base": wape(g, "pred_base"), "WAPE_cal": wape(g, "pred_cal")})).sort_values("faltas_dia", ascending=False)
pc["melhor"] = np.where(pc.WAPE_cal < pc.WAPE_base, "modelo", "base")
print("\n", pc.round(3).to_string())
print("\nCNES onde o modelo recalibrado ganha: %d de %d" % ((pc.melhor == "modelo").sum(), len(pc)))
