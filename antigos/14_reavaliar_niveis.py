"""Reavaliacao dos niveis de consistencia dos CNES (rodar a cada trimestre, ou quando chegar base nova).

Uso: python 14_reavaliar_niveis.py [--ate AAAA-MM-DD] [--base arquivo.parquet] [--saida niveis_cnes.csv]

Coorte: consultas em todos os anos 2019..ano de corte; >=100 consultas por ano completo; ativo em >=90% das
semanas completas desde 2019-01-07. Quebra de regime (taxa de falta mensal): segmentacao binaria, G>100,
|dif|>=5 p.p., segmentos >=6 meses.
Niveis: A = coorte sem quebra | B = coorte com 1-2 quebras sem colapso | C = coorte instavel (>=3 quebras, ou
colapso da taxa: regime <4% depois de regime >=10%) | D = fora da coorte.
Saidas: niveis_cnes.csv, historico_niveis/niveis_<data>.csv e lista de CNES que mudaram de nivel.
"""
import argparse
import os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="gercon_poa_censo_distancias.parquet")
ap.add_argument("--ate", default=None)
ap.add_argument("--saida", default="niveis_cnes.csv")
args = ap.parse_args()

d = pd.read_parquet(args.base, columns=["datahora_consulta", "situacaoagenda", "cnesexecutante"])
d["dia"] = pd.to_datetime(d.datahora_consulta)
ate = pd.Timestamp(args.ate) if args.ate else d.dia.max()
d = d[(d.dia >= "2019-01-01") & (d.dia <= ate)].copy()
d["ano"] = d.dia.dt.year
d["falta"] = (d.situacaoagenda == "FALTANTE").astype(int)
d["semana"] = d.dia.dt.to_period("W-SUN").dt.start_time
d["mes"] = d.dia.dt.to_period("M")

ult_sem = ate.to_period("W-SUN").start_time
if ate.dayofweek < 6:
    ult_sem -= pd.Timedelta(days=7)
ult_mes = ate.to_period("M") if ate.is_month_end else ate.to_period("M") - 1
anos = list(range(2019, ate.year + 1))
ano_pleno = [a for a in anos if a < ate.year or (ate.month == 12 and ate.day == 31)]

py = d.groupby(["cnesexecutante", "ano"]).size().unstack(fill_value=0).reindex(columns=anos, fill_value=0)
sem = pd.date_range("2019-01-07", ult_sem, freq="7D")
w = (d[d.semana.between(sem[0], sem[-1])].groupby(["cnesexecutante", "semana"]).size().unstack(fill_value=0)
       .reindex(index=py.index, columns=sem, fill_value=0) > 0)
res = pd.DataFrame(index=py.index)
res["consultas"] = py.sum(axis=1)
res["faltas"] = d.groupby("cnesexecutante").falta.sum()
res["todos_anos"] = (py > 0).all(axis=1)
res["vol100"] = (py[ano_pleno] >= 100).all(axis=1)
res["pct_sem_ativas"] = w.mean(axis=1).round(3)
res["coorte"] = res.todos_anos & res.vol100 & (res.pct_sem_ativas >= .90)

def ll(F, N):
    p = np.clip(np.where(N > 0, F / np.maximum(N, 1), 0), 1e-9, 1 - 1e-9)
    return F * np.log(p) + (N - F) * np.log(1 - p)
def segmenta(f, n, lo, hi, out, min_len=6):
    if hi - lo < 2 * min_len:
        return
    cf, cn = np.r_[0, np.cumsum(f)], np.r_[0, np.cumsum(n)]
    F, N = cf[hi] - cf[lo], cn[hi] - cn[lo]
    base_ll = ll(np.array(F), np.array(N))
    best = (0, None, 0, 0)
    for t in range(lo + min_len, hi - min_len + 1):
        F1, N1, F2, N2 = cf[t] - cf[lo], cn[t] - cn[lo], cf[hi] - cf[t], cn[hi] - cn[t]
        if N1 == 0 or N2 == 0:
            continue
        g = 2 * (ll(np.array(F1), np.array(N1)) + ll(np.array(F2), np.array(N2)) - base_ll)
        if g > best[0]:
            best = (g, t, F1 / N1, F2 / N2)
    if best[1] is not None and best[0] > 100 and abs(best[2] - best[3]) >= 0.05:
        out.append(best[1])
        segmenta(f, n, lo, best[1], out, min_len)
        segmenta(f, n, best[1], hi, out, min_len)

M = d[d.mes <= ult_mes].groupby(["cnesexecutante", "mes"]).agg(f=("falta", "sum"), n=("falta", "size"))
meses = pd.period_range("2019-01", ult_mes, freq="M")
info = {}
for c in res.index[res.coorte]:
    x = M.loc[c].reindex(meses, fill_value=0)
    cuts = []
    segmenta(x.f.values, x.n.values, 0, len(meses), cuts)
    cuts = sorted(cuts)
    b = [0] + cuts + [len(meses)]
    taxas = [x.f.values[a:z].sum() / max(x.n.values[a:z].sum(), 1) for a, z in zip(b[:-1], b[1:])]
    info[c] = {"n_quebras": len(cuts), "inicio_ultimo_regime": str(meses[cuts[-1]] if cuts else meses[0]),
               "meses_no_ultimo_regime": len(meses) - (cuts[-1] if cuts else 0),
               "taxa_ultimo_regime": round(taxas[-1], 3), "taxa_min_regime": round(min(taxas), 3), "taxa_max_regime": round(max(taxas), 3)}
res = res.join(pd.DataFrame(info).T)
def nivel(r):
    if not r.coorte:
        return "D"
    if r.n_quebras == 0:
        return "A"
    if r.n_quebras >= 3 or (r.taxa_min_regime < 0.04 and r.taxa_max_regime >= 0.10):
        return "C"
    return "B"
res["nivel"] = res.apply(nivel, axis=1)
res.index.name = "cnes"
res = res.sort_values(["nivel", "faltas"], ascending=[True, False])

os.makedirs("historico_niveis", exist_ok=True)
res.to_csv(f"historico_niveis/niveis_{ate:%Y-%m-%d}.csv")
anterior = None
if os.path.exists(args.saida):
    anterior = pd.read_csv(args.saida, dtype={"cnes": str}).set_index("cnes").nivel
res.to_csv(args.saida)

print("Corte: %s | semanas completas ate %s | meses completos ate %s" % (ate.date(), ult_sem.date(), ult_mes))
print("Niveis:", res.nivel.value_counts().sort_index().to_dict())
print("Cobertura dos niveis A+B: %.1f%% das consultas, %.1f%% das faltas" % (
    100 * res[res.nivel.isin(["A", "B"])].consultas.sum() / res.consultas.sum(),
    100 * res[res.nivel.isin(["A", "B"])].faltas.sum() / res.faltas.sum()))
for n in ("A", "B", "C"):
    print("nivel %s: %s" % (n, ", ".join(res.index[res.nivel == n])))
if anterior is not None:
    ch = pd.DataFrame({"antes": anterior, "agora": res.nivel}).dropna()
    ch = ch[ch.antes != ch.agora]
    print("\nMUDANCAS DE NIVEL desde a versao anterior:", "nenhuma" if ch.empty else "")
    if not ch.empty:
        print(ch.to_string())
