"""Coorte 2019-2026 restrita as CNES sob gestao da Prefeitura de Porto Alegre (flag_municipal.csv).

Mesmo criterio do 10_coorte_2019_2026.py, sem alterar limiares:
  coorte = consultas em todos os anos 2019..2026, >=100 consultas/ano em 2019-2025, ativo em >=90% das semanas.
  quebra de regime = segmentacao binaria na serie MENSAL, G>100, |dif de taxa|>=5 p.p., segmentos >=6 meses.
Diferenca: so entram CNES com municipal==True (natureza Municipio + gestor 431490 + razao social da prefeitura).
Saida: coorte_municipal_2019_2026.csv  (mesmas colunas do coorte_2019_2026.csv)
"""
import numpy as np
import pandas as pd

mun = pd.read_csv("flag_municipal.csv", dtype={"cnesexecutante": str})
MUN = set(mun.cnesexecutante[mun.municipal])
d = pd.read_parquet("gercon_poa_censo_distancias_com_natureza_juridica.parquet",
                    columns=["datahora_consulta", "situacaoagenda", "cnesexecutante"])
d = d[d.cnesexecutante.isin(MUN)].copy()
d["dia"] = pd.to_datetime(d.datahora_consulta)
d = d[(d.dia >= "2019-01-01") & (d.dia <= "2026-03-31")].copy()
d["ano"] = d.dia.dt.year
d["falta"] = (d.situacaoagenda == "FALTANTE").astype(int)
d["semana"] = d.dia.dt.to_period("W-SUN").dt.start_time
d["mes"] = d.dia.dt.to_period("M")

py = d.groupby(["cnesexecutante", "ano"]).size().unstack(fill_value=0)
sem = pd.date_range("2019-01-07", "2026-03-23", freq="7D")
w = (d[d.semana.between(sem[0], sem[-1])].groupby(["cnesexecutante", "semana"]).size().unstack(fill_value=0)
       .reindex(index=py.index, columns=sem, fill_value=0) > 0)
def lacuna(v):
    m = c = 0
    for x in v:
        c = 0 if x else c + 1
        m = max(m, c)
    return m
c = pd.DataFrame({"consultas": py.sum(axis=1), "pct_sem": w.mean(axis=1), "lacuna": w.apply(lambda r: lacuna(r.values), axis=1)})
c["faltas"] = d.groupby("cnesexecutante").falta.sum()
c["todos_anos"] = (py > 0).all(axis=1)
c["vol100"] = (py.drop(columns=[2026]) >= 100).all(axis=1)
c["coorte"] = c.todos_anos & c.vol100 & (c.pct_sem >= .90)

tot_c, tot_f = c.consultas.sum(), c.faltas.sum()
print("MUNICIPAIS 2019-01 a 2026-03 | %d CNES com consulta no periodo (de %d municipais) | %d consultas | %d faltas" % (len(c), len(MUN), tot_c, tot_f))
for nome, m in (("presentes nos 8 anos", c.todos_anos), ("+ >=100 consultas/ano (2019-25)", c.todos_anos & c.vol100), ("+ ativo >=90% das semanas = COORTE", c.coorte)):
    s = c[m]
    print("  %-36s %3d CNES | %4.1f%% consultas | %4.1f%% faltas" % (nome, len(s), 100 * s.consultas.sum() / tot_c, 100 * s.faltas.sum() / tot_f))

M = d.groupby(["cnesexecutante", "mes"]).agg(f=("falta", "sum"), n=("falta", "size"))
def ll(F, N):
    p = np.where(N > 0, F / np.maximum(N, 1), 0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return F * np.log(p) + (N - F) * np.log(1 - p)
def segmenta(f, n, lo, hi, out, min_len=6):
    if hi - lo < 2 * min_len:
        return
    cf, cn = np.r_[0, np.cumsum(f)], np.r_[0, np.cumsum(n)]
    F, N = cf[hi] - cf[lo], cn[hi] - cn[lo]
    base = ll(np.array(F), np.array(N))
    best = (0, None)
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
meses = pd.period_range("2019-01", "2026-03", freq="M")
res = []
for cn_ in c[c.coorte].sort_values("faltas", ascending=False).index:
    x = M.loc[cn_].reindex(meses, fill_value=0)
    cuts = []
    segmenta(x.f.values, x.n.values, 0, len(meses), cuts)
    cuts = sorted(cuts)
    bounds = [0] + cuts + [len(meses)]
    segs, taxas = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        N = x.n.values[a:b].sum()
        t_ = x.f.values[a:b].sum() / max(N, 1)
        taxas.append(t_)
        segs.append("%s>%s:%2.0f%%" % (meses[a].strftime("%y-%m"), meses[b - 1].strftime("%y-%m"), 100 * t_))
    res.append({"cnes": cn_, "faltas": c.faltas[cn_], "n_quebras": len(cuts),
                "ultimo_regime_inicio": str(meses[cuts[-1]]) if cuts else str(meses[0]),
                "meses_ultimo_regime": len(meses) - (cuts[-1] if cuts else 0),
                "so_pandemia": bool(cuts) and all(pd.Period("2020-03", "M") <= meses[t] <= pd.Period("2021-09", "M") for t in cuts),
                "taxa_ultimo_regime": taxas[-1], "regimes": " | ".join(segs)})
R = pd.DataFrame(res).set_index("cnes")
R["estavel"] = (R.n_quebras == 0)
print("\nCOORTE MUNICIPAL: %d CNES | sem quebra: %d | so pandemia: %d | com quebra fora da pandemia: %d" % (
    len(R), R.estavel.sum(), R.so_pandemia.sum(), ((R.n_quebras > 0) & ~R.so_pandemia).sum()))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(R[["faltas", "n_quebras", "regimes"]].to_string())
R.join(c[["consultas", "pct_sem", "lacuna"]]).to_csv("coorte_municipal_2019_2026.csv")
print("\nsalvo: coorte_municipal_2019_2026.csv")
