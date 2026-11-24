"""Consistencia temporal dos 221 CNES executantes: presenca por ano e continuidade semanal."""
import numpy as np
import pandas as pd

d = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["datahora_consulta", "situacaoagenda", "cnesexecutante"])
d["dia"] = pd.to_datetime(d.datahora_consulta)
d["ano"] = d.dia.dt.year
d["falta"] = (d.situacaoagenda == "FALTANTE").astype(int)
d["semana"] = d.dia.dt.to_period("W-SUN").dt.start_time
top = pd.read_csv("ranking_top20.csv", dtype={"cnesexecutante": str})["cnesexecutante"].tolist()

# ---- por ano ----
py = d.groupby(["cnesexecutante", "ano"]).size().unstack(fill_value=0)
anos = list(py.columns)
print("anos nos dados:", anos, "(2018 = ramp-up do registro; 2026 = ate 31/03)")
print("\nCNES com consultas em cada ano (qualquer volume):")
print((py > 0).sum().to_string())
n_anos = (py > 0).sum(axis=1)
print("\nDistribuicao: em quantos dos %d anos o CNES aparece" % len(anos))
print(n_anos.value_counts().sort_index().to_string())

# ---- continuidade semanal (jan/2019 a mar/2026) ----
sem = pd.date_range("2019-01-07", "2026-03-23", freq="7D")
w = d[(d.semana >= sem[0]) & (d.semana <= sem[-1])].groupby(["cnesexecutante", "semana"]).size().unstack(fill_value=0)
w = w.reindex(columns=sem, fill_value=0)
ativa = (w > 0)
def maior_lacuna(v):
    m = c = 0
    for x in v:
        c = 0 if x else c + 1
        m = max(m, c)
    return m
cons = pd.DataFrame({
    "n_anos": n_anos,
    "consultas": py.sum(axis=1),
    "pct_semanas_ativas": ativa.mean(axis=1),
    "maior_lacuna_sem": ativa.apply(lambda r: maior_lacuna(r.values), axis=1),
    "primeira_sem": ativa.apply(lambda r: sem[np.argmax(r.values)] if r.any() else pd.NaT, axis=1),
    "ultima_sem": ativa.apply(lambda r: sem[len(r) - 1 - np.argmax(r.values[::-1])] if r.any() else pd.NaT, axis=1),
}).reindex(py.index)
cons["top20"] = cons.index.isin(top)
cons["faltas"] = d.groupby("cnesexecutante").falta.sum()

todos = cons[cons.n_anos == len(anos)]
print("\nCNES presentes em TODOS os %d anos: %d de %d" % (len(anos), len(todos), len(cons)))

# volume minimo por ano (2018 e 2026 sao parciais: exigir menos)
min_ok = (py.drop(columns=[2018, 2026]) >= 100).all(axis=1) & (py[2018] > 0) & (py[2026] > 0)
print("... e com >=100 consultas em cada ano pleno (2019-2025): %d" % min_ok.sum())

print("\nCRITERIOS DE CONSISTENCIA (jan/2019 a mar/2026, %d semanas):" % len(sem))
for nome, m in (
    ("presente nos 9 anos", cons.n_anos == len(anos)),
    ("+ ativo em >=90%% das semanas", (cons.n_anos == len(anos)) & (cons.pct_semanas_ativas >= .90)),
    ("+ nenhuma lacuna > 8 semanas", (cons.n_anos == len(anos)) & (cons.pct_semanas_ativas >= .90) & (cons.maior_lacuna_sem <= 8)),
    ("+ >=100 consultas/ano em 2019-2025", (cons.n_anos == len(anos)) & (cons.pct_semanas_ativas >= .90) & (cons.maior_lacuna_sem <= 8) & min_ok.reindex(cons.index)),
):
    s = cons[m]
    print("  %-38s %3d CNES | %5.1f%% das consultas | %5.1f%% das faltas | do top20: %d" % (
        nome.replace("%%", "%"), len(s), 100 * s.consultas.sum() / cons.consultas.sum(),
        100 * s.faltas.sum() / cons.faltas.sum(), s.top20.sum()))

cons["consistente"] = (cons.n_anos == len(anos)) & (cons.pct_semanas_ativas >= .90) & (cons.maior_lacuna_sem <= 8)
print("\nTOP 20 - situacao de consistencia:")
t = cons[cons.top20].sort_values("faltas", ascending=False)
t = t.assign(primeira=t.primeira_sem.dt.date, ultima=t.ultima_sem.dt.date)
print(t[["n_anos", "pct_semanas_ativas", "maior_lacuna_sem", "primeira", "ultima", "consistente"]].round(2).to_string())

# ---- mudanca de regime dentro de cada CNES do top 20: volume e taxa por ano ----
print("\nTOP 20 - taxa de falta por ano (%) [volume = consultas/ano em centenas entre colchetes]")
tx = d[d.cnesexecutante.isin(top)].groupby(["cnesexecutante", "ano"]).agg(ag=("falta", "size"), f=("falta", "sum"))
tab = (100 * tx.f / tx.ag).unstack().round(0)
vol = (tx.ag / 100).unstack().round(0)
out = tab.astype("Int64").astype(str) + " [" + vol.astype("Int64").astype(str) + "]"
out = out.reindex(t.index)
print(out.to_string())

cons.to_csv("consistencia_cnes.csv")
print("\nsalvo: consistencia_cnes.csv")
