"""Painel semanal de faltas por CNES executante.

Unidade: (cnesexecutante, semana). Semana = segunda-feira (W-MON, rotulada pelo inicio),
definida pela data da consulta agendada (datahora_consulta).
Saida: painel_faltas_semanal.parquet (formato longo, grade completa por CNES).
"""

# Pre processamento do painel semanal de faltas por CNES executante.

import pandas as pd

SRC = "gercon_poa_censo_distancias.parquet"
OUT = "painel_faltas_semanal.parquet"

d = pd.read_parquet(SRC, columns=["datahora_consulta", "situacaoagenda", "cnesexecutante"])
d["datahora_consulta"] = pd.to_datetime(d["datahora_consulta"])
d["semana"] = d["datahora_consulta"].dt.to_period("W-SUN").dt.start_time  # inicio na segunda
d["falta"] = (d["situacaoagenda"] == "FALTANTE").astype("int32")

g = d.groupby(["cnesexecutante", "semana"], observed=True).agg(
    n_agendadas=("falta", "size"),
    n_faltas=("falta", "sum"),
).reset_index()
g["n_realizadas"] = g["n_agendadas"] - g["n_faltas"]

# Grade completa: semanas sem registro dentro da vida ativa do CNES viram 0 (nao NaN).
# Antes da 1a e depois da ultima semana com agenda o CNES nao e observado (nao e zero).
partes = []
for cnes, x in g.groupby("cnesexecutante"):
    idx = pd.date_range(x["semana"].min(), x["semana"].max(), freq="7D", name="semana")
    x = x.set_index("semana").reindex(idx)
    x["cnesexecutante"] = cnes
    partes.append(x.reset_index())
p = pd.concat(partes, ignore_index=True)
cols = ["n_agendadas", "n_faltas", "n_realizadas"]
p["sem_agenda"] = p["n_agendadas"].isna()  # semana inativa dentro do periodo ativo
p[cols] = p[cols].fillna(0).astype("int32")
p["taxa_falta"] = (p["n_faltas"] / p["n_agendadas"]).where(p["n_agendadas"] > 0)
p = p[["cnesexecutante", "semana", "n_faltas", "n_agendadas", "n_realizadas", "taxa_falta", "sem_agenda"]]
p = p.sort_values(["cnesexecutante", "semana"]).reset_index(drop=True)

# Salva o resultado
# p.to_parquet(OUT, index=False)

print("linhas:", len(p), "| CNES:", p.cnesexecutante.nunique())
print("semanas:", p.semana.min().date(), "->", p.semana.max().date())
print("total faltas:", p.n_faltas.sum(), "(esperado 163785)")
print("semanas sem agenda (zero-preenchidas):", int(p.sem_agenda.sum()))
tot = p.groupby("semana")["n_faltas"].sum()
print("faltas/semana (agregado): media %.0f, min %d, max %d" % (tot.mean(), tot.min(), tot.max()))
n = p.groupby("cnesexecutante").size()
print("semanas ativas por CNES: mediana %d, min %d, max %d" % (n.median(), n.min(), n.max()))
