"""Painel final: top 20 CNES executantes, semanal, 2019-01-07 a ultima semana completa.

Medidas por (cnesexecutante, semana da consulta):
  n_faltas                 eventos (consultas agendadas nao comparecidas)
  n_agendadas              consultas agendadas (populacao em risco)
  incidencia_1000          faltas por 1.000 consultas agendadas (risco de falta)
  n_pac_agendados          pacientes distintos com consulta agendada na semana
  n_pac_faltosos           pacientes distintos com >=1 falta na semana
  incid_pac_1000           pacientes faltosos por 1.000 pacientes agendados
"""
import pandas as pd

SRC = "gercon_poa_censo_distancias.parquet"
INICIO = pd.Timestamp("2019-01-07")       # 2018 = ramp-up do registro
POS_2020 = pd.Timestamp("2020-03-16")     # inicio da pandemia no RS

top = pd.read_csv("ranking_top20.csv", dtype={"cnesexecutante": str})["cnesexecutante"].tolist()

d = pd.read_parquet(SRC, columns=["idpaciente", "datahora_consulta", "situacaoagenda", "cnesexecutante"])
d = d[d.cnesexecutante.isin(top)].copy()
d["semana"] = pd.to_datetime(d["datahora_consulta"]).dt.to_period("W-SUN").dt.start_time
d["falta"] = d["situacaoagenda"] == "FALTANTE"

ult_dia = pd.to_datetime(d["datahora_consulta"]).max()
ult_sem_completa = ult_dia.to_period("W-SUN").start_time
if ult_dia.dayofweek != 4 and ult_dia.dayofweek != 6:  # termina antes de sexta -> semana parcial
    ult_sem_completa -= pd.Timedelta(days=7)
d = d[(d.semana >= INICIO) & (d.semana <= ult_sem_completa)]

g = d.groupby(["cnesexecutante", "semana"], observed=True)
t = g.agg(n_agendadas=("falta", "size"), n_faltas=("falta", "sum"),
          n_pac_agendados=("idpaciente", "nunique")).reset_index()
f = (d[d.falta].groupby(["cnesexecutante", "semana"], observed=True)["idpaciente"]
       .nunique().rename("n_pac_faltosos").reset_index())
t = t.merge(f, how="left", on=["cnesexecutante", "semana"])

# grade completa por CNES (semanas sem agenda = 0, marcadas), dentro da vida ativa
partes = []
for c, x in t.groupby("cnesexecutante"):
    idx = pd.date_range(x.semana.min(), x.semana.max(), freq="7D", name="semana")
    x = x.set_index("semana").reindex(idx)
    x["cnesexecutante"] = c
    partes.append(x.reset_index())
p = pd.concat(partes, ignore_index=True)
num = ["n_agendadas", "n_faltas", "n_pac_agendados", "n_pac_faltosos"]
p["sem_agenda"] = p["n_agendadas"].isna()
p[num] = p[num].fillna(0).astype("int32")
p["incidencia_1000"] = (1000 * p.n_faltas / p.n_agendadas).where(p.n_agendadas > 0)
p["incid_pac_1000"] = (1000 * p.n_pac_faltosos / p.n_pac_agendados).where(p.n_pac_agendados > 0)
p["pos_2020"] = (p.semana >= POS_2020).astype("int8")
p = p.sort_values(["cnesexecutante", "semana"]).reset_index(drop=True)
p.to_parquet("painel_final.parquet", index=False)
p.to_csv("painel_final.csv", index=False)

print("linhas:", len(p), "| CNES:", p.cnesexecutante.nunique())
print("semanas:", p.semana.min().date(), "->", p.semana.max().date(), "| ultimo dia dos dados:", ult_dia.date())
print("total faltas:", int(p.n_faltas.sum()), "| semanas sem agenda:", int(p.sem_agenda.sum()))
a = p.groupby("semana")[["n_faltas", "n_agendadas", "n_pac_faltosos"]].sum()
a["incid_1000"] = 1000 * a.n_faltas / a.n_agendadas
print("\nAgregado, media semanal por ano:")
print(a.groupby(a.index.year).mean().round(1).to_string())
print("\nSemanas 2024-12 a 2025-03 (agregado):")
print(a.loc["2024-11-25":"2025-03-10"].round(1).to_string())
