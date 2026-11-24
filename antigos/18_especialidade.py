"""Em qual especialidademae ocorrera a falta? Desagrega, por CNES-dia, o risco do modelo por paciente (V1, script 11).

Para cada CNES-dia (coorte A+B, 2023-2026) e especialidademae agendada nesse dia:
  esperado = soma das probabilidades de falta das consultas da especialidade, x k (o mesmo fator de recalibracao
             por CNES-dia do script 11: k = esperado_CNES-dia / soma das probabilidades, lido de risco_diario_coorte.csv)
  P(>=1)   = 1 - prod(1 - p_i)  (Poisson-binomial, exato)
Comparadores (mesma informacao disponivel em t-3, mesmas celulas):
  base_esp  = taxa de falta da especialidade na propria CNES nos 28 dias anteriores (defasagem 3) x agenda da especialidade
  base_part = esperado total do CNES-dia (V1) repartido pela fatia de agenda da especialidade (sem saber o risco individual)
Metricas: Brier, log-loss e AUC de P(>=1) por celula; calibracao; e, nos CNES-dias com falta, em quais especialidades
o modelo acerta: hit@1 (a especialidade no topo do ranking teve falta), recall@3 (fracao das faltas que cai nas 3 do topo).
Saidas: previsao_especialidade.csv, especialidade_metricas.csv, especialidade_por_esp.csv, especialidade_calibracao.csv
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC = "gercon_poa_censo_distancias.parquet"
LAG = 3
R = pd.read_csv("risco_diario_coorte.csv", dtype={"cnesexecutante": str})
R["dia"] = pd.to_datetime(R.dia)
raw = pd.read_parquet(SRC, columns=["cnesexecutante", "datahora_consulta", "situacaoagenda", "especialidademae"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
with open("p_modelos.pkl", "rb") as fh:
    P = pickle.load(fh)
assert len(P["V1"]) == len(raw)
raw["p"] = P["V1"]

# --- fator k por CNES-dia (recuperado: esperado / soma das probabilidades) ---
c = raw[raw.p.notna()].groupby(["cnesexecutante", "dia"]).p.sum().rename("soma_p").reset_index()
R = R.merge(c, on=["cnesexecutante", "dia"], how="left")
R["k"] = R.esperado / R.soma_p
assert R.k.notna().all(), "CNES-dias sem probabilidade"
print("CNES-dias:", len(R), "| k mediano %.3f" % R.k.median())

# --- celulas CNES-dia-especialidade (so os CNES-dias avaliados) ---
d = raw[raw.p.notna()].merge(R[["cnesexecutante", "dia", "k", "esperado"]], on=["cnesexecutante", "dia"])
d["pk"] = np.clip(d.p * d.k, 0, 0.995)
d["lq"] = np.log1p(-d.pk)
cell = (d.groupby(["cnesexecutante", "dia", "especialidademae"])
        .agg(ag_esp=("y", "size"), y=("y", "sum"), esperado=("pk", "sum"), lq=("lq", "sum"), esp_cnes=("esperado", "first"))
        .reset_index())
cell["p1"] = 1 - np.exp(cell.lq)
cell["ag_cnes"] = cell.groupby(["cnesexecutante", "dia"]).ag_esp.transform("sum")
cell["base_part"] = cell.esp_cnes * cell.ag_esp / cell.ag_cnes

# --- baseline: taxa da especialidade na CNES nos 28 dias anteriores (defasagem 3), historico completo ---
H = (raw[raw.cnesexecutante.isin(R.cnesexecutante.unique())]
     .groupby(["cnesexecutante", "especialidademae", "dia"]).agg(f=("y", "sum"), a=("y", "size")).reset_index())
parts = []
for (cn, es), x in H.groupby(["cnesexecutante", "especialidademae"]):
    x = x[["dia", "f", "a"]].set_index("dia").reindex(pd.date_range(x.dia.min(), R.dia.max(), freq="D")).fillna(0)
    num = x.f.shift(LAG).rolling(28, min_periods=1).sum()
    den = x.a.shift(LAG).rolling(28, min_periods=1).sum()
    parts.append(pd.DataFrame({"cnesexecutante": cn, "especialidademae": es, "taxa28": num / den.where(den > 0)})
                 .rename_axis("dia").reset_index())
cell = cell.merge(pd.concat(parts), on=["cnesexecutante", "especialidademae", "dia"], how="left")
# especialidade sem historico recente na CNES: usa a taxa da CNES-dia (V1) como ultimo recurso
cell["taxa28"] = cell.taxa28.fillna(cell.esp_cnes / cell.ag_cnes)
cell["base_esp"] = cell.taxa28 * cell.ag_esp
cell["p1_base_esp"] = 1 - np.exp(-cell.base_esp)
cell["p1_base_part"] = 1 - np.exp(-cell.base_part)
cell["yb"] = (cell.y >= 1).astype(int)
print("celulas CNES-dia-especialidade: %d | com falta: %.1f%%" % (len(cell), 100 * cell.yb.mean()))

# --- metricas por celula ---
def met(p, nome, y=cell.yb.values):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"modelo": nome, "Brier": np.mean((p - y) ** 2), "logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "AUC": roc_auc_score(y, p), "prev_media": p.mean(), "obs_media": y.mean()}

M = pd.DataFrame([met(cell.p1.values, "V1 por paciente (agregado)"),
                  met(cell.p1_base_esp.values, "base: taxa 28d da especialidade"),
                  met(cell.p1_base_part.values, "base: total do dia x fatia da agenda")])
# AUC dentro do CNES-dia (so dias com >=2 especialidades e pelo menos uma com falta e uma sem): ranking entre especialidades
def auc_dentro(col):
    out, w = [], []
    for _, g in cell.groupby(["cnesexecutante", "dia"]):
        if g.yb.nunique() == 2:
            out.append(roc_auc_score(g.yb, g[col])); w.append(len(g))
    return np.average(out, weights=w), len(out)
for nome, col in (("V1 por paciente (agregado)", "p1"), ("base: taxa 28d da especialidade", "p1_base_esp"),
                  ("base: total do dia x fatia da agenda", "p1_base_part")):
    a, n = auc_dentro(col)
    M.loc[M.modelo == nome, "AUC_dentro_do_dia"] = a
print("\nMETRICAS POR CELULA (menor Brier/logloss e maior AUC = melhor)")
print(M.round(4).to_string(index=False))

# --- hit@1 e recall@3 nos CNES-dias com falta ---
def ranking(col):
    h1, rec3, n = [], [], 0
    for _, g in cell.groupby(["cnesexecutante", "dia"]):
        if g.y.sum() == 0:
            continue
        g = g.sort_values(col, ascending=False, kind="stable")
        h1.append(g.y.iloc[0] >= 1)
        rec3.append(g.y.iloc[:3].sum() / g.y.sum())
    return np.mean(h1), np.mean(rec3), len(h1)
rk = []
for nome, col in (("V1 por paciente (agregado)", "esperado"), ("base: taxa 28d da especialidade", "base_esp"),
                  ("base: total do dia x fatia da agenda", "base_part"), ("referencia: maior agenda", "ag_esp")):
    h1, r3, n = ranking(col)
    rk.append({"ordenacao": nome, "hit@1": h1, "recall@3": r3, "cnes_dias_com_falta": n})
RK = pd.DataFrame(rk)
print("\nRANKING DE ESPECIALIDADES NOS CNES-DIAS COM FALTA (hit@1 = a 1a do ranking teve falta; recall@3 = % das faltas nas 3 primeiras)")
print(RK.round(3).to_string(index=False))

# --- calibracao de P(>=1) por especialidade-celula (V1) ---
cell["faixa"] = pd.cut(cell.p1, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
CAL = cell.groupby("faixa", observed=True).agg(n=("yb", "size"), prev_media=("p1", "mean"), observado=("yb", "mean")).reset_index()
print("\nCALIBRACAO de P(>=1 falta na especialidade no CNES-dia), V1")
print(CAL.round(3).to_string(index=False))

# --- por especialidade (as 15 maiores) ---
pe = cell.groupby("especialidademae").agg(celulas=("yb", "size"), faltas=("y", "sum"), esperado=("esperado", "sum"),
                                          base_esp=("base_esp", "sum"), ag=("ag_esp", "sum"))
pe["vies_V1"] = pe.esperado / pe.faltas
pe["vies_base"] = pe.base_esp / pe.faltas
pe["WAPE_V1"] = cell.assign(e=(cell.y - cell.esperado).abs()).groupby("especialidademae").e.sum() / pe.faltas
pe["WAPE_base"] = cell.assign(e=(cell.y - cell.base_esp).abs()).groupby("especialidademae").e.sum() / pe.faltas
pe = pe.sort_values("faltas", ascending=False)
print("\nPOR ESPECIALIDADE (15 com mais faltas; vies = previsto/observado)")
print(pe.head(15)[["celulas", "faltas", "vies_V1", "vies_base", "WAPE_V1", "WAPE_base"]].round(3).to_string())

out = cell[["cnesexecutante", "dia", "especialidademae", "ag_esp", "y", "esperado", "p1", "base_esp", "p1_base_esp"]].copy()
out.round(4).to_csv("previsao_especialidade.csv", index=False)
M.round(4).to_csv("especialidade_metricas.csv", index=False)
pe.round(4).to_csv("especialidade_por_esp.csv")
CAL.round(4).to_csv("especialidade_calibracao.csv", index=False)
RK.round(4).to_csv("especialidade_ranking.csv", index=False)
