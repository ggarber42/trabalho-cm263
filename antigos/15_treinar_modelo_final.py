"""Treina o modelo global por paciente (todas as CNES, todas as linhas ate --ate) e salva o pacote para producao.

Uso: python 15_treinar_modelo_final.py [--ate AAAA-MM-DD] [--saida modelo_final.joblib]
Retreinar periodicamente (ex.: mensal ou trimestral) com a base atualizada.
"""
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="gercon_poa_censo_distancias.parquet")
ap.add_argument("--ate", default=None)
ap.add_argument("--saida", default="modelo_final.joblib")
args = ap.parse_args()

NUM = ["idade", "n_consultas_previas", "n_faltas_previas", "taxa_falta_prev", "primeira_vez",
       "n_consultas_previas_especialidademae", "n_faltas_previas_especialidademae",
       "n_consultas_previas_motivo", "n_faltas_previas_motivo", "dias_ate_consulta",
       "favela_comunidade", "taxa_alf_geral", "renda_media_winsor", "distancia_km", "tempo_caminhando",
       "distancia_km_trans_publico", "tempo_desloc_trans_publico", "quant_bus", "baldiacoes"]
CAT = ["falta_recente", "ultima_previa_mesma_especialidademae", "ultima_previa_mesmo_motivo",
       "especialidademae", "cid_1dig", "possuiunidadeindicada", "dia_semana", "mes_consulta",
       "corregulador", "coordenadoria_saude", "distrito_sanitario", "status", "cnesexecutante"]
t0 = time.time()
raw = pd.read_parquet(args.base)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
ate = pd.Timestamp(args.ate) if args.ate else raw.dia.max()
raw = raw[raw.dia <= ate].copy()
raw["taxa_falta_prev"] = raw.n_faltas_previas / raw.n_consultas_previas.where(raw.n_consultas_previas > 0)
y = (raw.situacaoagenda == "FALTANTE").astype(int).values
X = raw[NUM + CAT].copy()
for c in CAT:
    X[c] = X[c].astype("category")
cats = {c: list(X[c].cat.categories) for c in CAT}
m = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, categorical_features="from_dtype", random_state=0)
m.fit(X, y)
joblib.dump({"model": m, "NUM": NUM, "CAT": CAT, "cats": cats, "treino_ate": str(ate.date()), "n_treino": int(len(y))}, args.saida)
print("modelo salvo em %s | treino ate %s | %d linhas | %d iteracoes | %.0fs" % (args.saida, ate.date(), len(y), m.n_iter_, time.time() - t0))
