"""Niveis A/B/C da coorte municipal (entrada: coorte_municipal_2019_2026.csv, do script 19).

A = estavel (nenhuma quebra de regime na taxa de falta).
B = mudanca moderada: 1-2 quebras e taxa do ultimo regime >= 5%.
C = instavel: >=3 quebras OU taxa do ultimo regime < 5% (taxa quase nula = provavel artefato de registro).
A regra B/C NAO estava no codigo original (11_modelo_final.py listou B e C a mao). Foi reconstruida aqui e
VALIDADA abaixo: reproduz 26/26 niveis da coorte original (coorte_2019_2026.csv + listas do script 11).
Saida: niveis_municipal.csv
"""
import numpy as np
import pandas as pd

def nivel(n_quebras, taxa_ult):
    if n_quebras == 0:
        return "A"
    return "C" if (n_quebras >= 3 or taxa_ult < 0.05) else "B"

# --- validacao no conjunto original ---
o = pd.read_csv("coorte_2019_2026.csv", dtype={"cnes": str})
B0 = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]
C0 = ["7513151", "2262630", "2264072", "2262355", "2262819", "2262401"]
o["real"] = o.cnes.map(lambda x: "B" if x in B0 else "C" if x in C0 else "A")
o["taxa_ult"] = o.regimes.str.split(" | ", regex=False).str[-1].str.extract(r":\s*(\d+)%")[0].astype(float) / 100
o["regra"] = [nivel(q, t) for q, t in zip(o.n_quebras, o.taxa_ult)]
print("VALIDACAO (coorte original): regra reproduz %d/%d niveis" % ((o.regra == o.real).sum(), len(o)))
print(o[o.regra != o.real][["cnes", "real", "regra", "n_quebras", "taxa_ult"]].to_string(index=False) or "")

# --- aplicacao ao recorte municipal ---
m = pd.read_csv("coorte_municipal_2019_2026.csv", dtype={"cnes": str})
m["nivel"] = [nivel(q, t) for q, t in zip(m.n_quebras, m.taxa_ultimo_regime)]
nome = pd.read_parquet("gercon_poa_censo_estabelecimentos.parquet", columns=["cnesexecutante", "NO_FANTASIA"]).rename(columns={"cnesexecutante": "cnes"})
m = m.merge(nome, on="cnes", how="left")
print("\nNIVEIS NO RECORTE MUNICIPAL")
print(m.groupby("nivel").agg(cnes=("cnes", "size"), consultas=("consultas", "sum")).to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(m.sort_values(["nivel", "consultas"], ascending=[True, False])[["cnes", "nivel", "NO_FANTASIA", "consultas", "n_quebras", "taxa_ultimo_regime", "regimes"]].to_string(index=False))
m.to_csv("niveis_municipal.csv", index=False)
print("\nsalvo: niveis_municipal.csv")
