"""Tabela de risco de faltas por CNES e dia (decisao 3 dias antes, agenda conhecida).

Uso (producao):  python prever_dia.py --agenda agenda.parquet [--saida risco_previsto.csv]
Uso (teste):     python prever_dia.py --demo 10 --modelo modelo_demo.joblib   (usa os ultimos 10 dias da base como se fossem a agenda)

agenda = consultas agendadas (mesmas colunas da base: cnesexecutante, datahora_consulta, historico do paciente ate t0,
         etc.); NAO precisa de situacaoagenda. Historico recente (ultimos 28 dias ate D-3) vem de --base.

Por nivel: A = modelo por paciente (p_i recalibrado pelo nivel recente do CNES) | B = baseline (taxa 28d x agenda).
Niveis C e D (fora da coorte) NAO recebem previsao. A dispersao extra (phi) vem de calibracao.json.
A tabela DESCREVE o risco (P(faltas >= k)); a decisao de quantos chamar e do gestor.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd

KS = (1, 2, 3, 4, 5, 6, 8, 10)
ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default="modelo_final.joblib")
ap.add_argument("--base", default="gercon_poa_censo_distancias.parquet")
ap.add_argument("--agenda", default=None)
ap.add_argument("--demo", type=int, default=None)
ap.add_argument("--niveis", default="niveis_cnes.csv")
ap.add_argument("--calibracao", default="calibracao.json")
ap.add_argument("--saida", default="risco_previsto.csv")
ap.add_argument("--ns", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)

bund = joblib.load(args.modelo)
model = bund["model"]
phi = json.load(open(args.calibracao))
niv = pd.read_csv(args.niveis, dtype={"cnes": str}).set_index("cnes").nivel

def preparar(df):
    df = df.copy()
    df["taxa_falta_prev"] = df.n_faltas_previas / df.n_consultas_previas.where(df.n_consultas_previas > 0)
    X = df[bund["NUM"] + bund["CAT"]].copy()
    for c in bund["CAT"]:
        X[c] = pd.Categorical(X[c].astype(object), categories=bund["cats"][c])
    return X

base = pd.read_parquet(args.base)
base["dia"] = pd.to_datetime(base.datahora_consulta)
base["y"] = (base.situacaoagenda == "FALTANTE").astype(int)
if args.agenda:
    ag = pd.read_parquet(args.agenda) if args.agenda.endswith(".parquet") else pd.read_csv(args.agenda, dtype={"cnesexecutante": str})
    ag["dia"] = pd.to_datetime(ag.datahora_consulta)
    ag["y_real"] = np.nan
elif args.demo:
    ag = base[base.dia > base.dia.max() - pd.Timedelta(days=args.demo)].copy()
    ag["y_real"] = ag.y
    if pd.Timestamp(bund["treino_ate"]) >= ag.dia.min():
        print("AVISO: o modelo foi treinado ate %s e ja viu parte destes dias; avaliacao otimista. Use --modelo treinado antes." % bund["treino_ate"])
else:
    raise SystemExit("informe --agenda arquivo ou --demo N")

ag["cnesexecutante"] = ag.cnesexecutante.astype(str)
ag["nivel"] = ag.cnesexecutante.map(niv).fillna("D")
fora = ag[~ag.nivel.isin(["A", "B"])]
ag = ag[ag.nivel.isin(["A", "B"])].copy()
datas = sorted(ag.dia.unique())
print("Agenda: %d consultas, %d CNES (A/B) em %d dia(s); %d consultas de CNES fora dos niveis A/B (%d CNES) sem previsao." % (
    len(ag), ag.cnesexecutante.nunique(), len(datas), len(fora), fora.cnesexecutante.nunique()))

# historico para o fator k (nivel A): pontuar com o modelo as linhas dos ultimos 28 dias ate D-3
ini = pd.Timestamp(min(datas)) - pd.Timedelta(days=30)
fim = pd.Timestamp(max(datas)) - pd.Timedelta(days=3)
hist = base[(base.dia >= ini) & (base.dia <= fim) & base.cnesexecutante.isin(niv.index[niv == "A"])].copy()
hist["p"] = model.predict_proba(preparar(hist))[:, 1] if len(hist) else []
ag_A = ag[ag.nivel == "A"].copy()
ag_A["p"] = model.predict_proba(preparar(ag_A))[:, 1] if len(ag_A) else []

linhas = []
for (c, t), g in ag.groupby(["cnesexecutante", "dia"]):
    n_ag = len(g)
    nivel = g.nivel.iloc[0]
    jan = base[(base.cnesexecutante == c) & (base.dia >= t - pd.Timedelta(days=30)) & (base.dia <= t - pd.Timedelta(days=3))]
    aviso = ""
    if nivel == "A":
        h = hist[(hist.cnesexecutante == c) & (hist.dia >= t - pd.Timedelta(days=30)) & (hist.dia <= t - pd.Timedelta(days=3))]
        sp, sy = h.p.sum(), h.y.sum()
        if sp < 5:
            k, aviso = 1.0, "historico_curto(k=1)"
        else:
            k = sy / sp
            if not 0.6 <= k <= 1.6:
                aviso = "k_atipico(%.2f)" % k
        p = np.clip(ag_A.loc[g.index, "p"].values * k, 0, 0.995)
        ph = max(phi["A_modelo"], 1e-6)
        gm = rng.gamma(1 / ph, ph, size=args.ns)
        pr = np.clip(p[:, None] * gm[None, :], 0, 0.995)
        sims = (rng.random(pr.shape) < pr).sum(0)
        esp, metodo = float(p.sum()), "modelo"
    else:
        if len(jan) < 30:
            aviso = "historico_curto"
        taxa = jan.y.sum() / max(len(jan), 1)
        esp = taxa * n_ag
        ph = max(phi["B_baseline"], 1e-6)
        sims = rng.poisson(esp * rng.gamma(1 / ph, ph, size=args.ns))
        metodo = "baseline"
    r = {"cnes": c, "data": pd.Timestamp(t).date(), "nivel": nivel, "metodo": metodo, "agendadas": n_ag,
         "esperado": round(esp, 2), "q10": np.quantile(sims, .1), "q50": np.quantile(sims, .5), "q90": np.quantile(sims, .9)}
    for kk in KS:
        r[f"P>={kk}"] = round((sims >= kk).mean(), 3)
    r["aviso"] = aviso
    r["faltas_reais"] = g.y_real.sum() if g.y_real.notna().any() else np.nan
    linhas.append(r)
out = pd.DataFrame(linhas).sort_values(["data", "nivel", "esperado"], ascending=[True, True, False])
out.to_csv(args.saida, index=False)
ult = out[out.data == out.data.max()]
pd.set_option("display.width", 220)
print("\nTabela de risco - %s (todas as datas em %s):" % (ult.data.iloc[0], args.saida))
print(ult.drop(columns=["data", "metodo"]).to_string(index=False))
if out.faltas_reais.notna().any():
    o = out.dropna(subset=["faltas_reais"])
    print("\nCONFERENCIA (demo, %d CNES-dias): faltas esperadas %.0f | reais %.0f | dentro de [q10,q90]: %.0f%%" % (
        len(o), o.esperado.sum(), o.faltas_reais.sum(), 100 * ((o.faltas_reais >= o.q10) & (o.faltas_reais <= o.q90)).mean()))
