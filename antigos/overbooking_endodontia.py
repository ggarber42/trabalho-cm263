"""Simulacao de overbooking, endodontia CNES 2237571: chamar m pacientes extras = limite inferior da previsao de faltas.
Conflito (excesso) ocorre quando faltas observadas < m (comparecem mais pessoas do que vagas)."""
import pickle, numpy as np, pandas as pd
CN, ESP = "2237571", "ODONTO ENDODONTIA"
R = pd.read_csv("risco_diario_coorte.csv", dtype={"cnesexecutante": str}); R["dia"] = pd.to_datetime(R.dia)
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["cnesexecutante","datahora_consulta","situacaoagenda","especialidademae"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int); raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["p"] = pickle.load(open("p_modelos.pkl","rb"))["V1"]
c = raw[(raw.cnesexecutante == CN) & raw.p.notna()]
r = R[R.cnesexecutante == CN].set_index("dia").join(c.groupby("dia").p.sum().rename("soma_p")); r["k"] = r.esperado / r.soma_p
c = c.join(r.k, on="dia"); c = c[c.k.notna() & (c.especialidademae == ESP)].copy()
c["pk"] = np.clip(c.p * c.k, 0, .995); c["sem"] = c.dia.dt.to_period("W-SUN").dt.end_time.dt.normalize()
def pmf(p):
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
ALF = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
def avalia(chave, nome):
    L = []
    for _, g in c.groupby(chave):
        cdf = np.cumsum(pmf(g.pk.values)); L.append((g.y.sum(), [int(np.searchsorted(cdf, a)) for a in ALF], g.pk.sum()))
    y = np.array([l[0] for l in L]); M = np.array([l[1] for l in L]); esp = np.array([l[2] for l in L])
    print("\n%s: %d periodos | faltas %d | media %.2f faltas/periodo" % (nome, len(y), y.sum(), y.mean()))
    print("%-34s %10s %12s %12s %14s" % ("regra: m extras = ", "sem excesso", "extras/periodo", "faltas aproveit.", "excesso medio*"))
    for j, a in enumerate(ALF):
        m = M[:, j]; conf = y < m
        print("quantil %2d%% do previsto (P(sem excesso)>=%d%%) %9.1f%% %12.2f %13.0f%% %14.2f" %
              (100*a, 100*(1-a), 100*(1-conf.mean()), m.mean(), 100*np.minimum(m, y).sum()/y.sum(), (m-y)[conf].mean() if conf.any() else 0))
    m = np.round(esp).astype(int); conf = y < m
    print("%-34s %9.1f%% %12.2f %13.0f%% %14.2f" % ("m = esperado arredondado", 100*(1-conf.mean()), m.mean(), 100*np.minimum(m, y).sum()/y.sum(), (m-y)[conf].mean()))
    m = M[:, 2] * 0 + np.array([int(np.searchsorted(np.cumsum(pmf(g.pk.values)), .90)) for _, g in c.groupby(chave)])
    conf = y < m
    print("%-34s %9.1f%% %12.2f %13.0f%% %14.2f" % ("m = limite SUPERIOR do IC80 (q90)", 100*(1-conf.mean()), m.mean(), 100*np.minimum(m, y).sum()/y.sum(), (m-y)[conf].mean()))
avalia("sem", "SEMANAL")
avalia("dia", "DIARIO")
print("\n* excesso medio = pacientes a mais que vagas, nos periodos com conflito")
