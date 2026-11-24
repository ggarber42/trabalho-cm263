import pickle, numpy as np, pandas as pd
CN = "2237571"; ALF = .10
R = pd.read_csv("risco_diario_coorte.csv", dtype={"cnesexecutante": str}); R["dia"] = pd.to_datetime(R.dia)
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["cnesexecutante","datahora_consulta","situacaoagenda","especialidademae"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int); raw["dia"] = pd.to_datetime(raw.datahora_consulta)
raw["p"] = pickle.load(open("p_modelos.pkl","rb"))["V1"]
c = raw[(raw.cnesexecutante == CN) & raw.p.notna()]
r = R[R.cnesexecutante == CN].set_index("dia").join(c.groupby("dia").p.sum().rename("soma_p")); r["k"] = r.esperado / r.soma_p
c = c.join(r.k, on="dia"); c = c[c.k.notna()].copy(); c["pk"] = np.clip(c.p * c.k, 0, .995)
def pmf(p):
    f = np.array([1.0])
    for q in p: f = np.convolve(f, [1 - q, q])
    return f
rows = []
for (e, d), g in c.groupby(["especialidademae", "dia"], observed=True):
    cdf = np.cumsum(pmf(g.pk.values)); rows.append((e, d, len(g), g.y.sum(), g.pk.sum(), int(np.searchsorted(cdf, ALF))))
D = pd.DataFrame(rows, columns=["esp", "dia", "agenda", "faltas", "esperado", "m"])
D["exc"] = (D.faltas < D.m); D["pac_excesso"] = (D.m - D.faltas).clip(lower=0); D["aprov"] = np.minimum(D.m, D.faltas)
def resumo(d, nome):
    print("%-26s dias-especialidade %4d | faltas %4d | extras chamados %4d | aproveitados %4d (%.0f%% das faltas) | dias com excesso %3d (%.1f%%) | pacientes em excesso %3d" %
          (nome, len(d), d.faltas.sum(), d.m.sum(), d.aprov.sum(), 100*d.aprov.sum()/d.faltas.sum(), d.exc.sum(), 100*d.exc.mean(), d.pac_excesso.sum()))
D26 = D[D.dia >= "2026-01-01"]
print("2026 (01/01 a %s), regra: extras = quantil 10%% da previsao diaria\n" % D26.dia.max().date())
resumo(D26[D26.esp == "ODONTO ENDODONTIA"], "Endodontia")
resumo(D26, "CNES 2237571 (todas)")
for a in ("2023","2024","2025"):
    resumo(D[D.dia.dt.year == int(a)][lambda x: x.esp == "ODONTO ENDODONTIA"], "Endodontia " + a)
e = D26[D26.esp == "ODONTO ENDODONTIA"]
print("\nEndodontia 2026: dias com agenda %d | dias com >=1 extra: %d | extras/dia (quando >0): %.2f | max extras num dia: %d" % (len(e), (e.m > 0).sum(), e.m[e.m > 0].mean(), e.m.max()))
print("Por mes (endodontia 2026):"); print(e.groupby(e.dia.dt.to_period("M")).agg(dias=("m", "size"), faltas=("faltas", "sum"), extras=("m", "sum"), aproveitados=("aprov", "sum"), dias_com_excesso=("exc", "sum"), pac_excesso=("pac_excesso", "sum")).to_string())
print("\nTop 8 especialidades da CNES em 2026 (extras chamados):")
t = D26.groupby("esp", observed=True).agg(faltas=("faltas", "sum"), extras=("m", "sum"), aprov=("aprov", "sum"), dias_exc=("exc", "sum"), dias=("m", "size")).sort_values("extras", ascending=False).head(8); print(t.to_string())
