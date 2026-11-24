"""Relatorio de decisao (backtest 2023-2026): o que teria acontecido se o gestor chamasse pessoas da fila
segundo cada nivel de risco aceito.

Regra: nivel de risco r = chance aceita de chamar MAIS pessoas do que as faltas do dia.
       chamar c(r) = menor c tal que P(faltas <= c) >= r  (quantil r da distribuicao prevista).
Resultado de um dia com c chamados e y faltas reais:
       preenchidas = min(c, y) | sem vaga (sobreoferta) = max(c - y, 0) | vagas perdidas = max(y - c, 0)
Metodos: nivel A -> modelo por paciente (+ phi) vs baseline (taxa 28d x agenda, + phi) | nivel B -> baseline (+ phi)
         regra ingenua: floor(m x media das faltas do mesmo dia da semana nas 4 semanas anteriores), m in {.25,.5,.75,1}
phi: um escalar por nivel/metodo (calibracao.json, estimado em 2023-2026: vazamento minimo, declarado no relatorio).
Saidas: relatorio_decisao.md, relatorio_decisao_fronteira.png, relatorio_decisao_niveis.csv, relatorio_decisao_por_cnes.csv
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
NS = 1000
RISCOS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
MULT = [0.25, 0.5, 0.75, 1.0]
rng = np.random.default_rng(11)

CO = pd.read_csv("coorte_2019_2026.csv", dtype={"cnes": str}).set_index("cnes")
A = list(CO.index[CO.estavel])
B = ["2237385", "9132562", "2237334", "7031092", "2237601", "2693801"]
tier = {**{c: "A" for c in A}, **{c: "B" for c in B}}
phi = json.load(open("calibracao.json"))
raw = pd.read_parquet("gercon_poa_censo_distancias.parquet", columns=["situacaoagenda", "datahora_consulta", "cnesexecutante"])
raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(int)
raw["dia"] = pd.to_datetime(raw.datahora_consulta)
P = pickle.load(open("p_modelos.pkl", "rb"))["V0"]
is_set = raw.cnesexecutante.isin(tier).values & (raw.dia >= "2023-01-01").values

S = raw.loc[is_set, ["cnesexecutante", "dia", "y"]].sort_values(["cnesexecutante", "dia"])
ordr = S.index.values
gk = S.groupby(["cnesexecutante", "dia"], sort=True)
grp = gk.ngroup().values
starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
D = gk.agg(y=("y", "sum"), ag=("y", "size")).reset_index()
NG = len(D)
D["nivel"] = D.cnesexecutante.map(tier)
p = P[ordr]
D["pred"] = np.add.reduceat(p, starts)

Bd = raw[raw.cnesexecutante.isin(tier)].groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size")).reset_index()
bl = []
for c_, x in Bd.groupby("cnesexecutante"):
    x = x.set_index("dia").reindex(pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")).fillna(0)
    pb = (x.f.shift(3).rolling(28, min_periods=14).sum() / x.ag.shift(3).rolling(28, min_periods=14).sum()) * x.ag
    nv = sum(x.f.shift(7 * j) for j in (1, 2, 3, 4)) / 4
    bl.append(pd.DataFrame({"cnesexecutante": c_, "pred_base": pb, "naive": nv}).rename_axis("dia").reset_index())
D = D.merge(pd.concat(bl), on=["cnesexecutante", "dia"], how="left")
k = np.full(NG, np.nan)
for c_, ix in D.groupby("cnesexecutante").indices.items():
    x = pd.DataFrame({"y": D.y.values[ix], "pred": D.pred.values[ix]}, index=D.dia.values[ix])
    cal = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D")).fillna(0)
    num = cal.y.shift(3).rolling(28, min_periods=14).sum()
    den = cal.pred.shift(3).rolling(28, min_periods=14).sum()
    k[ix] = (num / den.where(den > 0)).reindex(x.index).values
D["k"] = k
D["valid"] = D.k.notna() & D.pred_base.notna() & D.naive.notna()
V = D[D.valid].copy()
idx = np.flatnonzero(D.valid.values)
semanas = (V.dia.max() - V.dia.min()).days / 7
print("CNES-dias validos: %d (A %d, B %d) | %s a %s | %.0f semanas" % (
    len(V), (V.nivel == "A").sum(), (V.nivel == "B").sum(), V.dia.min().date(), V.dia.max().date(), semanas))

# ---------- distribuicoes simuladas ----------
def sim_modelo(days, ph):
    out = np.empty((len(days), NS), dtype=np.int16)
    for j, d_ in enumerate(days):
        r0 = starts[d_]; r1 = starts[d_ + 1] if d_ + 1 < NG else len(p)
        g = rng.gamma(1 / ph, ph, size=NS).astype(np.float32)
        pr = np.clip(p[r0:r1, None].astype(np.float32) * np.float32(D.k.values[d_]) * g[None, :], 0, 0.995)
        out[j] = (rng.random(pr.shape, dtype=np.float32) < pr).sum(0)
    return out
def sim_base(mu, ph):
    g = rng.gamma(1 / ph, ph, size=(len(mu), NS))
    return rng.poisson(mu[:, None] * g).astype(np.int16)

SIM = {}
mA = (D.nivel.values == "A")
mB = (D.nivel.values == "B")
dA = np.flatnonzero(D.valid.values & mA)
dB = np.flatnonzero(D.valid.values & mB)
SIM[("A", "modelo")] = (dA, sim_modelo(dA, phi["A_modelo"]))
SIM[("A", "baseline")] = (dA, sim_base(D.pred_base.values[dA], phi["A_baseline"]))
SIM[("B", "baseline")] = (dB, sim_base(D.pred_base.values[dB], phi["B_baseline"]))
print("simulacoes prontas")

def resumo(y, c, dias_cnes=None):
    fil = np.minimum(c, y); exc = np.maximum(c - y, 0)
    return {"cnes_dias": len(y), "faltas": int(y.sum()), "chamados": int(c.sum()), "preenchidas": int(fil.sum()),
            "pct_faltas_recuperadas": fil.sum() / y.sum(), "pct_dias_sobreoferta": float((c > y).mean()),
            "sem_vaga_total": int(exc.sum()), "chamados_por_semana": c.sum() / semanas,
            "preenchidas_por_semana": fil.sum() / semanas, "sem_vaga_por_semana": exc.sum() / semanas,
            "eficiencia": fil.sum() / c.sum() if c.sum() > 0 else np.nan}

rows = []
cnes_rows = []
for (t, met), (days, sims) in SIM.items():
    y = D.y.values[days]
    for r in RISCOS:
        c = np.quantile(sims, r, axis=1, method="inverted_cdf").astype(int)
        rows.append({"nivel": t, "metodo": met, "regra": f"risco {int(r * 100)}%", "param": r, **resumo(y, c)})
        if met == ("modelo" if t == "A" else "baseline"):
            cn = D.cnesexecutante.values[days]
            for cc in np.unique(cn):
                m = cn == cc
                cnes_rows.append({"cnes": cc, "nivel": t, "risco": r, "faltas_por_dia": y[m].mean(), "chamados_por_dia": c[m].mean(), **resumo(y[m], c[m])})
for t in ("A", "B"):
    days = SIM[(t, "baseline")][0]
    y = D.y.values[days]
    for m_ in MULT:
        c = np.floor(m_ * D.naive.values[days]).astype(int)
        rows.append({"nivel": t, "metodo": "regra ingenua", "regra": f"{m_:g} x media do mesmo dia da semana", "param": m_, **resumo(y, c)})
T = pd.DataFrame(rows)
T.to_csv("relatorio_decisao_niveis.csv", index=False)
CN = pd.DataFrame(cnes_rows)
CN.to_csv("relatorio_decisao_por_cnes.csv", index=False)

# ---------- figura: fronteira recuperadas x sobreoferta ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for a_, t in zip(ax, ("A", "B")):
    for met, cor, mk in (("modelo", "C0", "o"), ("baseline", "C1", "s"), ("regra ingenua", "C7", "^")):
        s = T[(T.nivel == t) & (T.metodo == met)].sort_values("param")
        if s.empty:
            continue
        a_.plot(100 * s.pct_dias_sobreoferta, 100 * s.pct_faltas_recuperadas, mk + "-", color=cor, label={"modelo": "modelo por paciente", "baseline": "baseline (taxa recente x agenda)", "regra ingenua": "regra ingenua (media do mesmo dia da semana)"}[met])
        if met != "regra ingenua":
            for _, r_ in s.iterrows():
                a_.annotate("%d%%" % round(100 * r_.param), (100 * r_.pct_dias_sobreoferta, 100 * r_.pct_faltas_recuperadas), textcoords="offset points", xytext=(4, -10), fontsize=7, color=cor)
    a_.set_xlabel("% dos dias com sobreoferta (chamou mais pessoas do que faltas)")
    a_.set_ylabel("% das faltas recuperadas (vagas preenchidas)")
    a_.set_title("Nivel %s - %s" % (t, "14 CNES (principal)" if t == "A" else "6 CNES (secundario)"))
    a_.grid(alpha=.3); a_.legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Curva de decisao 2023-2026: quanto se recupera para cada nivel de sobreoferta aceito (numeros = risco escolhido)", y=.995, fontsize=10)
plt.tight_layout()
plt.savefig("relatorio_decisao_fronteira.png", dpi=110)

# ---------- comparacao do modelo na mesma sobreoferta (interpolacao) ----------
def recup_em(t, met, x):
    s = T[(T.nivel == t) & (T.metodo == met)].sort_values("pct_dias_sobreoferta")
    return float(np.interp(x, s.pct_dias_sobreoferta, s.pct_faltas_recuperadas))
cmp = []
for x in (0.10, 0.20, 0.30):
    cmp.append({"sobreoferta": x, "modelo": recup_em("A", "modelo", x), "baseline": recup_em("A", "baseline", x),
                "ingenua": recup_em("A", "regra ingenua", x) if T[(T.nivel == "A") & (T.metodo == "regra ingenua")].pct_dias_sobreoferta.max() >= x >= T[(T.nivel == "A") & (T.metodo == "regra ingenua")].pct_dias_sobreoferta.min() else np.nan})
CMP = pd.DataFrame(cmp)
print("\nA: % das faltas recuperadas para o mesmo % de dias com sobreoferta"); print((CMP * 100).round(1).to_string(index=False))

# ---------- relatorio em markdown ----------
def md(df, fmt):
    cols = list(df.columns)
    lin = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lin.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in cols) + " |")
    return "\n".join(lin)
pct = lambda v: "%.0f%%" % (100 * v)
n1 = lambda v: "%.1f" % v
def tabela_nivel(t, met):
    s = T[(T.nivel == t) & (T.metodo == met) & T.regra.str.startswith("risco")].sort_values("param")
    d = pd.DataFrame({"Risco aceito": [f"{int(100 * v)}%" for v in s.param], "Chamados/semana": s.chamados_por_semana, "Vagas preenchidas/semana": s.preenchidas_por_semana,
                      "Faltas recuperadas": s.pct_faltas_recuperadas, "Dias com sobreoferta": s.pct_dias_sobreoferta,
                      "Pacientes sem vaga/semana": s.sem_vaga_por_semana, "Chamados que acharam vaga": s.eficiencia})
    return md(d, {"Chamados/semana": n1, "Vagas preenchidas/semana": n1, "Faltas recuperadas": pct, "Dias com sobreoferta": pct, "Pacientes sem vaga/semana": n1, "Chamados que acharam vaga": pct})
fa = V[V.nivel == "A"].y.sum() / semanas
fb = V[V.nivel == "B"].y.sum() / semanas
ex = CN[(CN.cnes == "2237571") & (CN.risco == 0.20)].iloc[0]
exB = CN[(CN.cnes == "2237385") & (CN.risco == 0.20)].iloc[0]
sel = CN[CN.risco.isin([0.10, 0.30])].copy()
pc = sel.pivot(index="cnes", columns="risco", values=["chamados_por_dia", "pct_faltas_recuperadas", "pct_dias_sobreoferta"])
pc.columns = [f"{a}_{int(100 * b)}" for a, b in pc.columns]
pc["faltas_por_dia"] = CN[CN.risco == 0.10].set_index("cnes").faltas_por_dia
pc["nivel"] = CN[CN.risco == 0.10].set_index("cnes").nivel
pc = pc.sort_values(["nivel", "faltas_por_dia"], ascending=[True, False]).reset_index()
tab_cnes = pd.DataFrame({"CNES": pc.cnes, "Nível": pc.nivel, "Faltas/dia": pc.faltas_por_dia,
                         "Risco 10%: chamar/dia": pc.chamados_por_dia_10, "recupera": pc.pct_faltas_recuperadas_10, "sobreoferta": pc.pct_dias_sobreoferta_10,
                         "Risco 30%: chamar/dia": pc.chamados_por_dia_30, "recupera ": pc.pct_faltas_recuperadas_30, "sobreoferta ": pc.pct_dias_sobreoferta_30})
tab_cnes_md = md(tab_cnes, {"Faltas/dia": n1, "Risco 10%: chamar/dia": n1, "Risco 30%: chamar/dia": n1, "recupera": pct, "sobreoferta": pct, "recupera ": pct, "sobreoferta ": pct})
cm = CMP.copy()
tab_cmp = md(pd.DataFrame({"Dias com sobreoferta aceitos": cm.sobreoferta, "Modelo": cm.modelo, "Baseline": cm.baseline, "Regra ingênua": cm.ingenua}),
             {"Dias com sobreoferta aceitos": pct, "Modelo": pct, "Baseline": pct, "Regra ingênua": lambda v: "n/d" if pd.isna(v) else pct(v)})

texto = f"""# Relatório de decisão — faltas e chamada da fila de espera
*Simulação retrospectiva com o histórico de {V.dia.min():%d/%m/%Y} a {V.dia.max():%d/%m/%Y} ({semanas:.0f} semanas, {len(V)} CNES-dias). Nenhuma regra foi aplicada na prática.*

## Como ler
Cada CNES recebe, todo dia, uma previsão de quantas faltas deve ter. Como as faltas variam por acaso, a previsão é uma **distribuição de probabilidade**, não um número só. O gestor escolhe o **risco que aceita**: a chance de chamar da fila **mais pessoas do que as faltas que de fato ocorrerão**, deixando alguém sem vaga naquele dia. Quanto maior o risco aceito, mais pessoas se chamam e mais vagas se preenchem, mas mais dias terminam com pacientes sem vaga. **Este relatório só descreve essas consequências; a escolha do nível de risco é do gestor de cada unidade.**

**Atenção à leitura do "risco aceito":** ele é um **teto** para a chance de sobreoferta prevista pelo modelo. Como o número de faltas é inteiro e pequeno, e em muitos dias a regra manda chamar zero pessoas, a frequência real de dias com sobreoferta (coluna "Dias com sobreoferta") fica abaixo do risco escolhido. Para decidir, use essa coluna, que mostra o que de fato aconteceu no histórico.

Definições: *vaga preenchida* = pessoa chamada que ocupou uma falta; *sobreoferta* = dia em que se chamou mais gente do que houve faltas; *paciente sem vaga* = quem foi chamado e não encontrou vaga.

## Nível A (14 CNES estáveis, principal) — modelo por paciente
Faltas reais: {fa:.0f} por semana neste conjunto.

{tabela_nivel("A", "modelo")}

## Nível B (6 CNES com mudança moderada, secundário) — baseline (taxa recente × agenda)
Faltas reais: {fb:.0f} por semana neste conjunto.

{tabela_nivel("B", "baseline")}

## O modelo ajuda na decisão?
Para o mesmo percentual de dias com sobreoferta (nível A), % das faltas recuperadas por cada método:

{tab_cmp}

Detalhe e curvas completas: `relatorio_decisao_fronteira.png` e `relatorio_decisao_niveis.csv`. A regra ingênua chama uma fração da média das faltas do mesmo dia da semana nas 4 semanas anteriores.

## Exemplo de leitura
CNES 2237571 (nível A, {ex.faltas_por_dia:.1f} faltas por dia): com risco aceito de 20%, a regra chamaria em média {ex.chamados_por_dia:.1f} pessoas por dia, recuperaria {100 * ex.pct_faltas_recuperadas:.0f}% das faltas e deixaria alguém sem vaga em {100 * ex.pct_dias_sobreoferta:.0f}% dos dias.
CNES 2237385 (nível B, {exB.faltas_por_dia:.1f} faltas por dia): no mesmo risco, chamaria {exB.chamados_por_dia:.1f} por dia, recuperaria {100 * exB.pct_faltas_recuperadas:.0f}% e teria sobreoferta em {100 * exB.pct_dias_sobreoferta:.0f}% dos dias.

## Resultado por CNES (riscos de 10% e 30%)
{tab_cnes_md}

Todos os riscos por CNES: `relatorio_decisao_por_cnes.csv`.

## O que este relatório NÃO inclui (leia antes de usar)
- **Vagas intercambiáveis.** Os números somam as faltas do CNES inteiro. Na prática, a vaga aberta por uma falta é de uma especialidade e de um horário específicos, e só serve a quem da fila precisa daquela especialidade. Isso reduz a recuperação real e não foi simulado.
- **Aceitação pelo paciente da fila.** Assume-se que quem é chamado em 3 dias consegue comparecer. Na prática, parte recusará ou não virá, e a recuperação real será menor.
- **Mudança de comportamento.** O modelo aprendeu com dados em que não se chamava a fila. Se a prática mudar o comportamento (de quem é chamado ou de quem falta), as previsões podem deixar de valer.
- **Pequenos CNES.** Com poucas faltas por dia, os resultados oscilam muito. CNES de nível C e D não têm previsão.
- **Calibração.** A dispersão extra do dia foi estimada com o mesmo período, o que é um vazamento mínimo (um número por nível). As probabilidades nas pontas da distribuição são levemente otimistas.
- **Backtest, não piloto.** Convém rodar em modo sombra por algumas semanas antes de agir.

## Arquivos
`relatorio_decisao_niveis.csv` (resultados por nível, método e risco) · `relatorio_decisao_por_cnes.csv` · `relatorio_decisao_fronteira.png`
"""
open("relatorio_decisao.md", "w", encoding="utf-8").write(texto)
print("\nrelatorio_decisao.md gerado")
sub = T[(T.nivel == "A") & (T.metodo == "modelo")][["regra", "chamados_por_semana", "preenchidas_por_semana", "pct_faltas_recuperadas", "pct_dias_sobreoferta", "sem_vaga_por_semana", "eficiencia"]]
print(sub.round(3).to_string(index=False))
