"""Selecao de variaveis do HGB por paciente (recorte municipal). Metodo: permutacao + ablacao + eliminacao retroativa.

Protocolo anti-vazamento: usa SO dados < 2023. Folds temporais: (treino < 2021 -> validacao 2021) e (treino < 2022 -> validacao 2022).
Treino = base toda (todas as CNES); validacao/metrica = consultas das CNES da coorte municipal (niveis_municipal.csv).
Modelo fixo para a selecao: HGB 200 iteracoes, lr 0.1, 31 folhas, min_samples_leaf 200, l2 1.0 (sem early stopping).
Metrica: Brier por consulta. delta = Brier(sem a variavel) - Brier(com); delta > 0 => a variavel ajuda.
Passos:
  1) importancia por permutacao (variavel e grupo; 5 repeticoes; queda de Brier e de AUC) no modelo completo;
  2) ablacao por grupo (re-treina sem o grupo);
  3) eliminacao retroativa por variavel: a cada rodada re-treina sem cada variavel, calcula delta e IC95% (bootstrap por CNES-mes);
     remove a de MENOR delta se o IC95% inclui <= 0 (importancia nao significativa); repete ate todas serem significativas;
  4) candidatas novas sobre o conjunto final: taxa_cnes_28d, taxa_prev_suav, e troca de cnesexecutante por taxa_cnes_28d.
Saidas (resultados_municipal/): importancia_permutacao.csv, ablacao_grupos.csv, eliminacao_retroativa.csv, candidatas_novas.csv,
  variaveis_finais.json, importancia_permutacao.png, relatorio_selecao_variaveis_municipal.md
"""
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from utils_municipal import (ALL_VARS, CAT, GRUPOS, bootstrap_delta, brier, carrega_pacientes, feature_extras, make_hgb, md_table)

warnings.filterwarnings("ignore")
T0 = time.time()
tic = lambda: "%.0fmin" % ((time.time() - T0) / 60)
OUT = "resultados_municipal/"
NJ = 3
raw, X = carrega_pacientes()
ex = feature_extras(raw)
yy = raw.y.values
ano = raw.dia.dt.year.values
FOLDS = [(2021, (ano < 2021), (ano == 2021) & raw.coorte.values), (2022, (ano < 2022), (ano == 2022) & raw.coorte.values)]
for _, tr, va in FOLDS:
    assert raw.dia[tr | va].max() < pd.Timestamp("2023-01-01"), "vazamento: dado >= 2023 na selecao"
CL = (raw.cnesexecutante.astype(str) + "_" + raw.dia.dt.to_period("M").astype(str)).values
print("folds:", [(a, int(t.sum()), int(v.sum())) for a, t, v in FOLDS], flush=True)

XF = pd.concat([X, ex], axis=1)
for c in ex.columns:
    XF[c] = XF[c].astype(float)
PARAMS = dict(max_iter=200, learning_rate=0.1)


def fit_fold(vars_, k):
    _, tr, va = FOLDS[k]
    m = make_hgb(**PARAMS).fit(XF.loc[tr, vars_], yy[tr])
    return m, m.predict_proba(XF.loc[va, vars_])[:, 1]


def avalia(vars_):
    """probabilidades de validacao (concatenadas dos 2 folds)"""
    r = Parallel(n_jobs=NJ, prefer="threads")(delayed(fit_fold)(vars_, k) for k in range(len(FOLDS)))
    return [m for m, _ in r], np.concatenate([p for _, p in r])


VA = np.concatenate([np.flatnonzero(v) for _, _, v in FOLDS])
Y_VA, CL_VA = yy[VA], CL[VA]
cur = list(ALL_VARS)
models, p_full = avalia(cur)
print("modelo completo: Brier %.5f AUC %.4f (%s)" % (brier(Y_VA, p_full), roc_auc_score(Y_VA, p_full), tic()), flush=True)
base_brier, base_auc = brier(Y_VA, p_full), roc_auc_score(Y_VA, p_full)

# ---------------- 1) permutacao ----------------
rng = np.random.default_rng(0)
Xva = [XF.loc[v, cur] for _, _, v in FOLDS]
yva = [yy[v] for _, _, v in FOLDS]


def perm_score(cols, rep):
    b, a = [], []
    pp = []
    for m, xv in zip(models, Xva):
        xp = xv.copy()
        order = np.random.default_rng(1000 + rep).permutation(len(xp))
        for c in cols:
            xp[c] = xp[c].iloc[order].values
        pp.append(m.predict_proba(xp)[:, 1])
    p = np.concatenate(pp)
    return brier(Y_VA, p) - base_brier, base_auc - roc_auc_score(Y_VA, p)

rows = []
for c in ALL_VARS:
    r = np.array([perm_score([c], k) for k in range(5)])
    rows.append({"tipo": "variavel", "nome": c, "d_brier": r[:, 0].mean(), "d_brier_dp": r[:, 0].std(), "d_auc": r[:, 1].mean()})
for g, cs in GRUPOS.items():
    r = np.array([perm_score(cs, k) for k in range(5)])
    rows.append({"tipo": "grupo", "nome": g, "d_brier": r[:, 0].mean(), "d_brier_dp": r[:, 0].std(), "d_auc": r[:, 1].mean()})
PI = pd.DataFrame(rows).sort_values(["tipo", "d_brier"], ascending=[True, False])
PI.to_csv(OUT + "importancia_permutacao.csv", index=False)
print("permutacao ok (%s)\n" % tic(), PI.round(5).to_string(index=False), flush=True)

# ---------------- 2) ablacao por grupo ----------------
rows = []
for g, cs in GRUPOS.items():
    keep = [v for v in cur if v not in cs]
    _, p = avalia(keep)
    d = (p - Y_VA) ** 2 - (p_full - Y_VA) ** 2
    m_, lo, hi = bootstrap_delta(d, CL_VA)
    rows.append({"grupo": g, "n_vars": len(cs), "d_brier": m_, "ic_lo": lo, "ic_hi": hi, "d_auc": base_auc - roc_auc_score(Y_VA, p)})
AG = pd.DataFrame(rows).sort_values("d_brier", ascending=False)
AG.to_csv(OUT + "ablacao_grupos.csv", index=False)
print("ablacao por grupo ok (%s)\n" % tic(), AG.round(5).to_string(index=False), flush=True)

# ---------------- 3) eliminacao retroativa ----------------
log, rodada = [], 0
p_cur = p_full
while len(cur) > 3:
    rodada += 1
    res = []
    for v in cur:
        keep = [c for c in cur if c != v]
        _, p = avalia(keep)
        d = (p - Y_VA) ** 2 - (p_cur - Y_VA) ** 2
        m_, lo, hi = bootstrap_delta(d, CL_VA, nboot=500)
        res.append((v, m_, lo, hi, p))
    res.sort(key=lambda t: t[1])
    v, m_, lo, hi, p = res[0]
    for rv, rm, rlo, rhi, _ in res:
        log.append({"rodada": rodada, "variavel": rv, "d_brier": rm, "ic_lo": rlo, "ic_hi": rhi, "removida": rv == v and lo <= 0})
    print("rodada %d (%d vars): menor delta = %s %.6f IC[%.6f, %.6f] %s (%s)" % (
        rodada, len(cur), v, m_, lo, hi, "-> REMOVE" if lo <= 0 else "-> para", tic()), flush=True)
    if lo > 0:
        break
    cur = [c for c in cur if c != v]
    p_cur = p
    pd.DataFrame(log).to_csv(OUT + "eliminacao_retroativa.csv", index=False)
EL = pd.DataFrame(log)
EL.to_csv(OUT + "eliminacao_retroativa.csv", index=False)
removidas = [r.variavel for r in EL[EL.removida].itertuples()]
print("FINAL: %d variaveis; removidas %d: %s" % (len(cur), len(removidas), removidas), flush=True)
b_fin, a_fin = brier(Y_VA, p_cur), roc_auc_score(Y_VA, p_cur)

# ---------------- 4) candidatas novas ----------------
cands = {"final": cur, "+taxa_cnes_28d": cur + ["taxa_cnes_28d"], "+taxa_prev_suav": cur + ["taxa_prev_suav"],
         "troca cnes -> taxa_cnes_28d": [c for c in cur if c != "cnesexecutante"] + ["taxa_cnes_28d"]}
rows = []
for nome, vs in cands.items():
    _, p = avalia(vs)
    d = (p - Y_VA) ** 2 - (p_cur - Y_VA) ** 2
    m_, lo, hi = bootstrap_delta(-d, CL_VA, nboot=500)  # ganho = Brier(final) - Brier(candidata)
    rows.append({"conjunto": nome, "brier": brier(Y_VA, p), "auc": roc_auc_score(Y_VA, p), "ganho_vs_final": m_, "ic_lo": lo, "ic_hi": hi})
CD = pd.DataFrame(rows)
CD.to_csv(OUT + "candidatas_novas.csv", index=False)
print(CD.round(5).to_string(index=False), flush=True)
# adota candidata so se o ganho for significativo (IC inferior > 0)
adotadas = CD[(CD.conjunto != "final") & (CD.ic_lo > 0)].sort_values("ganho_vs_final", ascending=False)
final = list(cands[adotadas.iloc[0].conjunto]) if len(adotadas) else list(cur)
json.dump({"variaveis": final, "removidas": removidas, "candidata_adotada": adotadas.iloc[0].conjunto if len(adotadas) else None,
           "brier_cv_completo": base_brier, "brier_cv_final": b_fin, "auc_cv_completo": base_auc, "auc_cv_final": a_fin},
          open(OUT + "variaveis_finais.json", "w"), indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = PI[PI.tipo == "variavel"].sort_values("d_brier")
    plt.figure(figsize=(7, 8)); plt.barh(v.nome, v.d_brier, xerr=v.d_brier_dp)
    plt.xlabel("aumento do Brier ao permutar (folds 2021-22, CNES municipais)"); plt.tight_layout()
    plt.savefig(OUT + "importancia_permutacao.png", dpi=130)
except Exception as e:
    print("grafico falhou:", e)

# ---------------- relatorio ----------------
md = ["# Seleção de variáveis do HGB por paciente — recorte municipal", "",
      "## Método", "",
      "- **Dados e protocolo:** somente consultas anteriores a 2023 (o teste 2023–2026 não foi tocado). Dois folds temporais: treino < 2021 → validação 2021; treino < 2022 → validação 2022. Treino com a base inteira; validação nas consultas das %d CNES da coorte municipal (`niveis_municipal.csv`)." % len(set(raw.cnesexecutante[raw.coorte])),
      "- **Modelo de referência:** `HistGradientBoostingClassifier`, 200 iterações, learning_rate 0,1, 31 folhas, min_samples_leaf 200, l2 1,0, sem early stopping.",
      "- **Métrica:** Brier por consulta (menor é melhor); AUC como apoio. delta = Brier(sem a variável) − Brier(com); delta > 0 significa que a variável ajuda.",
      "- **Passo 1, permutação:** embaralha a variável (ou o grupo) na validação, 5 repetições, e mede o aumento do Brier e a queda do AUC.",
      "- **Passo 2, ablação por grupo:** re-treina sem o grupo inteiro, o que evita subestimar variáveis correlacionadas.",
      "- **Passo 3, eliminação retroativa:** a cada rodada re-treina sem cada variável; remove a de menor delta se o IC 95% (bootstrap por CNES-mês, 500 réplicas) incluir 0 ou valores negativos; repete até todas as restantes serem significativas.",
      "- **Passo 4, candidatas novas** sobre o conjunto final (adotadas só se o ganho tiver IC 95% inferior > 0).", "",
      "## Resultado", "",
      "- Modelo completo (%d variáveis): Brier %.5f, AUC %.4f." % (len(ALL_VARS), base_brier, base_auc),
      "- Conjunto final (%d variáveis): Brier %.5f, AUC %.4f." % (len(cur), b_fin, a_fin),
      "- **Removidas (%d):** %s" % (len(removidas), ", ".join(removidas) or "nenhuma"),
      "- **Mantidas (%d):** %s" % (len(cur), ", ".join(cur)), "",
      "## Importância por permutação", "", md_table(PI.round(5)), "",
      "## Ablação por grupo", "", md_table(AG.round(5)), "",
      "## Eliminação retroativa", "", md_table(EL.round(6)), "",
      "## Candidatas novas", "", md_table(CD.round(5)), ""]
open(OUT + "relatorio_selecao_variaveis_municipal.md", "w").write("\n".join(md))
print("relatorio salvo (%s)" % tic())
