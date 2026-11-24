"""S8/S9 — Avaliação comparativa dos modelos (marco M4).

Objetivo
    A partir das previsões gravadas por S7: (1) tabela principal de métricas nas MESMAS linhas e com a
    MESMA distribuição preditiva (NB); (2) Diebold-Mariano (HAC semanal) contra o baseline e contra o melhor
    modelo, com Holm e bootstrap do ganho de WAPE; (3) subgrupos (nível A/B, ano, sem Materno Infantil);
    (4) comparação entre esquemas (no mesmo conjunto de dias de teste), defasagens e modos de agenda;
    (5) `experimentos.csv` com sementes, P, E, janela, hash dos dados e métricas por combinação.

Protocolo
    Nenhuma escolha de modelo/hiperparâmetro/esquema é feita aqui: o teste só descreve (regra 6 do CLAUDE.md).
    A dispersão da NB vem de cada divisão (estimada nos folds internos, anteriores ao teste).

Entradas   resultados/<escopo>_<perfil>/ (saída de S7), config.yaml
Saídas     tabelas/res_<escopo>_<perfil>_*.csv e resultados/<escopo>_<perfil>/experimentos.csv
Como rodar python3 etapas/S9_avaliacao.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.avaliacao import carrega_previsoes, interseccao, subgrupos, tabela_dm, tabela_metricas  # noqa: E402
from src.config import caminho, carrega_config  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402

REFERENCIA = "baseline_poisson"


def compara_fator(df: pd.DataFrame, fator: str, fixos: dict, periodo: tuple[str, str] | None,
                  ks: tuple[int, ...]) -> pd.DataFrame:
    """Compara níveis de um fator (esquema, L ou modo) nas MESMAS linhas.

    Parâmetros:
        df: previsões de todos os experimentos. fator: coluna ("variante", "L" ou "modo").
        fixos: demais colunas fixadas (ex.: {"L": 3, "modo": "conhecida"}).
        periodo: (início, fim) inclusivos do conjunto comum de dias de teste, ou None.
        ks: limiares do ECE.

    Retorna:
        DataFrame (fator, modelo, métricas, ganho_wape_vs_baseline). Só entram modelos presentes em TODOS
        os níveis do fator; as linhas são a interseção (cnes, dia) entre níveis e modelos. Vazio se não houver.
    """
    d = df.copy()
    for k, v in fixos.items():
        d = d[d[k] == v]
    if periodo:
        d = d[(d["dia"] >= pd.Timestamp(periodo[0])) & (d["dia"] <= pd.Timestamp(periodo[1]))]
    niveis = sorted(d[fator].unique(), key=str)
    if len(niveis) < 2:
        return pd.DataFrame()
    modelos = set.intersection(*[set(d[d[fator] == n]["modelo"]) for n in niveis])
    d = d[d["modelo"].isin(modelos)]
    if not modelos:
        return pd.DataFrame()
    piv = d.pivot_table(index=["cnes", "dia"], columns=[fator, "modelo"], values="mu", aggfunc="first")
    ok = piv.notna().all(axis=1)
    chaves = piv.index[ok]
    sub = d.set_index(["cnes", "dia"]).loc[chaves].reset_index()
    linhas = []
    for n in niveis:
        t = tabela_metricas(sub[sub[fator] == n], ks)
        t.insert(0, fator, n)
        if REFERENCIA in set(t["modelo"]):
            w0 = float(t.loc[t["modelo"] == REFERENCIA, "wape"].iloc[0])
            t["ganho_wape_vs_baseline"] = 1 - t["wape"] / w0
        t["n_linhas_comuns"] = len(chaves)
        linhas.append(t)
    return pd.concat(linhas, ignore_index=True)


def main() -> None:
    """Gera todas as tabelas da avaliação a partir das previsões."""
    cfg = carrega_config()
    t_ini = time.time()
    esc, perfil = cfg["escopo"], cfg["perfil_ativo"]
    reg = RegistroEtapa(f"S9_avaliacao_{esc}_{perfil}", caminho(cfg, cfg["saidas"]["logs"]), {"perfil": perfil})
    raiz = caminho(cfg, cfg["saidas"]["raiz"], f"{esc}_{perfil}")
    ks = tuple(cfg["avaliacao"]["ks"])
    df = carrega_previsoes(raiz)
    print(f"[S9] previsões: {len(df):,} linhas | grupos (variante, L, modo): "
          f"{df.groupby(['variante', 'L', 'modo']).ngroups} | modelos: {sorted(df['modelo'].unique())}")
    pre = f"res_{esc}_{perfil}_"
    pasta = caminho(cfg, cfg["saidas"]["tabelas"])
    saidas = []

    def grava(t: pd.DataFrame, nome: str) -> None:
        """Grava a tabela em tabelas/res_<escopo>_<perfil>_<nome>.csv e registra a saída para o manifesto."""
        arq = pasta / f"{pre}{nome}.csv"
        t.to_csv(arq, index=False)
        saidas.append(arq)

    # ---- 1. Todos os grupos: métricas na interseção + experimentos.csv ----
    exp_linhas, perdas_all = [], []
    for (var, L, modo), g in df.groupby(["variante", "L", "modo"]):
        sub, perdas = interseccao(g)
        if sub.empty:
            print(f"[S9] AVISO: interseção vazia em {(var, L, modo)}")
            continue
        perdas.insert(0, "variante", var); perdas.insert(1, "L", L); perdas.insert(2, "modo", modo)
        perdas_all.append(perdas)
        met = tabela_metricas(sub, ks)
        for _, r in met.iterrows():
            gm = g[g["modelo"] == r["modelo"]]
            metas = [json.loads(p.read_text(encoding="utf-8")) for p in
                     sorted((raiz / var / f"L{L}_{modo}" / r["modelo"]).glob("*.json"))]
            exp_linhas.append({"escopo": esc, "perfil": perfil, "variante": var, "esquema": gm["esquema"].iloc[0],
                               "L": L, "modo_agenda": modo, "modelo": r["modelo"],
                               "semente": cfg["semente"], "P": gm["P"].iloc[0], "E": gm["E"].iloc[0],
                               "janelas_meses": str(sorted({m.get("janela_meses") for m in metas}, key=str)),
                               "n_divisoes": len(metas), "hash_dados_sha256": metas[0]["hash_dados_sha256"],
                               "params_por_divisao": json.dumps({m["rotulo"]: m["params"] for m in metas}),
                               "tempo_total_s": round(sum(m["tempo_s"] for m in metas), 1),
                               **{k: r[k] for k in ("n", "wape", "vies", "crps", "cob80", "cob50", "larg80", "ece", "auc_celula")}})
    exp = pd.DataFrame(exp_linhas)
    exp.to_csv(raiz / "experimentos.csv", index=False)
    saidas.append(raiz / "experimentos.csv")
    grava(pd.concat(perdas_all, ignore_index=True), "linhas_perdidas")

    # ---- 2. Grupo principal ----
    var0, L0, modo0 = "expansiva", cfg["lag_principal"], cfg["agenda_modo"]
    g0 = df[(df["variante"] == var0) & (df["L"] == L0) & (df["modo"] == modo0)]
    assert not g0.empty, f"Sem resultados do grupo principal {(var0, L0, modo0)}"
    sub0, perdas0 = interseccao(g0)
    met0 = tabela_metricas(sub0, ks)
    w_base = float(met0.loc[met0["modelo"] == REFERENCIA, "wape"].iloc[0])
    met0["ganho_wape_vs_baseline"] = 1 - met0["wape"] / w_base
    grava(met0, "principal_metricas")
    dm0 = tabela_dm(sub0, REFERENCIA, met0, int(cfg["avaliacao"]["bootstrap"]), cfg["semente"])
    grava(dm0, "principal_dm")
    sg0 = subgrupos(sub0, cfg["estratos"]["materno_infantil"], ks)
    grava(sg0, "principal_subgrupos")
    print(met0[["modelo", "n", "wape", "vies", "crps", "cob80", "ganho_wape_vs_baseline"]].round(4).to_string(index=False))

    # ---- 3. Fatores: esquema, defasagem, modo de agenda ----
    comum = tuple(cfg["divisao"]["teste_comum"])
    t_esq = compara_fator(df, "variante", {"L": L0, "modo": modo0}, comum, ks)
    grava(t_esq, "esquemas_teste_comum")
    t_lag = compara_fator(df[df["variante"] == var0], "L", {"modo": modo0}, None, ks)
    grava(t_lag, "defasagens")
    t_modo = compara_fator(df[df["variante"] == var0], "modo", {"L": L0}, None, ks)
    grava(t_modo, "modos_agenda")

    # ---- 4. Métricas por consulta (modelo por paciente), só dos metadados ----
    cons = []
    for js in sorted(raiz.glob(f"{var0}/L{L0}_{modo0}/paciente_hgb/*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if "consulta" in m:
            cons.append({"rotulo": m["rotulo"], **m["consulta"]})
    grava(pd.DataFrame(cons), "paciente_metricas_consulta")

    grava_manifesto(f"S9_avaliacao_{esc}_{perfil}", [raiz / "experimentos.csv"], saidas, caminho(cfg, cfg["saidas"]["manifestos"]))
    reg.finaliza([raiz], saidas, {"linhas_previsoes": int(len(df)), "grupos": int(df.groupby(['variante', 'L', 'modo']).ngroups)})
    print(f"[S9] concluído em {time.time() - t_ini:.0f}s. Tabelas: {pasta}/{pre}*.csv")


if __name__ == "__main__":
    main()
