"""Avaliação comparativa dos modelos (S9): interseção de linhas, métricas, DM com Holm e subgrupos.

Etapa: S9. Todas as comparações são feitas nas MESMAS linhas (interseção entre modelos) e com a MESMA
família de distribuição preditiva (NB, dispersão por divisão estimada fora da amostra). Não executa nada
ao ser importado.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dm_test import bootstrap_ganho_wape, dm_hac, holm, perda_semanal
from src.metrics import MU_MIN, crps_nb, resume_modelo


def carrega_previsoes(raiz: Path) -> pd.DataFrame:
    """Lê todas as previsões gravadas por S7 em `raiz` e acrescenta os metadados de cada divisão.

    Parâmetros:
        raiz: pasta resultados/<escopo>_<perfil>.

    Retorna:
        DataFrame longo: cnes, dia, f, ag, expo, mu, nivel, variante, L, modo, modelo, rotulo, phi, P, E,
        janela_meses, esquema. Só contém agregados por CNES-dia (sem identificadores de paciente).
    """
    partes = []
    for js in sorted(Path(raiz).glob("*/L*_*/*/*.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        pq = js.with_suffix(".parquet")
        assert pq.exists(), f"Previsões ausentes para {js}"
        d = pd.read_parquet(pq)
        for k in ("variante", "L", "modo", "modelo", "rotulo", "phi", "P", "E", "janela_meses", "esquema"):
            d[k] = meta.get(k)
        d["dia"] = pd.to_datetime(d["dia"])
        partes.append(d)
    assert partes, f"Nenhuma previsão encontrada em {raiz}. Rode etapas/S7_experimentos.py"
    return pd.concat(partes, ignore_index=True)


def interseccao(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restringe `df` (vários modelos) às linhas (cnes, dia) em que TODOS os modelos têm previsão válida.

    Parâmetros:
        df: previsões de um grupo (mesma variante, L e modo), com coluna `modelo`.

    Retorna:
        (df restrito, tabela de linhas perdidas por modelo: n_linhas_uniao, n_validas_modelo, perdidas).

    Cuidados:
        Registra quantas linhas cada modelo perdeu (CLAUDE.md seção 4). Previsão NaN ou ausente = perdida.
    """
    piv = df.pivot(index=["cnes", "dia"], columns="modelo", values="mu")
    valido = piv.notna()
    todos = valido.all(axis=1)
    perdas = pd.DataFrame({"n_linhas_uniao": len(piv), "n_validas_modelo": valido.sum(),
                           "perdidas_do_modelo": (~valido).sum(), "n_na_interseccao": int(todos.sum())})
    chaves = piv.index[todos]
    sub = df.set_index(["cnes", "dia"]).loc[chaves].reset_index()
    return sub, perdas.rename_axis("modelo").reset_index()


def tabela_metricas(sub: pd.DataFrame, ks: tuple[int, ...]) -> pd.DataFrame:
    """Métricas de cada modelo na interseção (WAPE, viés, CRPS, cobertura, largura, ECE, AUC por célula)."""
    linhas = []
    for modelo, g in sub.groupby("modelo", sort=False):
        r = resume_modelo(g["f"].to_numpy(), g["mu"].to_numpy(), g["phi"].to_numpy(float), ks)
        linhas.append({"modelo": modelo, **r})
    return pd.DataFrame(linhas)


def _perdas_por_linha(sub: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta erro absoluto e CRPS por linha (cada modelo com a sua dispersão por divisão)."""
    sub = sub.copy()
    sub["erro_abs"] = (sub["f"] - sub["mu"]).abs()
    sub["crps"] = np.nan
    for modelo, ix in sub.groupby("modelo").indices.items():
        g = sub.iloc[ix]
        sub.iloc[ix, sub.columns.get_loc("crps")] = crps_nb(g["f"].to_numpy(), np.maximum(g["mu"].to_numpy(), MU_MIN),
                                                            g["phi"].to_numpy(float))
    return sub


def tabela_dm(sub: pd.DataFrame, referencia: str, metrica_ordem: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Diebold-Mariano (HAC, agregação semanal) de cada modelo contra o baseline e contra o melhor modelo.

    Parâmetros:
        sub: previsões na interseção (todas as linhas iguais entre modelos).
        referencia: nome do baseline (ex.: "baseline_poisson").
        metrica_ordem: saída de ``tabela_metricas`` (define o "melhor modelo" pelo menor WAPE).
        n_boot, seed: bootstrap em blocos semanais do ganho de WAPE vs. baseline.

    Retorna:
        DataFrame com: comparacao (vs_baseline | vs_melhor), modelo, referencia, perda (erro_abs | crps),
        diff_media (>0 = modelo melhor que a referência), p, p_holm, n_comparacoes, T_semanas e, para
        vs_baseline/erro_abs, ganho relativo de WAPE com IC 95% por bootstrap.

    Cuidados:
        Holm é aplicado separadamente em cada família (vs_baseline × perda e vs_melhor × perda); a coluna
        `n_comparacoes` informa o tamanho da família.
    """
    sub = _perdas_por_linha(sub)
    modelos = list(sub["modelo"].unique())
    melhor = metrica_ordem.sort_values("wape").iloc[0]["modelo"]
    semanal = {m: {p: perda_semanal(sub[sub["modelo"] == m], p) for p in ("erro_abs", "crps")} for m in modelos}
    y_sem = perda_semanal(sub[sub["modelo"] == modelos[0]].assign(yy=lambda x: x["f"]), "yy")
    linhas = []
    for comp, ref in (("vs_baseline", referencia), ("vs_melhor", melhor)):
        if ref not in modelos:
            continue
        for perda in ("erro_abs", "crps"):
            familia = []
            for m in modelos:
                if m == ref:
                    continue
                d = (semanal[ref][perda] - semanal[m][perda]).to_numpy()      # > 0 : m melhor que a referência
                r = dm_hac(d)
                lin = {"comparacao": comp, "modelo": m, "referencia": ref, "perda": perda, "diff_media": r["media"],
                       "estatistica": r["estatistica"], "p": r["p"], "T_semanas": r["T"]}
                if comp == "vs_baseline" and perda == "erro_abs":
                    b = bootstrap_ganho_wape(pd.DataFrame({"erro_abs": semanal[ref]["erro_abs"], "y": y_sem}),
                                             pd.DataFrame({"erro_abs": semanal[m]["erro_abs"], "y": y_sem}), n_boot, seed)
                    lin.update({"ganho_wape": b["ganho"], "ganho_ic_inf": b["ic_inf"], "ganho_ic_sup": b["ic_sup"]})
                familia.append(lin)
            if familia:
                aj = holm([x["p"] for x in familia])
                for x, pa in zip(familia, aj):
                    x["p_holm"] = float(pa)
                    x["n_comparacoes"] = len(familia)
                linhas += familia
    return pd.DataFrame(linhas)


def subgrupos(sub: pd.DataFrame, materno: list[str], ks: tuple[int, ...]) -> pd.DataFrame:
    """Métricas por nível (A/B), por ano e sem o Hospital Materno Infantil, para cada modelo.

    Parâmetros:
        sub: previsões na interseção; materno: CNES do estrato Materno Infantil; ks: limiares de ECE.

    Retorna:
        DataFrame: subgrupo, valor, modelo e métricas (n, wape, vies, crps, cob80, ...). Subgrupos com
        menos de 30 linhas são omitidos.
    """
    cortes = {"geral": sub, "sem_materno_infantil": sub[~sub["cnes"].isin(materno)]}
    for niv in sorted(sub["nivel"].dropna().unique()):
        cortes[f"nivel_{niv}"] = sub[sub["nivel"] == niv]
    for ano in sorted(sub["dia"].dt.year.unique()):
        cortes[f"ano_{ano}"] = sub[sub["dia"].dt.year == ano]
    linhas = []
    for nome, g in cortes.items():
        for modelo, h in g.groupby("modelo", sort=False):
            if len(h) < 30 or h["f"].sum() == 0:
                continue
            linhas.append({"subgrupo": nome, "modelo": modelo,
                           **resume_modelo(h["f"].to_numpy(), h["mu"].to_numpy(), h["phi"].to_numpy(float), ks)})
    return pd.DataFrame(linhas)
