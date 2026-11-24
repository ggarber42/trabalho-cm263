"""d01 — Auditoria dos resultados (pendências 3 e 4 de PROGRESSO_2026-11-23).

Objetivo:
(a) métricas de ETS/Croston/SARIMAX só nas linhas SEM fallback (D18), ao lado dos demais modelos
    nas mesmas linhas, para que os locais não "herdem" o desempenho da referência;
(b) teste prático de vazamento: altera (embaralha) o desfecho das consultas a partir de uma data
    de corte c e reconstrói o painel; nenhuma variável preditora de semanas s < c + h pode mudar.
Entradas: previsoes.parquet, consultas.parquet, estratos_cnes.csv.
Saídas: auditoria/metricas_sem_fallback.csv, auditoria/teste_vazamento.csv.
Uso: python3 etapas/d01_auditoria.py --config config.yaml --perfil completo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.metricas import vies, wape  # noqa: E402
from faltas.modelos_ml import colunas_preditoras  # noqa: E402
from faltas.painel import montar_painel  # noqa: E402

CHAVE = ["serie", "semana"]
LOCAIS = ["ets", "croston_tsb", "sarimax"]


def metricas_sem_fallback(prev: pd.DataFrame) -> pd.DataFrame:
    """WAPE, viés e MAE de todos os modelos do estrato consistentes nas linhas em que nenhum modelo
    local usou fallback. Retorna DataFrame[teste_ano, h, modelo, n, wape, vies, mae, wape_todas]."""
    out = []
    g0 = prev[prev["estrato"] == "consistentes"]
    for (ano, h), g in g0.groupby(["teste_ano", "h"]):
        fb = g[g["modelo"].isin(LOCAIS)].groupby(CHAVE)["fallback"].any()
        sem_fb = set(fb.index[~fb.astype(bool)])
        larga = g.pivot_table(index=CHAVE, columns="modelo", values="mu", aggfunc="first").dropna()
        y = g.groupby(CHAVE)["y"].first().reindex(larga.index)
        sel = larga.index.isin(list(sem_fb))
        for mod in larga.columns:
            yy, mm = y[sel].to_numpy(float), larga.loc[sel, mod].to_numpy(float)
            out.append({"teste_ano": ano, "h": h, "modelo": mod, "n": int(sel.sum()), "wape": wape(yy, mm),
                        "vies": vies(yy, mm), "mae": float(np.abs(yy - mm).mean()),
                        "wape_todas": wape(y.to_numpy(float), larga[mod].to_numpy(float))})
    return pd.DataFrame(out).sort_values(["h", "wape"])


def teste_vazamento(cons: pd.DataFrame, cfg: dict, corte: str, semente: int) -> pd.DataFrame:
    """Embaralha y das consultas com data_consulta ≥ corte, refaz o painel e compara as preditoras.
    ANTI-VAZAMENTO: semanas s < corte + h semanas não podem mudar (só usam informação até s − h).
    Retorna DataFrame[h, linhas_antes_do_limite, colunas_alteradas_antes, linhas_depois_alteradas]."""
    rng = np.random.default_rng(semente)
    alterado = cons.copy()
    m = alterado["data_consulta"] >= pd.Timestamp(corte)
    alterado.loc[m, "y"] = rng.permutation(alterado.loc[m, "y"].to_numpy())
    alterado.loc[m, "y"] = 1 - alterado.loc[m, "y"]          # garante mudança forte do desfecho
    res = []
    for h in cfg["avaliacao"]["horizontes"]:
        a, b = montar_painel(cons, h, cfg), montar_painel(alterado, h, cfg)
        cols = [c for c in colunas_preditoras(a) if c in a.columns]
        a, b = a.set_index(CHAVE).sort_index(), b.set_index(CHAVE).sort_index()
        assert a.index.equals(b.index), "Painéis com chaves diferentes"
        limite = pd.Timestamp(corte) + pd.Timedelta(weeks=h) - pd.Timedelta(days=7)
        antes = a.index.get_level_values("semana") < limite
        dif = ~np.isclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)
        res.append({"h": h, "corte": corte, "linhas_antes_do_limite": int(antes.sum()),
                    "colunas_alteradas_antes": ", ".join(np.array(cols)[dif[antes].any(axis=0)]) or "nenhuma",
                    "linhas_depois_alteradas": int(dif[~antes].any(axis=1).sum())})
    return pd.DataFrame(res)


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "d01_auditoria")
    pasta = saida / "auditoria"
    pasta.mkdir(exist_ok=True)

    prev = pd.read_parquet(saida / "previsoes.parquet")
    msf = metricas_sem_fallback(prev)
    msf.to_csv(pasta / "metricas_sem_fallback.csv", index=False)
    log.info(f"Métricas sem fallback: {msf['n'].iloc[0]:,} linhas por h")

    # DECISÃO: teste em uma CNES consistente (a de maior volume) para caber em poucos minutos.
    est = pd.read_csv(saida / "estratos_cnes.csv", dtype={"cnesexecutante": str})
    cnes = est[est["estrato"] == "consistentes"].sort_values("faltas_por_semana").iloc[-1]["cnesexecutante"]
    cons = pd.read_parquet(saida / "consultas.parquet")
    cons = cons[cons["cnesexecutante"].astype(str) == cnes].reset_index(drop=True)
    tv = teste_vazamento(cons, cfg, "2024-06-03", cfg["tuning"]["semente"])
    tv.to_csv(pasta / "teste_vazamento.csv", index=False)
    log.info(tv.to_string(index=False))
    assert (tv["colunas_alteradas_antes"] == "nenhuma").all(), "VAZAMENTO: preditoras mudaram antes do limite"
    log.manifesto([saida / "previsoes.parquet", saida / "consultas.parquet"],
                  [pasta / "metricas_sem_fallback.csv", pasta / "teste_vazamento.csv"])


if __name__ == "__main__":
    main()
