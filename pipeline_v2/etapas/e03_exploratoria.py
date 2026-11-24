"""e03 — Análise exploratória do treino e escolha do esquema de avaliação.

Objetivo: caracterizar cada série (CNES × especialidade-mãe) semanal usando SÓ o treino do teste principal
(semanas ≤ fim_treino), aplicar a regra pré-declarada de Cerqueira et al. (2020) e gravar o
esquema escolhido. Roda o teste de wavelets em R se o pacote locits estiver disponível.
Entradas: painel_h1.parquet.
Saídas: caracterizacao_series.csv, acf_series.csv, esquema.json, resumo_exploratoria.csv.
Uso: python3 etapas/e03_exploratoria.py --config config.yaml --perfil smoke
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.exploratoria import caracterizar_serie, decidir_esquema  # noqa: E402


def estratos_cnes(treino: pd.DataFrame, crit: dict) -> pd.DataFrame:
    """Classifica cada CNES em "consistentes" ou "outros" (D17) com as últimas
    crit["janela_semanas"] semanas do treino.

    Parâmetros: treino = painel (série-semana) restrito ao treino; crit = config.exploratoria.consistencia.
    Retorna: DataFrame[cnesexecutante, semanas_ativas, faltas_por_semana, estrato].
    ANTI-VAZAMENTO: recebe só semanas ≤ fim do treino.
    """
    por_cnes = treino.groupby(["cnesexecutante", "semana"])[["y", "n_total"]].sum().reset_index()
    ultimas = np.sort(por_cnes["semana"].unique())[-crit["janela_semanas"]:]
    jan = por_cnes[por_cnes["semana"].isin(ultimas)]
    r = jan.groupby("cnesexecutante").agg(semanas_ativas=("n_total", lambda n: int((n > 0).sum())),
                                          faltas=("y", "sum")).reset_index()
    r["faltas_por_semana"] = r["faltas"] / crit["janela_semanas"]
    todas = pd.DataFrame({"cnesexecutante": treino["cnesexecutante"].unique()})
    r = todas.merge(r, how="left").fillna({"semanas_ativas": 0, "faltas": 0, "faltas_por_semana": 0})
    ok = (r["semanas_ativas"] >= crit["min_semanas_ativas"]) & (r["faltas_por_semana"] >= crit["min_faltas_semana"])
    r["estrato"] = np.where(ok, "consistentes", "outros")
    return r.drop(columns="faltas")


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e03_exploratoria")
    principal = next(t for t in cfg["avaliacao"]["testes"] if t["principal"])
    fim_treino = pd.Timestamp(principal["fim_treino"])

    # DECISÃO: a caracterização usa y e n_total, que não dependem de h; lê o painel do h principal.
    p = pd.read_parquet(saida / f"painel_h{cfg['avaliacao']['horizonte_principal']}.parquet")
    # ANTI-VAZAMENTO: a exploratória que decide algo usa só o treino do teste principal.
    treino = p[p["semana"] <= fim_treino]
    log.info(f"Treino: {treino['semana'].min().date()} a {treino['semana'].max().date()}, {treino['serie'].nunique()} séries (CNES × especialidade)")

    linhas, acfs = [], []
    for serie, s in treino.groupby("serie"):
        s = s.sort_values("semana")
        y, n = s["y"].to_numpy(float), s["n_total"].to_numpy()
        car = caracterizar_serie(y, n, cfg["exploratoria"])
        car["serie"] = serie
        car["taxa_falta"] = s["y"].sum() / max(s["n_total"].sum(), 1)
        linhas.append(car)
        if len(y) >= 60 and y.std() > 0:
            r = acf(y, nlags=min(60, len(y) - 2), fft=True)
            acfs.append(pd.DataFrame({"serie": serie, "lag": range(len(r)), "acf": r}))
    car = pd.DataFrame(linhas).set_index("serie")
    # Estratos de CNES por consistência temporal (D17), só com o treino.
    estr = estratos_cnes(treino, cfg["exploratoria"]["consistencia"])
    estr.to_csv(saida / "estratos_cnes.csv", index=False)
    consist = set(estr.loc[estr["estrato"] == "consistentes", "cnesexecutante"])
    log.info(f"Estratos (D17): {len(consist)} CNES consistentes, {len(estr) - len(consist)} em 'outros'")
    car["cnesexecutante"] = car.index.str.split("|").str[0]
    # Modelos locais só nas séries longas das CNES consistentes.
    car["modelo_local"] = ((car["semanas_ativas"] >= cfg["exploratoria"]["min_semanas_ativas_modelo_local"])
                           & car["cnesexecutante"].isin(consist))

    # Teste de wavelets (opcional, R + locits)
    p_wav = None
    if shutil.which("Rscript"):
        ent, sai = saida / "_wavelet_entrada.csv", saida / "_wavelet_saida.csv"
        treino[["serie", "semana", "y"]].to_csv(ent, index=False)
        r = subprocess.run(["Rscript", str(Path(__file__).resolve().parents[1] / "r" / "teste_wavelet.R"), str(ent), str(sai)],
                           capture_output=True, text=True)
        log.info(f"Rscript: {(r.stdout + r.stderr).strip()[:300] or 'ok'}")
        if sai.exists():
            w = pd.read_csv(sai, dtype={"serie": str}).set_index("serie")
            car = car.join(w)
            p_wav = w["p_wavelet"]
    else:
        log.info("Rscript não encontrado: teste de wavelets não executado.")

    decisao = decidir_esquema(car, cfg["exploratoria"], p_wav)
    decisao["fim_treino_usado"] = str(fim_treino.date())
    log.info(f"Regra (Cerqueira et al. 2020): {decisao['teste']}; {decisao['n_series']} séries elegíveis; "
             f"fração não estacionária = {decisao['fracao_nao_estacionaria']}; esquema = {decisao['esquema']}")

    car.reset_index().to_csv(saida / "caracterizacao_series.csv", index=False)
    if acfs:
        pd.concat(acfs).to_csv(saida / "acf_series.csv", index=False)
    resumo = car.groupby("classe").agg(series=("media", "size"), media_faltas=("media", "mean")).reset_index()
    resumo.to_csv(saida / "resumo_exploratoria.csv", index=False)
    with open(saida / "esquema.json", "w", encoding="utf-8") as f:
        json.dump(decisao, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    log.manifesto([saida / f"painel_h{cfg['avaliacao']['horizonte_principal']}.parquet"], [saida / "caracterizacao_series.csv", saida / "esquema.json"])


if __name__ == "__main__":
    main()
