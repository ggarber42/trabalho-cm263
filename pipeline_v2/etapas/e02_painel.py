"""e02 — Painel CNES-semana e base por consulta.

Objetivo: montar, para cada horizonte h, o painel semanal (y, agenda conhecida na origem,
agregados do bloco B, defasagens, calendário) e a base por consulta com a semana e a marcação
de agenda conhecida.
Entradas: saidas/<perfil>/consultas.parquet.
Saídas: painel_h{h}.parquet, base_consulta.parquet, fracao_agenda_conhecida.csv.
Uso: python3 etapas/e02_painel.py --config config.yaml --perfil smoke --cnes 2237385
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.painel import chave_serie, montar_painel, origem_da_semana, semana_de  # noqa: E402


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e02_painel")
    cons = pd.read_parquet(saida / "consultas.parquet")
    if args.cnes:
        antes = len(cons)
        cons = cons[cons["cnesexecutante"].isin(args.cnes)].reset_index(drop=True)
        log.info(f"Restrito a {args.cnes}: {len(cons):,} de {antes:,} consultas")
        assert len(cons) > 0, "Nenhuma consulta para as CNES pedidas"

    saidas, fracoes = [], []
    for h in cfg["avaliacao"]["horizontes"]:
        p = montar_painel(cons, h, cfg)
        assert not p.duplicated(["serie", "semana"]).any(), "Painel com série-semana duplicada"
        caminho = saida / f"painel_h{h}.parquet"
        p.to_parquet(caminho, index=False)
        saidas.append(caminho)
        fr = p["n_agenda"].sum() / max(p["n_total"].sum(), 1)
        fracoes.append({"h": h, "fracao_agenda_conhecida": fr, "semanas_cnes": len(p),
                        "nao_avaliaveis": int((~p["avaliavel"]).sum())})
        log.info(f"h={h}: {len(p):,} série-semanas ({p['serie'].nunique()} séries); agenda conhecida na origem = {fr:.1%}; "
                 f"{int((~p['avaliavel']).sum())} semanas sem agenda nem falta (fora da avaliação)")
    pd.DataFrame(fracoes).to_csv(saida / "fracao_agenda_conhecida.csv", index=False)

    base = cons[cons["na_analise"]].copy()
    base["semana"] = semana_de(base["data_consulta"])
    base["serie"] = chave_serie(base["cnesexecutante"], base["especialidademae"])
    for h in cfg["avaliacao"]["horizontes"]:
        base[f"conhecida_h{h}"] = base["t0"] <= origem_da_semana(base["semana"], h)
    caminho = saida / "base_consulta.parquet"
    base.to_parquet(caminho, index=False)
    log.manifesto([saida / "consultas.parquet"], saidas + [caminho], {"cnes": args.cnes})


if __name__ == "__main__":
    main()
