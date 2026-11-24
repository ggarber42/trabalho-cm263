"""S4 — Gera, valida e descreve as divisões temporais dos quatro esquemas (marco M1).

Objetivo
    Para a base do escopo (ou a amostra do smoke), gerar com src/splits.py todas as divisões
    (holdout, expansiva, deslizante 12/24/36 meses, purged+embargo com P ∈ {0, 28, 90} e E ∈ {0, 7}),
    validar as regras anti-vazamento e gravar o relatório de n por divisão. Também calcula a purga
    RECOMENDADA a partir dos dados (antecedência consulta − t0 + L) para cada L.

Protocolo
    Divisões usam só a data da consulta (`datahora_consulta`). Nenhum modelo é treinado aqui.
    Escolha do esquema principal NÃO é feita aqui (portão G2, antes de olhar o teste).

Entradas   base (ou dados/derivados/amostra_smoke_<escopo>.parquet com --smoke), config.yaml
Saídas     tabelas/divisoes_<escopo>[_smoke].csv     n por divisão e fold
           tabelas/purga_recomendada_<escopo>[_smoke].csv
Como rodar python3 etapas/S4_divisoes.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.coorte import cnes_do_escopo  # noqa: E402
from src.splits import descreve_divisoes, gera_divisoes, purga_recomendada, valida_divisao  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402


def todas_as_divisoes(dias: pd.Series, cfg: dict) -> list:
    """Gera as divisões de todos os esquemas e variantes declaradas no config.yaml.

    Parâmetros:
        dias: datas de consulta (uma por linha).
        cfg: configuração (bloco `divisao`, `lag_principal`).

    Retorna:
        Lista de Divisao (cada uma com folds internos), rotuladas com a variante
        (ex.: "deslizante24m_2024", "purged_P28_E7_2024").

    Cuidados:
        # DECISÃO: a defasagem usada nas divisões é `lag_principal` (rascunho, L=3); a sensibilidade a L
        # altera só a purga recomendada (tabela separada), pois P = antecedência máxima + L.
    """
    dv = cfg["divisao"]
    L = cfg["lag_principal"]
    anos = dv["anos_teste"]
    saida = []
    saida += gera_divisoes(dias, "holdout", lag=L, corte_holdout=dv["holdout"]["corte"])
    saida += gera_divisoes(dias, "expansiva", anos_teste=anos, lag=L)
    for w in dv["deslizante"]["janelas_meses"]:
        for x in gera_divisoes(dias, "deslizante", anos_teste=anos, lag=L, janela_meses=w):
            x.rotulo = x.rotulo.replace("deslizante_", f"deslizante{w}m_")
            saida.append(x)
    for P in dv["purged_embargo"]["purga_dias"]:
        for E in dv["purged_embargo"]["embargo_dias"]:
            for x in gera_divisoes(dias, "purged_embargo", anos_teste=anos, lag=L, purga_dias=P, embargo_dias=E):
                x.rotulo = x.rotulo.replace("purged_embargo_", f"purged_P{P}_E{E}_")
                saida.append(x)
    return saida


def main() -> None:
    """Gera, valida e descreve as divisões; grava tabelas, manifesto e log."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="usa a amostra de CNES do smoke")
    args = ap.parse_args()
    cfg = carrega_config()
    escopo = cfg["escopo"]
    suf = "_smoke" if args.smoke else ""
    reg = RegistroEtapa(f"S4_divisoes_{escopo}{suf}", caminho(cfg, cfg["saidas"]["logs"]), {"smoke": args.smoke})

    cols = ["cnesexecutante", "datahora_consulta", "t0"]
    if args.smoke:
        arquivo = caminho(cfg, "dados", "derivados", f"amostra_smoke_{escopo}.parquet")
        assert arquivo.exists(), "Rode antes: python3 etapas/S2_amostra_smoke.py"
        d = pd.read_parquet(arquivo, columns=cols)
    else:
        arquivo = caminho(cfg, cfg["dados"]["base"])
        d = pd.read_parquet(arquivo, columns=cols)
        escopo_cnes = cnes_do_escopo(cfg)
        if escopo_cnes is not None:
            n0 = len(d)
            d = d[d["cnesexecutante"].isin(escopo_cnes)]
            print(f"[S4] filtro de escopo '{escopo}': removeu {n0 - len(d):,} linhas; restam {len(d):,}")
    d = d.reset_index(drop=True)
    dias = pd.to_datetime(d["datahora_consulta"])
    print(f"[S4] linhas: {len(d):,} | {dias.min().date()} a {dias.max().date()}")

    # Purga recomendada (antecedência consulta - t0 + L), para cada L do experimento.
    antec = (dias - pd.to_datetime(d["t0"])).dt.days
    prec = [{"L": L, "percentil": p, "antecedencia_dias": int(np.ceil(np.percentile(antec, p))),
             "P_recomendada": purga_recomendada(antec, L, p)} for L in cfg["lag"] for p in (99, 100)]
    t_prec = pd.DataFrame(prec)
    print(t_prec.to_string(index=False))

    divisoes = todas_as_divisoes(dias, cfg)
    for dv in divisoes:
        valida_divisao(dv, dias)          # levanta AssertionError em português se houver vazamento
    tab = descreve_divisoes(divisoes)
    n_rot = len({dv.rotulo for dv in divisoes})
    print(f"[S4] {n_rot} divisões geradas e validadas (nenhuma linha em dois conjuntos; treino anterior ao teste; "
          f"purga exata; folds internos para todos os anos de teste).")
    print(tab[tab.papel == "teste_externo"][["rotulo", "inicio", "n_treino", "n_teste_ou_validacao", "n_purgadas"]]
          .head(14).to_string(index=False))

    pasta = caminho(cfg, cfg["saidas"]["tabelas"])
    o1 = pasta / f"divisoes_{escopo}{suf}.csv"
    o2 = pasta / f"purga_recomendada_{escopo}{suf}.csv"
    tab.to_csv(o1, index=False)
    t_prec.to_csv(o2, index=False)
    grava_manifesto(f"S4_divisoes_{escopo}{suf}", [arquivo], [o1, o2], caminho(cfg, cfg["saidas"]["manifestos"]),
                    {"smoke": args.smoke})
    reg.finaliza([arquivo], [o1, o2], {"linhas": int(len(d)), "divisoes": n_rot})


if __name__ == "__main__":
    main()
