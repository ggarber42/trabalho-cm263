"""S2 — Coorte e níveis A/B/C definidos com dados até o corte de cada ano de teste (marco M1).

Objetivo
    Para cada ano de teste Y (config divisao.anos_teste), definir a coorte e os níveis usando
    SÓ consultas de 2019-01-01 até 31/12 de Y−1. Compara com a versão "global" (série inteira,
    a dos scripts 19/20) e com o arquivo legado niveis_municipal.csv.

Protocolo anti-vazamento
    Regra 5 do CLAUDE.md. Escopo vem do config.yaml (`escopo`), não do código.
    Nenhuma CNES é excluída por inatividade (decisão de 16/11/2026).

Entradas
    base (parquet), flag_municipal.csv (se escopo=municipal), niveis_municipal.csv (só para comparar).

Saídas (o escopo vai no nome do arquivo)
    dados/derivados/coorte_<escopo>_<corte>.csv     coorte+níveis por corte (e "global")
    tabelas/coorte_por_corte_<escopo>.csv           resumo: n CNES, consultas e níveis por corte
    tabelas/coorte_comparacao_global_<escopo>.csv   CNES que entram/saem/mudam de nível vs. global
    manifestos/S2_coorte_<escopo>.json, logs/S2_coorte_<escopo>_*.json

Como rodar (da raiz, sem -I)
    python3 etapas/S2_coorte_por_corte.py
    python3 etapas/S2_coorte_por_corte.py --smoke   # mesmas contas (é rápido); saídas com sufixo _smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.coorte import cnes_do_escopo, coorte_e_niveis, prepara_consultas  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402


def main() -> None:
    """Calcula coorte/níveis por corte e global, compara com o legado e grava tabelas."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="apenas sufixa as saídas (a etapa já é rápida)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    escopo = cfg["escopo"]
    sufixo = "_smoke" if args.smoke else ""
    reg = RegistroEtapa(f"S2_coorte_{escopo}{sufixo}", caminho(cfg, cfg["saidas"]["logs"]),
                        {"escopo": escopo, "smoke": args.smoke})

    arquivo = caminho(cfg, cfg["dados"]["base"])
    d = pd.read_parquet(arquivo, columns=["cnesexecutante", "datahora_consulta", "situacaoagenda"])
    escopo_cnes = cnes_do_escopo(cfg)
    cons = prepara_consultas(d, escopo_cnes)
    print(f"[S2] escopo={escopo} | consultas no escopo: {len(cons):,} | CNES: {cons['cnesexecutante'].nunique()}")

    cortes = {f"{y}-01-01": f"corte_{y}" for y in cfg["divisao"]["anos_teste"]}
    # Versão "global": toda a série (até 31/03/2026), como nos scripts 19/20 (só para comparação).
    fim = pd.Timestamp(cfg["dados"]["periodo_esperado"][1]) + pd.Timedelta(days=1)
    cortes[str(fim.date())] = "global"

    resultados = {}
    for corte, rotulo in cortes.items():
        r = coorte_e_niveis(cons, corte, cfg["coorte"])
        resultados[rotulo] = r
        print(f"[S2] {rotulo:10s} corte<{corte}: coorte={int(r['coorte'].sum()):3d} CNES "
              f"| níveis: {r['nivel'].value_counts().sort_index().to_dict()}")

    pasta_der = caminho(cfg, "dados", "derivados")
    pasta_tab = caminho(cfg, cfg["saidas"]["tabelas"])
    pasta_der.mkdir(parents=True, exist_ok=True)
    pasta_tab.mkdir(parents=True, exist_ok=True)
    saidas = []
    resumo, comp = [], []
    for rotulo, r in resultados.items():
        arq = pasta_der / f"coorte_{escopo}_{rotulo}{sufixo}.csv"
        r.to_csv(arq)
        saidas.append(arq)
        na = r[r["coorte"]]
        resumo.append({"corte": rotulo, "data_corte": r["corte"].iloc[0], "cnes_no_escopo_com_consulta": len(r),
                       "cnes_na_coorte": len(na), "consultas_da_coorte_ate_o_corte": int(na["consultas"].sum()),
                       **{f"nivel_{k}": int((na["nivel"] == k).sum()) for k in "ABC"}})
    g = resultados["global"]
    for rotulo, r in resultados.items():
        if rotulo == "global":
            continue
        entra = sorted(set(r.index[r["coorte"]]) - set(g.index[g["coorte"]]))
        sai = sorted(set(g.index[g["coorte"]]) - set(r.index[r["coorte"]]))
        comuns = sorted(set(r.index[r["coorte"]]) & set(g.index[g["coorte"]]))
        mudou = [c for c in comuns if r.loc[c, "nivel"] != g.loc[c, "nivel"]]
        comp.append({"corte": rotulo, "na_coorte_do_corte_e_no_global": len(comuns),
                     "so_no_corte (entram)": " ".join(entra), "so_no_global (saem)": " ".join(sai),
                     "mudam_de_nivel": " ".join(f"{c}:{r.loc[c, 'nivel']}->{g.loc[c, 'nivel']}" for c in mudou)})

    # Conferência contra o legado (niveis_municipal.csv): só faz sentido para escopo municipal.
    if escopo == "municipal":
        leg = pd.read_csv(caminho(cfg, cfg["dados"]["niveis_municipal"]), dtype={"cnes": str}).set_index("cnes")
        mesma_coorte = set(leg.index) == set(g.index[g["coorte"]])
        mesmos_niveis = bool((leg["nivel"] == g.loc[leg.index.intersection(g.index), "nivel"]).all()) if mesma_coorte else False
        print(f"[S2] versão global (com os limiares do config) reproduz o legado? "
              f"mesma coorte: {mesma_coorte} | mesmos níveis: {mesmos_niveis}")
        t_conf = pasta_tab / f"coorte_conferencia_legado_{escopo}{sufixo}.csv"
        pd.DataFrame([{"cnes_no_legado": len(leg), "mesma_coorte_que_o_legado": mesma_coorte,
                       "mesmos_niveis_que_o_legado": mesmos_niveis}]).to_csv(t_conf, index=False)
        saidas.append(t_conf)

    t_resumo = pasta_tab / f"coorte_por_corte_{escopo}{sufixo}.csv"
    t_comp = pasta_tab / f"coorte_comparacao_global_{escopo}{sufixo}.csv"
    pd.DataFrame(resumo).to_csv(t_resumo, index=False)
    pd.DataFrame(comp).to_csv(t_comp, index=False)
    saidas += [t_resumo, t_comp]
    print(pd.DataFrame(comp).to_string(index=False))

    grava_manifesto(f"S2_coorte_{escopo}{sufixo}", [arquivo], saidas, caminho(cfg, cfg["saidas"]["manifestos"]),
                    {"escopo": escopo})
    reg.finaliza([arquivo], saidas, {"consultas_escopo": int(len(cons))})
    print("[S2] concluído.")


if __name__ == "__main__":
    main()
