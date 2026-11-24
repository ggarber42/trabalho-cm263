"""S15 — Estimativa de tempo e memória do perfil COMPLETO, extrapolada do smoke (marco pós-M4).

Objetivo
    Ajudar a planejar a execução na máquina potente SEM treinar nada no conjunto inteiro: usa os tempos
    medidos nos JSON do smoke (tempo_s, n_treino, n_configs_avaliadas) e escala por (a) número de linhas
    (completo/smoke), (b) orçamento de tuning ((n_configs_completo+1)/(n_configs_smoke+1)) e (c) número de
    tarefas e divisões do perfil completo. Lê o nº de linhas do escopo completo direto do parquet (só contagem).

Aviso
    É uma ESTIMATIVA grosseira (±2x): o tempo real depende da CPU, do nº de threads por tarefa e do
    tamanho de cada ano de treino. A memória é medida carregando o contexto completo (sem treinar).

Entradas   resultados/<escopo>_smoke/**/*.json (rode antes: make smoke), config.yaml (perfil completo)
Saídas     tabelas/estimativa_recursos.csv e resumo no terminal
Como rodar python3 etapas/S15_estimativa_recursos.py [--jobs N] [--sem-memoria]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.coorte import cnes_do_escopo  # noqa: E402
from src.models import lista_modelos_disponiveis  # noqa: E402

EXPOENTE = {"rf": 1.15}      # DECISÃO: RF cresce um pouco mais que linear com o nº de linhas; os demais, linear


def tempos_smoke(raiz: Path) -> pd.DataFrame:
    """Lê os JSON do smoke e devolve o tempo médio por divisão (por modelo) e o nº médio de linhas de treino."""
    linhas = []
    for js in raiz.glob("*/L*_*/*/*.json"):
        m = json.loads(js.read_text(encoding="utf-8"))
        linhas.append({"modelo": m["modelo"], "tempo_s": m["tempo_s"], "n_treino": m["n_treino"], "n_cfg": m["n_configs_avaliadas"]})
    assert linhas, f"Sem resultados do smoke em {raiz}: rode `make smoke` antes"
    return pd.DataFrame(linhas).groupby("modelo").mean().reset_index()


def main() -> None:
    """Calcula a estimativa de CPU-horas, tempo de relógio e memória do perfil completo."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=None, help="processos paralelos previstos (padrão: núcleos − 1 desta máquina)")
    ap.add_argument("--sem-memoria", action="store_true", help="não carrega o contexto completo para medir RAM")
    args = ap.parse_args()
    cfg_s = carrega_config(perfil="smoke")
    cfg_c = carrega_config(perfil="completo")
    t_s = tempos_smoke(caminho(cfg_s, cfg_s["saidas"]["raiz"], f"{cfg_s['escopo']}_smoke"))

    d = pd.read_parquet(caminho(cfg_c, cfg_c["dados"]["base"]), columns=["cnesexecutante", "datahora_consulta"])
    esc = cnes_do_escopo(cfg_c)
    if esc is not None:
        d = d[d["cnesexecutante"].isin(esc)]
    n_consultas_full = len(d)
    n_celulas_full = d.drop_duplicates(["cnesexecutante", "datahora_consulta"]).shape[0]
    am = pd.read_parquet(caminho(cfg_s, "dados", "derivados", f"amostra_smoke_{cfg_s['escopo']}.parquet"), columns=["cnesexecutante", "datahora_consulta"])
    n_consultas_smoke = len(am)
    n_celulas_smoke = am.drop_duplicates(["cnesexecutante", "datahora_consulta"]).shape[0]
    print(f"escopo completo: {n_consultas_full:,} consultas / {n_celulas_full:,} CNES-dias | smoke: {n_consultas_smoke:,} / {n_celulas_smoke:,}")

    n_cfg_c = cfg_c["tuning"]["n_configs"]
    linhas, vistos = [], set()
    for b in cfg_c["experimento"]["blocos"]:
        ok, _ = lista_modelos_disponiveis(b["modelos"])
        for v in b["variantes"]:
            n_div = {"expansiva": 4, "holdout": 1}.get(v, 4)
            for L in b["lags"]:
                for modo in b["modos"]:
                    for m in ok:
                        if (v, L, modo, m) in vistos:
                            continue
                        vistos.add((v, L, modo, m))
                        r = t_s[t_s["modelo"] == m]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        razao = (n_consultas_full / n_consultas_smoke) if m == "paciente_hgb" else (n_celulas_full / n_celulas_smoke)
                        orcamento = 1.0 if m in ("naive", "baseline_poisson") else (n_cfg_c + 1) / (r["n_cfg"] + 1)
                        cpu = r["tempo_s"] * (razao ** EXPOENTE.get(m, 1.0)) * orcamento * n_div
                        linhas.append({"variante": v, "L": L, "modo": modo, "modelo": m, "divisoes": n_div, "cpu_s_estimado": cpu})
    t = pd.DataFrame(linhas)
    cpu_h = t["cpu_s_estimado"].sum() / 3600
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"tarefas: {len(t)} | CPU total estimada: {cpu_h:.1f} h | com {jobs} processos: ~{cpu_h / jobs * 1.2:.1f} h de relógio (+20% de folga)")
    por_modelo = t.groupby("modelo")["cpu_s_estimado"].sum().div(3600).round(2).rename("cpu_horas")
    print(por_modelo.to_string())
    destino = caminho(cfg_c, cfg_c["saidas"]["tabelas"], "estimativa_recursos.csv")
    t.to_csv(destino, index=False)
    if not args.sem_memoria:
        from src.dados import carrega_contexto
        ctx = carrega_contexto(cfg_c, 3, "conhecida")
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"memória: carregar o contexto completo (L=3) pico ~{pico:,.0f} MB por processo; "
              f"{jobs} processos ~{pico * jobs / 1024:.1f} GB no total (CNES-dias no painel: {len(ctx.P):,}; consultas: {len(ctx.raw):,})")
    print(f"[S15] detalhes em {destino}")


if __name__ == "__main__":
    main()
