"""S6/S7 — Tuning aninhado, treino final e previsão do teste para todos os experimentos do perfil (marco M4).

Objetivo
    Para cada combinação (variante de esquema, L, modo de agenda, modelo) e cada divisão de teste
    (ano ou holdout): escolher hiperparâmetros (e a janela, no deslizante) nos folds internos, treinar
    com a escolha, prever o teste e gravar previsões AGREGADAS por CNES-dia + metadados.

Protocolo
    Ver src/experimento.py (tuning só em folds anteriores ao teste; k_h e phi fora da amostra).
    Orçamento de tuning igual para todos os modelos (config tuning.n_configs). Sementes derivadas da
    semente do config. Nenhum resultado de teste é usado para escolher nada aqui.

Execução retomável
    Cada (variante, L, modo, modelo, divisão) grava <rotulo>.parquet e <rotulo>.json; se existirem, são
    PULADOS (use --refazer para refazer). Pode interromper e rodar de novo.

Entradas   config.yaml (perfil), base/amostra
Saídas     resultados/<escopo>_<perfil>/<variante>/L<L>_<modo>/<modelo>/<rotulo>.{parquet,json} e tuning_<rotulo>.csv
           logs/S7_experimentos_*.json
Como rodar
    python3 etapas/S7_experimentos.py                  # perfil do config (smoke por padrão)
    PERFIL=completo python3 etapas/S7_experimentos.py --jobs 11
    python3 etapas/S7_experimentos.py --estimativa     # só lista as tarefas e conta o que falta
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.dados import carrega_contexto  # noqa: E402
from src.experimento import executa_unidade  # noqa: E402
from src.models import lista_modelos_disponiveis  # noqa: E402
from src.splits import gera_divisoes, valida_divisao  # noqa: E402
from src.utils_io import RegistroEtapa, grava_json, sha256_arquivo  # noqa: E402

_CACHE: dict = {}      # contexto por (perfil, L, modo) — vive dentro de cada processo trabalhador


def raiz_resultados(cfg: dict) -> Path:
    """Pasta de resultados do perfil: resultados/<escopo>_<perfil>."""
    return caminho(cfg, cfg["saidas"]["raiz"], f"{cfg['escopo']}_{cfg['perfil_ativo']}")


def divisoes_da_variante(cfg: dict, ctx, variante: str) -> dict[str, dict]:
    """Gera as divisões (com folds internos) de uma variante de esquema do config, por rótulo de divisão.

    Parâmetros:
        cfg: configuração. ctx: Contexto (usa as datas de ctx.P). variante: chave de `variantes_esquema`.

    Retorna:
        {rotulo: {janela_meses ou None: Divisao}}. No deslizante há uma Divisao por janela candidata.

    Cuidados:
        # ANTI-VAZAMENTO: todas passam por ``valida_divisao`` antes de seguir.
    """
    spec = cfg["variantes_esquema"][variante]
    esq = spec["esquema"]
    dv = cfg["divisao"]
    dias = ctx.P["dia"]
    comum = dict(lag=ctx.lag, offsets_folds=cfg["tuning"]["folds_internos"])
    if esq == "deslizante":
        saida: dict[str, dict] = {}
        for w in cfg["tuning"]["janelas_meses"]:
            for d in gera_divisoes(dias, "deslizante", anos_teste=dv["anos_teste"], janela_meses=w, **comum):
                d.rotulo = d.rotulo.replace("deslizante_", f"{variante}_")
                saida.setdefault(d.rotulo, {})[w] = d
    else:
        extra = {}
        if esq == "holdout":
            extra["corte_holdout"] = dv["holdout"]["corte"]
        if esq == "purged_embargo":
            extra.update(purga_dias=spec.get("P", 0), embargo_dias=spec.get("E", 0))
        saida = {}
        for d in gera_divisoes(dias, esq, anos_teste=dv["anos_teste"], **extra, **comum):
            d.rotulo = d.rotulo.replace(f"{esq}_", f"{variante}_")
            saida[d.rotulo] = {None: d}
    for por_janela in saida.values():
        for d in por_janela.values():
            valida_divisao(d, dias)
    return saida


def tarefa(cfg: dict, variante: str, L: int, modo: str, modelo: str, refazer: bool, hash_dados: str) -> dict:
    """Executa todas as divisões de teste de (variante, L, modo, modelo); pula as já gravadas.

    Retorna:
        dict de resumo (n feitas, n puladas, tempo). Seguro para rodar em processo trabalhador.
    """
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    chave = (cfg["perfil_ativo"], L, modo)
    if chave not in _CACHE:
        _CACHE.clear()
        _CACHE[chave] = carrega_contexto(cfg, L, modo)
    ctx = _CACHE[chave]
    pasta = raiz_resultados(cfg) / variante / f"L{L}_{modo}" / modelo
    pasta.mkdir(parents=True, exist_ok=True)
    feitas = puladas = 0
    for rotulo, por_janela in divisoes_da_variante(cfg, ctx, variante).items():
        if (pasta / f"{rotulo}.json").exists() and not refazer:
            puladas += 1
            continue
        with threadpool_limits(limits=int(cfg["experimento"].get("threads_por_tarefa", 1))):
            pred, meta, tun = executa_unidade(ctx, modelo, por_janela, cfg, variante)
        meta.update({"escopo": cfg["escopo"], "perfil": cfg["perfil_ativo"], "hash_dados_sha256": hash_dados})
        pred.to_parquet(pasta / f"{rotulo}.parquet", index=False)
        tun.to_csv(pasta / f"tuning_{rotulo}.csv", index=False)
        grava_json(meta, pasta / f"{rotulo}.json")           # o JSON é gravado por último = marca de conclusão
        feitas += 1
    import resource
    pico_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024      # Linux: KB -> MB
    return {"variante": variante, "L": L, "modo": modo, "modelo": modelo, "feitas": feitas, "puladas": puladas,
            "tempo_s": round(time.time() - t0, 1), "pico_ram_mb": round(pico_mb)}


def lista_tarefas(cfg: dict) -> tuple[list[tuple], list[str]]:
    """Expande os blocos do config em tarefas únicas (variante, L, modo, modelo) e lista modelos indisponíveis."""
    tarefas, ausentes = [], set()
    for bloco in cfg["experimento"]["blocos"]:
        ok, falta = lista_modelos_disponiveis(bloco["modelos"])
        ausentes.update(falta)
        for v in bloco["variantes"]:
            for L in bloco["lags"]:
                for modo in bloco["modos"]:
                    for m in ok:
                        tarefas.append((v, L, modo, m))
    unicas = sorted(set(tarefas), key=lambda t: (t[1], t[2], t[0], t[3]))
    return unicas, sorted(ausentes)


def main() -> None:
    """Roda (ou retoma) todos os experimentos do perfil ativo."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=None, help="processos em paralelo (padrão: config experimento.n_jobs)")
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--estimativa", action="store_true", help="só lista tarefas e o que falta")
    args = ap.parse_args()
    cfg = carrega_config()
    reg = RegistroEtapa(f"S7_experimentos_{cfg['perfil_ativo']}", caminho(cfg, cfg["saidas"]["logs"]),
                        {"perfil": cfg["perfil_ativo"], "semente": cfg["semente"]})
    tarefas, ausentes = lista_tarefas(cfg)
    if ausentes:
        print(f"[S7] AVISO: modelos indisponíveis neste ambiente (biblioteca ausente): {ausentes}")
    import os
    n_jobs = args.jobs or cfg["experimento"].get("n_jobs") or max(1, (os.cpu_count() or 2) - 1)
    print(f"[S7] perfil={cfg['perfil_ativo']} | tarefas={len(tarefas)} | n_jobs={n_jobs} | saída: {raiz_resultados(cfg)}")
    if args.estimativa:
        for t in tarefas:
            print("  ", t)
        return
    arquivo = (caminho(cfg, "dados", "derivados", f"amostra_smoke_{cfg['escopo']}.parquet") if cfg.get("usa_amostra")
               else caminho(cfg, cfg["dados"]["base"]))
    hash_dados = sha256_arquivo(arquivo)
    t0 = time.time()
    if n_jobs == 1:
        resumo = []
        for i, t in enumerate(tarefas, 1):
            r = tarefa(cfg, *t, args.refazer, hash_dados)
            resumo.append(r)
            print(f"[S7] {i}/{len(tarefas)} {t} feitas={r['feitas']} puladas={r['puladas']} ({r['tempo_s']}s)", flush=True)
    else:
        from joblib import Parallel, delayed
        resumo = Parallel(n_jobs=n_jobs, verbose=5)(delayed(tarefa)(cfg, *t, args.refazer, hash_dados) for t in tarefas)
    print(f"[S7] pico de RAM de um processo trabalhador: {max(r['pico_ram_mb'] for r in resumo)} MB")
    print(f"[S7] concluído em {time.time() - t0:.0f}s | divisões feitas: {sum(r['feitas'] for r in resumo)} | "
          f"puladas (já prontas): {sum(r['puladas'] for r in resumo)}")
    reg.finaliza([arquivo], [raiz_resultados(cfg)], {"tarefas": len(tarefas), "modelos_indisponiveis": ausentes,
                                                    "feitas": sum(r["feitas"] for r in resumo)})


if __name__ == "__main__":
    main()
