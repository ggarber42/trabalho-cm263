"""S10 — Interpretabilidade: importância por permutação (variável e grupo) com estabilidade entre anos.

Objetivo
    Para modelos de árvore de série CNES-dia (config interpretabilidade.modelos), reajustar com os
    hiperparâmetros JÁ ESCOLHIDOS por S7 (arquivo .json de cada divisão) e medir quanto o WAPE do teste
    piora quando cada variável (ou grupo de variáveis) é embaralhada entre as linhas do teste.

Protocolo
    * Descritivo (regra 6 do CLAUDE.md): nada é escolhido a partir da importância.
    * O reajuste usa só o treino da divisão (mesma `Divisao` de S7); o embaralhamento atua só nas
      COLUNAS DE ENTRADA das linhas de teste, nunca no desfecho nem na exposição.
    * Sementes derivadas da semente do config; `n_repeticoes` embaralhamentos por variável.
    * Limitação declarada: importância por permutação com variáveis correlacionadas (r7/r14/r28/r91) divide
      o efeito entre elas; por isso os GRUPOS são a leitura principal. SHAP/ALE não foram implementados.

Entradas   resultados/<escopo>_<perfil>/ (S7), config.yaml
Saídas     tabelas/importancia_<escopo>_<perfil>.csv (variável × ano × L), ..._grupos.csv, ..._estabilidade.csv
Como rodar python3 etapas/S10_importancia.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.dados import AGENDA_VARS, CALENDARIO, TAXAS, carrega_contexto  # noqa: E402
from src.experimento import semente_derivada  # noqa: E402
from src.metrics import wape  # noqa: E402
from src.models import REGISTRO, lista_modelos_disponiveis  # noqa: E402
from src.models.series import FEATS_CAT, FEATS_NUM  # noqa: E402
from src.utils_io import RegistroEtapa  # noqa: E402
import S7_experimentos as s7  # noqa: E402

GRUPOS = {"taxas_defasadas": TAXAS, "agenda": AGENDA_VARS + ["expo_bruta"], "calendario": CALENDARIO}
MODELOS_PADRAO = ["hgb_poisson", "xgboost"]
N_REP = 5


def importancia(ctx, modelo, params: dict, div, seed: int, cfg: dict) -> pd.DataFrame:
    """Importância por permutação de um modelo ajustado em `div.treino` e avaliado em `div.teste`.

    Retorna:
        DataFrame (item, tipo ["variavel"|"grupo"], wape_base, wape_perm, delta_wape [média dos N_REP]).

    Cuidados:
        # ANTI-VAZAMENTO: ajuste só com o treino; embaralha apenas colunas de entrada das linhas de teste.
    """
    idx = ctx.linhas_da_coorte(div.teste, div.inicio_teste)
    k_h = ctx.k_h(div.treino)
    m = REGISTRO[modelo](params, seed).fit(ctx, div.treino, k_h)
    y = ctx.P["f"].to_numpy()[idx]
    expo = ctx.expo(idx, k_h)

    def wape_de(F2) -> float:
        """WAPE do teste com a matriz de entrada F2 (a exposição não muda)."""
        mu = m._taxa(dataclasses.replace(ctx, F=F2), idx) * expo
        ok = np.isfinite(mu)
        return wape(y[ok], mu[ok])

    base = wape_de(ctx.F)
    rng = np.random.default_rng(seed)
    itens = [(v, "variavel", [v]) for v in FEATS_NUM + FEATS_CAT] + [(g, "grupo", c) for g, c in GRUPOS.items()]
    linhas = []
    for nome, tipo, cols in itens:
        deltas = []
        for _ in range(N_REP):
            F2 = ctx.F.copy()
            perm = rng.permutation(len(idx))                 # mesma permutação de linhas para todas as colunas do grupo
            for c in cols:
                col = F2[c].to_numpy().copy()
                col[idx] = col[idx][perm]
                F2[c] = col
            deltas.append(wape_de(F2) - base)
        linhas.append({"item": nome, "tipo": tipo, "wape_base": base, "delta_wape": float(np.mean(deltas)),
                       "dp_repeticoes": float(np.std(deltas))})
    return pd.DataFrame(linhas)


def main() -> None:
    """Calcula as importâncias para cada modelo, L e ano de teste da variante expansiva (modo conhecida)."""
    cfg = carrega_config()
    base = s7.raiz_resultados(cfg)
    reg = RegistroEtapa(f"S10_importancia_{cfg['perfil_ativo']}", caminho(cfg, cfg["saidas"]["logs"]), {"perfil": cfg["perfil_ativo"]})
    pedidos = cfg.get("interpretabilidade", {}).get("modelos", MODELOS_PADRAO)
    modelos, ausentes = lista_modelos_disponiveis(pedidos)
    if ausentes:
        print(f"[S10] AVISO: modelos indisponíveis: {ausentes}")
    todas = []
    for L in cfg["lag"]:
        ctx = carrega_contexto(cfg, L, "conhecida")
        divs = s7.divisoes_da_variante(cfg, ctx, "expansiva")
        for modelo in modelos:
            for rotulo, por_janela in divs.items():
                js = base / "expansiva" / f"L{L}_conhecida" / modelo / f"{rotulo}.json"
                if not js.exists():
                    print(f"[S10] pulando {js.name}: execute S7 antes")
                    continue
                meta = json.loads(js.read_text(encoding="utf-8"))
                div = por_janela[None]
                seed = semente_derivada(cfg["semente"], "importancia", modelo, L, rotulo)
                t = importancia(ctx, modelo, meta["params"], div, seed, cfg)
                t["modelo"], t["L"], t["rotulo"], t["ano"] = modelo, L, rotulo, int(str(div.inicio_teste.date())[:4])
                todas.append(t)
                print(f"[S10] {modelo} L={L} {rotulo}: base WAPE {t['wape_base'].iloc[0]:.4f}")
    assert todas, "Nenhuma importância calculada (S7 não rodou?)"
    out = pd.concat(todas, ignore_index=True)
    saida = caminho(cfg, cfg["saidas"]["tabelas"])
    suf = f"{cfg['escopo']}_{cfg['perfil_ativo']}"
    out[out["tipo"] == "variavel"].to_csv(saida / f"importancia_{suf}.csv", index=False)
    out[out["tipo"] == "grupo"].to_csv(saida / f"importancia_{suf}_grupos.csv", index=False)
    # Estabilidade: correlação de postos (Spearman) da importância por variável entre pares de anos.
    est = []
    for (modelo, L), d in out[out["tipo"] == "variavel"].groupby(["modelo", "L"]):
        w = d.pivot(index="item", columns="ano", values="delta_wape")
        rho = w.corr(method="spearman")
        for a in rho.columns:
            for b in rho.columns:
                if a < b:
                    est.append({"modelo": modelo, "L": L, "ano_a": a, "ano_b": b, "spearman": float(rho.loc[a, b])})
    pd.DataFrame(est).to_csv(saida / f"importancia_{suf}_estabilidade.csv", index=False)
    top = out[(out["tipo"] == "grupo")].groupby(["modelo", "L", "item"])["delta_wape"].mean().round(4)
    print(top.to_string())
    reg.finaliza([base], [saida / f"importancia_{suf}.csv"], {"n_repeticoes": N_REP})


if __name__ == "__main__":
    main()
