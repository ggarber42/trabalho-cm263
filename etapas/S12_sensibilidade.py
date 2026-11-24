"""S12 — Robustez e sensibilidade (um fator por vez, a partir da configuração principal).

Objetivo
    (1) Consolidar em UMA tabela (`sensibilidade.csv`) o WAPE e o ganho sobre o baseline de cada modelo em cada
        nível dos fatores já avaliados por S9: esquema de divisão (no teste comum), defasagem L, modo de
        agenda e exclusão do Materno Infantil; e medir a ESTABILIDADE DO RANKING dos modelos (Spearman e
        Kendall em relação ao nível de referência).
    (2) Repetir com `sementes_repeticao` (≥ 5) os modelos estocásticos (rf, mlp, catboost) na configuração
        principal, reajustando com os hiperparâmetros já escolhidos por S7, para mostrar a variação por semente.

Protocolo
    Descritivo (regra 6 do CLAUDE.md): nenhuma escolha é feita aqui. A repetição por semente usa as mesmas
    divisões, o mesmo treino e o mesmo teste de S7; só a semente do estimador muda.
    Não cobertos (declarado): escopo geral × municipal, coorte global × por corte, poda de variáveis
    suspeitas de informação pós-desfecho (exigem novas execuções de S2/S7).

Entradas   tabelas/res_<escopo>_<perfil>_*.csv (S9), resultados/<escopo>_<perfil>/ (S7), config.yaml
Saídas     tabelas/sensibilidade_<escopo>_<perfil>.csv, ..._ranking.csv, ..._sementes.csv
Como rodar python3 etapas/S12_sensibilidade.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.dados import carrega_contexto  # noqa: E402
from src.experimento import semente_derivada  # noqa: E402
from src.metrics import wape  # noqa: E402
from src.models import REGISTRO, lista_modelos_disponiveis  # noqa: E402
from src.utils_io import RegistroEtapa  # noqa: E402
import S7_experimentos as s7  # noqa: E402

ESTOCASTICOS = ["rf", "mlp", "catboost"]
FATORES = {"esquema": ("esquemas_teste_comum", "variante"), "L": ("defasagens", "L"), "modo": ("modos_agenda", "modo")}


def consolida(tab: Path, pre: str) -> pd.DataFrame:
    """Empilha as tabelas de fator de S9 em (fator, nivel, modelo, n, wape, ganho_wape_vs_baseline, crps, cob80)."""
    partes = []
    for fator, (arq, col) in FATORES.items():
        f = tab / f"{pre}_{arq}.csv"
        if not f.exists():
            print(f"[S12] AVISO: {f.name} não existe (S9 não gerou este fator)")
            continue
        d = pd.read_csv(f).rename(columns={col: "nivel"})
        d["fator"] = fator
        partes.append(d[["fator", "nivel", "modelo", "n", "wape", "ganho_wape_vs_baseline", "crps", "cob80"]])
    f = tab / f"{pre}_principal_subgrupos.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["subgrupo"].isin(["geral", "sem_materno_infantil"])].rename(columns={"subgrupo": "nivel"})
        d["fator"] = "materno_infantil"
        d["ganho_wape_vs_baseline"] = np.nan
        base = d[d["modelo"] == "baseline_poisson"].set_index("nivel")["wape"]
        d["ganho_wape_vs_baseline"] = 1 - d["wape"] / d["nivel"].map(base)
        partes.append(d[["fator", "nivel", "modelo", "n", "wape", "ganho_wape_vs_baseline", "crps", "cob80"]])
    assert partes, "Nenhuma tabela de S9 encontrada; rode etapas/S9_avaliacao.py"
    return pd.concat(partes, ignore_index=True)


def estabilidade_ranking(sens: pd.DataFrame) -> pd.DataFrame:
    """Correlação de postos (Spearman e Kendall) do ranking por WAPE entre cada nível e o 1º nível do fator."""
    linhas = []
    for fator, d in sens.groupby("fator"):
        w = d.pivot(index="modelo", columns="nivel", values="wape").dropna()
        if w.shape[1] < 2 or len(w) < 3:
            continue
        ref = w.columns[0]
        for n in w.columns[1:]:
            linhas.append({"fator": fator, "referencia": ref, "nivel": n, "n_modelos": len(w),
                           "spearman": float(w[ref].corr(w[n], method="spearman")),
                           "kendall": float(w[ref].corr(w[n], method="kendall"))})
    return pd.DataFrame(linhas)


def repete_sementes(cfg: dict, base: Path) -> pd.DataFrame:
    """Reajusta rf/mlp/catboost com cada semente de `sementes_repeticao` na configuração principal.

    Retorna:
        DataFrame (modelo, L, semente, wape_pooled) — WAPE pooled sobre todos os anos de teste da variante
        expansiva (modo conhecida), com os hiperparâmetros escolhidos por S7.
    """
    modelos, ausentes = lista_modelos_disponiveis(ESTOCASTICOS)
    if ausentes:
        print(f"[S12] AVISO: modelos indisponíveis para a repetição: {ausentes}")
    linhas = []
    for L in cfg["lag"]:
        ctx = carrega_contexto(cfg, L, "conhecida")
        divs = s7.divisoes_da_variante(cfg, ctx, "expansiva")
        for modelo in modelos:
            for s in cfg["sementes_repeticao"]:
                ys, mus = [], []
                for rotulo, por_janela in divs.items():
                    js = base / "expansiva" / f"L{L}_conhecida" / modelo / f"{rotulo}.json"
                    if not js.exists():
                        continue
                    meta = json.loads(js.read_text(encoding="utf-8"))
                    div = por_janela[None]
                    idx = ctx.linhas_da_coorte(div.teste, div.inicio_teste)
                    k_h = ctx.k_h(div.treino)
                    sem = semente_derivada(cfg["semente"], "repeticao", modelo, L, s)
                    m = REGISTRO[modelo](meta["params"], sem).fit(ctx, div.treino, k_h)
                    mu = m.predict_mean(ctx, idx, k_h)
                    ys.append(ctx.P["f"].to_numpy()[idx])
                    mus.append(mu)
                if ys:
                    y, mu = np.concatenate(ys), np.concatenate(mus)
                    ok = np.isfinite(mu)
                    linhas.append({"modelo": modelo, "L": L, "semente": s, "wape_pooled": wape(y[ok], mu[ok])})
                    print(f"[S12] {modelo} L={L} semente {s}: WAPE {linhas[-1]['wape_pooled']:.4f}")
    return pd.DataFrame(linhas)


def main() -> None:
    """Gera as tabelas de sensibilidade e a repetição por semente."""
    cfg = carrega_config()
    base = s7.raiz_resultados(cfg)
    reg = RegistroEtapa(f"S12_sensibilidade_{cfg['perfil_ativo']}", caminho(cfg, cfg["saidas"]["logs"]), {"perfil": cfg["perfil_ativo"]})
    tab = caminho(cfg, cfg["saidas"]["tabelas"])
    suf = f"{cfg['escopo']}_{cfg['perfil_ativo']}"
    sens = consolida(tab, f"res_{suf}")
    sens.to_csv(tab / f"sensibilidade_{suf}.csv", index=False)
    rk = estabilidade_ranking(sens)
    rk.to_csv(tab / f"sensibilidade_{suf}_ranking.csv", index=False)
    print(rk.round(3).to_string(index=False))
    sem = repete_sementes(cfg, base)
    if len(sem):
        resumo = sem.groupby(["modelo", "L"])["wape_pooled"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        resumo.to_csv(tab / f"sensibilidade_{suf}_sementes.csv", index=False)
        print(resumo.round(4).to_string(index=False))
    reg.finaliza([base], [tab / f"sensibilidade_{suf}.csv"], {"sementes": cfg["sementes_repeticao"]})


if __name__ == "__main__":
    main()
