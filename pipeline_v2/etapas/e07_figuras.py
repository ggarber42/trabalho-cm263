"""e07 — Figuras dos resultados (observado × previsto, WAPE com IC95%, ganho com IC95%).

Objetivo: gerar, a partir dos arquivos do e05, as figuras do texto de Resultados, para cada
horizonte (h = 2 principal, h = 1 secundário) e estrato (consistentes, outros):
1. observado_previsto_h{h}.png — faltas semanais somadas (linhas da interseção do e05):
   observado, HGB (melhor modelo) e referência taxa × agenda;
2. wape_ic95_h{h}.png — WAPE no nível CNES-semana com IC95% (bootstrap em blocos);
3. ganho_ic95_h{h}.png — ganho de WAPE sobre a referência com IC95%; ponto cheio = significativo
   (p_holm < 0,05 e IC95% sem zero), vazado = não significativo.
Entradas: previsoes.parquet, metricas.csv, dm.csv.
Saídas: figuras/*.png e figuras/dados_observado_previsto.csv (os números de cada figura).
Uso: python3 etapas/e07_figuras.py --config config.yaml --perfil completo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402

CHAVE = ["serie", "semana"]
REF = "referencia_taxa_agenda"
MELHOR = "hgb_poisson"
ESTRATOS = ["consistentes", "outros"]
# DECISÃO: cores da paleta categórica validada (skill dataviz), em ordem fixa por entidade;
# o observado usa a tinta principal (não é uma série de modelo).
COR = {"observado": "#0b0b0b", MELHOR: "#2a78d6", REF: "#eb6834"}
TINTA2, GRADE = "#52514e", "#e4e3df"
NOMES = {"hgb_poisson": "HGB", "floresta": "Floresta", "mlp": "MLP", "glm_poisson": "GLM Poisson",
         "consulta_hgb": "HGB por consulta", "sarimax": "SARIMAX", "ets": "ETS", "croston_tsb": "Croston-TSB",
         REF: "Referência taxa × agenda", "referencia_mm4": "Média móvel 4 sem."}


def estilo(ax: plt.Axes) -> None:
    """Eixos discretos: sem bordas superior/direita, grade horizontal clara."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def serie_semanal(g: pd.DataFrame) -> pd.DataFrame:
    """Soma semanal de observado e previsto nas linhas da interseção (mesma regra do e05).
    Retorna DataFrame[semana, observado, <modelo>...]."""
    larga = g.pivot_table(index=CHAVE, columns="modelo", values="mu", aggfunc="first")
    cols = [m for m in larga.columns if larga[m].notna().any()]
    larga = larga[cols].dropna()
    y = g.groupby(CHAVE)["y"].first().reindex(larga.index)
    t = larga.assign(observado=y).groupby(level="semana").sum()
    return t[["observado", MELHOR, REF]].reset_index()


def fig_observado(prev: pd.DataFrame, h: int, pasta: Path) -> pd.DataFrame:
    """Figura 1 (dois painéis, um por estrato). Retorna os dados plotados."""
    fig, eixos = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    dados = []
    for ax, est in zip(eixos, ESTRATOS):
        t = serie_semanal(prev[(prev["h"] == h) & (prev["estrato"] == est)])
        dados.append(t.assign(h=h, estrato=est))
        ax.plot(t["semana"], t["observado"], color=COR["observado"], lw=2, label="Observado")
        ax.plot(t["semana"], t[MELHOR], color=COR[MELHOR], lw=2, label=NOMES[MELHOR])
        ax.plot(t["semana"], t[REF], color=COR[REF], lw=2, ls="--", label=NOMES[REF])
        estilo(ax)
        ax.set_title(f"CNES {est}", loc="left", fontsize=11)
        ax.set_ylabel("Faltas na semana", color=TINTA2)
        ax.set_ylim(bottom=0)
    eixos[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.32), fontsize=9)
    fig.suptitle(f"Faltas semanais observadas e previstas, teste 2024, h = {h}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(pasta / f"observado_previsto_h{h}.png", dpi=150)
    plt.close(fig)
    return pd.concat(dados)


def fig_wape(met: pd.DataFrame, h: int, pasta: Path) -> None:
    """Figura 2: WAPE (nível CNES) com IC95% por modelo, um painel por estrato."""
    m = met[(met["h"] == h) & (met["nivel"] == "cnes") & (met["recorte"] == "todas")]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, est in zip(eixos, ESTRATOS):
        d = m[m["estrato"] == est].sort_values("wape", ascending=False).reset_index(drop=True)
        cor = [COR[REF] if x == REF else (COR[MELHOR] if x == MELHOR else TINTA2) for x in d["modelo"]]
        ax.hlines(d.index, d["wape_ic_inf"], d["wape_ic_sup"], color=cor, lw=2)
        ax.scatter(d["wape"], d.index, color=cor, s=40, zorder=3)
        ref = d.loc[d["modelo"] == REF, "wape"]
        if len(ref):
            ax.axvline(ref.iloc[0], color=COR[REF], lw=1, ls=":")
        ax.set_yticks(d.index, [NOMES.get(x, x) for x in d["modelo"]])
        estilo(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRADE, linewidth=0.8)
        ax.set_title(f"CNES {est}", loc="left", fontsize=11)
        ax.set_xlabel("WAPE (menor é melhor) com IC95%", color=TINTA2)
    fig.suptitle(f"Erro por modelo, nível CNES-semana, teste 2024, h = {h}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(pasta / f"wape_ic95_h{h}.png", dpi=150)
    plt.close(fig)


def fig_ganho(dm: pd.DataFrame, h: int, pasta: Path) -> None:
    """Figura 3: ganho de WAPE sobre a referência com IC95%; cheio = significativo."""
    m = dm[dm["h"] == h]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, est in zip(eixos, ESTRATOS):
        d = m[m["estrato"] == est].sort_values("ganho_wape").reset_index(drop=True)
        cor = [COR[MELHOR] if x == MELHOR else TINTA2 for x in d["modelo"]]
        ax.hlines(d.index, 100 * d["ic95_inf"], 100 * d["ic95_sup"], color=cor, lw=2)
        sig = d["significativo"].astype(bool)
        ax.scatter(100 * d.loc[sig, "ganho_wape"], d.index[sig], color=[c for c, s in zip(cor, sig) if s], s=45, zorder=3,
                   label="Significativo (p_holm < 0,05)")
        ax.scatter(100 * d.loc[~sig, "ganho_wape"], d.index[~sig], facecolor="white", edgecolor=[c for c, s in zip(cor, sig) if not s],
                   s=45, zorder=3, lw=1.5, label="Não significativo")
        ax.axvline(0, color=TINTA2, lw=1)
        ax.set_yticks(d.index, [NOMES.get(x, x) for x in d["modelo"]])
        estilo(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRADE, linewidth=0.8)
        ax.set_title(f"CNES {est}", loc="left", fontsize=11)
        ax.set_xlabel("Ganho de WAPE sobre a referência (%) com IC95%", color=TINTA2)
    eixos[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Ganho sobre a referência taxa × agenda, teste 2024, h = {h}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(pasta / f"ganho_ic95_h{h}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e07_figuras")
    pasta = saida / "figuras"
    pasta.mkdir(exist_ok=True)
    prev = pd.read_parquet(saida / "previsoes.parquet")
    met = pd.read_csv(saida / "metricas.csv")
    dm = pd.read_csv(saida / "dm.csv")
    dados, saidas = [], []
    for h in cfg["avaliacao"]["horizontes"]:
        dados.append(fig_observado(prev, h, pasta))
        fig_wape(met, h, pasta)
        fig_ganho(dm, h, pasta)
        saidas += [pasta / f"{n}_h{h}.png" for n in ("observado_previsto", "wape_ic95", "ganho_ic95")]
    pd.concat(dados).to_csv(pasta / "dados_observado_previsto.csv", index=False)
    log.info(f"{len(saidas)} figuras em {pasta}")
    log.manifesto([saida / "previsoes.parquet", saida / "metricas.csv", saida / "dm.csv"], saidas)


if __name__ == "__main__":
    main()
