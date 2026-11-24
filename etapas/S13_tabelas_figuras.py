"""S13 — Tabelas (Markdown) e figuras do manuscrito, geradas SÓ a partir dos CSVs (marco M4).

Objetivo
    Um único script lê os CSVs de tabelas/ (auditoria, fluxograma, coorte, divisões, avaliação) e produz as
    tabelas em Markdown (tabelas/md/) e as figuras (figuras/, PNG e PDF vetorial) usadas em Resultados.
    Nenhum número é digitado à mão: tudo vem dos arquivos.

Entradas   tabelas/*.csv (S1, S2, S3.1, S4, S9), config.yaml (escopo, perfil)
Saídas     tabelas/md/<nome>.md, figuras/<nome>.png e .pdf
Como rodar python3 etapas/S13_tabelas_figuras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402

# Paleta de Okabe-Ito (acessível a daltônicos); um modelo = uma cor, a mesma em todas as figuras.
PALETA = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#999999"]
NOMES = {"baseline_poisson": "Baseline Poisson", "naive": "Naive (4 sem.)", "hgb_poisson": "HGB Poisson", "rf": "Floresta aleatória",
         "poisson_reg": "Poisson regularizado", "lightgbm": "LightGBM", "paciente_hgb": "HGB por paciente"}


def cor(modelo: str, ordem: list[str]) -> str:
    """Cor fixa de um modelo, pela ordem alfabética global (estável entre figuras)."""
    return PALETA[sorted(ordem).index(modelo) % len(PALETA)]


def salva(fig: plt.Figure, nome: str, pasta: Path) -> list[Path]:
    """Grava a figura em PNG (300 dpi) e PDF (vetorial) e fecha."""
    pasta.mkdir(parents=True, exist_ok=True)
    saidas = []
    for ext in ("png", "pdf"):
        arq = pasta / f"{nome}.{ext}"
        fig.savefig(arq, dpi=300, bbox_inches="tight")
        saidas.append(arq)
    plt.close(fig)
    return saidas


def md_tabela(df: pd.DataFrame, formatos: dict[str, str] | None = None) -> str:
    """Converte um DataFrame em tabela Markdown; `formatos` mapeia coluna -> formato (ex.: '{:.3f}')."""
    formatos = formatos or {}
    cab = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    linhas = []
    for _, r in df.iterrows():
        cel = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cel.append("" if pd.isna(v) else formatos.get(c, "{:.4g}").format(v))
            else:
                cel.append(str(v))
        linhas.append("| " + " | ".join(cel) + " |")
    return cab + "\n".join(linhas) + "\n"


def fig_fluxograma(fluxo: pd.DataFrame, pasta: Path) -> list[Path]:
    """Fluxograma de seleção (caixas empilhadas) a partir de tabelas/fluxograma_selecao.csv."""
    f = fluxo[fluxo["etapa"].str.match(r"^[BC]")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 1.3 * len(f) + 0.5))
    ax.axis("off")
    for i, r in f.iterrows():
        y = len(f) - i
        n_fmt = f"{int(r['n_consultas']):,}".replace(",", ".")      # separador de milhar em português (só no número)
        txt = f"{r['descricao']}\nn = {n_fmt} consultas | {int(r['n_cnes'])} CNES"
        ax.text(0.5, y, txt, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=PALETA[6]))
        if i < len(f) - 1:
            ax.annotate("", xy=(0.5, y - 0.65), xytext=(0.5, y - 0.35), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_xlim(0, 1); ax.set_ylim(0.3, len(f) + 0.7)
    return salva(fig, "fluxograma_selecao", pasta)


def fig_ganho(dm: pd.DataFrame, modelos: list[str], pasta: Path) -> list[Path]:
    """Ganho relativo de WAPE sobre o baseline (IC 95% por bootstrap semanal) e p ajustado por Holm."""
    d = dm[(dm["comparacao"] == "vs_baseline") & (dm["perda"] == "erro_abs")].sort_values("ganho_wape")
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.2))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.errorbar(100 * r["ganho_wape"], i, xerr=[[100 * (r["ganho_wape"] - r["ganho_ic_inf"])], [100 * (r["ganho_ic_sup"] - r["ganho_wape"])]],
                    fmt="o", color=cor(r["modelo"], modelos), capsize=3)
        ax.text(1.02, i, f"p(Holm) = {r['p_holm']:.2g}", va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#666", lw=1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([NOMES.get(m, m) for m in d["modelo"]])
    ax.set_xlabel("Ganho de WAPE sobre o baseline (%), IC 95%")
    ax.margins(x=0.1)
    return salva(fig, "ganho_wape_vs_baseline", pasta)


def fig_horizonte(lag: pd.DataFrame, modelos: list[str], pasta: Path) -> list[Path]:
    """WAPE em função da defasagem L (mesmas linhas entre L), uma linha por modelo."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, g in lag.groupby("modelo"):
        g = g.sort_values("L")
        ax.plot(g["L"], g["wape"], marker="o", label=NOMES.get(m, m), color=cor(m, modelos))
    ax.set_xlabel("Defasagem / horizonte L (dias)"); ax.set_ylabel("WAPE"); ax.set_xticks(sorted(lag["L"].unique()))
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    return salva(fig, "desempenho_vs_horizonte", pasta)


def fig_calibracao(met: pd.DataFrame, modelos: list[str], pasta: Path) -> list[Path]:
    """Cobertura empírica dos intervalos de 80% e 50% por modelo, contra o nível nominal."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _, r in met.iterrows():
        ax.scatter([0.5, 0.8], [r["cob50"], r["cob80"]], color=cor(r["modelo"], modelos), label=NOMES.get(r["modelo"], r["modelo"]), s=50)
    ax.plot([0.4, 0.9], [0.4, 0.9], color="#666", lw=1, ls="--")
    ax.set_xlabel("Cobertura nominal"); ax.set_ylabel("Cobertura observada")
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, fontsize=8)
    return salva(fig, "calibracao_cobertura", pasta)


def fig_esquemas(esq: pd.DataFrame, modelos: list[str], pasta: Path) -> list[Path]:
    """WAPE por esquema de divisão (mesmo conjunto de dias de teste), barras agrupadas por modelo."""
    p = esq.pivot(index="modelo", columns="variante", values="wape")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    larg = 0.8 / len(p.columns)
    for j, v in enumerate(p.columns):
        ax.bar(np.arange(len(p)) + j * larg, p[v], width=larg, label=v, color=PALETA[(j + 1) % len(PALETA)])
    ax.set_xticks(np.arange(len(p)) + 0.4 - larg / 2); ax.set_xticklabels([NOMES.get(m, m) for m in p.index], rotation=15)
    ax.set_ylabel("WAPE"); ax.set_ylim(p.min().min() * 0.9, p.max().max() * 1.05); ax.legend(frameon=False, fontsize=8)
    return salva(fig, "comparacao_esquemas", pasta)


def fig_importancia(grp: pd.DataFrame, pasta: Path) -> list[Path]:
    """Barras da importância por permutação (aumento do WAPE) por grupo de variáveis, por modelo e L (média dos anos)."""
    m = grp.groupby(["modelo", "L", "item"])["delta_wape"].mean().reset_index()
    combos = m[["modelo", "L"]].drop_duplicates().to_numpy().tolist()
    fig, eixos = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 3.2), sharey=True, squeeze=False)
    for ax, (mod, L) in zip(eixos[0], combos):
        d = m[(m["modelo"] == mod) & (m["L"] == L)].sort_values("delta_wape")
        ax.barh(d["item"], d["delta_wape"], color="#4C78A8")
        ax.set_title(f"{mod}, L={L}", fontsize=9)
        ax.set_xlabel("aumento do WAPE ao embaralhar")
    return salva(fig, "importancia_grupos", pasta)


def fig_overbooking(curva: pd.DataFrame, pasta: Path) -> list[Path]:
    """Benefício líquido (custo relativo do excesso c = 2) × risco aceito q, uma linha por modelo, um painel por L."""
    Ls = sorted(curva["L"].unique())
    fig, eixos = plt.subplots(1, len(Ls), figsize=(4.6 * len(Ls), 3.6), sharey=True, squeeze=False)
    for ax, L in zip(eixos[0], Ls):
        d = curva[(curva["L"] == L) & (curva["q"] > 0)]
        for mod, g in d.groupby("modelo"):
            ax.plot(g["q"], g["beneficio_liquido_c2"], marker="o", ms=3, lw=1, label=mod)
        ax.set_title(f"L={int(L)}", fontsize=9)
        ax.set_xlabel("risco aceito q (quantil da NB)")
    eixos[0][0].set_ylabel("recuperadas − 2 × excesso")
    eixos[0][-1].legend(fontsize=6, ncol=2)
    return salva(fig, "overbooking_beneficio", pasta)


def main() -> None:
    """Gera tabelas Markdown e figuras do perfil/escopo ativos."""
    cfg = carrega_config()
    esc, perfil = cfg["escopo"], cfg["perfil_ativo"]
    pre = f"res_{esc}_{perfil}_"
    tab = caminho(cfg, cfg["saidas"]["tabelas"])
    fig_dir = caminho(cfg, cfg["saidas"]["figuras"], perfil)
    md_dir = tab / "md" / perfil
    md_dir.mkdir(parents=True, exist_ok=True)
    reg = RegistroEtapa(f"S13_tabelas_figuras_{esc}_{perfil}", caminho(cfg, cfg["saidas"]["logs"]), {"perfil": perfil})
    saidas: list[Path] = []
    lê = lambda n: pd.read_csv(tab / f"{pre}{n}.csv")          # noqa: E731
    met, dm, sg = lê("principal_metricas"), lê("principal_dm"), lê("principal_subgrupos")
    esq, lag, modo = lê("esquemas_teste_comum"), lê("defasagens"), lê("modos_agenda")
    modelos = sorted(met["modelo"].unique())

    def md(nome: str, df: pd.DataFrame, fmt: dict | None = None) -> None:
        """Grava `df` como tabelas/md/<perfil>/<nome>.md (formatos por coluna em `fmt`) e registra a saída."""
        arq = md_dir / f"{nome}.md"
        arq.write_text(md_tabela(df, fmt), encoding="utf-8")
        saidas.append(arq)

    md("principal_metricas", met[["modelo", "n", "wape", "vies", "crps", "cob80", "cob50", "larg80", "ece", "auc_celula", "ganho_wape_vs_baseline"]],
       {c: "{:.3f}" for c in ["wape", "vies", "crps", "cob80", "cob50", "larg80", "ece", "auc_celula", "ganho_wape_vs_baseline"]})
    md("principal_dm", dm[["comparacao", "modelo", "referencia", "perda", "diff_media", "p", "p_holm", "n_comparacoes", "T_semanas"]],
       {"diff_media": "{:.3f}", "p": "{:.4f}", "p_holm": "{:.4f}"})
    md("subgrupos_wape", sg.pivot(index="modelo", columns="subgrupo", values="wape").reset_index(), {})
    md("esquemas", esq[["variante", "modelo", "n", "wape", "crps", "ganho_wape_vs_baseline"]], {"wape": "{:.3f}", "crps": "{:.3f}", "ganho_wape_vs_baseline": "{:.3f}"})
    md("defasagens", lag[["L", "modelo", "n", "wape", "ganho_wape_vs_baseline"]], {"wape": "{:.3f}", "ganho_wape_vs_baseline": "{:.3f}"})
    md("modos_agenda", modo[["modo", "modelo", "n", "wape", "vies", "crps"]], {"wape": "{:.3f}", "vies": "{:.3f}", "crps": "{:.3f}"})
    for nome in ("coorte_por_corte_" + esc, "fluxograma_selecao", "reproducao_baseline"):
        arq = tab / f"{nome}.csv"
        if arq.exists():
            md(nome, pd.read_csv(arq), {})
    if (tab / "fluxograma_selecao.csv").exists():
        saidas += fig_fluxograma(pd.read_csv(tab / "fluxograma_selecao.csv"), fig_dir)
    saidas += fig_ganho(dm, modelos, fig_dir)
    saidas += fig_horizonte(lag, modelos, fig_dir)
    saidas += fig_calibracao(met, modelos, fig_dir)
    if len(esq):
        saidas += fig_esquemas(esq, modelos, fig_dir)
    suf = f"{esc}_{perfil}"
    for nome, arq in (("importancia_grupos", f"importancia_{suf}_grupos"), ("importancia_estabilidade", f"importancia_{suf}_estabilidade"),
                      ("overbooking", f"overbooking_{suf}"), ("sensibilidade", f"sensibilidade_{suf}"),
                      ("sensibilidade_ranking", f"sensibilidade_{suf}_ranking"), ("sensibilidade_sementes", f"sensibilidade_{suf}_sementes")):
        if (tab / f"{arq}.csv").exists():
            md(nome, pd.read_csv(tab / f"{arq}.csv"), {})
    if (tab / f"importancia_{suf}_grupos.csv").exists():
        saidas += fig_importancia(pd.read_csv(tab / f"importancia_{suf}_grupos.csv"), fig_dir)
    if (tab / f"overbooking_{suf}_curva.csv").exists():
        saidas += fig_overbooking(pd.read_csv(tab / f"overbooking_{suf}_curva.csv"), fig_dir)
    grava_manifesto(f"S13_tabelas_figuras_{esc}_{perfil}", [tab / f"{pre}principal_metricas.csv"], saidas, caminho(cfg, cfg["saidas"]["manifestos"]))
    reg.finaliza([tab], saidas)
    print(f"[S13] {len(saidas)} arquivos gerados em {fig_dir} e {md_dir}")


if __name__ == "__main__":
    main()
