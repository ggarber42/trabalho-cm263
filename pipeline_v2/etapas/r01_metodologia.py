"""r01 — Gera entregaveis/metodologia_v2.md preenchendo o modelo com números dos resultados.

Objetivo: nenhum número da Metodologia é digitado à mão (CLAUDE.md, seção 6). O texto fica em
entregaveis/metodologia_v2_modelo.md com marcadores {{chave}}; este script calcula cada valor a
partir de saidas/<perfil>/ e de config.yaml e grava entregaveis/metodologia_v2.md.
Entradas: fluxograma.csv, consultas.parquet, painel_h*.parquet, fracao_agenda_conhecida.csv,
          esquema.json, estratos_cnes.csv, linhas_perdidas.csv, logs/e01_preparar.log,
          logs/e04_modelar.log, requirements.txt, config.yaml.
Saídas: ../entregaveis/metodologia_v2.md.
Uso: python3 etapas/r01_metodologia.py --config config.yaml --perfil completo
"""
from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]          # pasta do projeto (acima de pipeline_v2)
MODELO = RAIZ / "entregaveis" / "metodologia_v2_modelo.md"
DESTINO = RAIZ / "entregaveis" / "metodologia_v2.md"


def br(x: float, casas: int = 0) -> str:
    """Número no formato brasileiro (milhar com ponto, decimal com vírgula)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, casas: int = 1) -> str:
    """Proporção (0–1) como porcentagem brasileira."""
    return br(100 * x, casas) + "%"


def contagem_do_log(log: Path, rotulo: str) -> int:
    """Lê do log do e01 a linha '<rotulo>: N códigos ... completados'."""
    m = re.search(rf"{rotulo}: ([\d,.]+) códigos", log.read_text(encoding="utf-8"))
    assert m, f"Linha de correção de '{rotulo}' não encontrada em {log.name}"
    return int(m.group(1).replace(",", "").replace(".", ""))


def tabela_fluxo(f: pd.DataFrame) -> str:
    """Fluxograma em Markdown: passo, critério, linhas restantes, removidas, CNES."""
    linhas = ["| Passo | Critério | Consultas restantes | Removidas | CNES |", "|---|---|---|---|---|"]
    for _, r in f.iterrows():
        linhas.append(f"| {r['passo']} | {r['criterio']} | {br(r['linhas'])} | {br(r['removidas'])} | {r['cnes']} |")
    return "\n".join(linhas)


def valores(cfg: dict, saida: Path) -> dict[str, str]:
    """Calcula todos os valores do texto. Retorna dict {chave: texto}."""
    v: dict[str, str] = {}
    var, av, ex, tun = cfg["variaveis"], cfg["avaliacao"], cfg["exploratoria"], cfg["tuning"]
    log_e01 = saida / "logs" / "e01_preparar.log"
    v["cnes_corrigidos"] = br(contagem_do_log(log_e01, "cnesexecutante"))
    v["cnes_sol_corrigidos"] = br(contagem_do_log(log_e01, "cnes_solicitante"))

    fluxo = pd.read_csv(saida / "fluxograma.csv")
    v["tabela_fluxo"] = tabela_fluxo(fluxo)

    cons = pd.read_parquet(saida / "consultas.parquet", columns=["y", "na_analise", "cnesexecutante", "dias_desde_ultima_falta"])
    a = cons[cons["na_analise"]]
    assert len(a) == int(fluxo["linhas"].iloc[-1]), "consultas.parquet não bate com o fluxograma"
    v["n_consultas"], v["n_cnes"] = br(len(a)), br(a["cnesexecutante"].nunique())
    v["n_faltas"], v["pct_faltas"] = br(a["y"].sum()), pct(a["y"].mean())
    v["n_realizadas"] = br(len(a) - a["y"].sum())
    v["pct_ausente_dias"] = pct(a["dias_desde_ultima_falta"].isna().mean())

    p2 = pd.read_parquet(saida / "painel_h2.parquet", columns=["serie", "semana"])
    ini, fim = pd.Timestamp(cfg["periodo"]["inicio_analise"]), pd.Timestamp(cfg["periodo"]["fim_analise"])
    sem = p2.loc[(p2["semana"] >= ini) & (p2["semana"] <= fim), "semana"]
    v["n_semanas"], v["n_series"] = br(sem.nunique()), br(p2["serie"].nunique())

    fr = pd.read_csv(saida / "fracao_agenda_conhecida.csv").set_index("h")
    v["frac_h1"], v["frac_h2"] = pct(fr.loc[1, "fracao_agenda_conhecida"]), pct(fr.loc[2, "fracao_agenda_conhecida"])
    v["nao_avaliaveis_h2"] = br(fr.loc[2, "nao_avaliaveis"])

    esq = json.loads((saida / "esquema.json").read_text(encoding="utf-8"))
    assert esq["esquema"] == "rep_holdout", "O texto descreve holdout repetido; esquema.json diz outro"
    v["n_series_expl"], v["pct_nao_est"] = br(esq["n_series"]), pct(esq["fracao_nao_estacionaria"])
    rh = tun["rep_holdout"]
    v["n_dobras"], v["semanas_val"] = str(rh["n_repeticoes"]), str(rh["semanas_validacao"])

    c = ex["consistencia"]
    v["min_ativas"], v["janela"], v["min_faltas"] = str(c["min_semanas_ativas"]), str(c["janela_semanas"]), str(c["min_faltas_semana"])
    est = pd.read_csv(saida / "estratos_cnes.csv")
    cons_e = est["estrato"] == "consistentes"
    v["n_consistentes"], v["n_outros"] = str(int(cons_e.sum())), str(int((~cons_e).sum()))
    # faltas_por_semana é média na mesma janela para todas as CNES: a razão das somas é a fração das faltas.
    v["pct_faltas_consistentes"] = pct(est.loc[cons_e, "faltas_por_semana"].sum() / est["faltas_por_semana"].sum(), 0)
    v["min_local"] = str(ex["min_semanas_ativas_modelo_local"])

    # Fallback em h = 2: primeira ocorrência no log do e04 (bloco h = 2 vem primeiro, horizonte principal).
    txt = (saida / "logs" / "e04_modelar.log").read_text(encoding="utf-8")
    bloco_h2 = txt.split("h=2:", 1)[1].split("h=1:", 1)[0]
    m = re.search(r"ets: fallback para a referência em (\d+) de (\d+)", bloco_h2)
    assert m, "Contagem de fallback de h=2 não encontrada no log do e04"
    v["fb_h2"] = pct(int(m.group(1)) / int(m.group(2)), 0)

    lp = pd.read_csv(saida / "linhas_perdidas.csv")
    inter = lp[lp["h"] == 2].groupby("estrato")["na_intersecao"].first()
    v["inter_h2_cons"], v["inter_h2_out"] = br(inter["consistentes"]), br(inter["outros"])

    v.update(perfil=saida.name, te_m=str(var["te_suavizacao_m"]), lags=str(var["lags_y"]), lags_menos1=str(var["lags_y"] - 1),
             fourier=str(var["fourier_k"]), m_taxa=str(var["taxa_suavizacao_m"]), n_config=str(tun["n_configuracoes"]),
             semente=str(tun["semente"]), bloco=str(av["bootstrap_bloco_semanas"]), reamostragens=br(av["bootstrap_reamostragens"]))

    v["python"] = platform.python_version()
    req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").splitlines()
    libs = [r.replace("==", " ") for r in req if "==" in r and not r.startswith("#") and not r.lower().startswith("pytest")]
    v["bibliotecas"] = ", ".join(libs)
    return v


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "r01_metodologia")
    vals = valores(cfg, saida)
    texto = MODELO.read_text(encoding="utf-8")
    faltando = sorted(set(re.findall(r"\{\{(\w+)\}\}", texto)) - set(vals))
    assert not faltando, f"Marcadores sem valor: {faltando}"
    texto = re.sub(r"\{\{(\w+)\}\}", lambda m_: vals[m_.group(1)], texto)
    DESTINO.write_text(texto, encoding="utf-8")
    log.info(f"{len(vals)} valores preenchidos em {DESTINO.name}")
    log.manifesto([MODELO], [DESTINO])


if __name__ == "__main__":
    main()
