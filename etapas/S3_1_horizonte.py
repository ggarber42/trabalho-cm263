"""S3.1 — Estimativa e justificativa do horizonte h e do atraso de disponibilidade d (marco M2).

Objetivo
    Trocar o "L = 3" (escolha inicial sem justificativa) por evidência descritiva:
      1. antecedência do agendamento (consulta − t0): percentis, por ano, e fração com antecedência >= h;
      2. fração da agenda final do dia que já estava agendada em t − h, h ∈ {1,2,3,5,7,10,14}
         (por consultas e por CNES-dia); monotonia em h;
      3. atraso de registro do desfecho (d): a base NÃO tem campo de data de registro; fica como premissa;
      4. memória da série (ACF da taxa diária de falta), só para orientar lags de variáveis, não o h;
      5. quanto o "pico em 14 dias" pesa (se a antecedência vale exatamente 14 em grande parte das
         linhas, t0 pode ser nominal e a reconstrução da agenda deve ser lida com cautela).
    A curva desempenho × horizonte (item 4 do PIPELINE) vem dos experimentos (S9/S13), não daqui.

Protocolo
    Só descreve. NÃO decide L: o h principal é um critério operacional a declarar no portão G2.
    Nenhum dado de teste é usado para escolher nada (não há modelo aqui).

Entradas   base (escopo do config.yaml) ou amostra do smoke
Saídas     tabelas/antecedencia_agendamento_<escopo>.csv, tabelas/agenda_conhecida_por_h_<escopo>.csv,
           tabelas/atraso_registro.csv, tabelas/acf_taxa_diaria_<escopo>.csv,
           entregaveis/justificativa_horizonte.md
Como rodar python3 etapas/S3_1_horizonte.py [--smoke]
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
from src.panels import COLS_CONSULTA, agenda_conhecida_por_h, constroi_painel, prepara_consultas_painel  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402

HS = [1, 2, 3, 5, 7, 10, 14]


def tabela_antecedencia(cons: pd.DataFrame) -> pd.DataFrame:
    """Percentis e frações da antecedência do agendamento (consulta − t0), geral e por ano.

    Parâmetros:
        cons: consultas (saída de prepara_consultas_painel).

    Retorna:
        DataFrame longo: grupo ("geral" ou ano), métrica, valor. Inclui `frac_exatamente_14`, `frac_maior_14`
        e `frac_antec_ge_h` para h em HS.
    """
    linhas = []
    grupos = [("geral", cons)] + [(str(a), g) for a, g in cons.groupby(cons["dia"].dt.year)]
    for nome, g in grupos:
        a = g["antec"].to_numpy()
        for p in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            linhas.append({"grupo": nome, "metrica": f"percentil_{p}", "valor": float(np.percentile(a, p))})
        linhas.append({"grupo": nome, "metrica": "n_consultas", "valor": float(len(a))})
        linhas.append({"grupo": nome, "metrica": "frac_exatamente_14", "valor": float((a == 14).mean())})
        linhas.append({"grupo": nome, "metrica": "frac_maior_14", "valor": float((a > 14).mean())})
        for h in HS:
            linhas.append({"grupo": nome, "metrica": f"frac_antec_ge_{h}", "valor": float((a >= h).mean())})
    return pd.DataFrame(linhas)


def tabela_agenda_conhecida(cons: pd.DataFrame) -> pd.DataFrame:
    """Fração da agenda final já conhecida em t − h, geral e por ano da consulta.

    Parâmetros:
        cons: consultas.

    Retorna:
        DataFrame com grupo, h, n_consultas_conhecidas, n_consultas_finais, fracao, fracao_media_cnes_dia.
    """
    partes = []
    for nome, g in [("geral", cons)] + [(str(a), g) for a, g in cons.groupby(cons["dia"].dt.year)]:
        t = agenda_conhecida_por_h(g, HS)
        t.insert(0, "grupo", nome)
        partes.append(t)
    out = pd.concat(partes, ignore_index=True)
    geral = out[out["grupo"] == "geral"]["fracao"].to_numpy()
    assert (np.diff(geral) <= 1e-12).all(), "A fração conhecida deveria ser não crescente em h"
    return out


def acf_medio(painel: pd.DataFrame, max_lag: int = 28) -> pd.DataFrame:
    """Autocorrelação média da taxa diária de falta (dias com agenda) entre as CNES.

    Parâmetros:
        painel: painel CNES-dia (qualquer L; só `y_rate`, `cnes`, `dia` são usados).
        max_lag: maior defasagem em dias.

    Retorna:
        DataFrame com lag e acf_media (média das ACF por CNES, preenchendo dias sem agenda por interpolação
        linear só para esta descrição). Usada para orientar as defasagens de variáveis, não o horizonte.
    """
    from statsmodels.tsa.stattools import acf
    acfs = []
    for _, x in painel.groupby("cnes"):
        s = x.set_index("dia")["y_rate"].interpolate(limit_direction="both").dropna()
        if len(s) > 3 * max_lag:
            acfs.append(acf(s.to_numpy(), nlags=max_lag, fft=True))
    m = np.mean(acfs, axis=0)
    return pd.DataFrame({"lag": range(len(m)), "acf_media": m, "n_cnes": len(acfs)})


def escreve_justificativa(destino: Path, ant: pd.DataFrame, ag: pd.DataFrame, acf_t: pd.DataFrame, escopo: str, smoke: bool) -> None:
    """Gera entregaveis/justificativa_horizonte.md com os números lidos das tabelas (nada digitado à mão)."""
    def g(m: str, grupo: str = "geral") -> float:
        """Valor da métrica `m` na tabela de antecedência para o grupo ('geral' ou um ano)."""
        return float(ant[(ant["grupo"] == grupo) & (ant["metrica"] == m)]["valor"].iloc[0])
    ag_g = ag[ag["grupo"] == "geral"].set_index("h")
    anos = sorted(a for a in ant["grupo"].unique() if a != "geral")
    linhas = [
        "# Justificativa do horizonte (h) e do atraso de disponibilidade (d)" + (" — AMOSTRA SMOKE" if smoke else ""),
        "",
        f"Gerado por `etapas/S3_1_horizonte.py` (escopo `{escopo}`). **Descritivo: não escolhe L.** "
        "O h principal é um critério operacional, a declarar no portão G2 (com fonte) antes de ver o teste.",
        "",
        "## 1. Antecedência do agendamento (consulta − t0)",
        "",
        f"- Mediana {g('percentil_50'):.0f} dias; p75 {g('percentil_75'):.0f}; p95 {g('percentil_95'):.0f}; p99 {g('percentil_99'):.0f}; máximo {g('percentil_100'):.0f}.",
        f"- Fração com antecedência **exatamente 14 dias**: {100 * g('frac_exatamente_14'):.1f}%; maior que 14: {100 * g('frac_maior_14'):.1f}%.",
        "- Por ano, fração exatamente 14: " + "; ".join(f"{a}: {100 * g('frac_exatamente_14', a):.0f}%" for a in anos) + ".",
        "- **Atenção:** a concentração em 14 dias sugere regra de abertura de agenda (ou `t0` nominal). Confirmar com o serviço antes de",
        "  interpretar a reconstrução da agenda como 'o que existia em t − h'.",
        "",
        "## 2. Fração da agenda final já conhecida em t − h",
        "",
        "| h (dias) | consultas já agendadas | fração (consultas) | fração média por CNES-dia |",
        "|---|---|---|---|",
    ]
    for h, r in ag_g.iterrows():
        linhas.append(f"| {h} | {int(r['n_consultas_conhecidas']):,} | {100 * r['fracao']:.1f}% | {100 * r['fracao_media_cnes_dia']:.1f}% |".replace(",", "."))
    linhas += [
        "",
        "Se a fração for bem menor que 1, avaliar com a agenda FINAL (`ag`, `m_*`) é otimista; o painel traz as duas versões",
        "(`ag`/`m_*` = final; `ag_h`/`m_*_h` = conhecida em t − L) e `config.yaml: agenda_modo` escolhe qual entra nos modelos.",
        "",
        "## 3. Atraso de registro do desfecho (d)",
        "",
        "A base **não tem** campo com a data em que o desfecho foi registrado (ver `tabelas/atraso_registro.csv`). "
        "d fica como premissa (`config.yaml: atraso_d`, padrão 0 = o desfecho do dia t−L já é conhecido em t−L). "
        "**A confirmar com o serviço.** Sensibilidade: o painel aceita d > 0 (defasagem efetiva L + d).",
        "",
        "## 4. Memória da série (ACF média da taxa diária de falta)",
        "",
        "| lag | " + " | ".join(str(int(l)) for l in (1, 2, 3, 7, 14, 21, 28)) + " |",
        "|---|" + "---|" * 7,
        "| ACF | " + " | ".join(f"{acf_t.loc[acf_t['lag'] == l, 'acf_media'].iloc[0]:.3f}" for l in (1, 2, 3, 7, 14, 21, 28)) + " |",
        "",
        "Orienta os lags das variáveis (ex.: mesmo dia da semana), não o horizonte.",
        "",
        "## 5. Critério operacional (a preencher pelo usuário, com fonte)",
        "",
        "- Prazo de confirmação/lembrete ao paciente: _a registrar_.",
        "- Fechamento da agenda e encaixe de lista de espera: _a registrar_.",
        "- **h principal declarado:** _pendente (G2)_. Hoje o `config.yaml` usa `lag_principal: 3` como RASCUNHO.",
        "",
        "## 6. Curva desempenho × horizonte",
        "",
        "Vem dos experimentos com L ∈ {1, 3, 7, 14} (figura `figuras/desempenho_vs_horizonte.*`, etapa S13), como análise de",
        "sensibilidade; nunca como critério de menor erro no teste.",
    ]
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    """Calcula as tabelas da S3.1 e grava a justificativa."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = carrega_config()
    escopo = cfg["escopo"]
    suf = "_smoke" if args.smoke else ""
    reg = RegistroEtapa(f"S3_1_horizonte_{escopo}{suf}", caminho(cfg, cfg["saidas"]["logs"]), {"smoke": args.smoke})
    if args.smoke:
        arquivo = caminho(cfg, "dados", "derivados", f"amostra_smoke_{escopo}.parquet")
        assert arquivo.exists(), "Rode antes: python3 etapas/S2_amostra_smoke.py"
        d = pd.read_parquet(arquivo, columns=COLS_CONSULTA)
        escopo_cnes = None
    else:
        arquivo = caminho(cfg, cfg["dados"]["base"])
        d = pd.read_parquet(arquivo, columns=COLS_CONSULTA)
        escopo_cnes = cnes_do_escopo(cfg)
    cons = prepara_consultas_painel(d, escopo_cnes)
    print(f"[S3.1] consultas: {len(cons):,} | CNES: {cons['cnes'].nunique()}")
    t_ant = tabela_antecedencia(cons)
    t_ag = tabela_agenda_conhecida(cons)
    painel = constroi_painel(cons, cfg["lag_principal"])
    t_acf = acf_medio(painel)

    pasta = caminho(cfg, cfg["saidas"]["tabelas"])
    o_ant = pasta / f"antecedencia_agendamento_{escopo}{suf}.csv"
    o_ag = pasta / f"agenda_conhecida_por_h_{escopo}{suf}.csv"
    o_acf = pasta / f"acf_taxa_diaria_{escopo}{suf}.csv"
    o_atr = pasta / f"atraso_registro{suf}.csv"
    o_md = caminho(cfg, "entregaveis", f"justificativa_horizonte{suf}.md")
    t_ant.to_csv(o_ant, index=False)
    t_ag.to_csv(o_ag, index=False)
    t_acf.to_csv(o_acf, index=False)
    pd.DataFrame([{"campo_de_registro_do_desfecho": "inexistente na base",
                   "premissa_d_dias": cfg.get("atraso_d", 0), "status": "a confirmar com o serviço"}]).to_csv(o_atr, index=False)
    escreve_justificativa(o_md, t_ant, t_ag, t_acf, escopo, args.smoke)
    print(t_ag[t_ag["grupo"] == "geral"][["h", "fracao", "fracao_media_cnes_dia"]].round(4).to_string(index=False))
    print("antecedência exatamente 14 dias:", round(float(t_ant[(t_ant.grupo == "geral") & (t_ant.metrica == "frac_exatamente_14")]["valor"].iloc[0]), 4))
    saidas = [o_ant, o_ag, o_acf, o_atr, o_md]
    grava_manifesto(f"S3_1_horizonte_{escopo}{suf}", [arquivo], saidas, caminho(cfg, cfg["saidas"]["manifestos"]))
    reg.finaliza([arquivo], saidas, {"consultas": int(len(cons))})


if __name__ == "__main__":
    main()
