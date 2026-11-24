"""Painel CNES-dia com defasagem L parametrizável (S3).

Etapa: S3. Constrói, para cada CNES e cada dia do calendário, as faltas observadas `f`, a agenda `ag`,
taxas móveis defasadas, calendário e agregados da agenda. Não executa nada ao ser importado.

Regra central (anti-vazamento): toda variável que resume o PASSADO da série (taxas r7/r14/r28/r91,
rdow4, rd1) usa só dias <= t − L_efetivo, em que ``L_efetivo = L + d`` (d = atraso de registro do
desfecho, premissa do config.yaml; padrão 0). O que é "conhecido com a agenda" entra de duas formas:
  * versão FINAL (`ag`, `m_*`): a agenda do dia como ficou no fim (reprodutível, mas otimista);
  * versão CONHECIDA em t − L (`ag_h`, `m_*_h`): só as consultas já agendadas, isto é, com
    antecedência (consulta − t0) >= L. É o que existiria de fato na hora da previsão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGENDA_VARS = ["m_taxaprev", "m_primeira", "m_idade", "m_dias_ate"]
COLS_CONSULTA = ["cnesexecutante", "datahora_consulta", "t0", "situacaoagenda", "n_faltas_previas", "n_consultas_previas",
                 "primeira_vez", "idade", "dias_ate_consulta"]


def feriados(anos: range | list[int]) -> set[pd.Timestamp]:
    """Feriados nacionais (e o de Porto Alegre, 2/fev) mais Carnaval, Sexta Santa e Corpus Christi.

    Parâmetros:
        anos: anos desejados.

    Retorna:
        Conjunto de pd.Timestamp. Mesma lista do script legado (utils_municipal.feriados); 20/nov
        entra a partir de 2024 (Consciência Negra nacional).
    """
    from dateutil.easter import easter
    s: set[pd.Timestamp] = set()
    for y in anos:
        e = pd.Timestamp(easter(y))
        fixos = [(1, 1), (2, 2), (4, 21), (5, 1), (9, 7), (9, 20), (10, 12), (11, 2), (11, 15), (12, 25)]
        if y >= 2024:
            fixos.append((11, 20))
        for m, d in fixos:
            s.add(pd.Timestamp(y, m, d))
        for off in (-48, -47, -2, 60):
            s.add(e + pd.Timedelta(days=off))
    return s


def prepara_consultas_painel(d: pd.DataFrame, cnes: set[str] | None = None) -> pd.DataFrame:
    """Reduz a base por consulta às colunas do painel e padroniza tipos.

    Parâmetros:
        d: base com as colunas de ``COLS_CONSULTA`` (a taxa de falta prévia pode vir pronta como
            `taxa_falta_prev` ou ser calculada a partir de n_faltas_previas/n_consultas_previas).
        cnes: se informado, mantém só essas CNES.

    Retorna:
        DataFrame com cnes (str), dia, t0, y (int8), antec (consulta − t0 em dias), tp (taxa de falta
        prévia, NaN se sem histórico), primeira_vez, idade, dias_ate.

    Cuidados:
        Nenhum preenchimento de ausentes aqui (a legado usava 0,17 fixo, que é uma estatística global).
        Médias de agenda ignoram NaN; imputação, quando necessária, é feita só com o treino (nos modelos).
    """
    x = d if cnes is None else d[d["cnesexecutante"].isin(cnes)]
    dia = pd.to_datetime(x["datahora_consulta"])
    t0 = pd.to_datetime(x["t0"])
    if "taxa_falta_prev" in x.columns:
        tp = x["taxa_falta_prev"].to_numpy(dtype=float)
    else:
        tp = (x["n_faltas_previas"] / x["n_consultas_previas"].where(x["n_consultas_previas"] > 0)).to_numpy(dtype=float)
    out = pd.DataFrame({
        "cnes": x["cnesexecutante"].astype(str).to_numpy(),
        "dia": dia.to_numpy(),
        "t0": t0.to_numpy(),
        "y": (x["situacaoagenda"] == "FALTANTE").to_numpy().astype(np.int8),
        "antec": (dia - t0).dt.days.to_numpy(),
        "tp": tp,
        "primeira_vez": x["primeira_vez"].to_numpy(dtype=float),
        "idade": x["idade"].to_numpy(dtype=float),
        "dias_ate": x["dias_ate_consulta"].to_numpy(dtype=float),
    })
    assert (out["dia"] == out["dia"].dt.normalize()).all(), "datahora_consulta tem hora; o painel agrupa por dia"
    assert (out["antec"] >= 0).all(), "Há consultas com t0 posterior à data da consulta"
    return out


def _agrega_agenda(c: pd.DataFrame, sufixo: str) -> pd.DataFrame:
    """Agrega por CNES-dia: contagem de consultas e médias das variáveis de agenda (nomeadas com sufixo)."""
    g = c.groupby(["cnes", "dia"]).agg(ag=("y", "size"), m_taxaprev=("tp", "mean"), m_primeira=("primeira_vez", "mean"),
                                       m_idade=("idade", "mean"), m_dias_ate=("dias_ate", "mean"))
    g = g.rename(columns={k: k + sufixo for k in ["ag"] + AGENDA_VARS})
    return g


def constroi_painel(cons: pd.DataFrame, lag: int, atraso: int = 0, dia_final: pd.Timestamp | None = None) -> pd.DataFrame:
    """Constrói o painel CNES-dia completo (todos os dias entre a 1ª consulta da CNES e `dia_final`).

    Parâmetros:
        cons: saída de ``prepara_consultas_painel``.
        lag: L, horizonte de previsão em dias (1, 3, 7, 14...).
        atraso: d, atraso de registro do desfecho em dias (premissa; padrão 0).
        dia_final: último dia do painel (padrão: última data de ``cons``).

    Retorna:
        DataFrame ordenado por (cnes, dia) com colunas:
        f, ag, ag_h, m_* e m_*_h, r7, r14, r28, r91, rdow4, rd1, y_rate, dow, mes, fer, posfer, ano.
        `ag` = agenda final do dia; `ag_h` = consultas já agendadas em t − L (antec >= L).

    Cuidados:
        # ANTI-VAZAMENTO: ``L_ef = lag + atraso``; taxas usam ``shift(L_ef)`` e as de mesmo dia da semana
        # usam defasagens múltiplas de 7 >= L_ef (para L=14 não usam o dia t−7, que é mais recente que t−14).
        # Nenhum fillna com média global (a versão legada usava a média da série inteira).
    """
    assert lag >= 1 and atraso >= 0, "lag deve ser >= 1 e atraso >= 0"
    L_ef = lag + atraso
    final = cons.groupby(["cnes", "dia"]).agg(f=("y", "sum")).join(_agrega_agenda(cons, ""))
    conhecidas = cons[cons["antec"] >= lag]
    final = final.join(_agrega_agenda(conhecidas, "_h")[[c for c in ["ag_h"] + [v + "_h" for v in AGENDA_VARS]]], how="left")
    final["ag_h"] = final["ag_h"].fillna(0)
    final = final.reset_index()

    fim = dia_final if dia_final is not None else cons["dia"].max()
    k0 = int(np.ceil(L_ef / 7))                      # menor múltiplo de 7 que respeita a defasagem
    ks = [k0 + i for i in range(4)]                  # 4 semanas: t−7·k0, ..., t−7·(k0+3)
    fer = feriados(range(cons["dia"].min().year, fim.year + 1))
    partes = []
    for cnes, x in final.groupby("cnes"):
        idx = pd.date_range(x["dia"].min(), fim, freq="D")
        x = x.set_index("dia").reindex(idx)
        for col in ("f", "ag", "ag_h"):
            x[col] = x[col].fillna(0)
        f, a = x["f"], x["ag"]
        out = pd.DataFrame({"cnes": cnes, "f": f, "ag": a, "ag_h": x["ag_h"]})
        for v in AGENDA_VARS:
            out[v] = x[v]
            out[v + "_h"] = x[v + "_h"]
        for w, mp in ((7, 3), (14, 7), (28, 14), (91, 45)):
            num = f.shift(L_ef).rolling(w, min_periods=mp).sum()
            den = a.shift(L_ef).rolling(w, min_periods=mp).sum()
            out[f"r{w}"] = num / den.where(den > 0)
        nd = sum(f.shift(7 * k) for k in ks)
        dd = sum(a.shift(7 * k) for k in ks)
        out["rdow4"] = (nd / dd.where(dd > 0)).fillna(out["r28"])
        a7 = a.shift(7 * k0)
        out["rd1"] = (f.shift(7 * k0) / a7.where(a7 > 0)).fillna(out["r28"])
        out["y_rate"] = f / a.where(a > 0)
        partes.append(out.rename_axis("dia").reset_index())
    P = pd.concat(partes, ignore_index=True)
    P["dow"] = P["dia"].dt.dayofweek
    P["mes"] = P["dia"].dt.month
    P["ano"] = P["dia"].dt.year
    P["fer"] = P["dia"].isin(fer).astype(int)
    P["posfer"] = (P["dia"] - pd.Timedelta(days=1)).isin(fer).astype(int)
    P.attrs.update({"lag": lag, "atraso": atraso, "lag_efetivo": L_ef})
    return P.sort_values(["cnes", "dia"]).reset_index(drop=True)


def agenda_conhecida_por_h(cons: pd.DataFrame, horizontes: list[int]) -> pd.DataFrame:
    """Fração da agenda final do dia que já estava agendada em t − h, para vários h (S3.1).

    Parâmetros:
        cons: saída de ``prepara_consultas_painel``.
        horizontes: lista de h (dias).

    Retorna:
        DataFrame com h, n_consultas_conhecidas, n_consultas_finais, fracao (ponderada por consultas),
        fracao_media_cnes_dia (média simples das razões ag_h/ag por CNES-dia com ag > 0).
        A fração deve ser não crescente em h (monotonia testada em tests/).
    """
    linhas = []
    total = len(cons)
    ag_final = cons.groupby(["cnes", "dia"]).size()
    for h in horizontes:
        conhecidas = cons[cons["antec"] >= h]
        ag_h = conhecidas.groupby(["cnes", "dia"]).size().reindex(ag_final.index, fill_value=0)
        linhas.append({"h": h, "n_consultas_conhecidas": len(conhecidas), "n_consultas_finais": total,
                       "fracao": len(conhecidas) / total, "fracao_media_cnes_dia": float((ag_h / ag_final).mean())})
    return pd.DataFrame(linhas)
