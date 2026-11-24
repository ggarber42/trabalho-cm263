"""Coorte de CNES e níveis A/B/C definidos com dados ATÉ UM CORTE (módulo compartilhado).

Etapa: S2. Port fiel dos scripts legados 10, 19 e 20 (mesmos critérios e limiares), com uma
diferença decisiva: tudo é calculado só com consultas de `inicio` até ANTES da data de corte,
para que a coorte usada num teste do ano Y não conheça nada de Y em diante (regra 5 do CLAUDE.md).
Não executa nada ao ser importado. Limiares vêm do config.yaml (bloco `coorte`).

Regras (iguais aos scripts legados):
  * coorte = CNES com consulta em todos os anos da janela, >= min_consultas_ano em cada ano
    COMPLETO da janela e atividade em >= min_semanas_ativas das semanas completas da janela;
  * quebra de regime = segmentação binária na série MENSAL de (faltas, consultas), aceita se
    G > G_min, |diferença de taxa| >= dif_min e segmentos >= min_meses_segmento meses;
  * nível A = sem quebras; C = >= 3 quebras ou taxa do último regime < taxa_min_ultimo_regime;
    B = demais casos (1 a 2 quebras).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import caminho


def cnes_do_escopo(cfg: dict[str, Any]) -> set[str] | None:
    """Devolve o conjunto de CNES do escopo do config.yaml (None = todas as CNES).

    Parâmetros:
        cfg: configuração (chaves `escopo` e `dados.flag_municipal`).

    Retorna:
        set de códigos CNES (str) para ``municipal``; None para ``geral`` (sem restrição).

    Cuidados:
        ``criterio_x`` ainda não foi definido pelo usuário e levanta NotImplementedError,
        em vez de assumir um critério (CLAUDE.md seção 2: escopo é decisão em aberto).
    """
    escopo = cfg["escopo"]
    if escopo == "geral":
        return None
    if escopo == "municipal":
        mun = pd.read_csv(caminho(cfg, cfg["dados"]["flag_municipal"]), dtype={"cnesexecutante": str})
        return set(mun.loc[mun["municipal"], "cnesexecutante"])
    raise NotImplementedError("Escopo 'criterio_x' ainda não definido; defina o critério antes de usá-lo.")


def prepara_consultas(d: pd.DataFrame, cnes_escopo: set[str] | None) -> pd.DataFrame:
    """Reduz a base às colunas necessárias e aplica o escopo.

    Parâmetros:
        d: base com `cnesexecutante`, `datahora_consulta` e `situacaoagenda`.
        cnes_escopo: CNES do escopo (None = todas).

    Retorna:
        DataFrame com colunas cnesexecutante (str), dia (datetime64) e falta (int8).
        Nenhuma exclusão além do escopo (decisão de 16/11/2026: sem exclusão por inatividade).
    """
    x = d[["cnesexecutante", "datahora_consulta", "situacaoagenda"]].copy()
    if cnes_escopo is not None:
        x = x[x["cnesexecutante"].isin(cnes_escopo)]
    out = pd.DataFrame({
        "cnesexecutante": x["cnesexecutante"].astype(str).to_numpy(),
        "dia": pd.to_datetime(x["datahora_consulta"]).to_numpy(),
        "falta": (x["situacaoagenda"] == "FALTANTE").to_numpy().astype(np.int8),
    })
    return out


def _lacuna_maxima(v: np.ndarray) -> int:
    """Maior sequência de semanas consecutivas sem consulta (v é booleano por semana)."""
    maior = atual = 0
    for ativo in v:
        atual = 0 if ativo else atual + 1
        maior = max(maior, atual)
    return maior


def _loglik(F: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Log-verossimilhança binomial (sem constante) de F faltas em N consultas."""
    p = np.where(N > 0, F / np.maximum(N, 1), 0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return F * np.log(p) + (N - F) * np.log(1 - p)


def _segmenta(f: np.ndarray, n: np.ndarray, lo: int, hi: int, saida: list[int],
              g_min: float, dif_min: float, min_len: int) -> None:
    """Segmentação binária recursiva (mesma lógica do script 10); acrescenta cortes em ``saida``.

    Parâmetros:
        f, n: faltas e consultas por mês; lo, hi: intervalo [lo, hi) atual; saida: lista de
        índices de corte (modificada in-place); g_min, dif_min, min_len: limiares.
    """
    if hi - lo < 2 * min_len:
        return
    cf, cn = np.r_[0, np.cumsum(f)], np.r_[0, np.cumsum(n)]
    F, N = cf[hi] - cf[lo], cn[hi] - cn[lo]
    base = _loglik(np.array(F), np.array(N))
    melhor: tuple = (0.0, None)
    for t in range(lo + min_len, hi - min_len + 1):
        F1, N1, F2, N2 = cf[t] - cf[lo], cn[t] - cn[lo], cf[hi] - cf[t], cn[hi] - cn[t]
        if N1 == 0 or N2 == 0:
            continue
        g = 2 * (_loglik(np.array(F1), np.array(N1)) + _loglik(np.array(F2), np.array(N2)) - base)
        if g > melhor[0]:
            melhor = (g, t, F1 / N1, F2 / N2)
    if melhor[1] is not None and melhor[0] > g_min and abs(melhor[2] - melhor[3]) >= dif_min:
        saida.append(melhor[1])
        _segmenta(f, n, lo, melhor[1], saida, g_min, dif_min, min_len)
        _segmenta(f, n, melhor[1], hi, saida, g_min, dif_min, min_len)


def nivel_da_cnes(n_quebras: int, taxa_ultimo_regime: float, taxa_min: float = 0.05) -> str:
    """Classifica uma CNES em A, B ou C (regra reconstruída no script 20).

    Parâmetros:
        n_quebras: número de quebras de regime detectadas.
        taxa_ultimo_regime: taxa de falta do último regime (0 a 1).
        taxa_min: abaixo disso o último regime é tratado como provável artefato de registro.

    Retorna:
        "A", "B" ou "C".
    """
    if n_quebras == 0:
        return "A"
    return "C" if (n_quebras >= 3 or taxa_ultimo_regime < taxa_min) else "B"


def coorte_e_niveis(consultas: pd.DataFrame, corte: str | pd.Timestamp, cfg_coorte: dict[str, Any],
                    ) -> pd.DataFrame:
    """Calcula coorte e níveis A/B/C usando SOMENTE consultas com dia < corte.

    Parâmetros:
        consultas: saída de ``prepara_consultas`` (cnesexecutante, dia, falta).
        corte: data de corte (exclusiva). Para testar o ano Y use ``f"{Y}-01-01"``.
        cfg_coorte: bloco `coorte` do config.yaml (inicio, min_consultas_ano, min_semanas_ativas,
            G_min, dif_min_pp, min_meses_segmento, taxa_min_ultimo_regime).

    Retorna:
        DataFrame indexado por cnes (str) com colunas: consultas, faltas, pct_sem, lacuna,
        todos_anos, vol_min, coorte (bool), e, para as da coorte, n_quebras, taxa_ultimo_regime,
        regimes, nivel ("A"/"B"/"C"; NaN fora da coorte).

    Cuidados:
        # ANTI-VAZAMENTO: a primeira operação é descartar tudo com dia >= corte; nenhuma
        # estatística posterior usa linhas do período de teste. O teste automático
        # tests/test_coorte.py confere que alterar dados após o corte não muda o resultado.
        # DECISÃO: a janela começa em `coorte.inicio` (2019-01-01), pois 2018 tem cobertura parcial;
        # o último ano só entra no critério de volume se estiver completo antes do corte.
    """
    corte = pd.Timestamp(corte)
    inicio = pd.Timestamp(cfg_coorte["inicio"])
    d = consultas[(consultas["dia"] >= inicio) & (consultas["dia"] < corte)].copy()
    assert len(d) > 0, f"Nenhuma consulta entre {inicio.date()} e {corte.date()}"
    assert d["dia"].max() < corte, "Dados do corte em diante entraram na coorte"

    d["ano"] = d["dia"].dt.year
    d["semana"] = d["dia"].dt.to_period("W-SUN").dt.start_time
    d["mes"] = d["dia"].dt.to_period("M")

    # Anos da janela; o último é "parcial" se o corte não cair em 1º de janeiro.
    ultimo_dia = corte - pd.Timedelta(days=1)
    anos = list(range(inicio.year, ultimo_dia.year + 1))
    anos_completos = [a for a in anos if pd.Timestamp(f"{a + 1}-01-01") <= corte]

    por_ano = d.groupby(["cnesexecutante", "ano"]).size().unstack(fill_value=0).reindex(columns=anos, fill_value=0)

    # Semanas completas da janela: de a primeira segunda-feira em/após `inicio` até a última
    # semana inteira antes do corte (segunda <= corte - 7 dias).
    primeira_seg = inicio + pd.Timedelta(days=(7 - inicio.weekday()) % 7)
    ultima_seg = corte - pd.Timedelta(days=7)
    ultima_seg = ultima_seg - pd.Timedelta(days=ultima_seg.weekday())
    semanas = pd.date_range(primeira_seg, ultima_seg, freq="7D")
    ativo_sem = (d[d["semana"].between(semanas[0], semanas[-1])].groupby(["cnesexecutante", "semana"]).size()
                 .unstack(fill_value=0).reindex(index=por_ano.index, columns=semanas, fill_value=0) > 0)

    c = pd.DataFrame({
        "consultas": por_ano.sum(axis=1),
        "pct_sem": ativo_sem.mean(axis=1),
        "lacuna": ativo_sem.apply(lambda r: _lacuna_maxima(r.values), axis=1),
    })
    c["faltas"] = d.groupby("cnesexecutante")["falta"].sum()
    c["todos_anos"] = (por_ano > 0).all(axis=1)
    c["vol_min"] = por_ano[anos_completos].min(axis=1)
    c["coorte"] = (c["todos_anos"] & (c["vol_min"] >= cfg_coorte["min_consultas_ano"])
                   & (c["pct_sem"] >= cfg_coorte["min_semanas_ativas"]))

    # ---- Quebras de regime e níveis (só para quem está na coorte) ----
    meses = pd.period_range(inicio.to_period("M"), ultimo_dia.to_period("M"), freq="M")
    mensal = d.groupby(["cnesexecutante", "mes"]).agg(f=("falta", "sum"), n=("falta", "size"))
    linhas = {}
    for cnes in c.index[c["coorte"]]:
        x = mensal.loc[cnes].reindex(meses, fill_value=0)
        cortes: list[int] = []
        _segmenta(x["f"].to_numpy(), x["n"].to_numpy(), 0, len(meses), cortes,
                  cfg_coorte["G_min"], cfg_coorte["dif_min_pp"], cfg_coorte["min_meses_segmento"])
        cortes = sorted(cortes)
        limites = [0] + cortes + [len(meses)]
        taxas, rotulos = [], []
        for a, b in zip(limites[:-1], limites[1:]):
            nn = x["n"].to_numpy()[a:b].sum()
            t = x["f"].to_numpy()[a:b].sum() / max(nn, 1)
            taxas.append(t)
            rotulos.append(f"{meses[a].strftime('%y-%m')}>{meses[b - 1].strftime('%y-%m')}:{100 * t:.0f}%")
        linhas[cnes] = {"n_quebras": len(cortes), "taxa_ultimo_regime": taxas[-1],
                        "regimes": " | ".join(rotulos),
                        "nivel": nivel_da_cnes(len(cortes), taxas[-1], cfg_coorte["taxa_min_ultimo_regime"])}
    niv = pd.DataFrame.from_dict(linhas, orient="index")
    c = c.join(niv)
    c.index.name = "cnes"
    c["corte"] = str(corte.date())
    return c
