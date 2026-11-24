"""Etapa de preparação: leitura do bruto, padronização, filtro municipal, filtros de seleção e
histórico do paciente (opção B, D3). Substitui `preprocessar_gercon.ipynb` e
`Unificacao_GERCON_Porto_Alegre.ipynb`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Colunas do bruto que o pipeline usa. As demais não são lidas (D12).
COLUNAS_BRUTO = [
    "idsolicitacao", "data_solicitacao", "corregulador", "diasfilaespera", "complexidade",
    "especialidade", "especialidademae", "cid", "idpaciente", "idade", "cnes_solicitante",
    "tipounidade", "municipiosolicitante", "situacaoagenda", "datahora_consulta",
    "cnesexecutante", "possuiunidadeindicada", "ARQUIVO_ORIGEM",
]


def ler_bruto(caminho: Path) -> pd.DataFrame:
    """Lê o CSV unificado do GERCON com todas as colunas como texto.

    Retorna: DataFrame (uma linha por solicitação) com as colunas de COLUNAS_BRUTO.
    """
    return pd.read_csv(caminho, sep=";", dtype=str, encoding="utf-8-sig", usecols=COLUNAS_BRUTO)


def padronizar(df: pd.DataFrame, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Limpa texto, converte datas e números e corrige os códigos CNES (D5).

    - Datas truncadas ao dia: o arquivo de 2025 não tem hora; sem truncar, a regra data_ev < t0
      trataria anos diferentes de modo diferente.
    - CNES com zfill(7): arquivos antigos perderam o zero à esquerda.
    Retorna: cópia do DataFrame com `data_solicitacao`, `data_consulta` (datetime64, dia),
    `diasfilaespera` e `idade` numéricos.
    """
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].str.strip().replace("", pd.NA)
    df["data_solicitacao"] = pd.to_datetime(df["data_solicitacao"], format="mixed", errors="coerce").dt.normalize()
    df["data_consulta"] = pd.to_datetime(df["datahora_consulta"], format="mixed", errors="coerce").dt.normalize()
    df = df.drop(columns="datahora_consulta")
    df["diasfilaespera"] = pd.to_numeric(df["diasfilaespera"], errors="coerce")
    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    for c in ["cnesexecutante", "cnes_solicitante"]:
        curtos = df[c].notna() & (df[c].str.len() < 7)
        log(f"  {c}: {int(curtos.sum()):,} códigos com menos de 7 dígitos completados com zero")
        df[c] = df[c].str.zfill(7)
    return df


def contar_situacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem de todas as categorias de `situacaoagenda` por ano da consulta, antes dos filtros.

    Retorna: tabela larga (linhas = ano da consulta ou 'sem data'; colunas = situação, inclusive
    'sem situação').
    """
    ano = df["data_consulta"].dt.year.astype("Int64").astype(str).replace("<NA>", "sem data")
    sit = df["situacaoagenda"].fillna("sem situação")
    return pd.crosstab(ano, sit).rename_axis(index="ano_consulta", columns=None).reset_index()


def cnes_municipais(caminho_estab: Path, caminho_natjur: Path, cfg_sel: dict) -> pd.DataFrame:
    """Marca os estabelecimentos de gestão municipal (D2).

    Regra: natureza jurídica em `naturezas_municipais` E município gestor = `municipio_gestor`
    E razão social em `razoes_sociais`.
    Retorna: DataFrame[cnesexecutante, NO_RAZAO_SOCIAL, DS_NATUREZA_JUR, CO_MUNICIPIO_GESTOR, municipal].
    """
    kw = dict(sep=";", encoding="latin-1", dtype=str)
    estab = pd.read_csv(caminho_estab, usecols=["CO_CNES", "NO_RAZAO_SOCIAL", "CO_MUNICIPIO_GESTOR", "CO_NATUREZA_JUR"], **kw)
    natjur = pd.read_csv(caminho_natjur, **kw)
    estab = estab.merge(natjur, on="CO_NATUREZA_JUR", how="left", validate="m:1")
    por_natureza = estab["CO_NATUREZA_JUR"].isin(cfg_sel["naturezas_municipais"]) & (
        estab["CO_MUNICIPIO_GESTOR"] == cfg_sel["municipio_gestor"])
    por_razao = estab["NO_RAZAO_SOCIAL"].str.strip().isin(cfg_sel["razoes_sociais"])
    estab["municipal"] = por_natureza & por_razao
    estab = estab.rename(columns={"CO_CNES": "cnesexecutante"})
    estab["cnesexecutante"] = estab["cnesexecutante"].str.zfill(7)
    return estab[["cnesexecutante", "NO_RAZAO_SOCIAL", "DS_NATUREZA_JUR", "CO_MUNICIPIO_GESTOR", "municipal"]]


def aplicar_filtros(df: pd.DataFrame, cfg: dict, municipais: set[str],
                    log: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica os critérios de seleção em sequência, registrando o que cada um remove (D2, D4).

    Retorna: (df filtrado, fluxograma DataFrame[passo, criterio, linhas, removidas, cnes]).
    Cuidado: o período aplicado aqui é o do histórico (2018-01 em diante, até o fim da análise);
    a marcação do período de análise é feita depois do histórico.
    """
    sel, per = cfg["selecao"], cfg["periodo"]
    passos: list[tuple[str, pd.Series | None]] = [
        ("base bruta", None),
        (f"municipiosolicitante = {sel['municipiosolicitante']}", df["municipiosolicitante"].str.upper() == sel["municipiosolicitante"]),
        (f"tipounidade = {sel['tipounidade']}", df["tipounidade"] == sel["tipounidade"]),
        (f"complexidade = {sel['complexidade']}", df["complexidade"] == sel["complexidade"]),
        ("data da consulta preenchida", df["data_consulta"].notna()),
        (f"situacaoagenda em {sel['situacoes_desfecho']}", df["situacaoagenda"].isin(sel["situacoes_desfecho"])),
        ("executante de gestão municipal", df["cnesexecutante"].isin(municipais)),
        (f"consulta entre {per['inicio_historico']} e {per['fim_analise']}",
         df["data_consulta"].between(pd.Timestamp(per["inicio_historico"]), pd.Timestamp(per["fim_analise"]))),
    ]
    mascara = pd.Series(True, index=df.index)
    linhas = []
    anterior = len(df)
    for i, (nome, cond) in enumerate(passos):
        if cond is not None:
            mascara &= cond.fillna(False)
        n = int(mascara.sum())
        cnes = int(df.loc[mascara, "cnesexecutante"].nunique())
        linhas.append({"passo": i, "criterio": nome, "linhas": n, "removidas": anterior - n, "cnes": cnes})
        log(f"  [{i}] {nome}: {n:,} linhas ({anterior - n:,} removidas), {cnes} CNES")
        anterior = n
    return df[mascara].copy(), pd.DataFrame(linhas)


def _contar_antes(chave_ev: np.ndarray, chave_alvo: np.ndarray) -> np.ndarray:
    """Para chaves ordenadas (grupo, dia), devolve o índice do primeiro evento >= alvo."""
    return np.searchsorted(chave_ev, chave_alvo, side="left")


def calcular_historico(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as 5 variáveis de histórico do paciente (D14) sem vazamento.

    Usa apenas consultas da própria base filtrada (opção B) com data_consulta < t0, em que
    t0 = data_solicitacao + max(diasfilaespera, 0) é a data de agendamento.
    Parâmetros: df com idpaciente, data_solicitacao, diasfilaespera, data_consulta, y, especialidademae.
    Retorna: cópia com t0 e as colunas n_consultas_previas, n_faltas_previas,
    dias_desde_ultima_falta (NaN se nunca faltou), falta_recente (categoria),
    n_faltas_previas_especialidademae, ultima_previa_mesma_especialidademae
    (sem_previa | realizada | faltante).
    ANTI-VAZAMENTO: comparação estrita (<) no nível do dia; consultas no próprio dia do
    agendamento e a própria consulta nunca entram.
    """
    df = df.copy()
    df["t0"] = df["data_solicitacao"] + pd.to_timedelta(df["diasfilaespera"].clip(lower=0).fillna(0), unit="D")
    dia_ev = (df["data_consulta"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy(np.int64)
    dia_t0 = (df["t0"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy(np.int64)
    y = df["y"].to_numpy(np.int64)
    pac = pd.factorize(df["idpaciente"])[0].astype(np.int64)
    BASE = np.int64(1_000_000)  # dias cabem com folga; chave = grupo * BASE + dia

    def contagens(grupo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Número de eventos e de faltas anteriores a t0 dentro do grupo, e o índice do último."""
        ordem = np.lexsort((dia_ev, grupo))
        chave = grupo[ordem] * BASE + dia_ev[ordem]
        y_ord = y[ordem]
        acum = np.concatenate([[0], np.cumsum(y_ord)])
        ini = _contar_antes(chave, grupo * BASE)
        fim = _contar_antes(chave, grupo * BASE + dia_t0)
        n = fim - ini
        nf = acum[fim] - acum[ini]
        ultimo_y = np.where(n > 0, y_ord[np.maximum(fim - 1, 0)], -1)
        return n, nf, ultimo_y, ordem

    n, nf, _, _ = contagens(pac)
    # Própria consulta antes de t0 (erro de registro): retirar sua contribuição.
    propria = dia_ev < dia_t0
    n = n - propria
    nf = nf - (propria & (y == 1))

    # Última falta antes de t0: busca só entre eventos com falta.
    falta = y == 1
    chave_f = np.sort(pac[falta] * BASE + dia_ev[falta])
    ini_f = _contar_antes(chave_f, pac * BASE)
    fim_f = _contar_antes(chave_f, pac * BASE + dia_t0)
    # se a própria consulta é falta e está antes de t0, ela é o último elemento: pular
    fim_f = fim_f - (propria & falta)
    tem = fim_f > ini_f
    dia_ult = np.where(tem, chave_f[np.maximum(fim_f - 1, 0)] - pac * BASE, -1)
    dias_desde = np.where(tem, dia_t0 - dia_ult, np.nan)

    # Histórico na mesma especialidade-mãe.
    grupo_esp = pd.factorize(df["idpaciente"].astype(str) + "|" + df["especialidademae"].astype(str))[0].astype(np.int64)
    n_e, nf_e, ult_e, _ = contagens(grupo_esp)
    n_e = n_e - propria
    nf_e = nf_e - (propria & (y == 1))
    # Em caso de erro de registro com a própria consulta antes de t0, a "última" fica indefinida;
    # isso é raro e é tratado como sem_previa quando n_e cai a zero.
    ult_e = np.where(n_e > 0, ult_e, -1)

    df["n_consultas_previas"] = n.astype("int32")
    df["n_faltas_previas"] = nf.astype("int32")
    df["dias_desde_ultima_falta"] = dias_desde
    d = df["dias_desde_ultima_falta"]
    df["falta_recente"] = pd.Categorical(
        np.select([d.isna(), d <= 90, d <= 365], ["nunca_faltou", "ate_90d", "91_365d"], default="mais_de_365d"),
        categories=["nunca_faltou", "ate_90d", "91_365d", "mais_de_365d"])
    df["n_faltas_previas_especialidademae"] = nf_e.astype("int32")
    df["ultima_previa_mesma_especialidademae"] = pd.Categorical(
        np.select([ult_e == -1, ult_e == 1], ["sem_previa", "faltante"], default="realizada"),
        categories=["sem_previa", "realizada", "faltante"])
    return df


def variaveis_derivadas(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cria y, capítulo do CID, tempos, calendário, indicador de unidade indicada e período.

    Retorna: cópia com y (int8), cid_capitulo, tempo_fila, antecedencia, mes, dia_semana,
    possui_unidade_indicada (0/1) e na_analise (bool: consulta no período de análise).
    """
    df = df.copy()
    df["cid_capitulo"] = df["cid"].str[0].str.upper()
    df["tempo_fila"] = df["diasfilaespera"].clip(lower=0)
    df["antecedencia"] = (df["data_consulta"] - df["t0"]).dt.days
    df["mes"] = df["data_consulta"].dt.month.astype("int8")
    df["dia_semana"] = df["data_consulta"].dt.dayofweek.astype("int8")
    df["possui_unidade_indicada"] = (df["possuiunidadeindicada"].str.upper().str.startswith("S")).astype("int8")
    per = cfg["periodo"]
    df["na_analise"] = df["data_consulta"].between(pd.Timestamp(per["inicio_analise"]), pd.Timestamp(per["fim_analise"]))
    return df


# Colunas gravadas em consultas.parquet. idpaciente NÃO é gravado (D12).
COLUNAS_SAIDA = [
    "rid", "cnesexecutante", "data_solicitacao", "t0", "data_consulta", "y", "na_analise",
    "especialidademae", "especialidade", "cid_capitulo", "corregulador", "cnes_solicitante",
    "idade", "possui_unidade_indicada", "tempo_fila", "antecedencia", "mes", "dia_semana",
    "n_consultas_previas", "n_faltas_previas", "dias_desde_ultima_falta", "falta_recente",
    "n_faltas_previas_especialidademae", "ultima_previa_mesma_especialidademae", "ARQUIVO_ORIGEM",
]
