"""Painel série-semana (série = CNES × especialidade-mãe, D16), agenda conhecida na origem, defasagens, calendário e target encoding temporal.

Convenções (D8):
- semana = segunda-feira da semana ISO da consulta;
- para horizonte h (semanas), a origem da previsão da semana s é o domingo anterior à semana
  s − h + 1, isto é, origem = s − 7·(h − 1) − 1 dia;
- agenda conhecida = consultas da semana s com t0 ≤ origem;
- defasagens usam apenas semanas s − h, s − h − 1, ... (encerradas até a origem).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UM_DIA = pd.Timedelta(days=1)


def semana_de(datas: pd.Series) -> pd.Series:
    """Segunda-feira da semana ISO de cada data."""
    return (datas - pd.to_timedelta(datas.dt.dayofweek, unit="D")).dt.normalize()


def origem_da_semana(semana: pd.Series | pd.Timestamp, h: int):
    """Data de origem (domingo) da previsão da semana `semana` com horizonte h."""
    return semana - pd.Timedelta(days=7 * (h - 1)) - UM_DIA


def _pascoa(ano: int) -> pd.Timestamp:
    """Domingo de Páscoa (algoritmo de Meeus/Jones/Butcher)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    hh = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - hh - k) % 7
    m = (a + 11 * hh + 22 * l_) // 451
    mes = (hh + l_ - 7 * m + 114) // 31
    dia = ((hh + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(ano, mes, dia)


def feriados(anos: range) -> pd.DatetimeIndex:
    """Feriados nacionais, estadual do RS (20/set), de Porto Alegre (2/fev) e móveis (Carnaval,
    Sexta-feira Santa, Corpus Christi). Conhecidos antecipadamente, portanto sem risco de vazamento.
    DECISÃO: 20/nov (Consciência Negra) só a partir de 2024 (Lei 14.759/2023, feriado nacional).
    Retorna: DatetimeIndex ordenado, sem repetições."""
    datas = []
    for a in anos:
        fixos = [(1, 1), (2, 2), (4, 21), (5, 1), (9, 7), (9, 20), (10, 12), (11, 2), (11, 15), (12, 25)]
        if a >= 2024:
            fixos.append((11, 20))
        for m, d in fixos:
            datas.append(pd.Timestamp(a, m, d))
        p = _pascoa(a)
        datas += [p - pd.Timedelta(days=48), p - pd.Timedelta(days=47), p - pd.Timedelta(days=2), p + pd.Timedelta(days=60)]
    return pd.DatetimeIndex(sorted(set(datas)))


def calendario(semanas: pd.Series, fourier_k: int) -> pd.DataFrame:
    """Variáveis de calendário da semana (todas conhecidas de antemão).

    Parâmetros: semanas = Series de segundas-feiras; fourier_k = nº de pares seno/cosseno.
    Retorna: DataFrame alinhado ao índice de `semanas`, colunas
      mes, sen1..k, cos1..k, n_feriados (feriados em dias úteis), dias_uteis (5 − n_feriados),
      ponte (feriado numa terça ou quinta: a segunda/sexta tende a esvaziar),
      feriado_semana_anterior, feriado_semana_seguinte (efeito de semanas emendadas),
      recesso (semana com algum dia entre 24/dez e 1/jan).
    """
    fer = feriados(range(semanas.dt.year.min() - 1, semanas.dt.year.max() + 2))
    fer_util = fer[fer.dayofweek < 5]
    seg_do_feriado = fer_util - pd.to_timedelta(fer_util.dayofweek, unit="D")
    n_fer = pd.Series(1, index=seg_do_feriado).groupby(level=0).sum()
    eh_ponte = pd.Series(np.isin(fer_util.dayofweek, [1, 3]).astype(int), index=seg_do_feriado).groupby(level=0).max()

    out = pd.DataFrame(index=semanas.index)
    out["mes"] = (semanas + pd.Timedelta(days=3)).dt.month.astype("int8")
    t = semanas.dt.isocalendar().week.astype(float).to_numpy()
    for k in range(1, fourier_k + 1):
        out[f"sen{k}"] = np.sin(2 * np.pi * k * t / 52.18)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.18)
    sete = pd.Timedelta(days=7)
    out["n_feriados"] = semanas.map(n_fer).fillna(0).astype("int8").to_numpy()
    out["dias_uteis"] = (5 - out["n_feriados"]).astype("int8")
    out["ponte"] = semanas.map(eh_ponte).fillna(0).astype("int8").to_numpy()
    out["feriado_semana_anterior"] = (semanas - sete).map(n_fer).fillna(0).gt(0).astype("int8").to_numpy()
    out["feriado_semana_seguinte"] = (semanas + sete).map(n_fer).fillna(0).gt(0).astype("int8").to_numpy()
    # recesso: algum dia da semana (seg a dom) cai entre 24/12 e 01/01
    rec = np.zeros(len(semanas), dtype="int8")
    for d in range(7):
        dia = semanas + pd.Timedelta(days=d)
        rec |= (((dia.dt.month == 12) & (dia.dt.day >= 24)) | ((dia.dt.month == 1) & (dia.dt.day == 1))).to_numpy().astype("int8")
    out["recesso"] = rec
    return out


COLS_CALENDARIO = ["mes", "n_feriados", "dias_uteis", "ponte", "feriado_semana_anterior",
                   "feriado_semana_seguinte", "recesso"]


# Agregados da agenda conhecida na origem (Tabela 1 da Metodologia)
AGREGADOS_B = ["prop_falta_previa", "media_faltas_previas", "prop_ultima_esp_falta",
               "media_idade", "prop_unidade_indicada", "media_antecedencia"]


def chave_serie(cnes: pd.Series, esp: pd.Series) -> pd.Series:
    """Identificador da série (D16): "<cnes>|<especialidademae>" (especialidade ausente = "NA")."""
    return cnes.astype(str) + "|" + esp.astype(object).fillna("NA").astype(str)


def montar_painel(cons: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
    """Painel série-semana para o horizonte h; série = CNES × especialidade-mãe (D16).

    Parâmetros: cons = consultas.parquet (todas as linhas, inclusive o período só de histórico);
    h = horizonte em semanas; cfg = configuração.
    Retorna: DataFrame[serie, cnesexecutante, especialidademae, semana, h, y, n_total, n_agenda, AGREGADOS_B...,
    lag1..lagL, mm4, mm13, lag52, taxa_ag4, taxa_ag13, ref_taxa_agenda,
    COLS_CALENDARIO, sen*, cos*, avaliavel], uma linha por série e
    semana contígua desde a primeira semana com consulta da série até o fim da análise.
    `avaliavel` = falso quando a semana não tem agenda conhecida nem faltas (fica fora da avaliação).
    ANTI-VAZAMENTO: agenda conhecida usa t0 ≤ origem; defasagens começam em s − h.
    """
    per, var = cfg["periodo"], cfg["variaveis"]
    c = cons.copy()
    c["serie"] = chave_serie(c["cnesexecutante"], c["especialidademae"])
    c["semana"] = semana_de(c["data_consulta"])
    c["origem"] = origem_da_semana(c["semana"], h)
    c["conhecida"] = c["t0"] <= c["origem"]

    # Grade contígua de semanas por série (semanas sem consulta entram com y = 0).
    inicio = semana_de(pd.Series([pd.Timestamp(per["inicio_historico"])])).iloc[0]
    fim = semana_de(pd.Series([pd.Timestamp(per["fim_analise"])])).iloc[0]
    todas = pd.date_range(inicio, fim, freq="7D")
    primeira = c.groupby("serie")["semana"].min()
    grade = pd.concat([pd.DataFrame({"serie": ser, "semana": todas[todas >= s0]}) for ser, s0 in primeira.items()],
                      ignore_index=True)
    partes = grade["serie"].str.split("|", n=1, expand=True)
    grade["cnesexecutante"], grade["especialidademae"] = partes[0], partes[1]

    tot = c.groupby(["serie", "semana"]).agg(y=("y", "sum"), n_total=("y", "size"))
    k = c[c["conhecida"]].copy()
    k["falta_previa"] = (k["n_faltas_previas"] > 0).astype(float)
    k["ultima_esp_falta"] = (k["ultima_previa_mesma_especialidademae"].astype(str) == "faltante").astype(float)
    ag = k.groupby(["serie", "semana"]).agg(
        n_agenda=("y", "size"), prop_falta_previa=("falta_previa", "mean"),
        media_faltas_previas=("n_faltas_previas", "mean"), prop_ultima_esp_falta=("ultima_esp_falta", "mean"),
        media_idade=("idade", "mean"), prop_unidade_indicada=("possui_unidade_indicada", "mean"),
        media_antecedencia=("antecedencia", "mean"))
    p = grade.merge(tot, on=["serie", "semana"], how="left").merge(ag, on=["serie", "semana"], how="left")
    p[["y", "n_total", "n_agenda"]] = p[["y", "n_total", "n_agenda"]].fillna(0).astype(int)
    p = p.sort_values(["serie", "semana"]).reset_index(drop=True)

    # Defasagens: ANTI-VAZAMENTO — deslocamento mínimo de h semanas (grade contígua garante
    # que shift(k) = k semanas atrás).
    g = p.groupby("serie")["y"]
    for j in range(var["lags_y"]):
        p[f"lag{j + 1}"] = g.shift(h + j)
    p["mm4"] = g.transform(lambda s: s.shift(h).rolling(var["baseline_janela"], min_periods=var["baseline_janela"]).mean())
    p["mm13"] = g.transform(lambda s: s.shift(h).rolling(13, min_periods=13).mean())
    p["lag52"] = g.shift(52) if h <= 52 else np.nan

    # Taxa de falta por consulta já agendada na origem (D15): faltas ÷ agenda conhecida das
    # semanas encerradas (s − h, s − h − 1, ...). Cada semana passada usa a SUA agenda conhecida
    # na SUA origem, com o mesmo h, então a razão já absorve a fração de consultas marcadas
    # depois da origem. ANTI-VAZAMENTO: deslocamento mínimo de h semanas.
    # DECISÃO: suavização da janela de 4 semanas em direção à de 13 semanas com peso m
    # (config.variaveis.taxa_suavizacao_m), para semanas com pouca agenda.
    m = var.get("taxa_suavizacao_m", 20)
    ga = p.groupby("serie")["n_agenda"]
    soma = lambda g, w: g.transform(lambda s: s.shift(h).rolling(w, min_periods=w).sum())
    y4, a4, y13, a13 = soma(g, 4), soma(ga, 4), soma(g, 13), soma(ga, 13)
    p["taxa_ag13"] = y13 / a13.where(a13 > 0)
    # DECISÃO (D19): taxas cortadas em 1 (no máximo uma falta por consulta conhecida). Em h=2 a
    # agenda conhecida na origem é incompleta e a razão faltas/agenda conhecida chegava a 55,
    # inflando a referência (viés 1,12 no treino 2022–23; com o corte, 0,98). Escolhido só no treino.
    p["taxa_ag13"] = p["taxa_ag13"].clip(upper=1.0)
    p["taxa_ag4"] = ((y4 + m * p["taxa_ag13"]) / (a4 + m)).clip(upper=1.0)
    # Referência D11 revista: taxa suavizada × agenda conhecida da semana-alvo.
    p["ref_taxa_agenda"] = p["taxa_ag4"] * p["n_agenda"]
    p = pd.concat([p, calendario(p["semana"], var["fourier_k"])], axis=1)
    p["h"] = h
    p["avaliavel"] = (p["n_agenda"] > 0) | (p["y"] > 0)
    # Mantém só o período de análise (o anterior serviu para as defasagens).
    p = p[p["semana"] >= pd.Timestamp(per["inicio_analise"])].reset_index(drop=True)
    return p


def te_temporal(cons: pd.DataFrame, col: str, h: int, fim_treino: pd.Timestamp, m: float) -> pd.Series:
    """Target encoding temporal suavizado da coluna `col` (D13).

    Para uma consulta da semana s, usa só consultas de semanas ≤ s − h (desfecho conhecido na
    origem); para semanas posteriores a `fim_treino`, as estatísticas ficam congeladas no fim do
    treino (decisão do usuário: TE estimado só no treino).
    Valor = (faltas_c + m · taxa_geral) / (n_c + m); categoria sem histórico recebe taxa_geral.
    Retorna: Series float alinhada ao índice de `cons`.
    ANTI-VAZAMENTO: a própria consulta e as da mesma semana nunca entram.
    """
    c = cons[["semana", col, "y"]].copy()
    c[col] = c[col].fillna("NA")
    fim_sem = semana_de(pd.Series([pd.Timestamp(fim_treino)])).iloc[0]
    # semana "de corte" até a qual as estatísticas podem ir (inclusive)
    corte = c["semana"] - pd.Timedelta(days=7 * h)
    corte = corte.where(c["semana"] <= fim_sem, fim_sem)
    corte = corte.where(corte <= fim_sem, fim_sem)

    semanas = np.sort(c["semana"].unique())
    # acumulado global por semana
    glob = c.groupby("semana")["y"].agg(["sum", "size"]).reindex(semanas, fill_value=0).cumsum()
    # acumulado por categoria e semana
    cat = c.groupby([col, "semana"])["y"].agg(["sum", "size"]).groupby(level=0).cumsum().reset_index()

    # taxa global no corte
    idx_g = np.searchsorted(semanas, corte.to_numpy(), side="right") - 1
    gs = np.where(idx_g >= 0, glob["sum"].to_numpy()[np.maximum(idx_g, 0)], 0)
    gn = np.where(idx_g >= 0, glob["size"].to_numpy()[np.maximum(idx_g, 0)], 0)
    taxa_geral = np.where(gn > 0, gs / np.maximum(gn, 1), np.nan)

    # estatística da categoria no corte: merge_asof por categoria
    alvo = pd.DataFrame({"ordem": np.arange(len(c)), col: c[col].to_numpy(), "corte": corte.to_numpy()}).sort_values("corte")
    cat = cat.sort_values("semana")
    juntado = pd.merge_asof(alvo, cat, left_on="corte", right_on="semana", by=col, direction="backward").sort_values("ordem")
    fs = juntado["sum"].fillna(0).to_numpy()
    fn = juntado["size"].fillna(0).to_numpy()
    valor = (fs + m * taxa_geral) / (fn + m)
    return pd.Series(valor, index=cons.index, name=f"te_{col}")
