"""Esquemas de divisão temporal (S4): holdout, janela expansiva, janela deslizante, purged + embargo.

Etapa: S4. Um único gerador para os quatro esquemas e para os folds internos de tuning.
Não executa nada ao ser importado. Trabalha só com as DATAS das linhas (um array de datas
alinhado às linhas da base) e devolve índices POSICIONAIS (np.ndarray de inteiros), de modo que o
mesmo gerador serve para consultas e para CNES-dia.

Conceitos
  * Teste externo de um ano Y: [Y-01-01, Y+1-01-01) (para o último ano, até a última data da base).
  * Treino: o que vem antes do teste (esquemas "para frente"), menos a PURGA.
  * Purga P (dias): remove do treino as linhas com dia em [início_teste − P, início_teste). Uma
    consulta cujo desfecho só é conhecido perto do início do teste não estava disponível quando as
    linhas do teste foram agendadas/previstas. P deve cobrir o maior intervalo entre t0 e a
    consulta mais a defasagem L (ver ``purga_recomendada``).
  * Embargo E (dias): folga DEPOIS do fim do teste. Só tem efeito se o treino puder conter dados
    posteriores ao teste (``usar_futuro=True``, validação cruzada em blocos). Nos esquemas só para
    frente ele é implementado e registrado, mas não remove nada.
  * Folds internos de tuning: blocos de validação de 1 ano imediatamente anteriores ao início do
    teste externo (offsets -2 e -1 → para Y=2026, validações em 2024 e 2025), cada um com treino
    construído pelo MESMO esquema, só com dados anteriores ao bloco de validação.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

ESQUEMAS = ("holdout", "expansiva", "deslizante", "purged_embargo")


@dataclass
class Divisao:
    """Uma divisão treino/teste (com os folds internos de tuning).

    Atributos:
        esquema: um de ESQUEMAS.
        rotulo: identificador legível (ex.: "expansiva_2024").
        treino, teste: índices posicionais (np.ndarray[int]) das linhas.
        inicio_teste, fim_teste: limites do teste (fim exclusivo).
        parametros: dict com L, P, E, janela_meses, usar_futuro etc. (vai para experimentos.csv).
        n_purgadas, n_embargadas: linhas removidas do treino por purga e por embargo.
        folds_internos: lista de Divisao (validação dentro do período anterior ao teste).
    """
    esquema: str
    rotulo: str
    treino: np.ndarray
    teste: np.ndarray
    inicio_teste: pd.Timestamp
    fim_teste: pd.Timestamp
    parametros: dict = field(default_factory=dict)
    n_purgadas: int = 0
    n_embargadas: int = 0
    folds_internos: list["Divisao"] = field(default_factory=list)


def purga_recomendada(dias_agendamento_ate_consulta: Iterable[float], lag: int, percentil: float = 100.0) -> int:
    """Calcula o valor de P recomendado: antecedência (consulta − t0) alta + defasagem L.

    Parâmetros:
        dias_agendamento_ate_consulta: dias entre t0 e a consulta, por linha (ex.: (consulta − t0).dt.days).
        lag: defasagem L em dias.
        percentil: 100 = máximo (conservador); 99 = quase todas as linhas.

    Retorna:
        P em dias (int). Registre o percentil usado no log do experimento.
    """
    v = np.asarray(dias_agendamento_ate_consulta, dtype=float)
    assert v.size > 0, "Sem dados para calcular a purga"
    return int(np.ceil(np.percentile(v, percentil))) + int(lag)


def _limites_ano(ano: int, ultima_data: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Início e fim (exclusivo) de um ano-teste; o fim é cortado na última data observada + 1 dia."""
    ini = pd.Timestamp(f"{ano}-01-01")
    fim = min(pd.Timestamp(f"{ano + 1}-01-01"), ultima_data + pd.Timedelta(days=1))
    return ini, fim


def _monta_treino(dias: np.ndarray, inicio_teste: pd.Timestamp, fim_teste: pd.Timestamp, *, purga_dias: int,
                  embargo_dias: int, janela_meses: int | None, usar_futuro: bool) -> tuple[np.ndarray, int, int]:
    """Seleciona as linhas de treino para um teste [inicio_teste, fim_teste).

    Parâmetros:
        dias: array de datas (datetime64) de todas as linhas.
        inicio_teste, fim_teste: limites do teste (fim exclusivo).
        purga_dias: P; remove do treino linhas com dia em [inicio_teste − P, inicio_teste).
        embargo_dias: E; remove do treino linhas com dia em [fim_teste, fim_teste + E).
        janela_meses: se não for None, o treino só inclui linhas com dia >= inicio_teste − janela
            (janela deslizante); None = janela expansiva (todo o passado).
        usar_futuro: se True, linhas POSTERIORES ao teste (depois do embargo) também entram no treino
            (validação cruzada em blocos). Padrão False: esquemas só para frente.

    Retorna:
        (índices de treino, nº de linhas purgadas, nº de linhas embargadas).

    Cuidados:
        # ANTI-VAZAMENTO: com usar_futuro=False, nenhuma linha com dia >= inicio_teste − P entra no
        # treino. A purga é aplicada na seleção das linhas, nunca recalcula variáveis (as variáveis de
        # histórico já foram construídas com data_ev < t0 antes das divisões).
    """
    antes = dias < np.datetime64(inicio_teste)
    if janela_meses is not None:
        limite_inf = inicio_teste - pd.DateOffset(months=janela_meses)
        antes &= dias >= np.datetime64(limite_inf)
    em_purga = antes & (dias >= np.datetime64(inicio_teste - pd.Timedelta(days=purga_dias)))
    base = antes & ~em_purga
    n_embargadas = 0
    if usar_futuro:
        depois = dias >= np.datetime64(fim_teste)
        em_embargo = depois & (dias < np.datetime64(fim_teste + pd.Timedelta(days=embargo_dias)))
        n_embargadas = int(em_embargo.sum())
        base = base | (depois & ~em_embargo)
    return np.flatnonzero(base), int(em_purga.sum()), n_embargadas


def _folds_internos(dias: np.ndarray, esquema: str, inicio_teste: pd.Timestamp, ultima_data: pd.Timestamp,
                    offsets: Iterable[int], *, purga_dias: int, embargo_dias: int,
                    janela_meses: int | None, params: dict) -> list[Divisao]:
    """Folds internos de tuning: validação = ano (inicio_teste.year + offset), treino pelo mesmo esquema.

    Parâmetros:
        dias, esquema, inicio_teste, ultima_data: ver ``gera_divisoes``.
        offsets: deslocamentos em anos, ex.: [-2, -1]. Para teste em 2026 → validações em 2024 e 2025.
        purga_dias, embargo_dias, janela_meses: iguais aos do teste externo.
        params: parâmetros a registrar.

    Retorna:
        Lista de Divisao em que `teste` é o bloco de VALIDAÇÃO.

    Cuidados:
        # ANTI-VAZAMENTO: cada bloco de validação termina antes do início do teste externo
        # (offsets negativos) e seu treino termina antes do início da própria validação; nenhum
        # fold toca dados do teste externo ou posteriores (asserts abaixo).
    """
    folds = []
    for off in offsets:
        assert off < 0, "Offsets de folds internos devem ser negativos (anos anteriores ao teste)"
        ini_v, fim_v = _limites_ano(inicio_teste.year + off, ultima_data)
        assert fim_v <= inicio_teste, "Fold interno invade o teste externo"
        treino, n_p, _ = _monta_treino(dias, ini_v, fim_v, purga_dias=purga_dias, embargo_dias=embargo_dias,
                                       janela_meses=janela_meses, usar_futuro=False)
        valid = np.flatnonzero((dias >= np.datetime64(ini_v)) & (dias < np.datetime64(fim_v)))
        folds.append(Divisao(esquema, f"{esquema}_val{ini_v.year}", treino, valid, ini_v, fim_v,
                             {**params, "papel": "validacao_interna"}, n_p, 0))
    return folds


def gera_divisoes(dias: np.ndarray | pd.Series, esquema: str, *, anos_teste: Iterable[int] = (2023, 2024, 2025, 2026),
                  lag: int = 3, purga_dias: int = 0, embargo_dias: int = 0, janela_meses: int | None = None,
                  corte_holdout: str = "2023-01-01", usar_futuro: bool = False,
                  offsets_folds: Iterable[int] = (-2, -1)) -> list[Divisao]:
    """Gera as divisões treino/teste (e folds internos) de um esquema.

    Parâmetros:
        dias: datas de cada linha (datetime64), na ordem das linhas da base. Linhas sem data são proibidas.
        esquema: "holdout", "expansiva", "deslizante" ou "purged_embargo".
        anos_teste: anos de teste externo (expansiva, deslizante, purged_embargo). Ignorado no holdout.
        lag: defasagem L (registrada em `parametros`; usada em ``purga_recomendada``).
        purga_dias: P. No esquema "purged_embargo" é aplicado; nos demais é 0 por definição
            (se informado, é aplicado mesmo assim para permitir a análise de sensibilidade P ∈ {0, 28, 90}).
        embargo_dias: E (efeito só com usar_futuro=True).
        janela_meses: tamanho da janela (obrigatório em "deslizante"; ignorado nos demais).
        corte_holdout: data C do holdout (treino < C, teste >= C até o fim da base).
        usar_futuro: ver ``_monta_treino``; só deve ser True em validação cruzada em blocos.
        offsets_folds: anos dos folds internos relativos ao início do teste (padrão: -2 e -1).

    Retorna:
        Lista de Divisao, uma por ano de teste (um só elemento no holdout), cada uma com seus
        `folds_internos`. Índices são posicionais em `dias`.

    Cuidados:
        # ANTI-VAZAMENTO: ver ``_monta_treino`` e ``_folds_internos``. ``valida_divisao`` confere o
        # resultado (nenhuma linha em dois conjuntos, treino anterior ao teste, purga exata).
        # DECISÃO: o fim do último ano de teste é a última data da base (2026 é parcial).
    """
    assert esquema in ESQUEMAS, f"Esquema desconhecido: {esquema}. Use um de {ESQUEMAS}"
    dias_np = pd.to_datetime(pd.Series(np.asarray(dias))).to_numpy()
    assert not np.isnat(dias_np).any(), "Há datas ausentes; as divisões exigem data em todas as linhas"
    ultima = pd.Timestamp(dias_np.max())
    if esquema == "deslizante":
        assert janela_meses and janela_meses > 0, "Esquema deslizante exige janela_meses > 0"
    else:
        janela_meses = None
    params = {"esquema": esquema, "L": lag, "P": purga_dias, "E": embargo_dias, "janela_meses": janela_meses,
              "usar_futuro": usar_futuro}

    if esquema == "holdout":
        alvos = [(pd.Timestamp(corte_holdout), ultima + pd.Timedelta(days=1), f"holdout_{corte_holdout}")]
    else:
        alvos = []
        for y in anos_teste:
            ini, fim = _limites_ano(int(y), ultima)
            assert ini <= ultima, f"Ano de teste {y} não tem dados"
            alvos.append((ini, fim, f"{esquema}_{y}"))

    divisoes = []
    for ini, fim, rotulo in alvos:
        treino, n_p, n_e = _monta_treino(dias_np, ini, fim, purga_dias=purga_dias, embargo_dias=embargo_dias,
                                         janela_meses=janela_meses, usar_futuro=usar_futuro)
        teste = np.flatnonzero((dias_np >= np.datetime64(ini)) & (dias_np < np.datetime64(fim)))
        folds = _folds_internos(dias_np, esquema, ini, ultima, offsets_folds, purga_dias=purga_dias,
                                embargo_dias=embargo_dias, janela_meses=janela_meses, params=params)
        divisoes.append(Divisao(esquema, rotulo, treino, teste, ini, fim, dict(params), n_p, n_e, folds))
    return divisoes


def valida_divisao(div: Divisao, dias: np.ndarray | pd.Series, *, checar_folds: bool = True) -> None:
    """Confere as regras anti-vazamento de uma divisão; levanta AssertionError em português se violar.

    Parâmetros:
        div: divisão a validar.
        dias: as mesmas datas usadas em ``gera_divisoes``.
        checar_folds: se True, valida também cada fold interno (e que ele termina antes do teste externo).

    Verificações:
        1. nenhuma linha em dois conjuntos (treino ∩ teste = ∅);
        2. treino sem datas >= início do teste (exceto usar_futuro);
        3. nenhuma linha de treino dentro da janela de purga [início − P, início);
        4. teste contém exatamente as linhas com data em [início, fim);
        5. folds internos existem, terminam antes do teste externo e não usam dados dele.
    """
    d = pd.to_datetime(pd.Series(np.asarray(dias))).to_numpy()
    ini, fim = np.datetime64(div.inicio_teste), np.datetime64(div.fim_teste)
    P = int(div.parametros.get("P", 0))
    assert np.intersect1d(div.treino, div.teste).size == 0, f"{div.rotulo}: linha em treino e teste ao mesmo tempo"
    assert np.unique(div.treino).size == div.treino.size, f"{div.rotulo}: índices repetidos no treino"
    if not div.parametros.get("usar_futuro", False):
        assert div.treino.size == 0 or d[div.treino].max() < ini, \
            f"{div.rotulo}: treino contém datas posteriores ao início do teste"
    purga_ini = div.inicio_teste - pd.Timedelta(days=P)
    na_purga = (d[div.treino] >= np.datetime64(purga_ini)) & (d[div.treino] < ini)
    assert not na_purga.any(), f"{div.rotulo}: {int(na_purga.sum())} linhas de treino dentro da janela de purga"
    esperado = np.flatnonzero((d >= ini) & (d < fim))
    assert np.array_equal(np.sort(div.teste), esperado), f"{div.rotulo}: teste não é exatamente [início, fim)"
    if checar_folds:
        assert len(div.folds_internos) > 0, f"{div.rotulo}: sem folds internos de tuning"
        for f in div.folds_internos:
            assert f.fim_teste <= div.inicio_teste, f"{f.rotulo}: validação invade o teste externo de {div.rotulo}"
            assert f.teste.size > 0 and f.treino.size > 0, f"{f.rotulo}: fold vazio"
            assert d[f.teste].max() < ini, f"{f.rotulo}: validação usa dados do teste externo"
            valida_divisao(f, dias, checar_folds=False)


def restringe_teste_comum(div: Divisao, dias: np.ndarray | pd.Series, inicio: str, fim: str) -> np.ndarray:
    """Devolve os índices do teste de `div` que caem no conjunto COMUM de dias de teste [inicio, fim].

    Parâmetros:
        div: divisão.
        dias: datas das linhas.
        inicio, fim: limites inclusivos do período comum (config divisao.teste_comum).

    Retorna:
        np.ndarray de índices posicionais (subconjunto de div.teste). A comparação entre esquemas só
        vale neste conjunto (CLAUDE.md 3.1, regra 4).
    """
    d = pd.to_datetime(pd.Series(np.asarray(dias))).to_numpy()
    ok = (d[div.teste] >= np.datetime64(pd.Timestamp(inicio))) & (d[div.teste] <= np.datetime64(pd.Timestamp(fim)))
    return div.teste[ok]


def descreve_divisoes(divisoes: list[Divisao]) -> pd.DataFrame:
    """Tabela com o tamanho de cada divisão e dos folds internos (para o relatório de S4).

    Retorna:
        DataFrame com: esquema, rotulo, papel (teste_externo/validacao_interna), inicio, fim, n_treino,
        n_teste, n_purgadas, n_embargadas, L, P, E, janela_meses.
    """
    linhas = []
    for dv in divisoes:
        for papel, x in [("teste_externo", dv)] + [("validacao_interna", f) for f in dv.folds_internos]:
            linhas.append({"esquema": x.esquema, "rotulo": x.rotulo, "papel": papel,
                           "teste_externo_ref": dv.rotulo, "inicio": x.inicio_teste.date(), "fim_exclusivo": x.fim_teste.date(),
                           "n_treino": len(x.treino), "n_teste_ou_validacao": len(x.teste),
                           "n_purgadas": x.n_purgadas, "n_embargadas": x.n_embargadas,
                           "L": x.parametros.get("L"), "P": x.parametros.get("P"), "E": x.parametros.get("E"),
                           "janela_meses": x.parametros.get("janela_meses")})
    return pd.DataFrame(linhas)
