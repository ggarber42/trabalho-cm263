"""S1 — Ingestão e auditoria dos dados (marco M0).

Objetivo
    Descrever a base analítica ANTES de qualquer filtro do estudo: categorias de
    `situacaoagenda`, `status` e `possuiunidadeindicada`; duplicatas; datas inválidas;
    ausentes por variável e por ano; distribuição de `dias_ate_consulta`; conferência
    das variáveis de histórico contra a regra `data_ev < t0`; atividade das CNES por
    ano; fluxograma de seleção gerado por código; dicionário de variáveis.

Protocolo / limites
    * Só LEITURA da base. Nenhum filtro é aplicado aos dados do estudo (decisão de
      16/11/2026: 2018 é mantido e nenhuma CNES é excluída por inatividade).
    * A contagem de TODAS as categorias de `situacaoagenda` antes dos filtros do
      `reestruturacao.qmd` não é recuperável a partir do parquet, que já contém só
      REALIZADA e FALTANTE. Essa limitação é registrada no fluxograma e no dicionário.
    * Nenhum identificador de paciente é impresso ou gravado (só contagens).

Entradas (caminhos em config.yaml)
    base, flag_municipal, niveis_municipal, dados/dicionario_manual.csv.

Saídas
    tabelas/auditoria_dados.csv            tabela longa (bloco, item, categoria, n, pct, obs)
    tabelas/ausentes_por_variavel_ano.csv  ausentes por variável e ano
    tabelas/atividade_cnes.csv             atividade por CNES e ano (descritiva)
    tabelas/fluxograma_selecao.csv         fluxograma de seleção (n por etapa)
    dicionario_variaveis.md                nome, tipo, momento em que a variável é conhecida
    manifestos/S1_auditoria.json, logs/S1_auditoria_*.json
    (no modo --smoke todos levam o sufixo _smoke e não sobrescrevem os completos)

Como rodar (da raiz do projeto, SEM a opção -I)
    python3 etapas/S1_auditoria.py            # base completa (~1 min)
    python3 etapas/S1_auditoria.py --smoke    # amostra de pacientes, todos os anos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite "import src..." quando o script é rodado a partir de etapas/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto, sha256_arquivo  # noqa: E402

BIG = 100_000  # maior que qualquer número de dias desde 1970 (~20.800); usado na chave paciente×dia


# --------------------------------------------------------------------------------------
# Utilitários de montagem da tabela longa
# --------------------------------------------------------------------------------------
def linha(bloco: str, item: str, categoria: object, n: float, total: float | None = None,
          obs: str = "") -> dict:
    """Monta uma linha da tabela longa de auditoria.

    Parâmetros:
        bloco: grupo da checagem (ex.: "desfecho").
        item: o que foi medido (ex.: "situacaoagenda").
        categoria: categoria ou limite (ex.: "FALTANTE"); use "" se não houver.
        n: contagem ou valor.
        total: denominador para calcular a porcentagem; None = sem porcentagem.
        obs: comentário curto.

    Retorna:
        dict com as colunas bloco, item, categoria, n, pct, obs.
    """
    pct = round(100.0 * n / total, 4) if total else np.nan
    return {"bloco": bloco, "item": item, "categoria": str(categoria), "n": n, "pct": pct, "obs": obs}


def contagem_categorias(serie: pd.Series, bloco: str, item: str, obs: str = "") -> list[dict]:
    """Conta TODAS as categorias de uma coluna, incluindo ausentes (rotulados como <NA>).

    Parâmetros:
        serie: coluna a contar.
        bloco, item, obs: rótulos da tabela longa.

    Retorna:
        Lista de linhas (uma por categoria) com n e pct sobre o total da coluna.
    """
    total = len(serie)
    cont = serie.astype("object").where(serie.notna(), "<NA>").value_counts()
    return [linha(bloco, item, cat, int(n), total, obs) for cat, n in cont.items()]


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------
def carrega_base(cfg: dict, smoke: bool) -> tuple[pd.DataFrame, dict]:
    """Lê o parquet analítico; no modo smoke, mantém uma amostra de PACIENTES inteiros.

    Parâmetros:
        cfg: configuração (config.yaml).
        smoke: se True, amostra uma fração de pacientes (config smoke.fracao_pacientes).

    Retorna:
        (DataFrame com todas as colunas, dict com contagens da carga).

    Cuidados:
        # ANTI-VAZAMENTO: a amostra é por paciente (não por linha) para preservar o
        # histórico completo de cada paciente e permitir a conferência de `data_ev < t0`.
        # DECISÃO: semente vem de config.yaml (`semente`); fração em `smoke.fracao_pacientes`.
    """
    arquivo = caminho(cfg, cfg["dados"]["base"])
    d = pd.read_parquet(arquivo)
    contagens = {"linhas_parquet": int(len(d)), "colunas_parquet": int(d.shape[1])}
    if smoke:
        rng = np.random.default_rng(cfg["semente"])
        pacientes = d["idpaciente"].unique()
        n_amostra = max(1, int(len(pacientes) * cfg["smoke"]["fracao_pacientes"]))
        escolhidos = set(rng.choice(pacientes, size=n_amostra, replace=False))
        d = d[d["idpaciente"].isin(escolhidos)].reset_index(drop=True)
        contagens["linhas_smoke"] = int(len(d))
        contagens["pacientes_smoke"] = int(n_amostra)
        anos = set(pd.to_datetime(d["datahora_consulta"]).dt.year.unique())
        esperado = set(range(2018, 2027))
        assert esperado <= anos, f"Amostra do smoke não cobre todos os anos: faltam {sorted(esperado - anos)}"
    return d, contagens


# --------------------------------------------------------------------------------------
# Blocos de auditoria
# --------------------------------------------------------------------------------------
def audita_categorias(d: pd.DataFrame) -> list[dict]:
    """Contagem de categorias de situacaoagenda, status e possuiunidadeindicada (sem filtro).

    Parâmetros:
        d: base completa (ou amostra no smoke).

    Retorna:
        Linhas da tabela longa. Inclui uma linha de aviso: o parquet só tem REALIZADA e
        FALTANTE porque o filtro é feito antes, no reestruturacao.qmd.
    """
    out = contagem_categorias(d["situacaoagenda"], "desfecho", "situacaoagenda",
                              "parquet já filtrado no reestruturacao.qmd (só REALIZADA e FALTANTE)")
    out += contagem_categorias(d["status"], "categorias", "status", "status do cálculo de rota")
    out += contagem_categorias(d["possuiunidadeindicada"], "categorias", "possuiunidadeindicada")
    # Ausência de status coincide com as 7 variáveis de distância? (indício de que não é pós-desfecho)
    cols_dist = ["distancia_km", "tempo_caminhando", "distancia_km_trans_publico",
                 "tempo_desloc_trans_publico", "quant_bus", "baldiacoes"]
    status_na = d["status"].isna()
    coincide = int((status_na == d[cols_dist].isna().all(axis=1)).sum())
    out.append(linha("categorias", "status_NA_coincide_com_distancias_NA", "linhas coincidentes",
                     coincide, len(d), "100% = status descreve o cálculo de rota, não o desfecho"))
    # Taxa de falta por status (informativo; diferença grande exigiria investigar)
    y = (d["situacaoagenda"] == "FALTANTE")
    por_status = y.groupby(d["status"].astype("object").where(d["status"].notna(), "<NA>")).agg(["sum", "size"])
    for cat, r in por_status.iterrows():
        out.append(linha("categorias", "taxa_falta_por_status", cat, int(r["sum"]), int(r["size"]),
                         "pct = taxa de falta dentro da categoria"))
    por_pui = y.groupby(d["possuiunidadeindicada"].astype("object")).agg(["sum", "size"])
    for cat, r in por_pui.iterrows():
        out.append(linha("categorias", "taxa_falta_por_possuiunidadeindicada", cat, int(r["sum"]),
                         int(r["size"]), "pct = taxa de falta dentro da categoria"))
    return out


def audita_duplicatas(d: pd.DataFrame) -> list[dict]:
    """Conta duplicatas por identificador, por linha inteira e por chave composta.

    Parâmetros:
        d: base.

    Retorna:
        Linhas da tabela longa. Chave composta = paciente + data da solicitação + data da
        consulta + especialidade (possível dupla marcação). Nenhuma linha é removida.
    """
    n = len(d)
    chave = ["idpaciente", "data_solicitacao", "datahora_consulta", "especialidademae"]
    return [
        linha("duplicatas", "idsolicitacao_repetido", "", int(d["idsolicitacao"].duplicated().sum()), n),
        linha("duplicatas", "rid_repetido", "", int(d["rid"].duplicated().sum()), n),
        linha("duplicatas", "linha_inteira_repetida", "", int(d.duplicated().sum()), n),
        linha("duplicatas", "chave_composta_repetida", "paciente+solicitacao+consulta+especialidade",
              int(d.duplicated(chave).sum()), n, "mantidas na base; só descritivo"),
    ]


def audita_datas(d: pd.DataFrame, periodo: list[str]) -> list[dict]:
    """Valida datas: ausentes, ordem lógica, consistência de dias_ate_consulta e período esperado.

    Parâmetros:
        d: base.
        periodo: [início, fim] esperados (config dados.periodo_esperado).

    Retorna:
        Linhas da tabela longa com contagens de violações (0 esperado) e a distribuição por ano.
    """
    n = len(d)
    sol = pd.to_datetime(d["data_solicitacao"])
    con = pd.to_datetime(d["datahora_consulta"])
    t0 = pd.to_datetime(d["t0"])
    ini, fim = pd.Timestamp(periodo[0]), pd.Timestamp(periodo[1])
    out = [
        linha("datas", "data_solicitacao_ausente", "", int(sol.isna().sum()), n),
        linha("datas", "datahora_consulta_ausente", "", int(con.isna().sum()), n),
        linha("datas", "t0_ausente", "", int(t0.isna().sum()), n),
        linha("datas", "solicitacao_depois_da_consulta", "", int((sol > con).sum()), n),
        linha("datas", "t0_depois_da_consulta", "", int((t0 > con).sum()), n),
        linha("datas", "t0_antes_da_solicitacao", "", int((t0 < sol).sum()), n),
        linha("datas", "dias_ate_consulta_negativo", "", int((d["dias_ate_consulta"] < 0).sum()), n),
        linha("datas", "dias_ate_consulta_diferente_de_consulta_menos_solicitacao", "",
              int((d["dias_ate_consulta"] != (con - sol).dt.days).sum()), n),
        linha("datas", "consulta_fora_do_periodo_esperado", f"{periodo[0]}..{periodo[1]}",
              int(((con < ini) | (con > fim)).sum()), n),
        linha("datas", "primeira_consulta", str(con.min().date()), 1),
        linha("datas", "ultima_consulta", str(con.max().date()), 1),
    ]
    por_ano = con.dt.year.value_counts().sort_index()
    y = (d["situacaoagenda"] == "FALTANTE").groupby(con.dt.year).mean()
    for ano, qtd in por_ano.items():
        out.append(linha("ano", "consultas_por_ano", ano, int(qtd), n, f"taxa de falta no ano = {y[ano]:.4f}"))
    return out


def audita_dias_ate_consulta(d: pd.DataFrame, horizontes: list[int]) -> list[dict]:
    """Distribuição de dias_ate_consulta (percentis) e fração com antecedência >= h.

    Parâmetros:
        d: base.
        horizontes: lista de h (dias) para a fração de consultas com dias_ate_consulta >= h.

    Retorna:
        Linhas da tabela longa. Nota: dias_ate_consulta inclui a fila de espera; a
        antecedência do AGENDAMENTO (consulta − t0) é medida separadamente, pois é a que
        importa para saber o que estava agendado em t − h (etapa S3.1).
    """
    n = len(d)
    out = []
    dias = d["dias_ate_consulta"]
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        out.append(linha("dias_ate_consulta", "percentil", p, float(np.percentile(dias, p))))
    antec = (pd.to_datetime(d["datahora_consulta"]) - pd.to_datetime(d["t0"])).dt.days
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        out.append(linha("antecedencia_agendamento", "percentil (consulta - t0)", p, float(np.percentile(antec, p))))
    for h in horizontes:
        out.append(linha("antecedencia_agendamento", "consultas_com_antecedencia_>=h", h,
                         int((antec >= h).sum()), n, "pct = fração agendada com pelo menos h dias de antecedência"))
    return out


def audita_ausentes(d: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Ausentes por variável (total) e por variável × ano da consulta.

    Parâmetros:
        d: base.

    Retorna:
        (linhas da tabela longa só com as variáveis que têm ausentes,
         DataFrame longo variavel/ano/n_linhas/n_ausentes/pct_ausentes com TODAS as variáveis).
    """
    ano = pd.to_datetime(d["datahora_consulta"]).dt.year.rename("ano")
    isna = d.isna()
    partes = []
    for col in d.columns:
        g = isna[col].groupby(ano).agg(["sum", "size"]).reset_index()
        g.columns = ["ano", "n_ausentes", "n_linhas"]
        g.insert(0, "variavel", col)
        partes.append(g)
    tab = pd.concat(partes, ignore_index=True)
    tab["pct_ausentes"] = (100 * tab["n_ausentes"] / tab["n_linhas"]).round(4)
    tab = tab[["variavel", "ano", "n_linhas", "n_ausentes", "pct_ausentes"]]
    totais = isna.sum()
    linhas = [linha("ausentes", col, "", int(n), len(d)) for col, n in totais.items() if n > 0]
    return linhas, tab


def verifica_historico(d: pd.DataFrame) -> list[dict]:
    """Recalcula n_consultas_previas e n_faltas_previas com a regra data_ev < t0 e compara.

    Parâmetros:
        d: base (no smoke, amostra de pacientes inteiros, o que mantém a conta válida).

    Retorna:
        Linhas da tabela longa: linhas que batem, divergentes e viés médio (recalculado − original).

    Cuidados:
        # ANTI-VAZAMENTO: confirma que o histórico usa só eventos ANTERIORES a t0 e que a
        # própria consulta nunca entra (data da consulta >= t0). Pequenas divergências podem
        # vir de linhas removidas depois do cálculo no R (ex.: filtros posteriores); por isso
        # é um relato, não um assert. Implementação vetorizada: chave = paciente*BIG + dia,
        # contagem por busca binária (np.searchsorted) dentro do bloco de cada paciente.
    """
    pid = pd.factorize(d["idpaciente"])[0].astype(np.int64)
    dia = pd.to_datetime(d["datahora_consulta"]).values.astype("datetime64[D]").astype(np.int64)
    t0 = pd.to_datetime(d["t0"]).values.astype("datetime64[D]").astype(np.int64)
    y = (d["situacaoagenda"] == "FALTANTE").to_numpy(dtype=np.int64)
    chave = pid * BIG + dia
    ordem = np.argsort(chave, kind="stable")
    chave_ord = chave[ordem]
    cum_y = np.concatenate([[0], np.cumsum(y[ordem])])
    inicio = np.searchsorted(chave_ord, pid * BIG, side="left")            # 1º evento do paciente
    fim = np.searchsorted(chave_ord, pid * BIG + t0, side="left")           # eventos com dia < t0
    n_prev = fim - inicio
    n_falt = cum_y[fim] - cum_y[inicio]
    n = len(d)
    orig_n = d["n_consultas_previas"].to_numpy(dtype=np.int64)
    orig_f = d["n_faltas_previas"].to_numpy(dtype=np.int64)
    return [
        linha("historico", "n_consultas_previas_igual_ao_recalculo", "", int((n_prev == orig_n).sum()), n),
        linha("historico", "n_faltas_previas_igual_ao_recalculo", "", int((n_falt == orig_f).sum()), n),
        linha("historico", "n_consultas_previas_divergente", "", int((n_prev != orig_n).sum()), n),
        linha("historico", "vies_medio_n_consultas (recalculo - original)", "", float((n_prev - orig_n).mean()),
              obs="valor em consultas, não em %"),
        linha("historico", "historico_original_maior_que_recalculo", "", int((orig_n > n_prev).sum()), n,
              "se > 0, o original contém eventos não anteriores a t0 (possível vazamento)"),
    ]


def classifica_atividade(d: pd.DataFrame, flag_mun: set[str]) -> pd.DataFrame:
    """Classifica as CNES por atividade (DESCRITIVO; não exclui nada).

    Parâmetros:
        d: base.
        flag_mun: conjunto de CNES municipais.

    Retorna:
        DataFrame indexado por cnesexecutante com n por ano, primeira/última consulta,
        flag municipal e classe (inativo, parou em 2024, sem 2026, novo, reativado, ativo).

    Cuidados:
        Usa a série inteira e, portanto, NÃO pode ser usado para filtrar treino ou teste
        (regra 5 do CLAUDE.md). É só uma descrição da base.
    """
    con = pd.to_datetime(d["datahora_consulta"])
    tab = (pd.crosstab(d["cnesexecutante"], con.dt.year)
           .rename(columns=lambda a: f"n_{a}"))
    for ano in (2024, 2025, 2026):
        if f"n_{ano}" not in tab:
            tab[f"n_{ano}"] = 0
    g = con.groupby(d["cnesexecutante"]).agg(primeira="min", ultima="max")
    tab = tab.join(g)
    tab["total"] = tab.filter(regex=r"^n_\d{4}$").sum(axis=1)

    def classe(r: pd.Series) -> str:
        """Classe de atividade de uma CNES a partir das contagens de 2024, 2025, 2026 e da 1ª consulta (descritivo)."""
        if r.n_2024 == 0 and r.n_2025 == 0 and r.n_2026 == 0:
            return "1_inativo (0 em 2024, 2025 e 2026)"
        if r.n_2025 == 0 and r.n_2026 == 0:
            return "2_parou em 2024 (0 em 2025 e 2026)"
        if r.n_2026 == 0:
            return "3_sem 2026 (ativo em 2025)"
        if r.primeira.year >= 2025:
            return "4_novo (1a consulta em 2025+)"
        if r.n_2024 == 0:
            return "5_ativo, mas 0 em 2024 (reativado)"
        return "6_ativo (2024, 2025 e 2026)"

    tab["classe"] = tab.apply(classe, axis=1)
    tab["municipal"] = tab.index.isin(flag_mun)
    tab["primeira"] = tab["primeira"].dt.date
    tab["ultima"] = tab["ultima"].dt.date
    return tab


def monta_fluxograma(d: pd.DataFrame, cfg: dict, ativ: pd.DataFrame) -> pd.DataFrame:
    """Fluxograma de seleção (n de consultas e de CNES por etapa), gerado por código.

    Parâmetros:
        d: base.
        cfg: configuração (caminhos de flag_municipal e niveis_municipal).
        ativ: tabela de atividade (para registrar as CNES por classe).

    Retorna:
        DataFrame com colunas etapa, descricao, n_consultas, n_cnes, removidas, fonte.
        As etapas anteriores ao parquet (filtros do reestruturacao.qmd) entram como linhas
        SEM número, porque o arquivo bruto não está disponível: declarado, não inventado.
    """
    linhas = []
    amont = ["tipounidade == CENTRO DE SAUDE/UNIDADE BASICA (solicitante)",
             "remoção de solicitantes sem população do setor (exceto lista de exceções)",
             "datahora_consulta ausente (sem agendamento)",
             "situacaoagenda fora de REALIZADA e FALTANTE"]
    for i, txt in enumerate(amont):
        linhas.append({"etapa": f"A{i + 1}", "descricao": txt, "n_consultas": np.nan, "n_cnes": np.nan,
                       "removidas": np.nan, "fonte": "reestruturacao.qmd (não reexecutado; arquivo bruto ausente)"})
    n0 = len(d)
    linhas.append({"etapa": "B0", "descricao": "Base analítica (parquet), com 2018 mantido",
                   "n_consultas": n0, "n_cnes": d["cnesexecutante"].nunique(), "removidas": 0,
                   "fonte": "parquet (verificado pelo código)"})
    y_ok = d["situacaoagenda"].isin(["REALIZADA", "FALTANTE"])
    linhas.append({"etapa": "B1", "descricao": "Desfecho observável (REALIZADA ou FALTANTE)",
                   "n_consultas": int(y_ok.sum()), "n_cnes": d.loc[y_ok, "cnesexecutante"].nunique(),
                   "removidas": int((~y_ok).sum()), "fonte": "código"})
    mun = pd.read_csv(caminho(cfg, cfg["dados"]["flag_municipal"]), dtype={"cnesexecutante": str})
    cnes_mun = set(mun.loc[mun["municipal"], "cnesexecutante"])
    dm = d[d["cnesexecutante"].isin(cnes_mun)]
    linhas.append({"etapa": "C1", "descricao": "Escopo municipal (flag_municipal.csv)",
                   "n_consultas": len(dm), "n_cnes": dm["cnesexecutante"].nunique(),
                   "removidas": n0 - len(dm), "fonte": "código"})
    nv = pd.read_csv(caminho(cfg, cfg["dados"]["niveis_municipal"]), dtype={"cnes": str})
    dc = dm[dm["cnesexecutante"].isin(set(nv["cnes"]))]
    linhas.append({"etapa": "C2", "descricao": "Coorte estável municipal (níveis A+B, versão global)",
                   "n_consultas": len(dc), "n_cnes": dc["cnesexecutante"].nunique(),
                   "removidas": len(dm) - len(dc),
                   "fonte": "niveis_municipal.csv (definida com a série inteira; será refeita por corte no M1)"})
    for classe, n_c in ativ["classe"].value_counts().sort_index().items():
        linhas.append({"etapa": "D", "descricao": f"Informativo, atividade das CNES: {classe}",
                       "n_consultas": int(ativ.loc[ativ["classe"] == classe, "total"].sum()),
                       "n_cnes": int(n_c), "removidas": 0, "fonte": "descritivo; nenhuma CNES é excluída"})
    return pd.DataFrame(linhas)


def escreve_dicionario(d: pd.DataFrame, cfg: dict, destino: Path) -> list[str]:
    """Gera o dicionário de variáveis em Markdown a partir do esquema da base e do CSV manual.

    Parâmetros:
        d: base (para tipo, ausentes e nº de valores distintos).
        cfg: configuração (caminho de dados.dicionario_manual).
        destino: arquivo .md de saída.

    Retorna:
        Lista de variáveis do parquet SEM entrada no dicionário manual (vazia = completo).
    """
    man = pd.read_csv(caminho(cfg, cfg["dados"]["dicionario_manual"]), sep=";", dtype=str).set_index("variavel")
    sem_entrada = [c for c in d.columns if c not in man.index]
    linhas = [
        "# Dicionário de variáveis",
        "",
        "Gerado por `etapas/S1_auditoria.py` (esquema do parquet + `dados/dicionario_manual.csv`).",
        "Coluna **Momento** = quando a variável passa a ser conhecida (regra 8 do CLAUDE.md).",
        "Coluna **Risco pós-desfecho**: `nao`, `suspeita`, `suspeita_baixa` ou `a_confirmar` (pendente de confirmação com o serviço).",
        "",
        "| Variável | Tipo | Ausentes (%) | Distintos | Grupo | Momento | Risco pós-desfecho | Observação |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for col in d.columns:
        s = d[col]
        tipo = str(s.dtype)
        pct_na = 100 * s.isna().mean()
        nd = int(s.nunique(dropna=True))
        if col in man.index:
            m = man.loc[col]
            grupo, momento, risco, obs = m["grupo"], m["momento_disponivel"], m["risco_pos_desfecho"], m["fonte_ou_observacao"]
        else:
            grupo = momento = risco = "?"
            obs = "SEM ENTRADA NO DICIONÁRIO MANUAL"
        # Identificadores não mostram nº de valores distintos exato por segurança (LGPD): só ordem de grandeza.
        if col in ("idpaciente", "idsolicitacao"):
            nd_txt = "(oculto)"
        else:
            nd_txt = f"{nd:,}".replace(",", ".")
        linhas.append(f"| `{col}` | {tipo} | {pct_na:.2f} | {nd_txt} | {grupo} | {momento} | {risco} | {obs} |")
    linhas += [
        "",
        "## Notas",
        "",
        "- O parquet contém só `situacaoagenda` ∈ {REALIZADA, FALTANTE}. As demais categorias foram removidas no",
        "  `reestruturacao.qmd`; a contagem delas está no arquivo bruto, que não está disponível (pendência).",
        "- O script que gerou as distâncias (`status`, `distancia_km`, ...) não está versionado (pendência).",
        "- Variáveis de histórico seguem `data_ev < t0`; a conferência está em `tabelas/auditoria_dados.csv` (bloco `historico`).",
    ]
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return sem_entrada


# --------------------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------------------
def main() -> None:
    """Executa a auditoria completa e grava tabelas, dicionário, manifesto e log."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="amostra de pacientes (todos os anos), saídas com sufixo _smoke")
    ap.add_argument("--config", default=None, help="caminho alternativo do config.yaml")
    args = ap.parse_args()

    cfg = carrega_config(args.config)
    sufixo = "_smoke" if args.smoke else ""
    reg = RegistroEtapa("S1_auditoria" + sufixo, caminho(cfg, cfg["saidas"]["logs"]),
                        {"smoke": args.smoke, "semente": cfg["semente"]})
    pasta_tab = caminho(cfg, cfg["saidas"]["tabelas"])
    pasta_tab.mkdir(parents=True, exist_ok=True)

    print(f"[S1] lendo a base ({'smoke' if args.smoke else 'completa'})...", flush=True)
    d, contagens = carrega_base(cfg, args.smoke)
    arquivo_base = caminho(cfg, cfg["dados"]["base"])
    hash_base = sha256_arquivo(arquivo_base)  # hash do parquet COMPLETO (também no smoke)
    print(f"[S1] linhas lidas: {len(d):,} | colunas: {d.shape[1]} | sha256: {hash_base[:16]}...")

    linhas = [linha("base", "linhas", "", len(d)), linha("base", "colunas", "", d.shape[1]),
              linha("base", "sha256_parquet", hash_base, 1),
              linha("base", "cnes_executantes_distintas", "", int(d["cnesexecutante"].nunique())),
              linha("base", "pacientes_distintos", "", int(d["idpaciente"].nunique()), obs="só a contagem; sem identificadores")]
    linhas += audita_categorias(d)
    linhas += audita_duplicatas(d)
    linhas += audita_datas(d, cfg["dados"]["periodo_esperado"])
    linhas += audita_dias_ate_consulta(d, [1, 2, 3, 5, 7, 10, 14])
    lin_aus, tab_aus = audita_ausentes(d)
    linhas += lin_aus
    linhas += verifica_historico(d)

    mun = pd.read_csv(caminho(cfg, cfg["dados"]["flag_municipal"]), dtype={"cnesexecutante": str})
    ativ = classifica_atividade(d, set(mun.loc[mun["municipal"], "cnesexecutante"]))
    for classe, n_c in ativ["classe"].value_counts().sort_index().items():
        linhas.append(linha("atividade_cnes", "cnes_por_classe", classe, int(n_c), len(ativ), "descritivo"))

    aud = pd.DataFrame(linhas)
    fluxo = monta_fluxograma(d, cfg, ativ)

    saidas = {
        "auditoria": pasta_tab / f"auditoria_dados{sufixo}.csv",
        "ausentes": pasta_tab / f"ausentes_por_variavel_ano{sufixo}.csv",
        "atividade": pasta_tab / f"atividade_cnes{sufixo}.csv",
        "fluxo": pasta_tab / f"fluxograma_selecao{sufixo}.csv",
        "dicionario": caminho(cfg, f"dicionario_variaveis{sufixo}.md"),
    }
    aud.to_csv(saidas["auditoria"], index=False)
    tab_aus.to_csv(saidas["ausentes"], index=False)
    ativ.to_csv(saidas["atividade"])
    fluxo.to_csv(saidas["fluxo"], index=False)
    sem_entrada = escreve_dicionario(d, cfg, saidas["dicionario"])

    # ---- Resumo curto no terminal (só agregados) ----
    print("\n[S1] desfecho:\n" + aud[(aud.item == "situacaoagenda")][["categoria", "n", "pct"]].to_string(index=False))
    print("\n[S1] status:\n" + aud[(aud.bloco == "categorias") & (aud.item == "status")][["categoria", "n", "pct"]].to_string(index=False))
    print("\n[S1] checagens com valor 0 esperado:")
    chk = aud[aud.item.isin(["solicitacao_depois_da_consulta", "t0_depois_da_consulta", "t0_antes_da_solicitacao",
                             "dias_ate_consulta_negativo", "idsolicitacao_repetido", "rid_repetido",
                             "linha_inteira_repetida", "chave_composta_repetida",
                             "dias_ate_consulta_diferente_de_consulta_menos_solicitacao",
                             "consulta_fora_do_periodo_esperado", "historico_original_maior_que_recalculo"])]
    print(chk[["item", "n"]].to_string(index=False))
    print("\n[S1] histórico recalculado:\n" + aud[aud.bloco == "historico"][["item", "n", "pct"]].to_string(index=False))
    if sem_entrada:
        print(f"\n[S1] ATENÇÃO: variáveis sem entrada no dicionário manual: {sem_entrada}")

    # ---- Manifesto e log ----
    entradas = [arquivo_base, caminho(cfg, cfg["dados"]["flag_municipal"]),
                caminho(cfg, cfg["dados"]["niveis_municipal"]), caminho(cfg, cfg["dados"]["dicionario_manual"])]
    saidas_lista = list(saidas.values())
    grava_manifesto("S1_auditoria" + sufixo, entradas, saidas_lista, caminho(cfg, cfg["saidas"]["manifestos"]),
                    {"smoke": args.smoke, "variaveis_sem_entrada_no_dicionario": sem_entrada})
    log = reg.finaliza(entradas, saidas_lista, {**contagens, "linhas_auditoria": len(aud)})
    print(f"\n[S1] concluído. Saídas em {pasta_tab} e {saidas['dicionario'].name}; log: {log.name}")


if __name__ == "__main__":
    main()
