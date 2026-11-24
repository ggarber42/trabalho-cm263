"""Contexto de modelagem: base por consulta, painel CNES-dia, features e coortes por corte (S3/S5).

Etapa: S3/S5. Carrega os dados UMA vez por (defasagem L, modo de agenda) e entrega aos modelos tudo o que
eles precisam, sem que o modelo precise saber de arquivos ou de datas de corte. Não executa nada ao ser
importado.

LGPD: o contexto contém linhas por paciente em memória, mas nada daqui grava identificadores; as saídas
dos experimentos são agregadas por CNES-dia.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import caminho
from src.coorte import cnes_do_escopo, coorte_e_niveis
from src.panels import AGENDA_VARS, constroi_painel, prepara_consultas_painel

CALENDARIO = ["dow", "mes", "fer", "posfer"]
TAXAS = ["r7", "r14", "r28", "r91", "rdow4", "rd1"]


def colunas_necessarias(cfg: dict[str, Any]) -> list[str]:
    """Colunas do parquet exigidas pelo painel e pelo modelo por paciente (sem `taxa_falta_prev`, que é derivada)."""
    pac = [c for c in cfg["paciente"]["numericas"] + cfg["paciente"]["categoricas"] if c != "taxa_falta_prev"]
    base = ["cnesexecutante", "datahora_consulta", "t0", "situacaoagenda", "n_faltas_previas", "n_consultas_previas",
            "primeira_vez", "idade", "dias_ate_consulta"]
    return list(dict.fromkeys(base + pac))


@dataclass
class Contexto:
    """Tudo o que os modelos usam para um par (L, modo de agenda).

    Atributos:
        lag, modo: defasagem L e modo de agenda ("conhecida" ou "final").
        P: tabela de modelagem = painel CNES-dia com ag > 0 (índice posicional 0..n−1 é a "linha" das divisões).
        F: features de série alinhadas a P (taxas, calendário, agenda do modo, `expo_bruta`).
        raw: base por consulta (cnes, dia, y, conhecida, linha_painel), alinhada a X.
        X: variáveis do modelo por paciente (categóricas como dtype category).
        cons_coorte: consultas (cnesexecutante, dia, falta) para definir coortes por corte.
        cfg: configuração.
    """
    lag: int
    modo: str
    P: pd.DataFrame
    F: pd.DataFrame
    raw: pd.DataFrame
    X: pd.DataFrame
    cons_coorte: pd.DataFrame
    cfg: dict
    _coortes: dict = field(default_factory=dict)

    def k_h(self, idx_treino: np.ndarray) -> float:
        """Fator k_h = Σ ag / Σ ag_h no TREINO, que reescala a agenda conhecida em t − L para a agenda final.

        # ANTI-VAZAMENTO: calculado só com as linhas de treino da divisão (passado). Em modo "final" vale 1.
        """
        if self.modo == "final":
            return 1.0
        ag = self.P["ag"].to_numpy()[idx_treino].sum()
        ag_h = self.P["ag_h"].to_numpy()[idx_treino].sum()
        assert ag_h > 0, "Treino sem nenhuma consulta agendada com antecedência >= L"
        return float(ag / ag_h)

    def expo(self, idx: np.ndarray, k_h: float) -> np.ndarray:
        """Exposição (nº esperado de consultas do dia na previsão): ag final, ou ag_h × k_h (modo conhecida)."""
        col = "ag" if self.modo == "final" else "ag_h"
        return self.P[col].to_numpy()[idx] * (1.0 if self.modo == "final" else k_h)

    def membros_coorte(self, corte: pd.Timestamp) -> pd.Series:
        """Nível (A/B/C) das CNES da coorte definida com dados ANTERIORES a `corte` (cache).

        # ANTI-VAZAMENTO: usa src.coorte.coorte_e_niveis, que descarta tudo com dia >= corte.
        """
        chave = str(pd.Timestamp(corte).date())
        if chave not in self._coortes:
            r = coorte_e_niveis(self.cons_coorte, chave, self.cfg["coorte"])
            self._coortes[chave] = r.loc[r["coorte"], "nivel"]
        return self._coortes[chave]

    def linhas_da_coorte(self, idx: np.ndarray, corte: pd.Timestamp) -> np.ndarray:
        """Restringe linhas do painel às CNES da coorte do corte dado."""
        membros = list(self.membros_coorte(corte).index)
        return idx[np.isin(self.P["cnes"].to_numpy()[idx], membros)]


def _features_serie(P: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Monta a matriz de features de série (taxas defasadas, calendário e agenda do modo)."""
    suf = "" if modo == "final" else "_h"
    F = P[TAXAS + CALENDARIO].copy()
    for v in AGENDA_VARS:
        F[v] = P[v + suf]
    F["expo_bruta"] = P["ag"] if modo == "final" else P["ag_h"]
    return F


def carrega_contexto(cfg: dict[str, Any], lag: int, modo: str) -> Contexto:
    """Carrega a base do escopo (ou da amostra do smoke) e monta o contexto de modelagem.

    Parâmetros:
        cfg: configuração (escopo, perfil, paciente, atraso_d).
        lag: defasagem L (1, 3, 7, 14).
        modo: "conhecida" ou "final".

    Retorna:
        Contexto pronto para os modelos.

    Cuidados:
        # ANTI-VAZAMENTO: nenhuma imputação com estatística global aqui; ausentes ficam NaN e cada modelo
        # imputa/padroniza só com o treino da sua divisão. A categoria de variáveis categóricas é o
        # vocabulário da base inteira (só nomes de categorias, sem ligação com o desfecho).
    """
    assert modo in ("conhecida", "final"), f"modo de agenda desconhecido: {modo}"
    cols = colunas_necessarias(cfg)
    if cfg.get("usa_amostra"):
        arq = caminho(cfg, "dados", "derivados", f"amostra_smoke_{cfg['escopo']}.parquet")
        assert arq.exists(), "Rode antes: python3 etapas/S2_amostra_smoke.py"
        d = pd.read_parquet(arq, columns=cols)
    else:
        d = pd.read_parquet(caminho(cfg, cfg["dados"]["base"]), columns=cols)
        escopo = cnes_do_escopo(cfg)
        if escopo is not None:
            n0 = len(d)
            d = d[d["cnesexecutante"].isin(escopo)]
            print(f"[dados] filtro de escopo '{cfg['escopo']}': removeu {n0 - len(d):,} linhas; restam {len(d):,}", flush=True)
    d = d.reset_index(drop=True)
    cons = prepara_consultas_painel(d)
    P_all = constroi_painel(cons, lag, cfg.get("atraso_d", 0))
    P = P_all[P_all["ag"] > 0].reset_index(drop=True)
    F = _features_serie(P, modo)

    # Linhas por consulta -> linha do painel (para somar probabilidades por CNES-dia).
    chave = pd.MultiIndex.from_arrays([P["cnes"], P["dia"]])
    pos = chave.get_indexer(pd.MultiIndex.from_arrays([cons["cnes"], cons["dia"]]))
    assert (pos >= 0).all(), "Há consultas sem linha no painel (cnes, dia)"
    raw = pd.DataFrame({"cnes": cons["cnes"].to_numpy(), "dia": cons["dia"].to_numpy(), "y": cons["y"].to_numpy(),
                        "conhecida": (cons["antec"] >= lag).to_numpy(), "linha_painel": pos})
    X = d[[c for c in cfg["paciente"]["numericas"] + cfg["paciente"]["categoricas"] if c != "taxa_falta_prev"]].copy()
    if "taxa_falta_prev" in cfg["paciente"]["numericas"]:
        X["taxa_falta_prev"] = (d["n_faltas_previas"] / d["n_consultas_previas"].where(d["n_consultas_previas"] > 0)).to_numpy()
    X = X[cfg["paciente"]["numericas"] + cfg["paciente"]["categoricas"]]
    for c in cfg["paciente"]["categoricas"]:
        X[c] = X[c].astype("category")
    cons_coorte = pd.DataFrame({"cnesexecutante": cons["cnes"].to_numpy(), "dia": cons["dia"].to_numpy(),
                                "falta": cons["y"].to_numpy()})
    return Contexto(lag, modo, P, F, raw, X, cons_coorte, cfg)
