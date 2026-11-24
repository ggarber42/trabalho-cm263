"""Testes do marco M0 (S0 e S1): configuração, hash, auditoria e dicionário.

Rodar: ``python3 -m pytest tests -q`` (da raiz do projeto).
Os testes de dados usam a base real em modo leitura; os de função usam dados sintéticos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "etapas"))

from src.config import caminho, carrega_config  # noqa: E402
from src.utils_io import sha256_arquivo  # noqa: E402
import S1_auditoria as s1  # noqa: E402


def test_config_tem_semente_e_sem_caminho_absoluto():
    """O config.yaml precisa ter semente fixa e só caminhos relativos."""
    cfg = carrega_config()
    assert isinstance(cfg["semente"], int)
    for chave in ("base", "estabelecimentos", "flag_municipal", "niveis_municipal", "dicionario_manual"):
        assert not Path(cfg["dados"][chave]).is_absolute(), f"Caminho absoluto em dados.{chave}"
    assert cfg["escopo"] in ("municipal", "geral", "criterio_x")


def test_sha256_estavel(tmp_path):
    """O hash do mesmo arquivo é sempre igual e muda se o conteúdo muda."""
    a = tmp_path / "a.txt"
    a.write_text("abc")
    h1 = sha256_arquivo(a)
    assert h1 == sha256_arquivo(a)
    a.write_text("abd")
    assert h1 != sha256_arquivo(a)


def _base_sintetica() -> pd.DataFrame:
    """Dois pacientes com histórico conhecido para testar a regra data_ev < t0."""
    # Paciente A: consultas em 01/02, 15/02 (falta), 10/03. t0 de cada uma = data - 5 dias.
    # Consulta 10/03 (t0 = 05/03): eventos anteriores a t0 = 01/02 e 15/02 -> n=2, faltas=1.
    # Consulta 15/02 (t0 = 10/02): anteriores = 01/02 -> n=1, faltas=0.
    # Consulta 01/02 (t0 = 27/01): nenhuma anterior -> n=0.
    # Paciente B: duas consultas no mesmo dia 20/03; t0 = 15/03 -> uma não conta a outra.
    linhas = [
        ("A", "2024-02-01", "2024-01-27", "REALIZADA", 0, 0),
        ("A", "2024-02-15", "2024-02-10", "FALTANTE", 1, 0),
        ("A", "2024-03-10", "2024-03-05", "REALIZADA", 2, 1),
        ("B", "2024-03-20", "2024-03-15", "REALIZADA", 0, 0),
        ("B", "2024-03-20", "2024-03-15", "FALTANTE", 0, 0),
    ]
    return pd.DataFrame(linhas, columns=["idpaciente", "datahora_consulta", "t0", "situacaoagenda",
                                         "n_consultas_previas", "n_faltas_previas"])


def test_verifica_historico_regra_data_ev_menor_que_t0():
    """O recálculo do histórico bate com os valores esperados à mão (regra data_ev < t0)."""
    res = {r["item"]: r["n"] for r in s1.verifica_historico(_base_sintetica())}
    assert res["n_consultas_previas_igual_ao_recalculo"] == 5
    assert res["n_faltas_previas_igual_ao_recalculo"] == 5
    assert res["historico_original_maior_que_recalculo"] == 0


def test_verifica_historico_detecta_vazamento():
    """Se o 'original' contar um evento a mais (posterior a t0), a checagem acusa."""
    d = _base_sintetica()
    d.loc[0, "n_consultas_previas"] = 1   # paciente A na 1ª consulta não tinha histórico
    res = {r["item"]: r["n"] for r in s1.verifica_historico(d)}
    assert res["historico_original_maior_que_recalculo"] == 1


def test_contagem_categorias_inclui_ausentes():
    """Ausentes aparecem como categoria própria e a soma fecha com o total."""
    s = pd.Series(["a", "a", None, "b"])
    linhas = s1.contagem_categorias(s, "x", "y")
    assert sum(l["n"] for l in linhas) == 4
    assert any(l["categoria"] == "<NA>" and l["n"] == 1 for l in linhas)


@pytest.mark.skipif(not (RAIZ / "tabelas" / "auditoria_dados.csv").exists(),
                    reason="auditoria completa ainda não rodada")
def test_auditoria_completa_fecha_com_o_total():
    """Na auditoria completa, a soma das categorias de situacaoagenda é o total de linhas."""
    aud = pd.read_csv(RAIZ / "tabelas" / "auditoria_dados.csv")
    total = aud.loc[(aud.bloco == "base") & (aud.item == "linhas"), "n"].iloc[0]
    soma = aud.loc[(aud.bloco == "desfecho") & (aud.item == "situacaoagenda"), "n"].sum()
    assert soma == total
    soma_status = aud.loc[(aud.bloco == "categorias") & (aud.item == "status"), "n"].sum()
    assert soma_status == total


@pytest.mark.skipif(not (RAIZ / "dicionario_variaveis_smoke.md").exists() and
                    not (RAIZ / "dicionario_variaveis.md").exists(), reason="dicionário ainda não gerado")
def test_dicionario_cobre_todas_as_variaveis():
    """Nenhuma variável do parquet pode ficar sem entrada no dicionário manual."""
    arq = RAIZ / "dicionario_variaveis.md"
    if not arq.exists():
        arq = RAIZ / "dicionario_variaveis_smoke.md"
    assert "SEM ENTRADA NO DICIONÁRIO MANUAL" not in arq.read_text(encoding="utf-8")


def test_smoke_cobre_todos_os_anos():
    """A amostra do smoke contém consultas de 2018 a 2026 (o carregamento já tem o assert)."""
    cfg = carrega_config()
    d, _ = s1.carrega_base(cfg, smoke=True)
    anos = set(pd.to_datetime(d["datahora_consulta"]).dt.year)
    assert set(range(2018, 2027)) <= anos
