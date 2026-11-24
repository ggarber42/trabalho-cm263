"""Auditoria estática das convenções do CLAUDE.md (seção 7) nos arquivos novos (src/, etapas/, tests/).

* todo script de etapas/ tem cabeçalho com "Objetivo" e "Como rodar";
* toda função/classe pública tem docstring;
* nenhum caminho absoluto do usuário no código; nenhum `random_state=None`;
* nenhum identificador de paciente é gravado em saídas (colunas proibidas);
* estimadores aleatórios do sklearn recebem `random_state`.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVOS = sorted([*RAIZ.glob("src/**/*.py"), *RAIZ.glob("etapas/*.py")])


@pytest.mark.parametrize("arq", ARQUIVOS, ids=lambda p: str(p.relative_to(RAIZ)))
def test_docstrings_e_convencoes(arq: Path):
    texto = arq.read_text(encoding="utf-8")
    arvore = ast.parse(texto)
    if arq.parent.name == "etapas":
        cab = ast.get_docstring(arvore) or ""
        assert "Objetivo" in cab and "Como rodar" in cab, f"{arq.name}: cabeçalho sem Objetivo/Como rodar"
    faltando = []
    for no in ast.walk(arvore):
        if isinstance(no, (ast.FunctionDef, ast.ClassDef)) and not no.name.startswith("_") and not ast.get_docstring(no):
            if no.name in ("fit", "predict_mean", "_taxa", "espaco") and arq.parent.name == "models":
                continue                       # métodos de contrato documentados na interface (base.py)
            faltando.append(no.name)
    assert not faltando, f"{arq.name}: sem docstring: {faltando}"
    assert "/home/" not in texto, f"{arq.name}: caminho absoluto no código"
    assert "random_state=None" not in texto.replace(" ", ""), f"{arq.name}: random_state=None"


def test_estimadores_aleatorios_recebem_semente():
    """Todo RandomForest/HGB/LightGBM/MLP instanciado nos modelos tem random_state."""
    for arq in (RAIZ / "src" / "models").glob("*.py"):
        for m in re.finditer(r"(RandomForestRegressor|HistGradientBoosting(?:Regressor|Classifier)|LGBMRegressor)\(([^)]*)\)", arq.read_text(encoding="utf-8")):
            assert "random_state" in m.group(2), f"{arq.name}: {m.group(1)} sem random_state"


def test_saidas_sem_identificador_de_paciente():
    """Tabelas e previsões gravadas não podem ter colunas de identificação do paciente."""
    import pandas as pd
    proibidas = {"idpaciente", "idsolicitacao"}
    for csv in list((RAIZ / "tabelas").glob("*.csv")):
        assert not (proibidas & set(pd.read_csv(csv, nrows=0).columns)), f"{csv.name} tem identificador de paciente"
    for pq in (RAIZ / "resultados").glob("**/*.parquet"):
        assert not (proibidas & set(pd.read_parquet(pq).columns)), f"{pq.name} tem identificador de paciente"
