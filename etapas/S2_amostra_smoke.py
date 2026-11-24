"""S2 — Amostra do teste de fumaça (marco M1).

Objetivo
    Extrair do parquet as linhas das CNES listadas em config.yaml (`smoke.cnes`) e verificar que
    CADA uma tem consultas em todos os anos 2019–2026 (e, quando houver, 2018). Os esquemas de
    divisão e os testes de vazamento precisam de vários anos, não só 2025–2026.

Entradas   base (parquet), config.yaml: smoke.cnes
Saídas     dados/derivados/amostra_smoke_<escopo>.parquet (ignorado pelo git: contém linhas de paciente)
           tabelas/amostra_smoke_<escopo>.csv  (n de consultas por CNES e ano, só agregados)
Como rodar python3 etapas/S2_amostra_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import caminho, carrega_config  # noqa: E402
from src.utils_io import RegistroEtapa, grava_manifesto  # noqa: E402


def extrai_amostra(base: pd.DataFrame, cnes: list[str], anos_exigidos: range) -> pd.DataFrame:
    """Seleciona as linhas das CNES da amostra e confere a cobertura por ano.

    Parâmetros:
        base: base completa por consulta (coluna `cnesexecutante` e `datahora_consulta`).
        cnes: lista de CNES da amostra (config smoke.cnes); não pode ser vazia.
        anos_exigidos: anos em que TODA CNES precisa ter ao menos uma consulta (ex.: range(2019, 2027)).

    Retorna:
        DataFrame com todas as colunas da base, só das CNES escolhidas.

    Cuidados:
        Falha com mensagem clara se alguma CNES não existir ou faltar algum ano.
    """
    assert len(cnes) > 0, "smoke.cnes está vazio no config.yaml"
    amostra = base[base["cnesexecutante"].isin(cnes)].copy()
    ausentes = set(cnes) - set(amostra["cnesexecutante"].unique())
    assert not ausentes, f"CNES da amostra sem nenhuma consulta na base: {sorted(ausentes)}"
    anos = pd.to_datetime(amostra["datahora_consulta"]).dt.year
    cob = pd.crosstab(amostra["cnesexecutante"], anos)
    for c in cnes:
        faltam = [a for a in anos_exigidos if cob.loc[c].get(a, 0) == 0]
        assert not faltam, f"CNES {c} sem consultas nos anos {faltam}: não serve para o smoke"
    return amostra


def main() -> None:
    """Grava a amostra do smoke e a tabela de cobertura por CNES e ano."""
    cfg = carrega_config()
    escopo = cfg["escopo"]
    reg = RegistroEtapa(f"S2_amostra_smoke_{escopo}", caminho(cfg, cfg["saidas"]["logs"]), {"cnes": cfg["smoke"]["cnes"]})
    arquivo = caminho(cfg, cfg["dados"]["base"])
    base = pd.read_parquet(arquivo)
    amostra = extrai_amostra(base, [str(c) for c in cfg["smoke"]["cnes"]], range(2019, 2027))
    anos = pd.to_datetime(amostra["datahora_consulta"]).dt.year
    cob = pd.crosstab(amostra["cnesexecutante"], anos)
    cob["total"] = cob.sum(axis=1)
    out_pq = caminho(cfg, "dados", "derivados", f"amostra_smoke_{escopo}.parquet")
    out_csv = caminho(cfg, cfg["saidas"]["tabelas"], f"amostra_smoke_{escopo}.csv")
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_parquet(out_pq, index=False)
    cob.to_csv(out_csv)
    print(cob.to_string())
    print(f"[amostra] {len(amostra):,} linhas ({100 * len(amostra) / len(base):.1f}% da base); "
          f"taxa de falta {100 * (amostra['situacaoagenda'] == 'FALTANTE').mean():.1f}%")
    grava_manifesto(f"S2_amostra_smoke_{escopo}", [arquivo], [out_pq, out_csv], caminho(cfg, cfg["saidas"]["manifestos"]))
    reg.finaliza([arquivo], [out_pq, out_csv], {"linhas_amostra": int(len(amostra))})


if __name__ == "__main__":
    main()
