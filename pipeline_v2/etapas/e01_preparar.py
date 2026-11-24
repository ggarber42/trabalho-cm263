"""e01 — Preparação da base de consultas.

Objetivo: ler o CSV bruto do GERCON, registrar as categorias de `situacaoagenda`, padronizar
datas e códigos, marcar as CNES municipais, aplicar os critérios de seleção (com fluxograma) e
calcular o histórico do paciente sem vazamento (opção B: só na base filtrada).
Entradas: dados.gercon, dados.cnes_estabelecimentos, dados.cnes_natureza_juridica (config.yaml).
Saídas (saidas/<perfil>/): consultas.parquet (sem idpaciente), fluxograma.csv,
contagens_situacao.csv, cnes_municipais.csv, logs/.
Uso: python3 etapas/e01_preparar.py --config config.yaml --perfil completo
(--cnes não se aplica aqui: o histórico usa todas as CNES municipais.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas import preparacao as prep  # noqa: E402


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e01_preparar")

    log.info("Lendo o bruto...")
    bruto = prep.ler_bruto(cfg["dados"]["gercon"])
    log.info(f"  {len(bruto):,} linhas")

    log.info("Padronizando datas e códigos...")
    df = prep.padronizar(bruto, log=log.info)
    del bruto

    # Contagem de todas as situações antes de qualquer filtro (regra do projeto).
    contagens = prep.contar_situacoes(df)
    contagens.to_csv(saida / "contagens_situacao.csv", index=False)

    log.info("Marcando CNES municipais...")
    cad = prep.cnes_municipais(cfg["dados"]["cnes_estabelecimentos"], cfg["dados"]["cnes_natureza_juridica"], cfg["selecao"])
    municipais = set(cad.loc[cad["municipal"], "cnesexecutante"])
    presentes = set(df["cnesexecutante"].dropna())
    sem_cadastro = presentes - set(cad["cnesexecutante"])
    log.info(f"  {len(municipais):,} CNES municipais no cadastro; {len(sem_cadastro)} CNES da base sem cadastro (tratadas como não municipais)")
    cad[cad["cnesexecutante"].isin(presentes)].to_csv(saida / "cnes_municipais.csv", index=False)

    log.info("Aplicando critérios de seleção:")
    df["y"] = (df["situacaoagenda"] == "FALTANTE").astype("int8")
    df, fluxo = prep.aplicar_filtros(df, cfg, municipais, log=log.info)

    log.info("Calculando histórico do paciente (data_ev < t0)...")
    df = prep.calcular_historico(df)
    df = prep.variaveis_derivadas(df, cfg)
    df = df.sort_values(["data_consulta", "cnesexecutante"]).reset_index(drop=True)
    df["rid"] = range(len(df))

    n_analise = int(df["na_analise"].sum())
    fluxo.loc[len(fluxo)] = {"passo": len(fluxo), "criterio": "período de análise (o restante só alimenta o histórico)",
                             "linhas": n_analise, "removidas": len(df) - n_analise,
                             "cnes": int(df.loc[df["na_analise"], "cnesexecutante"].nunique())}
    log.info(f"  período de análise: {n_analise:,} consultas; taxa de falta {df.loc[df['na_analise'], 'y'].mean():.3f}")
    fluxo.to_csv(saida / "fluxograma.csv", index=False)

    assert "idpaciente" not in prep.COLUNAS_SAIDA, "idpaciente não pode sair da preparação"
    assert df["rid"].is_unique
    caminho = saida / "consultas.parquet"
    df[prep.COLUNAS_SAIDA].to_parquet(caminho, index=False)
    log.manifesto([cfg["dados"]["gercon"], cfg["dados"]["cnes_estabelecimentos"]],
                  [caminho, saida / "fluxograma.csv", saida / "contagens_situacao.csv"])


if __name__ == "__main__":
    main()
