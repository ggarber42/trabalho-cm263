"""Leitura do config.yaml e resolução de caminhos (módulo compartilhado).

Etapa: S0. Objetivo: ser o único ponto onde o código descobre parâmetros e caminhos,
para que nenhum caminho absoluto apareça nos scripts. Não executa nada ao ser importado.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Raiz do projeto = pasta que contém o config.yaml (um nível acima de src/).
RAIZ = Path(__file__).resolve().parent.parent


def _merge(base: dict, extra: dict) -> dict:
    """Merge profundo: valores de `extra` sobrescrevem os de `base` (dicts são mesclados; listas, substituídas)."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def carrega_config(caminho: str | Path | None = None, perfil: str | None = None) -> dict[str, Any]:
    """Lê o config.yaml, aplica o perfil ativo e devolve o dicionário de parâmetros.

    Parâmetros:
        caminho: caminho do YAML; se None, usa ``<raiz>/config.yaml``.
        perfil: nome do perfil (smoke ou completo). Se None, usa a variável de ambiente PERFIL ou a
            chave `perfil` do YAML. O perfil sobrescreve (merge profundo) as demais chaves.

    Retorna:
        dict com todos os parâmetros, mais a chave ``_raiz`` (str) com a raiz do projeto.

    Cuidados:
        Falha com mensagem clara se o arquivo não existir ou se faltar a semente
        (regra: nenhuma aleatoriedade sem semente registrada).
    """
    arquivo = Path(caminho) if caminho else RAIZ / "config.yaml"
    assert arquivo.exists(), f"config.yaml não encontrado em {arquivo}"
    with open(arquivo, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("semente") is not None, "config.yaml sem 'semente': defina uma semente fixa"
    escolhido = perfil or os.environ.get("PERFIL") or cfg.get("perfil", "smoke")
    assert escolhido in cfg.get("perfis", {}), f"Perfil desconhecido: {escolhido}"
    cfg = _merge(cfg, cfg["perfis"][escolhido])
    cfg["perfil_ativo"] = escolhido
    cfg["_raiz"] = str(RAIZ)
    return cfg


def caminho(cfg: dict[str, Any], *partes: str) -> Path:
    """Resolve um caminho relativo (dado em partes) contra a raiz do projeto.

    Parâmetros:
        cfg: dicionário devolvido por ``carrega_config``.
        partes: pedaços do caminho relativo, ex.: ``caminho(cfg, cfg["dados"]["base"])``.

    Retorna:
        ``Path`` absoluto. Se alguma parte já for absoluta, ela é mantida (não recomendado).
    """
    return Path(cfg["_raiz"]).joinpath(*partes)
