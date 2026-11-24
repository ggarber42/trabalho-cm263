"""Leitura da configuração, argumentos comuns das etapas, log e manifesto (hashes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def _mesclar(base: dict, extra: dict) -> dict:
    """Mescla dicionários de forma recursiva (extra sobrescreve base)."""
    saida = dict(base)
    for chave, valor in extra.items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _mesclar(saida[chave], valor)
        else:
            saida[chave] = valor
    return saida


def carregar_config(caminho: str | Path) -> dict[str, Any]:
    """Lê o YAML de configuração (com suporte a `herda:`) e resolve os caminhos.

    Parâmetros: caminho (str | Path) do YAML.
    Retorna: dict com as chaves do YAML; caminhos de `dados` e `saidas` viram Path absolutos,
    resolvidos em relação à pasta do arquivo de configuração.
    """
    caminho = Path(caminho).resolve()
    with open(caminho, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "herda" in cfg:
        base = carregar_config(caminho.parent / cfg.pop("herda"))
        base.pop("_pasta", None)
        cfg = _mesclar(_desresolver(base), cfg)
    pasta = caminho.parent
    cfg["_pasta"] = pasta
    for chave, valor in cfg["dados"].items():
        cfg["dados"][chave] = (pasta / valor).resolve()
    cfg["saidas"] = (pasta / str(cfg["saidas"])).resolve()
    return cfg


def _desresolver(cfg: dict) -> dict:
    """Converte Paths de volta a str para permitir nova resolução após a herança."""
    out = dict(cfg)
    out["dados"] = {k: str(v) for k, v in cfg["dados"].items()}
    out["saidas"] = str(cfg["saidas"])
    return out


def argumentos(descricao: str) -> argparse.Namespace:
    """Argumentos comuns a todas as etapas.

    --config: YAML (padrão config.yaml); --perfil: nome da pasta de saída (smoke | completo);
    --cnes: lista de CNES para restringir a modelagem (teste rápido; vazio = todas).
    """
    p = argparse.ArgumentParser(description=descricao)
    p.add_argument("--config", default=str(RAIZ / "config.yaml"))
    p.add_argument("--perfil", default="completo")
    p.add_argument("--cnes", nargs="*", default=[])
    return p.parse_args()


def pasta_saida(cfg: dict, perfil: str) -> Path:
    """Retorna (e cria) saidas/<perfil>/."""
    pasta = Path(cfg["saidas"]) / perfil
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def sha256(caminho: Path) -> str:
    """SHA-256 de um arquivo, lido em blocos."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


class Registro:
    """Log simples da etapa: imprime e grava em <saida>/logs/<etapa>.log, e escreve um manifesto
    JSON com entradas, saídas (com hash), parâmetros e tempo."""

    def __init__(self, pasta: Path, etapa: str):
        self.pasta = pasta
        self.etapa = etapa
        (pasta / "logs").mkdir(exist_ok=True)
        self.arquivo = open(pasta / "logs" / f"{etapa}.log", "w", encoding="utf-8")
        self.inicio = time.time()
        self.info(f"Etapa {etapa} | Python {platform.python_version()} | {' '.join(sys.argv)}")

    def info(self, msg: str) -> None:
        print(msg, flush=True)
        self.arquivo.write(msg + "\n")
        self.arquivo.flush()

    def manifesto(self, entradas: list[Path], saidas: list[Path], params: dict | None = None,
                  hash_entradas: bool = True) -> None:
        dados = {
            "etapa": self.etapa,
            "segundos": round(time.time() - self.inicio, 1),
            "entradas": {str(p): (sha256(p) if hash_entradas else None) for p in entradas},
            "saidas": {str(p): sha256(p) for p in saidas},
            "parametros": params or {},
        }
        with open(self.pasta / "logs" / f"manifesto_{self.etapa}.json", "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False, default=str)
        self.info(f"Concluído em {dados['segundos']} s")
