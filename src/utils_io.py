"""Hash, manifestos e log de etapas (módulo compartilhado).

Etapa: S0. Objetivo: cada etapa do pipeline grava (a) um manifesto com o SHA-256 das
entradas e saídas e (b) um log JSON com parâmetros, semente, tempo e contagens.
Não executa nada ao ser importado. (Nome utils_io.py para não sombrear o módulo io do Python.)
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_arquivo(arquivo: str | Path, bloco: int = 1 << 20) -> str:
    """Calcula o SHA-256 de um arquivo lendo em blocos (não carrega tudo na memória).

    Parâmetros:
        arquivo: caminho do arquivo.
        bloco: tamanho do bloco de leitura em bytes.

    Retorna:
        str hexadecimal de 64 caracteres.
    """
    h = hashlib.sha256()
    with open(arquivo, "rb") as f:
        while True:
            dados = f.read(bloco)
            if not dados:
                break
            h.update(dados)
    return h.hexdigest()


def grava_json(objeto: Any, destino: str | Path) -> None:
    """Grava um objeto serializável em JSON (UTF-8, indentado), criando a pasta se preciso."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(objeto, f, ensure_ascii=False, indent=2, default=str)


def grava_manifesto(etapa: str, entradas: list[Path], saidas: list[Path],
                    pasta_manifestos: Path, extra: dict | None = None) -> Path:
    """Grava o manifesto de uma etapa com o hash de entradas e saídas.

    Parâmetros:
        etapa: identificador da etapa (ex.: "S1_auditoria").
        entradas, saidas: listas de arquivos (devem existir).
        pasta_manifestos: pasta de destino.
        extra: informações adicionais (ex.: modo smoke).

    Retorna:
        Caminho do manifesto gravado (``<etapa>.json``).
    """
    manifesto = {
        "etapa": etapa,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "sistema": platform.platform(),
        "entradas": {str(p): sha256_arquivo(p) for p in entradas},
        "saidas": {str(p): sha256_arquivo(p) for p in saidas},
        "extra": extra or {},
    }
    destino = Path(pasta_manifestos) / f"{etapa}.json"
    grava_json(manifesto, destino)
    return destino


class RegistroEtapa:
    """Cronometra uma etapa e grava o log em ``logs/<etapa>_<data>.json``.

    Uso: ``reg = RegistroEtapa("S1_auditoria", pasta_logs, parametros)``; no fim,
    ``reg.finaliza(entradas, saidas, contagens)``.
    """

    def __init__(self, etapa: str, pasta_logs: Path, parametros: dict) -> None:
        self.etapa = etapa
        self.pasta_logs = Path(pasta_logs)
        self.parametros = parametros
        self.inicio = time.time()

    def finaliza(self, entradas: list[Path], saidas: list[Path], contagens: dict | None = None) -> Path:
        """Grava o log com entradas, saídas, parâmetros, contagens e duração em segundos."""
        log = {
            "etapa": self.etapa,
            "parametros": self.parametros,
            "entradas": [str(p) for p in entradas],
            "saidas": [str(p) for p in saidas],
            "contagens": contagens or {},
            "duracao_s": round(time.time() - self.inicio, 2),
            "python": sys.version.split()[0],
        }
        destino = self.pasta_logs / f"{self.etapa}_{datetime.now():%Y%m%d_%H%M%S}.json"
        grava_json(log, destino)
        return destino
