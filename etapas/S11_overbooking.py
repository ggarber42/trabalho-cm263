"""S11 — Utilidade de decisão: regra de overbooking por quantil da distribuição preditiva.

Objetivo
    Para cada modelo, simular a regra "agendar k encaixes extras por CNES-dia", com k = quantil q da
    distribuição preditiva do nº de faltas (risco aceito: P(faltas < k) ≤ q). Comparar com a regra sem
    previsão (k = 0). Mede, nas MESMAS linhas para todos os modelos:
      * recuperadas  = Σ min(k, faltas)        (consultas que seriam perdidas e foram ocupadas)
      * excesso      = Σ max(k − faltas, 0)    (pacientes a mais que o previsto: sobreoferta)
      * dias_com_excesso (fração de CNES-dias com k > faltas)
      * beneficio_liquido(c) = recuperadas − c·excesso, para custos relativos c declarados abaixo.

Protocolo
    Só DESCREVE o teste (regra 6 do CLAUDE.md): nada é escolhido aqui. A dispersão phi vem de cada divisão
    (estimada nos folds internos). Custos c são declarados, não estimados: c = 1 (excesso custa o mesmo que
    uma consulta recuperada vale), 2 e 3 (espera/consulta atrasada pesa mais).

Entradas   resultados/<escopo>_<perfil>/ (saída de S7), config.yaml
Saídas     tabelas/overbooking_<escopo>_<perfil>.csv, ..._por_cnes.csv, ..._curva.csv
Como rodar python3 etapas/S11_overbooking.py            (PERFIL=completo para o perfil completo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.avaliacao import carrega_previsoes, interseccao  # noqa: E402
from src.config import caminho, carrega_config  # noqa: E402
from src.metrics import quantis  # noqa: E402
from src.utils_io import RegistroEtapa  # noqa: E402

QUANTIS = (0.05, 0.10, 0.20, 0.30, 0.50)       # DECISÃO: riscos aceitos declarados antes de olhar o teste
CUSTOS = (1.0, 2.0, 3.0)                       # DECISÃO: custo relativo do excesso (ver cabeçalho)
Q_PRINCIPAL = 0.20


def simula(d: pd.DataFrame, q: float) -> pd.Series:
    """Resultado da regra de overbooking de quantil q sobre as linhas de `d` (cnes-dia, mu, phi, f).

    Retorna:
        Series com n, k_medio, recuperadas, excesso, dias_com_excesso, beneficio_liquido_c{1,2,3} e
        recuperacao (recuperadas ÷ faltas totais).
    """
    k = np.zeros(len(d)) if q == 0 else quantis(d["mu"].to_numpy(), d["phi"].to_numpy(), q)
    f = d["f"].to_numpy()
    rec, exc = np.minimum(k, f), np.maximum(k - f, 0)
    out = {"n": len(d), "k_medio": float(k.mean()), "faltas": float(f.sum()), "recuperadas": float(rec.sum()),
           "excesso": float(exc.sum()), "dias_com_excesso": float((k > f).mean()),
           "recuperacao": float(rec.sum() / f.sum())}
    for c in CUSTOS:
        out[f"beneficio_liquido_c{c:g}"] = float(rec.sum() - c * exc.sum())
    return pd.Series(out)


def main() -> None:
    """Gera as tabelas de overbooking (por modelo, por CNES e curva de risco aceito × benefício)."""
    cfg = carrega_config()
    base = caminho(cfg, cfg["saidas"]["raiz"], f"{cfg['escopo']}_{cfg['perfil_ativo']}")
    reg = RegistroEtapa(f"S11_overbooking_{cfg['perfil_ativo']}", caminho(cfg, cfg["saidas"]["logs"]), {"perfil": cfg["perfil_ativo"]})
    df = carrega_previsoes(base)
    df = df[(df["variante"] == "expansiva") & (df["modo"] == "conhecida")]
    linhas, por_cnes, curva = [], [], []
    for L, dL in df.groupby("L"):
        comum, perdas = interseccao(dL)                     # mesmas linhas para todos os modelos (regra do CLAUDE.md)
        print(f"[S11] L={L}: {comum['modelo'].nunique()} modelos, {len(comum) // max(comum['modelo'].nunique(), 1)} CNES-dias comuns")
        for modelo, dm in comum.groupby("modelo"):
            for q in (0.0,) + QUANTIS:
                if q == 0 and modelo != sorted(comum["modelo"].unique())[0]:
                    continue                                # regra sem previsão: uma linha só por L
                r = simula(dm, q)
                r["L"], r["modelo"], r["q"] = int(L), ("sem_previsao" if q == 0 else modelo), q
                curva.append(r)
                if q == Q_PRINCIPAL:
                    for c, dc in dm.groupby("cnes"):
                        rc = simula(dc, q)
                        rc["L"], rc["modelo"], rc["cnes"] = int(L), modelo, c
                        por_cnes.append(rc)
    saida = caminho(cfg, cfg["saidas"]["tabelas"])
    sufixo = f"{cfg['escopo']}_{cfg['perfil_ativo']}"
    curva = pd.DataFrame(curva)
    curva.to_csv(saida / f"overbooking_{sufixo}_curva.csv", index=False)
    pd.DataFrame(por_cnes).to_csv(saida / f"overbooking_{sufixo}_por_cnes.csv", index=False)
    principal = pd.concat([curva[curva["q"] == Q_PRINCIPAL], curva[curva["q"] == 0]]).sort_values(["L", "beneficio_liquido_c2"], ascending=[True, False])
    principal.to_csv(saida / f"overbooking_{sufixo}.csv", index=False)
    print(principal[["L", "modelo", "k_medio", "recuperacao", "dias_com_excesso", "beneficio_liquido_c2"]].round(3).to_string(index=False))
    reg.finaliza([base], [saida / f"overbooking_{sufixo}.csv"], {"q_principal": Q_PRINCIPAL})


if __name__ == "__main__":
    main()
