"""e05 — Avaliação no teste: mesmas linhas para todos, métricas, Diebold-Mariano e bootstrap.

Objetivo: para cada (ano de teste, h), restringir à interseção de linhas com previsão válida de
todos os modelos, calcular WAPE, viés, CRPS, cobertura e largura dos intervalos (binomial
negativa comum), testar cada modelo contra a referência (DM + Holm) e estimar o IC95% do ganho
de WAPE por bootstrap de blocos de semanas. Também por CNES e sem o Materno Infantil.
Dois níveis (D16): "serie" (CNES × especialidade-mãe × semana, onde os modelos são ajustados) e
"cnes" (soma das especialidades na CNES-semana; só métricas pontuais, porque a soma de binomiais
negativas não é binomial negativa).
Entradas: previsoes.parquet, previsoes_consulta.parquet.
Saídas: metricas.csv, metricas_por_cnes.csv, dm.csv, dm_pares.csv (entre os modelos melhores que a referência), linhas_perdidas.csv, metricas_consulta.csv.
Uso: python3 etapas/e05_avaliar.py --config config.yaml --perfil smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.metricas import bootstrap_ganho_wape, bootstrap_metricas, diebold_mariano, holm, resumo_metricas, vies, wape  # noqa: E402

REF = "referencia_taxa_agenda"   # D15; referencia_mm4 fica só descritiva
CHAVE = ["serie", "semana"]


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e05_avaliar")
    av = cfg["avaliacao"]
    prev = pd.read_parquet(saida / "previsoes.parquet")
    # DECISÃO: nível de confiança 95% (alfa = 0,05): diferença significativa se p_holm < 0,05 e
    # o IC95% do ganho não contém 0. Margem de erro = metade da largura do IC95%.
    conf = av.get("nivel_confianca", 0.95)
    alfa = 1 - conf
    boot = (av["bootstrap_reamostragens"], av["bootstrap_bloco_semanas"], cfg["tuning"]["semente"], conf)

    metricas, por_cnes, dms, perdidas, dms_pares, dms_mm4 = [], [], [], [], [], []
    for (ano, h, estrato), g in prev.groupby(["teste_ano", "h", "estrato"]):
        larga = g.pivot_table(index=CHAVE, columns="modelo", values="mu", aggfunc="first")
        y = g.groupby(CHAVE)["y"].first()
        cnes_de = g.groupby("serie")["cnesexecutante"].first()
        # DECISÃO (D17): cada estrato de CNES ("consistentes", "outros") é avaliado à parte, na
        # interseção das linhas com previsão de todos os modelos ajustados naquele estrato.
        conjuntos = {estrato: [m for m in larga.columns if larga[m].notna().any()]}
        for conjunto, cols in conjuntos.items():
            larga_c = larga[cols]
            validas = larga_c.notna()
            # Mesmas linhas para todos: interseção. Registrar quantas cada modelo perderia.
            inter = validas.all(axis=1)
            for mod in larga_c.columns:
                perdidas.append({"teste_ano": ano, "h": h, "estrato": conjunto, "modelo": mod, "linhas_uniao": len(larga_c),
                                 "validas_modelo": int(validas[mod].sum()), "perdidas_modelo": int((~validas[mod]).sum()),
                                 "na_intersecao": int(inter.sum())})
            L, yy = larga_c[inter], y[inter]
            log.info(f"Teste {ano}, h={h}, {conjunto}: {inter.sum():,} de {len(larga_c):,} série-semanas na interseção")
            phis = g.groupby("modelo")["phi"].first()
            blocos = g.groupby("modelo")["bloco"].first()
            recortes = {"todas": slice(None)}
            cnes_linha = cnes_de.reindex(L.index.get_level_values("serie")).to_numpy()
            fora = [c for c in av["cnes_relatar_a_parte"] if c in set(cnes_linha)]
            if fora:
                recortes["sem_materno_infantil"] = ~np.isin(cnes_linha, fora)
            for nome_rec, sel in recortes.items():
                Lr, yr = L[sel], yy[sel]
                for mod in L.columns:
                    r = resumo_metricas(yr.to_numpy(float), Lr[mod].to_numpy(float), float(phis[mod]))
                    r["mae"] = float(np.abs(yr.to_numpy(float) - Lr[mod].to_numpy(float)).mean())
                    # IC95% e margem de erro (bootstrap em blocos de semanas).
                    tb = pd.DataFrame({"semana": Lr.index.get_level_values("semana"), "y": yr.to_numpy(float), mod: Lr[mod].to_numpy(float)})
                    r.update(bootstrap_metricas(tb, mod, *boot))
                    metricas.append({"teste_ano": ano, "h": h, "estrato": conjunto, "nivel": "serie", "recorte": nome_rec, "modelo": mod, "bloco": blocos[mod], **r})
                # Nível CNES-semana: soma das especialidades (mesmas linhas da interseção).
                agg = Lr.assign(y=yr.to_numpy(), cnes=cnes_linha[sel] if not isinstance(sel, slice) else cnes_linha)
                agg = agg.reset_index().groupby(["cnes", "semana"])[list(L.columns) + ["y"]].sum()
                for mod in L.columns:
                    ya, ma = agg["y"].to_numpy(float), agg[mod].to_numpy(float)
                    tb = pd.DataFrame({"semana": agg.index.get_level_values("semana"), "y": ya, mod: ma})
                    metricas.append({"teste_ano": ano, "h": h, "estrato": conjunto, "nivel": "cnes", "recorte": nome_rec, "modelo": mod, "bloco": blocos[mod],
                                     "n": len(ya), "wape": wape(ya, ma), "vies": vies(ya, ma), "mae": float(np.abs(ya - ma).mean()),
                                     **bootstrap_metricas(tb, mod, *boot)})
            for cnes in pd.unique(cnes_linha):
                sel = cnes_linha == cnes
                for mod in L.columns:
                    r = resumo_metricas(yy[sel].to_numpy(float), L.loc[sel, mod].to_numpy(float), float(phis[mod]))
                    por_cnes.append({"teste_ano": ano, "h": h, "estrato": conjunto, "cnesexecutante": cnes, "modelo": mod, **r})

            # Diebold-Mariano contra a referência, com perda absoluta agregada por semana.
            tab = L.reset_index().assign(y=yy.to_numpy())
            err = tab.set_index("semana")[L.columns].sub(tab.set_index("semana")["y"], axis=0).abs().groupby(level=0).sum()
            linhas_dm = []
            for mod in L.columns:
                if mod in (REF, "referencia_mm4"):
                    continue
                est, p = diebold_mariano(err[mod].to_numpy(), err[REF].to_numpy(), h)
                lo, hi = bootstrap_ganho_wape(tab, mod, REF, av["bootstrap_reamostragens"], av["bootstrap_bloco_semanas"], cfg["tuning"]["semente"])
                ganho = 1 - (tab[mod] - tab["y"]).abs().sum() / (tab[REF] - tab["y"]).abs().sum()
                linhas_dm.append({"teste_ano": ano, "h": h, "estrato": conjunto, "modelo": mod, "bloco": blocos[mod], "ganho_wape": ganho,
                                  "ic95_inf": lo, "ic95_sup": hi, "dm": est, "p": p})
            d = pd.DataFrame(linhas_dm).assign(referencia=REF)
            # D20: comparação secundária, declarada (não escolhida pelo teste), contra a média
            # móvel de 4 semanas — no treino 2022–23 a referência taxa × agenda foi melhor que mm4
            # em todos os estratos, por isso continua sendo a principal.
            sec = []
            for mod in L.columns:
                if mod in (REF, "referencia_mm4"):
                    continue
                est, p = diebold_mariano(err[mod].to_numpy(), err["referencia_mm4"].to_numpy(), h)
                lo, hi = bootstrap_ganho_wape(tab, mod, "referencia_mm4", *boot[:3])
                ganho = 1 - (tab[mod] - tab["y"]).abs().sum() / (tab["referencia_mm4"] - tab["y"]).abs().sum()
                sec.append({"teste_ano": ano, "h": h, "estrato": conjunto, "modelo": mod, "bloco": blocos[mod], "ganho_wape": ganho,
                            "ic95_inf": lo, "ic95_sup": hi, "dm": est, "p": p})
            ds = pd.DataFrame(sec).assign(referencia="referencia_mm4")
            ds["p_holm"] = holm(ds["p"])
            ds["n_comparacoes"] = ds["p"].notna().sum()
            ds["moe_ganho"] = (ds["ic95_sup"] - ds["ic95_inf"]) / 2
            ds["significativo"] = (ds["p_holm"] < alfa) & ((ds["ic95_inf"] > 0) | (ds["ic95_sup"] < 0))
            dms_mm4.append(ds)
            d["p_holm"] = holm(d["p"])
            d["n_comparacoes"] = d["p"].notna().sum()
            d["moe_ganho"] = (d["ic95_sup"] - d["ic95_inf"]) / 2
            d["significativo"] = (d["p_holm"] < alfa) & ((d["ic95_inf"] > 0) | (d["ic95_sup"] < 0))
            dms.append(d)

            # D18: comparações par a par entre os modelos com WAPE menor que a referência
            # (DM bilateral + IC95% bootstrap do ganho do 1º sobre o 2º; Holm entre os pares).
            melhores = d.loc[d["ganho_wape"] > 0].sort_values("ganho_wape", ascending=False)["modelo"].tolist()
            pares = []
            for i, a_ in enumerate(melhores):
                for b_ in melhores[i + 1:]:
                    est, p = diebold_mariano(err[a_].to_numpy(), err[b_].to_numpy(), h)
                    lo, hi = bootstrap_ganho_wape(tab, a_, b_, av["bootstrap_reamostragens"], av["bootstrap_bloco_semanas"], cfg["tuning"]["semente"])
                    ganho = 1 - (tab[a_] - tab["y"]).abs().sum() / (tab[b_] - tab["y"]).abs().sum()
                    pares.append({"teste_ano": ano, "h": h, "estrato": conjunto, "modelo_a": a_, "modelo_b": b_,
                                  "ganho_wape_a_sobre_b": ganho, "ic95_inf": lo, "ic95_sup": hi, "dm": est, "p": p})
            if pares:
                dp = pd.DataFrame(pares)
                dp["p_holm"] = holm(dp["p"])
                dp["n_comparacoes"] = len(dp)
                dp["moe_ganho"] = (dp["ic95_sup"] - dp["ic95_inf"]) / 2
                dp["significativo"] = (dp["p_holm"] < alfa) & ((dp["ic95_inf"] > 0) | (dp["ic95_sup"] < 0))
                dms_pares.append(dp)

    pd.DataFrame(metricas).to_csv(saida / "metricas.csv", index=False)
    pd.DataFrame(por_cnes).to_csv(saida / "metricas_por_cnes.csv", index=False)
    pd.concat(dms).to_csv(saida / "dm.csv", index=False)
    pd.concat(dms_mm4).to_csv(saida / "dm_vs_mm4.csv", index=False)
    if dms_pares:
        pd.concat(dms_pares).to_csv(saida / "dm_pares.csv", index=False)
    pd.DataFrame(perdidas).to_csv(saida / "linhas_perdidas.csv", index=False)

    # Tabela principal: WAPE ± margem de erro [IC95%] e ganho sobre a referência ± margem.
    mt = pd.DataFrame(metricas)
    mt = mt[mt["recorte"] == "todas"][["teste_ano", "h", "estrato", "nivel", "modelo", "wape", "wape_ic_inf", "wape_ic_sup", "wape_moe", "vies", "mae"]]
    dd = pd.concat(dms)[["teste_ano", "h", "estrato", "modelo", "ganho_wape", "ic95_inf", "ic95_sup", "moe_ganho", "p", "p_holm", "significativo"]]
    tp = mt.merge(dd, on=["teste_ano", "h", "estrato", "modelo"], how="left").sort_values(["h", "estrato", "nivel", "wape"])
    tp.to_csv(saida / "tabela_principal.csv", index=False)

    # Métricas por consulta (modelo por consulta, consultas conhecidas na origem).
    pc = pd.read_parquet(saida / "previsoes_consulta.parquet")
    mc = []
    for (ano, h), g in pc.groupby(["teste_ano", "h"]):
        yv, pv = g["y"].to_numpy(), np.clip(g["p"].to_numpy(), 1e-6, 1 - 1e-6)
        auc = roc_auc_score(yv, pv) if 0 < yv.mean() < 1 else np.nan
        mc.append({"teste_ano": ano, "h": h, "n": len(g), "brier": float(((pv - yv) ** 2).mean()), "auc": auc,
                   "logloss": log_loss(yv, pv, labels=[0, 1])})
    pd.DataFrame(mc).to_csv(saida / "metricas_consulta.csv", index=False)
    log.manifesto([saida / "previsoes.parquet"], [saida / "metricas.csv", saida / "dm.csv"])


if __name__ == "__main__":
    main()
