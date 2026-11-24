"""e04 — Tuning (só no treino) e previsão do teste para todos os modelos.

Objetivo: para cada (ano de teste, horizonte h), gerar as dobras de validação com o esquema de
esquema.json, escolher a melhor configuração de cada modelo (8 por modelo, mesmo orçamento),
estimar a dispersão φ fora da amostra e prever as semanas de teste com origem móvel.
Modelos (D15: todos recebem a agenda conhecida na origem; não há mais blocos A/B):
- referências: referencia_taxa_agenda (taxa de falta suavizada das 4 semanas encerradas × agenda
  conhecida; referência dos testes) e referencia_mm4 (média móvel de faltas, só descritiva);
- locais: ets e croston_tsb (sobre a taxa por consulta agendada, convertida de volta em contagem),
  sarimax (agenda e agregados como regressores);
- globais: glm_poisson, hgb_poisson, floresta, mlp (alvo = taxa, peso = agenda);
- consulta_hgb (por consulta, somado por CNES-semana).
Entradas: painel_h{h}.parquet, base_consulta.parquet, caracterizacao_series.csv, esquema.json.
Saídas: previsoes.parquet, tuning.csv, diagnostico_residuos.csv, previsoes_consulta.parquet.
Uso: python3 etapas/e04_modelar.py --config config.yaml --perfil smoke --cnes 2237385
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from faltas import modelos_ml as ml  # noqa: E402
from faltas import modelos_series as ms  # noqa: E402
from faltas.config import Registro, argumentos, carregar_config, pasta_saida  # noqa: E402
from faltas.divisoes import gerar_dobras  # noqa: E402
from faltas.metricas import phi_momentos, wape  # noqa: E402
from faltas.painel import AGREGADOS_B, te_temporal  # noqa: E402

LOCAIS = ["ets", "croston_tsb", "sarimax"]
REF = "referencia_taxa_agenda"
GLOBAIS = ["glm_poisson", "hgb_poisson", "floresta", "mlp"]
CHAVE = ["serie", "semana"]   # série = CNES × especialidade-mãe (D16)


def exogenas_locais(s: pd.DataFrame, nome: str, n_ajuste: int) -> np.ndarray | None:
    """Exógenas dos modelos locais: no sarimax, Fourier, calendário, agenda e agregados.
    Ausentes (semana sem agenda) são preenchidos com a média do período de ajuste."""
    if nome in ("ets", "croston_tsb"):
        return None
    cols = [c for c in s.columns if c.startswith(("sen", "cos"))]
    X = s[cols + ["n_feriados", "ponte", "recesso"]].astype(float).copy()
    if nome == "sarimax":
        X["log_agenda"] = np.log1p(s["n_agenda"])
        for c in AGREGADOS_B:
            X[c] = s[c].fillna(s[c].iloc[:n_ajuste].mean()).fillna(0)
    return X.to_numpy(float)


def prever_local_taxa(nome, y, n_ag, X, n_ajuste, alvos, h, p, m):
    """ETS e Croston/TSB não aceitam exógenas: modelam a taxa suavizada por consulta agendada
    r_t = (y_t + m·r̄) / (n_agenda_t + m) e devolvem a contagem y = r·(n_agenda + m) − m·r̄
    (inversa exata), usando a agenda conhecida da semana-alvo.
    r̄ = Σy / Σn_agenda nas n_ajuste semanas iniciais (só passado do ajuste).
    ANTI-VAZAMENTO: r̄ usa só o período de ajuste; n_agenda do alvo é conhecida na origem.
    Retorna: (np.ndarray de contagens previstas, resultado do ajuste ou None)."""
    den = n_ag[:n_ajuste].sum()
    rbar = y[:n_ajuste].sum() / den if den > 0 else 0.0
    r = (y + m * rbar) / (n_ag + m)
    mu_r, res = ms.prever_local(nome, r, X, n_ajuste, alvos, h, p)
    mu = np.clip(mu_r * (n_ag[alvos] + m) - m * rbar, 0, None)
    return mu, res


def _prever(nome, y, n_ag, X, n_ajuste, alvos, h, p, m):
    if nome in ("ets", "croston_tsb"):
        return prever_local_taxa(nome, y, n_ag, X, n_ajuste, alvos, h, p, m)
    return ms.prever_local(nome, y, X, n_ajuste, alvos, h, p)


def _locais_uma_cnes(cnes, s, dobras, semanas_teste, h, presel, m):
    """Tuning e previsão dos modelos locais de UMA CNES (roda em paralelo entre CNES).
    presel: dict {nome: grade pré-selecionada por AICc} já calculado (cache entre horizontes) ou {}.
    Retorna: (prev, tun, diag, presel_usada)."""
    import warnings
    warnings.filterwarnings("ignore")
    prev, tun, diag, usada = [], [], [], {}
    s = s.sort_values("semana").reset_index(drop=True)
    pos = {d: i for i, d in enumerate(s["semana"])}
    y = s["y"].to_numpy(float)
    n_ag = s["n_agenda"].to_numpy(float)
    aval = s["avaliavel"].to_numpy()
    alvos_teste = [pos[d] for d in semanas_teste if d in pos and aval[pos[d]]]
    n_treino = pos[semanas_teste[0]] if semanas_teste[0] in pos else len(s)
    for nome in LOCAIS:
        bloco = "agenda"
        X = exogenas_locais(s, nome, n_treino)
        if nome == "sarimax":
            # AICc só no treino; não depende de h, por isso fica em cache.
            grade = presel.get(nome) or ms.preselecionar_sarima(y, X, n_treino)
            usada[nome] = grade
        elif nome == "ets":
            grade = ms.grade_ets()
        else:
            grade = ms.grade_croston()
        melhor, melhor_w, melhor_val = None, np.inf, None
        for p in grade:
            ys, ms_ = [], []
            for d in dobras:
                na = pos.get(d.inicio_validacao)
                alvos = [pos[w] for w in d.validacao if w in pos and aval[pos[w]]]
                if na is None or na < 60 or not alvos:
                    continue
                mu, _ = _prever(nome, y, n_ag, X, na, alvos, h, p, m)
                ys.append(y[alvos]); ms_.append(mu)
            if not ys:
                continue
            yy, mm = np.concatenate(ys), np.concatenate(ms_)
            w = wape(yy[np.isfinite(mm)], mm[np.isfinite(mm)]) if np.isfinite(mm).mean() > 0.9 else np.inf
            tun.append({"modelo": nome, "bloco": bloco, "serie": cnes, "config": json.dumps(p, default=str), "wape_validacao": w})
            if w < melhor_w:
                melhor, melhor_w, melhor_val = p, w, (yy, mm)
        if melhor is None:
            continue
        mu, res = _prever(nome, y, n_ag, X, n_treino, alvos_teste, h, melhor, m)
        prev.append(pd.DataFrame({"serie": cnes, "semana": s["semana"].iloc[alvos_teste].to_numpy(),
                                  "modelo": nome, "bloco": bloco, "mu": mu, "_yv": np.nan}))
        prev.append(pd.DataFrame({"serie": cnes, "semana": pd.NaT, "modelo": nome, "bloco": bloco,
                                  "mu": melhor_val[1], "_yv": melhor_val[0]}))  # validação, para φ
        diag.append({"modelo": nome, "serie": cnes, "config": json.dumps(melhor, default=str),
                     "wape_validacao": melhor_w, "ljung_box_p": ms.ljung_box(res) if res is not None else np.nan})
    return prev, tun, diag, usada


def rodar_locais(painel, car, dobras, semanas_teste, h, cfg, log, cache_presel):
    """Modelos locais em paralelo entre CNES (joblib, n_jobs de config.tuning.n_jobs).
    cache_presel: dict {(cnes, fim_treino): {nome: grade}} reaproveitado entre horizontes."""
    from joblib import Parallel, delayed
    elegiveis = set(car.index[car["modelo_local"]])
    grupos = []
    for cnes, s in painel.groupby("serie"):
        if cnes not in elegiveis:
            log.info(f"  {cnes}: sem dados suficientes para modelos locais (linhas perdidas)")
            continue
        grupos.append((cnes, s))
    chave_fim = str(semanas_teste[0].date())
    res = Parallel(n_jobs=cfg["tuning"].get("n_jobs", -1))(
        delayed(_locais_uma_cnes)(cnes, s, dobras, semanas_teste, h, cache_presel.get((cnes, chave_fim), {}),
                                  cfg["variaveis"]["taxa_suavizacao_m"])
        for cnes, s in grupos)
    prev, tun, diag = [], [], []
    for (cnes, _), (p_, t_, d_, usada) in zip(grupos, res):
        prev += p_; tun += t_; diag += d_
        cache_presel[(cnes, chave_fim)] = usada
    return prev, tun, diag


def completar_com_fallback(pl: list[pd.DataFrame], painel: pd.DataFrame, em_teste: pd.Series, log) -> list[pd.DataFrame]:
    """D18: nas séries sem modelo local (curtas demais) ou com previsão falha, o modelo local
    devolve a previsão da referência taxa × agenda (fallback), marcada em `fallback` = True.
    Só as linhas de teste são completadas; as de validação (para φ) ficam como estão.
    Retorna: lista de DataFrames de previsão (validação + teste completado)."""
    if not pl:
        return pl
    todas = pd.concat(pl, ignore_index=True)
    val = todas[todas["semana"].isna()]
    teste = todas[todas["semana"].notna()]
    alvo = painel.loc[em_teste, CHAVE + ["ref_taxa_agenda"]]
    saida = [val]
    for nome in LOCAIS:
        t = alvo.merge(teste[teste["modelo"] == nome][CHAVE + ["mu"]], on=CHAVE, how="left")
        t["fallback"] = t["mu"].isna()
        t["mu"] = t["mu"].where(~t["fallback"], t["ref_taxa_agenda"])
        log.info(f"  {nome}: fallback para a referência em {int(t['fallback'].sum())} de {len(t)} linhas de teste")
        saida.append(t.drop(columns="ref_taxa_agenda").assign(modelo=nome, bloco="agenda", _yv=np.nan))
    return saida


def linhas_validas(p: pd.DataFrame) -> pd.Series:
    """Linhas usadas no ajuste e na avaliação: semana avaliável e médias/taxas defasadas disponíveis."""
    return p["avaliavel"] & p["mm4"].notna() & p["taxa_ag4"].notna()


def rodar_globais(painel, dobras, semanas_teste, cfg, semente):
    """Tuning global (8 configurações) por WAPE agregado nas dobras e previsão do teste."""
    prev, tun = [], []
    ok = linhas_validas(painel)
    bloco = "agenda"
    X = ml.preparar_X(painel)
    e = ml.exposicao(painel)
    y = painel["y"].to_numpy(float)
    for nome in GLOBAIS:
        melhor, melhor_w, melhor_val = None, np.inf, None
        for p in ml.configuracoes(nome, cfg["tuning"]["n_configuracoes"], semente):
            ys, mus = [], []
            for d in dobras:
                ia = (ok & painel["semana"].isin(d.ajuste)).to_numpy()
                iv = (ok & painel["semana"].isin(d.validacao)).to_numpy()
                if ia.sum() < 30 or iv.sum() == 0:
                    continue
                mus.append(ml.ajustar_prever_global(nome, p, X[ia], y[ia], e[ia], X[iv], e[iv], semente))
                ys.append(y[iv])
            yy, mm = np.concatenate(ys), np.concatenate(mus)
            w = wape(yy, mm)
            tun.append({"modelo": nome, "bloco": bloco, "serie": "todas", "config": json.dumps(p, default=str), "wape_validacao": w})
            if w < melhor_w:
                melhor, melhor_w, melhor_val = p, w, (yy, mm)
        treino = (ok & (painel["semana"] < semanas_teste[0])).to_numpy()
        teste = (ok & painel["semana"].isin(semanas_teste)).to_numpy()
        mu = ml.ajustar_prever_global(nome, melhor, X[treino], y[treino], e[treino], X[teste], e[teste], semente)
        prev.append(painel.loc[teste, CHAVE].assign(modelo=nome, bloco=bloco, mu=mu, _yv=np.nan))
        prev.append(pd.DataFrame({"serie": "todas", "semana": pd.NaT, "modelo": nome, "bloco": bloco,
                                  "mu": melhor_val[1], "_yv": melhor_val[0]}))
        tun.append({"modelo": nome, "bloco": bloco, "serie": "todas", "config": json.dumps(melhor, default=str),
                    "wape_validacao": melhor_w, "escolhida": True})
    return prev, tun


def _com_te(base: pd.DataFrame, h: int, congela: pd.Timestamp, m: float, cols: list[str]) -> pd.DataFrame:
    b = base.copy()
    for c in cols:
        b[f"te_{c}"] = te_temporal(b, c, h, congela, m).to_numpy()
    return b


def rodar_consulta(base, painel, dobras, semanas_teste, h, cfg, semente):
    """Modelo por consulta: tuning por Brier nas dobras; fator k = Σy/Σp nas dobras (fora da
    amostra) para compensar a agenda ainda não conhecida na origem; previsão do teste."""
    m, cols = cfg["variaveis"]["te_suavizacao_m"], cfg["variaveis"]["te_colunas"]
    col_conh = f"conhecida_h{h}"
    ok = linhas_validas(painel)
    tun, melhor, melhor_b, melhor_val = [], None, np.inf, None
    for p in ml.configuracoes("consulta_hgb", cfg["tuning"]["n_configuracoes"], semente):
        briers, agreg = [], []
        for d in dobras:
            # ANTI-VAZAMENTO: TE congelado na véspera da validação.
            b = _com_te(base, h, d.inicio_validacao - pd.Timedelta(days=1), m, cols)
            ia = b["semana"].isin(d.ajuste)
            iv = b["semana"].isin(d.validacao) & b[col_conh]
            if ia.sum() < 200 or iv.sum() == 0:
                continue
            pr = ml.ajustar_prever_consulta(p, b[ia], b[iv], semente)
            briers.append(((pr - b.loc[iv, "y"].to_numpy()) ** 2))
            soma = b.loc[iv, CHAVE].assign(p=pr).groupby(CHAVE)["p"].sum()
            pv = painel[ok & painel["semana"].isin(d.validacao)].set_index(CHAVE)
            agreg.append(pd.DataFrame({"y": pv["y"], "p": soma.reindex(pv.index).fillna(0)}))
        brier = float(np.concatenate(briers).mean())
        tun.append({"modelo": "consulta_hgb", "bloco": "agenda", "serie": "todas", "config": json.dumps(p, default=str), "brier_validacao": brier})
        if brier < melhor_b:
            melhor, melhor_b, melhor_val = p, brier, pd.concat(agreg)
    k = melhor_val["y"].sum() / max(melhor_val["p"].sum(), 1e-9)
    fim_treino = semanas_teste[0] - pd.Timedelta(days=1)
    b = _com_te(base, h, fim_treino, m, cols)
    ia = b["semana"] < semanas_teste[0]
    ip = b["semana"].isin(semanas_teste) & b[col_conh]
    pr = ml.ajustar_prever_consulta(melhor, b[ia], b[ip], semente)
    cons_prev = b.loc[ip, ["rid", "serie", "cnesexecutante", "semana", "especialidademae", "y"]].assign(p=pr, k=k)
    soma = cons_prev.groupby(CHAVE)["p"].sum() * k
    pt = painel[ok & painel["semana"].isin(semanas_teste)].set_index(CHAVE)
    prev = [pd.DataFrame({"mu": soma.reindex(pt.index).fillna(0)}).reset_index().assign(modelo="consulta_hgb", bloco="agenda", _yv=np.nan),
            pd.DataFrame({"serie": "todas", "semana": pd.NaT, "modelo": "consulta_hgb", "bloco": "agenda",
                          "mu": melhor_val["p"].to_numpy() * k, "_yv": melhor_val["y"].to_numpy()})]
    tun.append({"modelo": "consulta_hgb", "bloco": "agenda", "serie": "todas", "config": json.dumps(melhor, default=str),
                "brier_validacao": melhor_b, "fator_k": k, "escolhida": True})
    return prev, tun, cons_prev


def main() -> None:
    args = argumentos(__doc__.splitlines()[0])
    cfg = carregar_config(args.config)
    saida = pasta_saida(cfg, args.perfil)
    log = Registro(saida, "e04_modelar")
    semente = cfg["tuning"]["semente"]
    np.random.seed(semente)
    esquema = json.load(open(saida / "esquema.json", encoding="utf-8"))["esquema"]
    car = pd.read_csv(saida / "caracterizacao_series.csv", dtype={"serie": str}).set_index("serie")
    base_all = pd.read_parquet(saida / "base_consulta.parquet")
    estr = pd.read_csv(saida / "estratos_cnes.csv", dtype={"cnesexecutante": str})
    log.info(f"Esquema de validação (da exploratória): {esquema}; semente {semente}")

    todas_prev, todos_tun, todos_diag, todas_cons = [], [], [], []
    cache_presel: dict = {}
    for teste in cfg["avaliacao"]["testes"]:
        for h in cfg["avaliacao"]["horizontes"]:
            t0 = time.time()
            painel = pd.read_parquet(saida / f"painel_h{h}.parquet")
            base = base_all
            if args.cnes:
                painel = painel[painel["cnesexecutante"].isin(args.cnes)].reset_index(drop=True)
                base = base_all[base_all["cnesexecutante"].isin(args.cnes)].reset_index(drop=True)
            fim = pd.Timestamp(teste["fim_treino"])
            sem_treino = pd.DatetimeIndex(painel.loc[painel["semana"] <= fim, "semana"].unique())
            sem_teste = pd.DatetimeIndex(sorted(painel.loc[painel["semana"].dt.year == teste["ano"], "semana"].unique()))
            # ANTI-VAZAMENTO: o teste começa depois do fim do treino.
            assert sem_teste.min() > fim, "Teste começa antes do fim do treino"
            dobras = gerar_dobras(sem_treino, esquema, cfg["tuning"], semente)
            log.info(f"Teste {teste['ano']}, h={h}: {len(sem_treino)} semanas de treino, {len(sem_teste)} de teste, "
                     f"{len(dobras)} dobras ({[str(d.inicio_validacao.date()) for d in dobras]})")

            # D17: cada estrato de CNES é modelado à parte (tuning e ajuste próprios).
            painel_tot, base_tot = painel, base
            for estrato in ["consistentes", "outros"]:
                cnes_e = set(estr.loc[estr["estrato"] == estrato, "cnesexecutante"])
                painel = painel_tot[painel_tot["cnesexecutante"].isin(cnes_e)].reset_index(drop=True)
                base = base_tot[base_tot["cnesexecutante"].isin(cnes_e)].reset_index(drop=True)
                if painel.empty:
                    log.info(f"  [{estrato}] nenhuma CNES neste estrato")
                    continue
                log.info(f"  [{estrato}] {len(cnes_e & set(painel['cnesexecutante']))} CNES, {painel['serie'].nunique()} séries")
                ok = linhas_validas(painel)
                # Referências: colunas já calculadas no painel só com semanas encerradas na origem.
                em_teste = ok & painel["semana"].isin(sem_teste)
                val_ref = painel[ok & painel["semana"].isin(np.concatenate([d.validacao for d in dobras]))]
                partes = []
                for nome_ref, col in [(REF, "ref_taxa_agenda"), ("referencia_mm4", "mm4")]:
                    partes.append(painel.loc[em_teste, CHAVE].assign(modelo=nome_ref, bloco="ref",
                                                                     mu=painel.loc[em_teste, col].to_numpy(), _yv=np.nan))
                    partes.append(pd.DataFrame({"serie": "todas", "semana": pd.NaT, "modelo": nome_ref, "bloco": "ref",
                                                "mu": val_ref[col].to_numpy(), "_yv": val_ref["y"].to_numpy()}))
                pl, tl, dl = (rodar_locais(painel, car, dobras, sem_teste, h, cfg, log, cache_presel)
                                  if estrato == "consistentes" else ([], [], []))  # D17: locais só nas consistentes
                if estrato == "consistentes":
                    pl = completar_com_fallback(pl, painel, em_teste, log)
                log.info(f"  modelos locais: {time.time() - t0:.0f} s")
                pg, tg = rodar_globais(painel, dobras, sem_teste, cfg, semente)
                log.info(f"  modelos globais: {time.time() - t0:.0f} s")
                pc, tc, cons = rodar_consulta(base, painel, dobras, sem_teste, h, cfg, semente)
                prev = pd.concat(partes + pl + pg + pc, ignore_index=True)

                # φ por modelo, estimado nas previsões de validação (fora da amostra).
                val = prev[prev["semana"].isna()]
                phis = val.groupby("modelo").apply(lambda g: phi_momentos(g["_yv"].to_numpy(), g["mu"].to_numpy()), include_groups=False)
                prev = prev[prev["semana"].notna()].drop(columns="_yv")
                prev["phi"] = prev["modelo"].map(phis)
                prev = prev.merge(painel[CHAVE + ["cnesexecutante", "especialidademae", "y"]], on=CHAVE, how="left", validate="m:1")
                prev["teste_ano"], prev["h"], prev["estrato"] = teste["ano"], h, estrato
                todas_prev.append(prev)
                for t in tl + tg + tc:
                    t.update(teste_ano=teste["ano"], h=h, estrato=estrato)
                todos_tun += tl + tg + tc
                todos_diag += [dict(d, teste_ano=teste["ano"], h=h, estrato=estrato) for d in dl]
                todas_cons.append(cons.assign(teste_ano=teste["ano"], h=h, estrato=estrato))
                log.info(f"  [{estrato}] {prev['modelo'].nunique()} modelos, {len(prev):,} previsões, {time.time() - t0:.0f} s; "
                         f"φ: {', '.join(f'{k}={v:.3f}' for k, v in phis.items())}")

    previsoes = pd.concat(todas_prev, ignore_index=True)
    assert previsoes["mu"].notna().mean() > 0, "Nenhuma previsão válida"
    previsoes.to_parquet(saida / "previsoes.parquet", index=False)
    pd.DataFrame(todos_tun).to_csv(saida / "tuning.csv", index=False)
    pd.DataFrame(todos_diag).to_csv(saida / "diagnostico_residuos.csv", index=False)
    pd.concat(todas_cons).to_parquet(saida / "previsoes_consulta.parquet", index=False)
    log.manifesto([saida / f"painel_h{h}.parquet" for h in cfg["avaliacao"]["horizontes"]] + [saida / "base_consulta.parquet"],
                  [saida / "previsoes.parquet", saida / "tuning.csv"], {"esquema": esquema, "semente": semente, "cnes": args.cnes})


if __name__ == "__main__":
    main()
