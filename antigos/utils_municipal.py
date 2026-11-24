"""Funcoes compartilhadas dos scripts 21-24 (recorte municipal). Nao executa nada ao ser importado."""
import numpy as np
import pandas as pd

SRC = "gercon_poa_censo_distancias_com_natureza_juridica.parquet"
NUM = ["idade", "n_consultas_previas", "n_faltas_previas", "taxa_falta_prev", "primeira_vez",
       "n_consultas_previas_especialidademae", "n_faltas_previas_especialidademae",
       "n_consultas_previas_motivo", "n_faltas_previas_motivo", "dias_ate_consulta",
       "favela_comunidade", "taxa_alf_geral", "renda_media_winsor", "distancia_km", "tempo_caminhando",
       "distancia_km_trans_publico", "tempo_desloc_trans_publico", "quant_bus", "baldiacoes"]
CAT = ["falta_recente", "ultima_previa_mesma_especialidademae", "ultima_previa_mesmo_motivo",
       "especialidademae", "cid_1dig", "possuiunidadeindicada", "dia_semana", "mes_consulta",
       "corregulador", "coordenadoria_saude", "distrito_sanitario", "status", "cnesexecutante"]
ALL_VARS = NUM + CAT
GRUPOS = {
    "historico_paciente": ["n_consultas_previas", "n_faltas_previas", "taxa_falta_prev", "primeira_vez",
                           "n_consultas_previas_especialidademae", "n_faltas_previas_especialidademae",
                           "n_consultas_previas_motivo", "n_faltas_previas_motivo", "falta_recente",
                           "ultima_previa_mesma_especialidademae", "ultima_previa_mesmo_motivo"],
    "perfil": ["idade"],
    "territorio": ["favela_comunidade", "taxa_alf_geral", "renda_media_winsor", "coordenadoria_saude",
                   "distrito_sanitario", "corregulador"],
    "acesso": ["distancia_km", "tempo_caminhando", "distancia_km_trans_publico", "tempo_desloc_trans_publico",
               "quant_bus", "baldiacoes"],
    "agenda": ["dias_ate_consulta", "dia_semana", "mes_consulta"],
    "servico": ["especialidademae", "cid_1dig", "cnesexecutante"],
    "administrativas": ["status", "possuiunidadeindicada"],
}
assert sorted(sum(GRUPOS.values(), [])) == sorted(ALL_VARS)


def carrega_pacientes():
    """Base por consulta com y, dia, flags (municipal, coorte) e X pronto (categoricas como category)."""
    cols = list(dict.fromkeys([v for v in ALL_VARS if v != "taxa_falta_prev"] + ["situacaoagenda", "datahora_consulta", "t0", "n_faltas_previas",
                                         "n_consultas_previas"]))
    raw = pd.read_parquet(SRC, columns=cols)
    raw["y"] = (raw.situacaoagenda == "FALTANTE").astype(np.int8)
    raw["dia"] = pd.to_datetime(raw.datahora_consulta)
    raw["taxa_falta_prev"] = raw.n_faltas_previas / raw.n_consultas_previas.where(raw.n_consultas_previas > 0)
    mun = pd.read_csv("flag_municipal.csv", dtype={"cnesexecutante": str})
    nv = pd.read_csv("niveis_municipal.csv", dtype={"cnes": str})
    raw["municipal"] = raw.cnesexecutante.isin(set(mun.cnesexecutante[mun.municipal])).values
    raw["coorte"] = raw.cnesexecutante.isin(set(nv.cnes)).values
    raw["nivel"] = raw.cnesexecutante.map(dict(zip(nv.cnes, nv.nivel)))
    X = raw[ALL_VARS].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return raw, X


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def make_hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    base = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, categorical_features="from_dtype", random_state=0)
    base.update(kw)
    return HistGradientBoostingClassifier(**base)


def bootstrap_delta(d, cluster, nboot=1000, seed=0):
    """IC95% da media de d (diferenca de erro quadratico por linha) com bootstrap por cluster (ex.: CNES-mes)."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"d": d, "c": cluster}).groupby("c").d.agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    k = len(s)
    idx = rng.integers(0, k, (nboot, k))
    est = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def feature_extras(raw):
    """Candidatas novas: taxa da CNES nos 28 dias anteriores (defasagem 3) e taxa previa suavizada do paciente."""
    d = raw.groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), a=("y", "size")).reset_index()
    out = []
    for c, x in d.groupby("cnesexecutante"):
        x = x.set_index("dia").reindex(pd.date_range(d.dia.min(), d.dia.max(), freq="D")).fillna(0)
        num = x.f.shift(3).rolling(28, min_periods=14).sum()
        den = x.a.shift(3).rolling(28, min_periods=14).sum()
        out.append(pd.DataFrame({"cnesexecutante": c, "taxa_cnes_28d": num / den.where(den > 0)}).rename_axis("dia").reset_index())
    e = raw[["cnesexecutante", "dia"]].merge(pd.concat(out), on=["cnesexecutante", "dia"], how="left")
    prior = raw.n_faltas_previas.sum() / max(raw.n_consultas_previas.sum(), 1)
    return pd.DataFrame({"taxa_cnes_28d": e.taxa_cnes_28d.values,
                         "taxa_prev_suav": ((raw.n_faltas_previas + 2 * prior) / (raw.n_consultas_previas + 2)).values},
                        index=raw.index)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


# ======================= painel de series CNES-dia (coorte municipal) =======================
LAG = 3
AGENDA_COLS = ["m_taxaprev", "m_primeira", "m_idade", "m_dias_ate"]


def feriados(anos):
    from dateutil.easter import easter
    s = set()
    for y in anos:
        e = pd.Timestamp(easter(y))
        fixos = [(1, 1), (2, 2), (4, 21), (5, 1), (9, 7), (9, 20), (10, 12), (11, 2), (11, 15), (12, 25)]
        if y >= 2024:
            fixos.append((11, 20))
        for m, d in fixos:
            s.add(pd.Timestamp(y, m, d))
        for off in (-48, -47, -2, 60):
            s.add(e + pd.Timedelta(days=off))
    return s


def painel_series(raw):
    """Painel diario completo por CNES da coorte: faltas f, agenda ag, taxas defasadas (so dados <= t-3), calendario e
    agregados da agenda do dia (m_*), conhecidos com a agenda. Igual ao painel L do script 17, mais os agregados."""
    c = raw[raw.coorte].copy()
    c["tp"] = c.taxa_falta_prev.fillna(0.17)
    Bd = c.groupby(["cnesexecutante", "dia"]).agg(f=("y", "sum"), ag=("y", "size"), m_taxaprev=("tp", "mean"),
                                                   m_primeira=("primeira_vez", "mean"), m_idade=("idade", "mean"),
                                                   m_dias_ate=("dias_ate_consulta", "mean")).reset_index()
    full_idx = pd.date_range(Bd.dia.min(), Bd.dia.max(), freq="D")
    FER = feriados(range(full_idx.min().year, full_idx.max().year + 1))
    rows = []
    for c_, x in Bd.groupby("cnesexecutante"):
        x = x.set_index("dia").reindex(full_idx)
        ag_cols = x[AGENDA_COLS]
        x = x[["f", "ag"]].fillna(0)
        f, a = x.f, x.ag
        d = pd.DataFrame({"cnes": c_, "f": f, "ag": a})
        for w, mp in ((7, 3), (14, 7), (28, 14), (91, 45)):
            num = f.shift(LAG).rolling(w, min_periods=mp).sum()
            den = a.shift(LAG).rolling(w, min_periods=mp).sum()
            d[f"r{w}"] = num / den.where(den > 0)
        nd = sum(f.shift(7 * k) for k in (1, 2, 3, 4))
        dd = sum(a.shift(7 * k) for k in (1, 2, 3, 4))
        d["rdow4"] = (nd / dd.where(dd > 0)).fillna(d.r28)
        d["rd1"] = (f.shift(7) / a.shift(7).where(a.shift(7) > 0)).fillna(d.r28)
        d["y_rate"] = f / a.where(a > 0)
        for k in AGENDA_COLS:
            d[k] = ag_cols[k].values
        rows.append(d.rename_axis("dia").reset_index())
    L = pd.concat(rows, ignore_index=True)
    L["dow"] = L.dia.dt.dayofweek
    L["mes"] = L.dia.dt.month
    L["fer"] = L.dia.isin(FER).astype(int)
    L["posfer"] = (L.dia - pd.Timedelta(days=1)).isin(FER).astype(int)
    for k in ("r7", "r14", "r91"):
        L[k] = L[k].fillna(L.r28)
    for k in AGENDA_COLS:
        L[k] = L[k].fillna(L[k].mean())
    return L.sort_values(["cnes", "dia"]).reset_index(drop=True)
