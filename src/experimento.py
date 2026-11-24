"""Execução de UMA unidade de experimento: tuning aninhado + treino final + previsão do teste (S6/S7).

Unidade = (escopo, variante de esquema, L, modo de agenda, modelo, divisão de teste). Não executa nada ao
ser importado. Todas as divisões vêm de src/splits.py; coortes vêm de src/coorte.py (só dados anteriores).

Protocolo anti-vazamento
  * Hiperparâmetros (e a janela do esquema deslizante) são escolhidos SÓ nos folds internos, que terminam
    antes do início do teste externo (validação interna: treino < v, validação = ano v).
  * Dispersão phi da distribuição preditiva é estimada nas previsões dos folds internos do melhor modelo.
  * k_h (reescala da agenda conhecida) é estimado só no treino de cada ajuste.
  * O teste externo só é tocado depois da escolha, uma única vez.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import estima_dispersao, metricas_consulta
from src.models import REGISTRO
from src.models.base import amostra_configs
from src.splits import Divisao


def semente_derivada(semente: int, *partes: Any) -> int:
    """Semente determinística (SHA-256, não o hash() do Python, que varia entre execuções) para (semente, partes...)."""
    h = hashlib.sha256((str(semente) + "|" + "|".join(map(str, partes))).encode()).hexdigest()
    return int(h[:8], 16)


def _pontua_config(ctx, cls, params: dict, folds: list[Divisao], seed: int) -> tuple[float, list[float], np.ndarray, np.ndarray]:
    """Avalia uma configuração nos folds internos e devolve o critério pooled e as previsões de validação.

    Parâmetros:
        ctx: Contexto; cls: classe do modelo; params: hiperparâmetros; folds: folds internos (Divisao);
        seed: semente do modelo.

    Retorna:
        (critério pooled [menor é melhor], critério por fold, y de validação, mu de validação), com y/mu
        concatenados dos folds, só nas linhas da coorte do corte de cada fold.

    Cuidados:
        # ANTI-VAZAMENTO: cada fold treina só com linhas anteriores à sua validação (já garantido por
        # src/splits.py) e usa a coorte definida com dados anteriores ao início da validação.
    """
    num_t = den_t = 0.0
    por_fold, ys, mus = [], [], []
    f_all = ctx.P["f"].to_numpy()
    for fold in folds:
        idx_val = ctx.linhas_da_coorte(fold.teste, fold.inicio_teste)
        if len(idx_val) == 0:
            continue
        k_h = ctx.k_h(fold.treino)
        m = cls(params, seed).fit(ctx, fold.treino, k_h)
        mu = m.predict_mean(ctx, idx_val, k_h)
        num, den = m.criterio_validacao(ctx, idx_val, mu, f_all[idx_val])
        num_t, den_t = num_t + num, den_t + den
        por_fold.append(num / den if den else np.nan)
        ys.append(f_all[idx_val])
        mus.append(mu)
    assert den_t > 0, "Folds internos sem nenhuma linha de validação na coorte"
    return num_t / den_t, por_fold, np.concatenate(ys), np.concatenate(mus)


def executa_unidade(ctx, nome_modelo: str, divs_por_janela: dict[int | None, Divisao], cfg: dict[str, Any],
                    variante: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Tuning aninhado, treino final e previsão do teste para um modelo e uma divisão de teste.

    Parâmetros:
        ctx: Contexto (L e modo já fixados). nome_modelo: chave de REGISTRO.
        divs_por_janela: {janela_meses ou None: Divisao}. Para o esquema deslizante há uma Divisao por janela
            candidata; a janela vence pela validação interna (hiperparâmetro). Nos demais, só a chave None.
        cfg: configuração (tuning.n_configs, semente). variante: nome da variante de esquema.

    Retorna:
        (previsões do teste [cnes, dia, f, ag, expo, mu, nivel], meta [dict serializável], tabela de tuning).

    Cuidados:
        # ANTI-VAZAMENTO: ver cabeçalho do módulo. A escolha usa só `folds_internos`; o teste é usado
        # uma única vez, no ajuste final com a configuração e janela já escolhidas.
        # DECISÃO: mesmo orçamento (n_configs) para todos os modelos ajustáveis; grades menores são
        # percorridas por inteiro; sementes derivadas de (semente, modelo, variante, divisão).
    """
    t0 = time.time()
    cls = REGISTRO[nome_modelo]
    ref = next(iter(divs_por_janela.values()))
    seed = semente_derivada(cfg["semente"], nome_modelo, variante, ref.rotulo)
    rng = np.random.default_rng(seed)
    n_cfg = int(cfg["tuning"]["n_configs"])
    configs = amostra_configs(cls.espaco(), n_cfg, rng) if cls.tunavel else [{}]
    janelas = [w for w in divs_por_janela if w is not None]
    if janelas:                                     # janela como hiperparâmetro (esquema deslizante)
        configs = [{**c, "_janela": int(rng.choice(janelas))} for c in configs]

    linhas_tuning = []
    melhor = None
    for i, c in enumerate(configs):
        params = {k: v for k, v in c.items() if k != "_janela"}
        w = c.get("_janela")
        folds = divs_por_janela[w].folds_internos
        crit, por_fold, y_val, mu_val = _pontua_config(ctx, cls, params, folds, seed)
        linhas_tuning.append({"config": i, "janela_meses": w, "criterio": crit, "por_fold": str(np.round(por_fold, 5).tolist()),
                              "params": str(params)})
        if melhor is None or crit < melhor["crit"]:
            melhor = {"crit": crit, "params": params, "janela": w, "y_val": y_val, "mu_val": mu_val}

    ok = np.isfinite(melhor["mu_val"])
    phi = estima_dispersao(melhor["y_val"][ok], melhor["mu_val"][ok])

    div = divs_por_janela[melhor["janela"]]
    idx_te = ctx.linhas_da_coorte(div.teste, div.inicio_teste)
    k_h = ctx.k_h(div.treino)
    m = cls(melhor["params"], seed).fit(ctx, div.treino, k_h)
    mu = m.predict_mean(ctx, idx_te, k_h)
    P = ctx.P
    nivel = ctx.membros_coorte(div.inicio_teste)
    pred = pd.DataFrame({"cnes": P["cnes"].to_numpy()[idx_te], "dia": P["dia"].to_numpy()[idx_te],
                         "f": P["f"].to_numpy()[idx_te], "ag": P["ag"].to_numpy()[idx_te],
                         "expo": ctx.expo(idx_te, k_h), "mu": mu})
    pred["nivel"] = pred["cnes"].map(nivel)
    meta = {"modelo": nome_modelo, "variante": variante, "rotulo": div.rotulo, "L": ctx.lag, "modo": ctx.modo,
            "esquema": div.esquema, "P": div.parametros.get("P"), "E": div.parametros.get("E"),
            "janela_meses": melhor["janela"], "semente": int(cfg["semente"]), "semente_derivada": seed,
            "params": melhor["params"], "criterio_validacao": melhor["crit"], "phi": phi, "k_h": k_h,
            "n_configs_avaliadas": len(configs), "n_treino": int(len(div.treino)), "n_teste": int(len(idx_te)),
            "n_mu_nan": int((~np.isfinite(mu)).sum()), "inicio_teste": str(div.inicio_teste.date()),
            "fim_teste_exclusivo": str(div.fim_teste.date()), "tempo_s": round(time.time() - t0, 2)}
    if m.p_consulta_ is not None:                         # modelo por paciente: métricas por consulta (agregadas)
        y_c, p_c = m.p_consulta_
        meta["consulta"] = metricas_consulta(y_c, p_c)
    return pred, meta, pd.DataFrame(linhas_tuning)
