# -*- coding: utf-8 -*-
"""
gercon_comum.py  -  modulo comum aos 3 scripts (regressao logistica, random forest, xgboost).

Faz: leitura do banco, preparo das variaveis, divisao temporal (treino / validacao / teste),
pre-processamento (MinMax, One-Hot, Target Encoder), tratamento de desbalanceamento (class_weight,
sub-amostragem, super-amostragem, SMOTE), validacao cruzada ANINHADA, selecao de hiperparametros,
avaliacao na validacao e no teste (metricas + matriz de confusao) e gravacao dos CSVs.

Os 3 scripts (01_..., 02_..., 03_...) importam este arquivo: mantenha os 4 arquivos na mesma pasta.
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, TargetEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix,
                             accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

T0 = time.time()
def log(*a): print('[%5.1f min]' % ((time.time() - T0) / 60), *a, flush=True)

# ---------------------------------------------------------------------------------------------
# 1. COLUNAS
# ---------------------------------------------------------------------------------------------
COLS_LEITURA = ['idsolicitacao', 'idpaciente', 'datahora_consulta', 'situacaoagenda', 'corregulador', 'especialidademae', 'cid_3dig', 'cid_1dig',
    'possuiunidadeindicada', 'dias_ate_consulta', 'dia_semana', 'mes_consulta', 'idade', 'n_consultas_previas', 'n_faltas_previas', 'falta_recente',
    'primeira_vez', 'n_consultas_previas_especialidademae', 'n_faltas_previas_especialidademae', 'ultima_previa_mesma_especialidademae',
    'n_consultas_previas_motivo', 'n_faltas_previas_motivo', 'ultima_previa_mesmo_motivo', 'cnes_solicitante', 'cnesexecutante',
    'favela_comunidade', 'taxa_alf_geral', 'renda_media_winsor', 'coordenadoria_saude', 'distrito_sanitario', 'status',
    'distancia_km', 'tempo_caminhando', 'distancia_km_trans_publico', 'tempo_desloc_trans_publico', 'quant_bus', 'baldiacoes']
COLS_DIST = ['distancia_km', 'tempo_caminhando', 'distancia_km_trans_publico', 'tempo_desloc_trans_publico', 'quant_bus', 'baldiacoes']
COLS_CAT_TEXTO = ['corregulador', 'especialidademae', 'cid_3dig', 'cid_1dig', 'possuiunidadeindicada', 'dia_semana', 'mes_consulta', 'falta_recente',
    'ultima_previa_mesma_especialidademae', 'ultima_previa_mesmo_motivo', 'cnes_solicitante', 'cnesexecutante', 'coordenadoria_saude', 'distrito_sanitario']

# conjuntos de variaveis (num = MinMax | ohe = One-Hot | esp = One-Hot da especialidade (max 21 niveis) | te = Target Encoder)
CONJUNTOS = {
 'completo': dict(
    num=['dias_log', 'idade', 'favela_comunidade', 'taxa_alf_geral', 'renda_media_winsor', 'n_consultas_previas', 'n_faltas_previas', 'primeira_vez',
         'n_consultas_previas_especialidademae', 'n_faltas_previas_especialidademae', 'n_consultas_previas_motivo', 'n_faltas_previas_motivo',
         'taxa_falta_prev', 'int_taxaprev_x_espera', 'distancia_km', 'tempo_caminhando', 'distancia_km_trans_publico', 'tempo_desloc_trans_publico',
         'quant_bus', 'baldiacoes', 'dist_invalida'],
    renda_q=['renda_media_winsor'],
    ohe=['corregulador', 'possuiunidadeindicada', 'coordenadoria_saude', 'distrito_sanitario', 'cid_1dig', 'dia_semana', 'mes_consulta', 'falta_recente',
         'ultima_previa_mesma_especialidademae', 'ultima_previa_mesmo_motivo'],
    esp=['especialidademae'],
    te=['cid_3dig', 'cnes_solicitante', 'cnesexecutante', 'int_esp_exec', 'int_esp_idade']),
 'enxuto': dict(      # conjunto final da analise (12 variaveis + 3 interacoes)
    num=['dias_log', 'idade', 'n_consultas_previas', 'n_faltas_previas', 'taxa_falta_prev', 'tempo_desloc_trans_publico', 'int_taxaprev_x_espera'],
    renda_q=[], ohe=['corregulador', 'possuiunidadeindicada', 'cid_1dig', 'falta_recente'], esp=['especialidademae'],
    te=['cnesexecutante', 'int_esp_exec', 'int_esp_idade']),
}

# ---------------------------------------------------------------------------------------------
# 2. LEITURA E PREPARO
# ---------------------------------------------------------------------------------------------
def carregar_dados(args):
    d = pd.read_parquet(args.dados, columns=COLS_LEITURA)
    log('banco lido: %d linhas' % len(d))
    for c in COLS_CAT_TEXTO + ['situacaoagenda', 'status']: d[c] = d[c].astype(object).where(d[c].notna(), 'NA').astype(str)
    d = d[d.situacaoagenda.isin(['FALTANTE', 'REALIZADA'])]
    if args.executantes_csv and os.path.exists(args.executantes_csv):
        ex = pd.read_csv(args.executantes_csv, dtype=str); d = d[d.cnesexecutante.isin(ex.cnesexecutante)]
        log('recorte municipal (%d executantes na lista): %d linhas' % (ex.cnesexecutante.nunique(), len(d)))
    else: log('ATENCAO: sem lista de executantes -> usando TODAS as linhas do banco')
    d = d.reset_index(drop=True)
    d['y'] = (d.situacaoagenda == 'FALTANTE').astype(int); d['dc'] = pd.to_datetime(d.datahora_consulta)
    if args.amostra and args.amostra < len(d):
        _, ix = train_test_split(np.arange(len(d)), test_size=args.amostra, stratify=d.y, random_state=args.semente); d = d.iloc[np.sort(ix)].reset_index(drop=True)
        log('AMOSTRA estratificada pelo desfecho: %d linhas' % len(d))
    # distancias invalidas (rota inexistente, >60 km, >480 min a pe) viram ausentes; o imputador (mediana do treino) trata dentro do pipeline
    inval = d.status.isin(['sem_rota', 'NA', 'ausente']) | (d.distancia_km > 60) | (d.tempo_caminhando > 480)
    d['dist_invalida'] = inval.astype(int)
    for c in COLS_DIST: d.loc[inval, c] = np.nan
    d['dias_log'] = np.log1p(d.dias_ate_consulta)
    d['taxa_falta_prev'] = (d.n_faltas_previas + 0.44) / (d.n_consultas_previas + 2)          # taxa previa suavizada
    d['int_taxaprev_x_espera'] = d.taxa_falta_prev * d.dias_log
    fx = pd.cut(d.idade, [-1, 17, 39, 59, 200], labels=['0-17', '18-39', '40-59', '60+']).astype(str)
    d['int_esp_exec'] = d.especialidademae + '|' + d.cnesexecutante                           # interacoes (entram no Target Encoder)
    d['int_esp_idade'] = d.especialidademae + '|' + fx
    log('falta geral: %.1f%% | periodo %s a %s' % (100 * d.y.mean(), d.dc.min().date(), d.dc.max().date()))
    return d

def dividir(d, args):
    """treino / validacao / teste TEMPORAIS: teste = consultas a partir de --corte_teste; validacao = ultimo --frac_val do restante (por data)"""
    corte = pd.Timestamp(args.corte_teste); dev = d[d.dc < corte].sort_values('dc', kind='stable'); te = d[d.dc >= corte]
    nv = int(round(len(dev) * args.frac_val)); tr, va = dev.iloc[:len(dev) - nv], dev.iloc[len(dev) - nv:]
    for n, x in [('treino', tr), ('validacao', va), ('teste', te)]:
        log('%-9s %7d linhas | %6d pacientes | falta %.1f%% | %s a %s' % (n, len(x), x.idpaciente.nunique(), 100 * x.y.mean(), x.dc.min().date(), x.dc.max().date()))
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

# ---------------------------------------------------------------------------------------------
# 3. PRE-PROCESSAMENTO (ajustado SOMENTE no treino de cada dobra)
# ---------------------------------------------------------------------------------------------
def preprocessador(conjunto='completo'):
    s = CONJUNTOS[conjunto]; t = []
    t.append(('num', SkPipeline([('imp', SimpleImputer(strategy='median')), ('mm', MinMaxScaler())]), s['num']))
    if s['renda_q']: t.append(('renda_q', SkPipeline([('imp', SimpleImputer(strategy='median')), ('kb', KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile'))]), s['renda_q']))
    t.append(('esp', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=21, sparse_output=False), s['esp']))
    t.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), s['ohe']))
    t.append(('te', TargetEncoder(target_type='binary', cv=5, random_state=42), s['te']))
    return ColumnTransformer(t)
def colunas_usadas(conjunto):
    s = CONJUNTOS[conjunto]; return list(dict.fromkeys(s['num'] + s['renda_q'] + s['esp'] + s['ohe'] + s['te']))

# ---------------------------------------------------------------------------------------------
# 4. DESBALANCEAMENTO: 'bal' e um passo do pipeline (so atua nos dados de TREINO de cada dobra; nunca no teste)
# ---------------------------------------------------------------------------------------------
SAMPLERS = {'sub_amostragem': RandomUnderSampler(random_state=0), 'super_amostragem': RandomOverSampler(random_state=0), 'smote': SMOTE(random_state=0)}
def montar_espaco(params, chave_peso, peso_neutro, peso_ativo, balanceamentos):
    """lista de dicionarios: 1 por estrategia de balanceamento (nenhum | class_weight | sub_amostragem | super_amostragem | smote)"""
    esp = []
    for b in balanceamentos:
        p = {k: list(v) for k, v in params.items()}
        if b == 'nenhum': p.update({'bal': ['passthrough'], chave_peso: [peso_neutro]})
        elif b == 'class_weight': p.update({'bal': ['passthrough'], chave_peso: [peso_ativo]})
        else: p.update({'bal': [SAMPLERS[b]], chave_peso: [peso_neutro]})
        esp.append(p)
    return esp
def rotulo_bal(p, chave_peso, peso_neutro):
    b = p['bal']
    if not isinstance(b, str): return {RandomUnderSampler: 'sub_amostragem', RandomOverSampler: 'super_amostragem', SMOTE: 'smote'}[type(b)]
    return 'nenhum' if p[chave_peso] == peso_neutro else 'class_weight'
def params_texto(p): return {k: (type(v).__name__ if k == 'bal' and not isinstance(v, str) else v) for k, v in p.items()}

# ---------------------------------------------------------------------------------------------
# 5. METRICAS
# ---------------------------------------------------------------------------------------------
def metricas(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b > 0 else np.nan
    try: auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    except ValueError: auc = pr = np.nan
    k10 = max(int(len(y) * .10), 1); k20 = max(int(len(y) * .20), 1); o = np.argsort(-p, kind='stable')
    return dict(n=len(y), taxa_falta=y.mean(), limiar=thr, AUC=auc, PR_AUC=pr, Brier=brier_score_loss(y, p), LogLoss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
        acuracia=accuracy_score(y, pred), acuracia_balanceada=balanced_accuracy_score(y, pred), precisao=sd(tp, tp + fp), sensibilidade=sd(tp, tp + fn),
        especificidade=sd(tn, tn + fp), VPN=sd(tn, tn + fn), F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        lift_top10=y[o[:k10]].mean() / y.mean() if y.mean() > 0 else np.nan, precisao_top20=y[o[:k20]].mean(), sensibilidade_top20=y[o[:k20]].sum() / max(y.sum(), 1),
        VN=tn, FP=fp, FN=fn, VP=tp)
MET_KEYS = list(metricas(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]), .5))
def escolher_limiar(y, p, criterio):
    if criterio == '0.5': return 0.5
    y = np.asarray(y); best = (-9, 0.5)
    for t in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))):
        pr = (p >= t).astype(int)
        s = matthews_corrcoef(y, pr) if criterio == 'mcc' else f1_score(y, pr, zero_division=0)
        if len(np.unique(pr)) > 1 and s > best[0]: best = (s, t)
    return float(best[1]) if best[0] > -9 else 0.5

# ---------------------------------------------------------------------------------------------
# 6. ARGUMENTOS
# ---------------------------------------------------------------------------------------------
def ler_args(modelo, desc):
    here = os.path.dirname(os.path.abspath(__file__)); ex = os.path.join(here, 'executantes_municipais.csv')
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dados', default='gercon_poa_censo_distancias.parquet', help='caminho do banco (parquet)')
    ap.add_argument('--saida', default='resultados', help='pasta onde os CSVs serao gravados')
    ap.add_argument('--executantes_csv', default=ex if os.path.exists(ex) else None, help='lista de executantes municipais (recorte); use "" para usar o banco todo')
    ap.add_argument('--amostra', type=int, default=0, help='usa so N linhas (estratificadas) - para testar; 0 = banco todo')
    ap.add_argument('--corte_teste', default='2024-04-08', help='consultas a partir desta data formam o TESTE')
    ap.add_argument('--frac_val', type=float, default=0.20, help='fracao final (por data) do periodo anterior ao corte usada como VALIDACAO')
    ap.add_argument('--conjunto', default='completo', choices=list(CONJUNTOS), help='conjunto de variaveis')
    ap.add_argument('--balanceamento', default='nenhum,class_weight,sub_amostragem,super_amostragem,smote', help='estrategias testadas (separadas por virgula)')
    ap.add_argument('--outer', type=int, default=5, help='dobras externas da CV aninhada')
    ap.add_argument('--inner', type=int, default=3, help='dobras internas (selecao de hiperparametros)')
    ap.add_argument('--n_iter', type=int, default=15, help='combinacoes de hiperparametros sorteadas por busca')
    ap.add_argument('--metrica', default='average_precision', choices=['roc_auc', 'average_precision', 'neg_brier_score'], help='metrica otimizada na busca')
    ap.add_argument('--limiar', default='mcc', choices=['mcc', 'f1', '0.5'], help='criterio do limiar de decisao (escolhido so com dados de treino)')
    ap.add_argument('--max_linhas_busca', type=int, default=60000, help='a CV aninhada e a busca usam no maximo N linhas do treino (o ajuste final usa o treino todo)')
    ap.add_argument('--sem_aninhado', action='store_true', help='pula a CV aninhada (so busca + validacao + teste)')
    ap.add_argument('--sem_refit_treino_val', action='store_true', help='nao reajusta com treino+validacao antes do teste')
    ap.add_argument('--n_jobs', type=int, default=2); ap.add_argument('--semente', type=int, default=42)
    a = ap.parse_args(); a.modelo = modelo; a.balanceamentos = [b.strip() for b in a.balanceamento.split(',') if b.strip()]
    for b in a.balanceamentos:
        if b not in ['nenhum', 'class_weight'] + list(SAMPLERS): ap.error('balanceamento invalido: ' + b)
    os.makedirs(a.saida, exist_ok=True); return a

# ---------------------------------------------------------------------------------------------
# 7. PIPELINE COMPLETO
# ---------------------------------------------------------------------------------------------
def rodar(args, estimador, params, chave_peso, peso_neutro, peso_ativo_fn, n_jobs_modelo_key=None):
    d = carregar_dados(args); tr, va, te = dividir(d, args); cols = colunas_usadas(args.conjunto)
    pw = peso_ativo_fn(tr.y)
    pipe = Pipeline([('pre', preprocessador(args.conjunto)), ('bal', 'passthrough'), ('m', estimador)])
    espaco = montar_espaco(params, chave_peso, peso_neutro, pw, args.balanceamentos)
    # subamostra do treino para CV aninhada e busca (agrupada por paciente nao e necessaria aqui: amostra de linhas)
    if len(tr) > args.max_linhas_busca:
        _, ix = train_test_split(np.arange(len(tr)), test_size=args.max_linhas_busca, stratify=tr.y, random_state=args.semente); tb = tr.iloc[np.sort(ix)].reset_index(drop=True)
        log('CV aninhada e busca usam %d das %d linhas de treino' % (len(tb), len(tr)))
    else: tb = tr
    Xb, yb, gb = tb[cols], tb.y.values, tb.idpaciente.values
    inner = StratifiedGroupKFold(n_splits=args.inner, shuffle=True, random_state=args.semente)
    def nova_busca():
        return RandomizedSearchCV(clone(pipe), espaco, n_iter=args.n_iter, scoring=args.metrica, cv=inner, n_jobs=args.n_jobs, random_state=args.semente, refit=True, error_score='raise')
    linhas_met = []; linhas_cm = []; folds = []
    # ---- A) CV ANINHADA: externa avalia o PROCEDIMENTO (busca interna + limiar), nunca vendo a dobra externa na busca
    if not args.sem_aninhado:
        outer = StratifiedGroupKFold(n_splits=args.outer, shuffle=True, random_state=args.semente + 1)
        for i, (a, b) in enumerate(outer.split(Xb, yb, gb), 1):
            t = time.time(); s = nova_busca(); s.fit(Xb.iloc[a], yb[a], groups=gb[a]); best = s.best_estimator_
            oof = cross_val_predict(clone(best), Xb.iloc[a], yb[a], groups=gb[a], cv=inner, method='predict_proba')[:, 1]; thr = escolher_limiar(yb[a], oof, args.limiar)
            m = metricas(yb[b], best.predict_proba(Xb.iloc[b])[:, 1], thr); m.update(dobra_externa=i, **{'melhor_' + k: v for k, v in params_texto(s.best_params_).items()}, balanceamento=rotulo_bal(s.best_params_, chave_peso, peso_neutro))
            folds.append(m); log('CV aninhada %d/%d | AUC %.4f | PR-AUC %.4f | balanceamento escolhido: %s | %.1f min' % (i, args.outer, m['AUC'], m['PR_AUC'], m['balanceamento'], (time.time() - t) / 60))
        F = pd.DataFrame(folds); num = list(MET_KEYS)
        medias = F[num].mean(numeric_only=True); dps = F[num].std(numeric_only=True)
        linhas_met += [dict(etapa='cv_aninhada_media', **medias.to_dict()), dict(etapa='cv_aninhada_dp', **dps.to_dict())]
        F.to_csv(os.path.join(args.saida, args.modelo + '_cv_aninhada_dobras.csv'), index=False)
    # ---- B) busca final de hiperparametros (todo o treino da busca)
    log('busca final de hiperparametros (%d combinacoes x %d dobras)' % (args.n_iter, args.inner)); s = nova_busca(); s.fit(Xb, yb, groups=gb); best_params = s.best_params_
    R = pd.DataFrame(s.cv_results_); R['balanceamento'] = [rotulo_bal(p, chave_peso, peso_neutro) for p in R.params]; R['params'] = [str(params_texto(p)) for p in R.params]
    R = R.sort_values('rank_test_score'); keep = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'balanceamento', 'params']
    R[keep].to_csv(os.path.join(args.saida, args.modelo + '_busca_hiperparametros.csv'), index=False)
    cb = R.groupby('balanceamento').agg(melhor_score=('mean_test_score', 'max'), score_medio=('mean_test_score', 'mean'), combinacoes=('mean_test_score', 'size')).sort_values('melhor_score', ascending=False)
    cb.insert(0, 'metrica', args.metrica); cb.to_csv(os.path.join(args.saida, args.modelo + '_comparacao_balanceamento.csv')); log('melhor combinacao:', params_texto(best_params)); print(cb.round(4).to_string())
    # ---- C) limiar: so com dados de treino (previsoes fora-da-dobra)
    best = clone(pipe).set_params(**best_params); oof = cross_val_predict(clone(best), Xb, yb, groups=gb, cv=inner, method='predict_proba')[:, 1]; thr = escolher_limiar(yb, oof, args.limiar)
    log('limiar de decisao (%s, treino): %.4f' % (args.limiar, thr))
    # ---- D) ajuste no treino completo e VALIDACAO
    if n_jobs_modelo_key: best.set_params(**{n_jobs_modelo_key: args.n_jobs})
    t = time.time(); best.fit(tr[cols], tr.y.values); pv = best.predict_proba(va[cols])[:, 1]; mv = metricas(va.y.values, pv, thr); linhas_met.append(dict(etapa='validacao', **mv)); log('VALIDACAO | AUC %.4f | PR-AUC %.4f | MCC %.4f | %.1f min' % (mv['AUC'], mv['PR_AUC'], mv['MCC'], (time.time() - t) / 60))
    # ---- E) TESTE (reajusta com treino + validacao, mesmos hiperparametros e limiar)
    if not args.sem_refit_treino_val:
        full = pd.concat([tr, va], ignore_index=True); best.fit(full[cols], full.y.values)
    pt = best.predict_proba(te[cols])[:, 1]; mt = metricas(te.y.values, pt, thr); linhas_met.append(dict(etapa='teste', **mt)); log('TESTE | AUC %.4f | PR-AUC %.4f | MCC %.4f' % (mt['AUC'], mt['PR_AUC'], mt['MCC']))
    # ---- saidas
    M = pd.DataFrame(linhas_met); cm_cols = ['VN', 'FP', 'FN', 'VP']
    M.drop(columns=cm_cols).round(5).to_csv(os.path.join(args.saida, args.modelo + '_metricas.csv'), index=False)
    cm = M[M.etapa.isin(['validacao', 'teste'])][['etapa'] + cm_cols].copy()
    for e in ['validacao', 'teste']:
        r = cm[cm.etapa == e].iloc[0]; print('\nMatriz de confusao (%s) | limiar %.4f\n                 previsto REALIZADA   previsto FALTA\n  real REALIZADA %14d %16d\n  real FALTA     %14d %16d' % (e, thr, r.VN, r.FP, r.FN, r.VP))
    cm.to_csv(os.path.join(args.saida, args.modelo + '_matriz_confusao.csv'), index=False)
    H = pd.DataFrame([dict(modelo=args.modelo, conjunto=args.conjunto, balanceamento=rotulo_bal(best_params, chave_peso, peso_neutro), limiar=thr, criterio_limiar=args.limiar, metrica_busca=args.metrica, **params_texto(best_params))]); H.to_csv(os.path.join(args.saida, args.modelo + '_hiperparametros.csv'), index=False)
    pd.DataFrame({'idsolicitacao': te.idsolicitacao.values, 'y': te.y.values, 'prob_falta': pt, 'previsto': (pt >= thr).astype(int)}).to_csv(os.path.join(args.saida, args.modelo + '_predicoes_teste.csv'), index=False)
    # consolidado entre modelos (cada script acrescenta/atualiza as proprias linhas)
    f = os.path.join(args.saida, 'resultados_comparacao_modelos.csv'); novo = M[M.etapa.isin(['cv_aninhada_media', 'validacao', 'teste'])].copy(); novo.insert(0, 'modelo', args.modelo)
    if os.path.exists(f): ant = pd.read_csv(f); novo = pd.concat([ant[ant.modelo != args.modelo], novo], ignore_index=True)
    novo.round(5).to_csv(f, index=False); log('FIM | arquivos gravados em', os.path.abspath(args.saida)); return M
