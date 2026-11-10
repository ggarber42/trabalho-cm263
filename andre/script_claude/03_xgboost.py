# -*- coding: utf-8 -*-
"""XGBoost: pre-processamento (MinMax, One-Hot, Target Encoder), balanceamento, CV aninhada, validacao, teste.
Exemplo:  python 03_xgboost.py --dados gercon_poa_censo_distancias.parquet --saida resultados
Teste rapido:  python 03_xgboost.py --amostra 1000 --outer 3 --inner 2 --n_iter 6 --saida teste_rapido"""
from xgboost import XGBClassifier
from gercon_comum import ler_args, rodar
a = ler_args('xgboost', __doc__)
params = {'m__n_estimators': [200, 400, 600], 'm__learning_rate': [0.03, 0.05, 0.1], 'm__max_depth': [3, 4, 6], 'm__min_child_weight': [1, 5, 20],
          'm__subsample': [0.7, 0.9, 1.0], 'm__colsample_bytree': [0.5, 0.8, 1.0], 'm__reg_lambda': [1, 5, 10]}
est = XGBClassifier(tree_method='hist', eval_metric='logloss', n_jobs=1, random_state=42)
rodar(a, est, params, 'm__scale_pos_weight', 1.0, lambda y: float((y == 0).sum() / max((y == 1).sum(), 1)), n_jobs_modelo_key='m__n_jobs')   # class_weight = scale_pos_weight (neg/pos)
