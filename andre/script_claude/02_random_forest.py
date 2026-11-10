# -*- coding: utf-8 -*-
"""Random Forest: pre-processamento (MinMax, One-Hot, Target Encoder), balanceamento, CV aninhada, validacao, teste.
Exemplo:  python 02_random_forest.py --dados gercon_poa_censo_distancias.parquet --saida resultados
Teste rapido:  python 02_random_forest.py --amostra 1000 --outer 3 --inner 2 --n_iter 6 --saida teste_rapido"""
from sklearn.ensemble import RandomForestClassifier
from gercon_comum import ler_args, rodar
a = ler_args('random_forest', __doc__)
params = {'m__n_estimators': [200, 400], 'm__max_depth': [None, 10, 20], 'm__min_samples_leaf': [5, 15, 30, 60], 'm__max_features': ['sqrt', 0.3]}
est = RandomForestClassifier(n_jobs=1, random_state=42)                            # n_jobs=1 dentro da busca (a busca ja usa --n_jobs); o ajuste final usa --n_jobs
rodar(a, est, params, 'm__class_weight', None, lambda y: 'balanced_subsample', n_jobs_modelo_key='m__n_jobs')
