# -*- coding: utf-8 -*-
"""Regressao logistica: pre-processamento (MinMax, One-Hot, Target Encoder), balanceamento, CV aninhada, validacao, teste.
Exemplo:  python 01_regressao_logistica.py --dados gercon_poa_censo_distancias.parquet --saida resultados
Teste rapido:  python 01_regressao_logistica.py --amostra 1000 --outer 3 --inner 2 --n_iter 6 --saida teste_rapido"""
from sklearn.linear_model import LogisticRegression
from gercon_comum import ler_args, rodar
a = ler_args('regressao_logistica', __doc__)
params = {'m__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}                      # penalidade L2 (lbfgs); C = inverso da regularizacao
est = LogisticRegression(max_iter=3000, solver='lbfgs')
rodar(a, est, params, 'm__class_weight', None, lambda y: 'balanced')
