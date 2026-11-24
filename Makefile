# Makefile — ponto de entrada do pipeline (M0-M4 + empacotamento). Rodar da raiz do projeto.
#   make smoke            amostra de 4 CNES, todos os anos e esquemas, poucos minutos (testes incluídos)
#   make full JOBS=11     perfil COMPLETO (todas as CNES do escopo; retomável); só na máquina potente
#   make estimativa       tempo e memória do completo, extrapolados do smoke (rode `make smoke` antes)
# PYTHON padrão: o venv do projeto (.venv, criado por `make venv`); sem ele, python3. Pode ser trocado: make smoke PYTHON=...
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
JOBS   ?=
ARGS_S7 = $(if $(JOBS),--jobs $(JOBS),)

.PHONY: smoke full test venv lock env confere-hash estimativa limpa-smoke \
        auditoria auditoria-smoke coorte amostra divisoes divisoes-smoke horizonte horizonte-smoke \
        reproduz experimentos avaliacao figuras importancia overbooking sensibilidade

# ---------------- perfis (variável PERFIL lida por src/config.py) ----------------
smoke: export PERFIL=smoke
smoke: test auditoria-smoke coorte amostra divisoes-smoke horizonte-smoke reproduz experimentos avaliacao importancia overbooking sensibilidade figuras

full: export PERFIL=completo
full: confere-hash auditoria coorte horizonte divisoes reproduz experimentos avaliacao importancia overbooking sensibilidade figuras

# ---------------- ambiente ----------------
# Cria o venv do projeto em .venv (ignorado pelo git) com Python 3.12 e as versões exatas do lock.
# DECISÃO (24/11/2026, pedido do usuário): venv dentro da pasta do projeto. Se a pasta for sincronizada
# com o Drive, excluir .venv/ da sincronização.
PY_BASE ?= python3.12
venv:
	$(PY_BASE) -m venv .venv
	.venv/bin/pip install --upgrade pip
	grep -v '^#' requirements.lock | .venv/bin/pip install -r /dev/stdin
	@echo "Pronto: make smoke (usa .venv automaticamente)"

lock:
	$(PYTHON) etapas/S0_lock_ambiente.py

env:
	$(PYTHON) etapas/S0_lock_ambiente.py --so-mostrar

confere-hash:
	$(PYTHON) etapas/S0_confere_hash.py

# ---------------- etapas ----------------
test:
	$(PYTHON) -m pytest tests -q

auditoria-smoke:
	$(PYTHON) etapas/S1_auditoria.py --smoke
auditoria:
	$(PYTHON) etapas/S1_auditoria.py

coorte:
	$(PYTHON) etapas/S2_coorte_por_corte.py
amostra:
	$(PYTHON) etapas/S2_amostra_smoke.py

horizonte-smoke: amostra
	$(PYTHON) etapas/S3_1_horizonte.py --smoke
horizonte:
	$(PYTHON) etapas/S3_1_horizonte.py

divisoes-smoke: amostra
	$(PYTHON) etapas/S4_divisoes.py --smoke
divisoes:
	$(PYTHON) etapas/S4_divisoes.py

reproduz:
	$(PYTHON) etapas/S5_reproduz_baseline.py

# Tuning aninhado + treino + previsão. RETOMÁVEL: o que já tem JSON/parquet é pulado.
experimentos:
	$(PYTHON) etapas/S7_experimentos.py $(ARGS_S7)

avaliacao:
	$(PYTHON) etapas/S9_avaliacao.py
importancia:
	$(PYTHON) etapas/S10_importancia.py
overbooking:
	$(PYTHON) etapas/S11_overbooking.py
sensibilidade:
	$(PYTHON) etapas/S12_sensibilidade.py
figuras:
	$(PYTHON) etapas/S13_tabelas_figuras.py

estimativa:
	$(PYTHON) etapas/S15_estimativa_recursos.py $(if $(JOBS),--jobs $(JOBS),)

# Apaga SÓ os resultados do smoke (para refazer do zero).
limpa-smoke:
	rm -rf resultados/municipal_smoke
