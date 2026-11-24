# Pipeline v2 — previsão semanal de faltas por CNES executante (GERCON, Porto Alegre)

Pasta autocontida: do CSV bruto do GERCON às tabelas de comparação de modelos.
Desenho: `../docs/superpowers/specs/2026-11-22-redesenho-metodologico-design.md`;
decisões: `../DECISOES_REDESENHO_2026-11-22.md` (D1–D20); texto: `../entregaveis/metodologia_v2.md`.

## 1. Rodar em outra máquina (passo a passo)

1. **Copie a pasta `pipeline_v2/` inteira** para a máquina nova.
2. **Coloque os dados brutos** em `pipeline_v2/dados/brutos/` (os nomes estão em `config.yaml`):
   - `gercon_solicitacoes_executante_porto_alegre_2018_2026.csv` (GERCON unificado, `;`, UTF-8)
   - `tbEstabelecimento202609.csv` e `tbNaturezaJuridica202609.csv` (CNES, competência 09/2026,
     arquivo `BASE_DE_DADOS_CNES_202609.ZIP` em https://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp)
3. **Crie um ambiente Python 3.12** fora de pastas sincronizadas (Drive/Dropbox):
   ```bash
   python3.12 -m venv ~/venvs/faltas
   source ~/venvs/faltas/bin/activate
   pip install -r requirements.txt
   ```
4. **(Opcional) Teste de wavelets em R** (Nason 2013). Sem ele, a regra usa ADF + KPSS e isso
   fica registrado em `esquema.json`:
   ```bash
   Rscript -e 'install.packages("locits")'
   ```
   Se o pacote não instalar (pode estar arquivado no CRAN), siga sem ele.
5. **Teste rápido com uma CNES** (≈ 20–25 min; medido em 22/11/2026 com 12 núcleos):
   ```bash
   ./executar.sh --perfil smoke --cnes 2237385
   ```
6. **Rodada completa** (todas as CNES municipais; ≈ 70 min medidos em 23/11/2026 com 12 núcleos,
   a maior parte nos modelos locais ETS/Croston/SARIMAX das CNES consistentes):
   ```bash
   ./executar.sh --perfil completo
   ```
   Para rodar uma etapa isolada: `python3 etapas/e04_modelar.py --perfil completo`.
   Se os dados estiverem em outro lugar, crie um YAML como `config_local.yaml` (que só troca
   os caminhos) e passe `--config meu_config.yaml` em todas as etapas.

## 2. Etapas

| Etapa | O que faz | Principais saídas (`saidas/<perfil>/`) |
|---|---|---|
| `e01_preparar` | Lê o bruto, conta as situações, padroniza datas (dia) e CNES (7 dígitos), marca CNES municipais, aplica os filtros com fluxograma e calcula o histórico do paciente (`data_ev < t0`, só na base filtrada) | `consultas.parquet` (sem `idpaciente`), `fluxograma.csv`, `contagens_situacao.csv`, `cnes_municipais.csv` |
| `e02_painel` | Painel série (CNES × especialidade-mãe) × semana por horizonte (h = 2 principal, h = 1 secundário) (agenda conhecida na origem, agregados, defasagens, calendário) | `painel_h1.parquet`, `painel_h2.parquet`, `base_consulta.parquet`, `fracao_agenda_conhecida.csv` |
| `e03_exploratoria` | Caracteriza as séries **só no treino**, aplica a regra de Cerqueira et al. (2020) e define os estratos consistentes/outros (D17) | `caracterizacao_series.csv`, `acf_series.csv`, `esquema.json`, `estratos_cnes.csv` |
| `e04_modelar` | Por estrato: tuning no holdout repetido do treino (8 configurações por modelo), previsão do teste 2024 com origem móvel, fallback dos locais para a referência (D18) | `previsoes.parquet`, `tuning.csv`, `diagnostico_residuos.csv`, `previsoes_consulta.parquet` |
| `e05_avaliar` | Interseção de linhas, métricas com IC95% e margem de erro (D19), Diebold-Mariano + Holm contra a referência e a mm4 (D20) e entre pares | `tabela_principal.csv`, `metricas.csv`, `dm.csv`, `dm_vs_mm4.csv`, `dm_pares.csv`, `linhas_perdidas.csv`, `metricas_consulta.csv` |
| `e06_especialidade` | Erro por especialidade-mãe, por estrato e modelo | `especialidade_resumo.csv` |
| `e07_figuras` | Observado × previsto (soma semanal), WAPE com IC95% e ganho sobre a referência com IC95%, por h e estrato | `figuras/*.png`, `figuras/dados_observado_previsto.csv` |
| `d01_auditoria` | Métricas sem fallback e teste prático de vazamento (desfecho alterado não pode mudar preditoras passadas) | `auditoria/metricas_sem_fallback.csv`, `auditoria/teste_vazamento.csv` |
| `r01_metodologia` | Preenche `../entregaveis/metodologia_v2_modelo.md` com os números dos resultados | `../entregaveis/metodologia_v2.md` |

Cada etapa grava `logs/<etapa>.log` e `logs/manifesto_<etapa>.json` (SHA-256 das entradas e saídas).

## 3. Proteções contra vazamento (onde estão)

- Histórico do paciente: `faltas/preparacao.py::calcular_historico` (estritamente antes de t0).
- Agenda conhecida e defasagens: `faltas/painel.py::montar_painel` (t0 ≤ origem; lags a partir de s − h).
- Target encoding: `faltas/painel.py::te_temporal` (semanas ≤ s − h; congelado no fim do treino).
- Exploratória e escolha do esquema: só semanas ≤ fim do treino do teste principal.
- Dobras: `faltas/divisoes.py` (só semanas do treino); teste com `assert` de início após o treino.
- Modelos locais: origem móvel com parâmetros fixos (`faltas/modelos_series.py`).
- Pré-processamento dos globais (log1p, mediana, min-max) ajustado só no treino: `faltas/preprocessamento.py`.
- Teste prático de vazamento: `etapas/d01_auditoria.py` (falha com "VAZAMENTO" se uma preditora passada mudar).
Os testes em `tests/` cobrem esses pontos e rodam antes de cada execução do `executar.sh`.

## 4. Simplificações e pendências conhecidas

- *Gradient boosting* via `HistGradientBoosting` do scikit-learn (sem LightGBM/XGBoost/CatBoost),
  para reduzir dependências.
- Regressão de contagem: Poisson penalizada; a binomial negativa entra como distribuição
  preditiva comum a todos os modelos (φ por momentos, fora da amostra).
- `agenda conhecida` conta só consultas que terminaram REALIZADA/FALTANTE (as que depois
  foram transferidas não estão na base de análise) — limitação a declarar.
- Wavelets (Nason 2013) não executaram nesta máquina; a regra usou ADF + KPSS (registrado em `esquema.json`).
- Intervalos de previsão conservadores (cobertura 98–99% no IC95), por contagens pequenas e discretas (D20).
- Avaliação em um único ano de teste (2024).
- Critério municipal (D2) reproduz o recorte antigo (111 CNES); sem a lista de razões sociais
  entrariam 8 CNES "Hospital Nossa Senhora da Conceição" e o HPS — a confirmar.
