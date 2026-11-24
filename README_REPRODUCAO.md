# Reprodução do pipeline em outra máquina (perfil COMPLETO)

Pipeline de previsão de absenteísmo (GERCON Porto Alegre) — CMP263. Este guia leva da máquina nova até as
tabelas e figuras. **Nada aqui precisa de internet além da instalação do ambiente.**

## 1. O que levar para a máquina nova
1. O repositório git desta pasta (código, `config.yaml`, listas de CNES, CSVs pequenos, testes):
   `git clone <caminho>` ou copie a pasta **sem** `__pycache__`.
2. Os dados, que **não** vão no git (LGPD e tamanho). Coloque na **raiz do projeto** (mesma pasta do `config.yaml`):
   - `gercon_poa_censo_distancias_com_natureza_juridica.parquet` (~42 MB)
   - `gercon_poa_censo_estabelecimentos.parquet`
3. Os CSVs de entrada (já no git): `flag_municipal.csv`, `niveis_municipal.csv`, `coorte_2019_2026.csv`, `comparacao_series.csv`.

## 2. Ambiente (Python 3.12, Linux)
```bash
make venv                                   # cria .venv na pasta do projeto (Python 3.12, ignorado pelo git)
                                            # e instala as versões exatas do requirements.lock (inclui lightgbm etc.)
PY=.venv/bin/python                         # Makefile e run_all.sh já usam .venv por padrão
make env                                    # mostra versões, nº de núcleos e RAM
# make lock                                 # só se mudar o ambiente: regrava requirements.lock
```
- Se a pasta for sincronizada com o Drive, exclua `.venv/` da sincronização.
- Sem `lightgbm` o pipeline roda do mesmo jeito: o modelo `lightgbm` é pulado com um aviso.
- `lightgbm`, `xgboost` e `catboost` são opcionais (`requirements-extra.txt`); sem elas o modelo correspondente é pulado.

## 3. Conferir as entradas (hashes SHA-256)
```bash
python3 etapas/S0_confere_hash.py           # compara com dados/hashes_esperados.json
```
Tem de dizer `entradas conferem`. Se algum arquivo estiver `DIFERENTE`, **pare**: os resultados não são comparáveis.

## 4. Teste de fumaça (poucos minutos) — obrigatório antes do completo
```bash
make smoke PYTHON=$PY
```
Roda testes (~115), auditoria, coorte, divisões, S3.1, reprodução do baseline legado, experimentos (4 CNES, 3
configurações de tuning, todos os esquemas, L ∈ {3,14}), avaliação e figuras. Esperado: todos os testes passam
e `[S5] REPRODUZ o legado dentro da tolerância`.

## 5. Estimativa de tempo e memória do completo
```bash
make estimativa JOBS=11 PYTHON=$PY
```
Extrapola dos tempos do smoke (±2x). Na máquina de origem (12 núcleos / 15 GB): **~88 CPU-horas ≈ 10 h de relógio com
11 processos (rf 23 h, paciente_hgb 10 h, hgb 8 h, hgb_k 8 h, sarimax 16 h, sarima 8 h de CPU); ~0,7 GB de RAM por processo** (≈ 8 GB no total). Estimativa ±2x, para L ∈ {3,14}; S10–S12 somam pouco.

## 6. Rodar o completo
```bash
./run_all.sh 11 $PY                         # ou: make full JOBS=11 PYTHON=$PY
```
Ordem: conferência de hashes → auditoria → coorte por corte → S3.1 → divisões → reprodução do baseline →
**experimentos (S7, o longo)** → avaliação (S9) → tabelas e figuras (S13). Logs em `logs/`.

**Retomável:** cada (variante, L, modo, modelo, divisão) grava `resultados/municipal_completo/.../<rotulo>.json` por
último; se a execução cair, rode `./run_all.sh` de novo e o que já terminou é pulado. Para refazer um experimento:
apague seus arquivos ou use `python3 etapas/S7_experimentos.py --refazer`. `--estimativa` lista as tarefas.

Parâmetros que você pode mudar em `config.yaml` (perfil `completo`): `tuning.n_configs` (24), `experimento.n_jobs`,
`experimento.threads_por_tarefa`, `experimento.blocos` (quais variantes, L, modos e modelos entram).

## 7. Onde estão os resultados
| O quê | Onde |
|---|---|
| Previsões e metadados por experimento | `resultados/municipal_completo/<variante>/L<L>_<modo>/<modelo>/` |
| Registro de todas as execuções (sementes, P, E, janela, hash, métricas) | `resultados/municipal_completo/experimentos.csv` |
| Tabelas CSV da avaliação | `tabelas/res_municipal_completo_*.csv` |
| Tabelas em Markdown e figuras (PNG/PDF) | `tabelas/md/completo/`, `figuras/completo/` |
| Auditoria, fluxograma, coorte, divisões, S3.1 | `tabelas/`, `entregaveis/justificativa_horizonte.md` |
| Manifestos com hash das saídas | `manifestos/` |

Conferência final: `tabelas/reproducao_baseline.csv` (baseline do pipeline novo × legado) e
`tabelas/res_municipal_completo_linhas_perdidas.csv` (linhas perdidas por modelo).

## 8. Limitações e decisões em aberto (leia antes de interpretar)
- **Cenários de L (decisão de 17/11/2026):** só **L = 3** (operacional) e **L = 14**; ambos rodam completos (todos os blocos).
  **G2 congelado em 17/11/2026** (`entregaveis/plano_analise.md`): esquema principal `expansiva`, `lag_principal: 3`,
  `agenda_modo: conhecida`, período comum 2024-01 a 2026-03 e métrica primária WAPE + CRPS; as demais variantes são sensibilidade.
- **Horizonte:** a S3.1 (`entregaveis/justificativa_horizonte.md`) mostra ~56% das consultas com antecedência de exatamente 14 dias e
  só ~92% da agenda final agendada 3 dias antes (56% a 14). Confirmar com o serviço a regra de abertura de agenda e o atraso de
  registro do desfecho (a base não tem esse campo).
- **Modelos agora no pipeline (17/11/2026):** naive, baseline, ETS, SARIMA, SARIMAX, GLM-NB, Poisson regularizado, HGB, HGB com
  recalibração k, floresta, LightGBM, XGBoost, CatBoost, MLP e HGB por paciente. SARIMA/SARIMAX só no modo `conhecida`
  (custo), grade de 12 configurações e ajuste nos últimos 730 dias do treino. O k do `hgb_poisson_k` usa, no início do teste,
  previsões dentro da amostra na janela de 28 dias (declarado no código).
- **S10/S11/S12 implementadas:** importância por permutação (variável e grupo; SHAP/ALE não), overbooking por quantil com custos
  declarados, sensibilidade por fator + estabilidade do ranking + repetição por semente (rf, mlp, catboost). Não cobertos:
  escopo geral, coorte global e remoção de variáveis suspeitas (exigem novas execuções).
- **Seleção de variáveis:** o modelo por paciente usa as 30 variáveis candidatas (sem `status` e `possuiunidadeindicada`).
- **Coorte** definida por corte de ano (só dados anteriores). **DM "vs melhor":** o melhor é identificado no mesmo teste; descritivo.
- **Scripts anteriores ao pipeline** estão em `antigos/` (não fazem parte do `make`; para rodá-los, execute de dentro de `antigos/`).
- **Dados:** arquivo bruto e script das distâncias confirmados pelo usuário (17/11/2026), fora das conferências. `t0 = data_solicitacao + max(diasfilaespera, 0)` é a data de agendamento (`diasfilaespera` = tempo até o agendamento, não até a consulta). Ainda em aberto: atraso de registro do desfecho, `possuiunidadeindicada`, `idade`.

## 9. Se algo der errado
| Sintoma | O que fazer |
|---|---|
| `ENTRADAS DIFERENTES` | Copie de novo o parquet da máquina de origem; confira o `sha256sum`. |
| `ModuleNotFoundError: src` | Rode da raiz do projeto e **sem** `python -I`. |
| Falta de memória | Reduza `experimento.n_jobs` (cada processo usa ~0,7 GB + o tamanho dos modelos). |
| Execução interrompida | Rode `./run_all.sh` de novo (retomável). |
| `[S5] NÃO reproduz` | Pare e investigue antes de seguir (versão de pandas/numpy diferente?). |
