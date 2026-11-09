import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PREPARAÇÃO E LIMPEZA DE VARIÁVEIS (FASE 1)
# ==============================================================================
# Assumindo que seu DataFrame original se chama 'df'

df = pd.read_parquet("dados/processados/gercon_poa_pronto.parquet", engine='pyarrow')

df_clean = df.copy()

# Veja primeiro quais são os dois nomes que estão na sua coluna:
print("Categorias presentes:", df_clean["situacaoagenda"].unique())

# Classe positiva (1) = FALTANTE (desfecho de interesse, classe minoritária).
# Classe negativa (0) = REALIZADA.
categoria_positiva = "FALTANTE"
categoria_negativa = "REALIZADA"

# Garante que não existam outras categorias (ou NAs) sendo codificadas como 0
valores_observados = set(df_clean["situacaoagenda"].dropna().astype(str).unique())
if (
    df_clean["situacaoagenda"].isna().any()
    or not valores_observados <= {categoria_positiva, categoria_negativa}
):
  raise ValueError(
      "situacaoagenda contém valores inesperados ou NAs:"
      f" {valores_observados}"
  )

y = (df_clean["situacaoagenda"] == categoria_positiva).astype(int).values

print(f"Classe Positiva (1) definida como: '{categoria_positiva}' | Proporção de 1s: {y.mean():.2%}")

# 1.1 Truncar CID para os 3 primeiros dígitos
df_clean["cid_3dig"] = (
    df_clean["cid"].astype(str).str.upper().str.strip().str[:3]
)

# 1.2 Criar varaivel de interação entre cid_3dig e especialidademae
df_clean["cid_especialidade"] = (
    df_clean["cid_3dig"].astype(str)
    + "_"
    + df_clean["especialidademae"].astype(str)
)

# 1.3 Remover idpaciente, cid original e as variáveis de distância colineares
cols_para_remover = [
    "situacaoagenda",
    "idpaciente",
    "cid",
    "tempo_caminhando",
    "distancia_km_trans_publico",  # nome exato da sua tabela (17)
    "tempo_desloc_trans_publico",
    "baldiacoes",
    "cnes_solicitante",
    "cnesexecutante"
]

X = df_clean.drop(
    columns=[c for c in cols_para_remover if c in df_clean.columns]
)

# ==============================================================================
# 2. DIVISÃO CORRETA: APENAS TREINO (80%) E TESTE FINAL (20%)
# (X_train = treino+validação; a validação nasce via K-Fold dentro dele)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(
    f"Desenvolvimento/Treino (para K-Fold): {X_train.shape[0]} linhas (80%) |"
    f" Teste Final (Holdout): {X_test.shape[0]} linhas (20%)"
)

# ==============================================================================
# 3. FUNÇÃO CONSTRUTORA DO PIPELINE (SEM BALANCEAMENTO / COM SMOTE+TOMEK)
# ==============================================================================

class KNNCategoricalImputer(BaseEstimator, TransformerMixin):
  """Imputa NAs de uma coluna categórica via KNN (k vizinhos) usando
  outras colunas categóricas como preditoras (one-hot + distância euclidiana).
  O ajuste usa apenas as linhas de treino com o alvo observado."""

  def __init__(self, coluna_alvo, colunas_preditoras, n_neighbors=5):
    self.coluna_alvo = coluna_alvo
    self.colunas_preditoras = colunas_preditoras
    self.n_neighbors = n_neighbors

  def fit(self, X, y=None):
    obs = X[self.coluna_alvo].notna().to_numpy()
    self.ohe_ = OneHotEncoder(handle_unknown="ignore")
    Z = self.ohe_.fit_transform(
        X.loc[obs, self.colunas_preditoras].astype(str)
    )
    self.knn_ = KNeighborsClassifier(
        n_neighbors=self.n_neighbors, algorithm="brute"
    )
    self.knn_.fit(Z, X.loc[obs, self.coluna_alvo].astype(str).to_numpy())
    return self

  def transform(self, X):
    X = X.copy()
    col = X[self.coluna_alvo].astype(object)
    faltando = col.isna().to_numpy()
    if faltando.any():
      Z = self.ohe_.transform(
          X.loc[faltando, self.colunas_preditoras].astype(str)
      )
      col.loc[faltando] = self.knn_.predict(Z)
    X[self.coluna_alvo] = col
    return X


def construir_pipeline(usar_smotetomek: bool, modelo_nome: str):
  # Listas de colunas por grupo
  cols_num = [
      "diasfilaespera",
      "idade",
      "distancia_km",
      "quant_bus",
      "censo_favela_comunidade",
      "censo_taxa_alfabetizacao",
      "censo_renda_media_winsor",
  ]

  cols_target_enc = [
      "cid_3dig",
      "especialidademae",
      "censo_distrito_sanitario",
      "cid_especialidade",
  ]

  cols_onehot_enc = [
      "corregulador",
      "possuiunidadeindicada",
      "censo_coordenadoria_saude",
  ]

  # Etapa 1a: imputação KNN (k=5) dos NAs de corregulador
  imputador_corregulador = KNNCategoricalImputer(
      coluna_alvo="corregulador",
      colunas_preditoras=["cid_3dig", "especialidademae"],
      n_neighbors=5,
  )

  # Etapa 1b: transformações por grupo de variáveis
  transformador = ColumnTransformer(
      transformers=[
          ("num", MinMaxScaler(), cols_num),
          (
              "te",
              TargetEncoder(
                  smooth="auto", cv=5, target_type="binary", random_state=42
              ),
              cols_target_enc,
          ),
          (
              "ohe",
              OneHotEncoder(
                  drop="if_binary",
                  handle_unknown="ignore",
                  sparse_output=False,
              ),
              cols_onehot_enc,
          ),
      ],
      remainder="drop",
  )

  # Definição dos 2 Modelos Simples e seus espaços de hiperparâmetros
  if modelo_nome == "LogisticRegression":
    clf = LogisticRegression(max_iter=500, random_state=42)
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
  elif modelo_nome == "DecisionTree":
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        "clf__max_depth": [5, 10, 15, None],
        "clf__min_samples_leaf": [20, 50, 100],
    }

  # Montagem da sequência:
  # Imputação -> Transformação -> (UnderSampler -> SMOTETomek) -> Modelo
  steps = [
      ("imp_corregulador", imputador_corregulador),
      ("transf", transformador),
  ]

  if usar_smotetomek:
    # Sequência Híbrida: Primeiro reduz majoritária (Under), depois cria sintéticos e limpa fronteira (SMOTE + Tomek)
    steps.append((
        "under",
        RandomUnderSampler(sampling_strategy=0.5, random_state=42),
    ))
    steps.append((
        "smote_tomek",
        SMOTETomek(
            sampling_strategy=0.8,
            tomek=TomekLinks(n_jobs=-1),
            random_state=42,
        ),
    ))

  steps.append(("clf", clf))

  return ImbPipeline(steps), param_grid


# ==============================================================================
# 4. FUNÇÕES DE MÉTRICAS E ESCOLHA DE LIMIAR
# ==============================================================================


def limiar_por_mcc(y_true, y_prob):
  """Limiar (grade 0,05-0,95) que maximiza o MCC."""
  grade = np.linspace(0.05, 0.95, 91)
  mccs = [matthews_corrcoef(y_true, y_prob >= t) for t in grade]
  return float(grade[int(np.argmax(mccs))])


def calcular_todas_metricas(y_true, y_prob, limiar=0.5):
  y_pred = (y_prob >= limiar).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
  especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
  vp_positivo = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision (VPP)
  vp_negativo = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # VPN

  return {
      "Limiar": limiar,
      "ROC_AUC": roc_auc_score(y_true, y_prob),
      "PR_AUC": average_precision_score(y_true, y_prob),
      "F1_Score": f1_score(y_true, y_pred),
      "Sensibilidade_Recall": sensibilidade,
      "Especificidade": especificidade,
      "Precisao_VPP": vp_positivo,
      "VPN": vp_negativo,
      "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
      "Acuracia_Global": accuracy_score(y_true, y_pred),
      "MCC": matthews_corrcoef(y_true, y_pred),
      "G_Mean": np.sqrt(sensibilidade * especificidade),
      "Brier_Score": brier_score_loss(y_true, y_prob),
      "TN": int(tn),
      "FP": int(fp),
      "FN": int(fn),
      "TP": int(tp),
  }


# ==============================================================================
# 5. VALIDAÇÃO (K-FOLD DENTRO DE TREINO+VALIDAÇÃO) -> ESCOLHA DE HIPERPARÂMETROS
# ==============================================================================
# Fluxo:
#   (a) Para cada cenário (sem balanceamento / SMOTETomek) e modelo, testa TODAS
#       as combinações de hiperparâmetros em K folds de X_train (treino+validação).
#   (b) Escolhe a melhor combinação pelo ROC AUC médio de validação.
#   (c) Escolhe o limiar de decisão (MCC) com as probabilidades de validação
#       (out-of-fold) da combinação escolhida.
#   (d) Reajusta a melhor combinação em 100% de X_train e avalia UMA vez no teste.
# Tudo é gravado em disco (pasta DIR_SAIDA) para rastrear cada etapa.

DIR_SAIDA = Path("resultados_modelagem")
DIR_SAIDA.mkdir(exist_ok=True)

N_FOLDS = 5
cv_validacao = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits_validacao = list(cv_validacao.split(X_train, y_train))

cenarios = {
    "sem_balanceamento": False,
    "smotetomek": True,
}
modelos = ["LogisticRegression", "DecisionTree"]


def ajustar_e_prever_fold(cenario, usar_smote, mod_nome, params, fold, tr, va):
  """Ajusta uma combinação de hiperparâmetros em um fold e prevê a validação."""
  pipe, _ = construir_pipeline(usar_smote, mod_nome)
  pipe.set_params(**params)
  t0 = time.time()
  pipe.fit(X_train.iloc[tr], y_train[tr])
  prob = pipe.predict_proba(X_train.iloc[va])[:, 1]
  return {
      "cenario": cenario,
      "modelo": mod_nome,
      "params": params,
      "fold": fold,
      "va_idx": va,
      "prob": prob,
      "tempo_s": time.time() - t0,
  }


# --- 5.1 Lista de tarefas (cenário x modelo x combinação x fold) ---
tarefas = []
for cenario, usar_smote in cenarios.items():
  for mod_nome in modelos:
    _, grade = construir_pipeline(usar_smote, mod_nome)
    for params in ParameterGrid(grade):
      for fold, (tr, va) in enumerate(splits_validacao, start=1):
        tarefas.append((cenario, usar_smote, mod_nome, params, fold, tr, va))

print(f"Validação: {len(tarefas)} ajustes (cenário x modelo x hiperparâmetros x fold)")

saidas = Parallel(n_jobs=-1, verbose=5)(
    delayed(ajustar_e_prever_fold)(*t) for t in tarefas
)

# --- 5.2 Métricas por combinação e fold ---
df_saidas = pd.DataFrame(saidas)
df_saidas["params_str"] = df_saidas["params"].astype(str)

linhas_folds = []
linhas_resumo = []
probs_oof = {}  # (cenario, modelo, params_str) -> vetor de prob. out-of-fold

for (cenario, mod_nome, params_str), g in df_saidas.groupby(
    ["cenario", "modelo", "params_str"], sort=False
):
  g = g.sort_values("fold")
  # Probabilidades out-of-fold (cada linha de X_train prevista 1x)
  prob_oof = np.zeros(len(y_train))
  for _, r in g.iterrows():
    prob_oof[r["va_idx"]] = r["prob"]
  probs_oof[(cenario, mod_nome, params_str)] = prob_oof

  limiar = limiar_por_mcc(y_train, prob_oof)

  metricas_folds = []
  for _, r in g.iterrows():
    y_va = y_train[r["va_idx"]]
    m = calcular_todas_metricas(y_va, r["prob"], limiar)
    metricas_folds.append(m)
    linhas_folds.append({
        "etapa": "validacao",
        "cenario": cenario,
        "modelo": mod_nome,
        "params": params_str,
        "fold": r["fold"],
        "n_validacao": len(y_va),
        "prevalencia_validacao": y_va.mean(),
        "tempo_ajuste_s": round(r["tempo_s"], 2),
        **m,
    })

  df_m = pd.DataFrame(metricas_folds)
  resumo = {
      "etapa": "validacao_resumo",
      "cenario": cenario,
      "modelo": mod_nome,
      "params": params_str,
      "limiar_mcc_oof": limiar,
  }
  for c in df_m.columns.drop(["Limiar", "TN", "FP", "FN", "TP"]):
    resumo[f"{c}_mean"] = df_m[c].mean()
    resumo[f"{c}_sd"] = df_m[c].std()
  linhas_resumo.append(resumo)

df_validacao_folds = pd.DataFrame(linhas_folds)
df_validacao_resumo = pd.DataFrame(linhas_resumo)

# Ranking dentro de cada cenário x modelo, pelo ROC AUC médio de validação
df_validacao_resumo["rank_roc_auc"] = (
    df_validacao_resumo.groupby(["cenario", "modelo"])["ROC_AUC_mean"]
    .rank(ascending=False, method="first")
    .astype(int)
)
df_validacao_resumo["selecionado"] = df_validacao_resumo["rank_roc_auc"] == 1

df_validacao_folds.to_csv(DIR_SAIDA / "01_validacao_folds.csv", index=False)
df_validacao_resumo.to_csv(DIR_SAIDA / "02_validacao_resumo.csv", index=False)

display(
    df_validacao_resumo.sort_values(
        ["cenario", "modelo", "rank_roc_auc"]
    )[
        [
            "cenario",
            "modelo",
            "params",
            "rank_roc_auc",
            "ROC_AUC_mean",
            "ROC_AUC_sd",
            "PR_AUC_mean",
            "MCC_mean",
            "limiar_mcc_oof",
            "selecionado",
        ]
    ]
)

# ==============================================================================
# 6. TESTE FINAL: REAJUSTE DA MELHOR COMBINAÇÃO EM TODO X_train + AVALIAÇÃO
# ==============================================================================
linhas_teste = []
previsoes_teste = {"y_test": y_test}
modelos_finais = {}

for _, sel in df_validacao_resumo[df_validacao_resumo["selecionado"]].iterrows():
  cenario, mod_nome = sel["cenario"], sel["modelo"]
  # recupera o dicionário de parâmetros original
  params = df_saidas.loc[
      (df_saidas["cenario"] == cenario)
      & (df_saidas["modelo"] == mod_nome)
      & (df_saidas["params_str"] == sel["params"]),
      "params",
  ].iloc[0]

  print(f"Teste final: {cenario} | {mod_nome} | {params}")
  pipe, _ = construir_pipeline(cenarios[cenario], mod_nome)
  pipe.set_params(**params)
  t0 = time.time()
  pipe.fit(X_train, y_train)
  tempo = time.time() - t0

  limiar = sel["limiar_mcc_oof"]  # definido SÓ com dados de validação

  prob_test = pipe.predict_proba(X_test)[:, 1]
  m_test = calcular_todas_metricas(y_test, prob_test, limiar)

  linhas_teste.append({
      "etapa": "teste",
      "cenario": cenario,
      "modelo": mod_nome,
      "params": sel["params"],
      "n_treino": len(y_train),
      "n_teste": len(y_test),
      "prevalencia_teste": y_test.mean(),
      "tempo_ajuste_s": round(tempo, 2),
      "ROC_AUC_validacao_mean": sel["ROC_AUC_mean"],
      "ROC_AUC_validacao_sd": sel["ROC_AUC_sd"],
      **m_test,
  })
  previsoes_teste[f"prob_{cenario}_{mod_nome}"] = prob_test
  modelos_finais[f"{cenario}_{mod_nome}"] = pipe

df_teste = pd.DataFrame(linhas_teste)
df_teste.to_csv(DIR_SAIDA / "03_teste_resultados.csv", index=False)
pd.DataFrame(previsoes_teste).to_parquet(
    DIR_SAIDA / "04_teste_probabilidades.parquet", index=False
)
joblib.dump(modelos_finais, DIR_SAIDA / "05_modelos_finais.joblib")

# Comparação validação x teste (tabela única)
df_resultados = df_teste[
    [
        "cenario",
        "modelo",
        "params",
        "ROC_AUC_validacao_mean",
        "ROC_AUC_validacao_sd",
        "ROC_AUC",
        "PR_AUC",
        "Limiar",
        "Sensibilidade_Recall",
        "Especificidade",
        "Precisao_VPP",
        "VPN",
        "MCC",
        "Balanced_Acc",
        "Brier_Score",
    ]
].rename(columns={"ROC_AUC": "ROC_AUC_teste", "PR_AUC": "PR_AUC_teste"})
df_resultados.to_csv(DIR_SAIDA / "06_comparacao_validacao_teste.csv", index=False)
display(df_resultados)

# ==============================================================================
# 7. REGISTRO DA EXECUÇÃO (CONFIGURAÇÃO PARA REPRODUTIBILIDADE)
# ==============================================================================
config = {
    "data_execucao": datetime.now().isoformat(timespec="seconds"),
    "arquivo_dados": "dados/processados/gercon_poa_pronto.parquet",
    "classe_positiva": categoria_positiva,
    "prevalencia_total": float(y.mean()),
    "n_total": int(len(y)),
    "n_treino_validacao": int(len(y_train)),
    "n_teste": int(len(y_test)),
    "prevalencia_treino": float(y_train.mean()),
    "prevalencia_teste": float(y_test.mean()),
    "split_treino_teste": {"test_size": 0.20, "stratify": True, "random_state": 42},
    "validacao": {
        "tipo": "StratifiedKFold",
        "n_folds": N_FOLDS,
        "random_state": 42,
        "criterio_selecao": "ROC_AUC medio de validacao",
        "limiar": "maximiza MCC nas probabilidades out-of-fold de validacao",
    },
    "cenarios": cenarios,
    "modelos": {
        m: {
            k: [str(v) for v in vs]
            for k, vs in construir_pipeline(False, m)[1].items()
        }
        for m in modelos
    },
    "pipeline": {
        "imputacao": "KNN k=5 em corregulador (cid_3dig, especialidademae)",
        "numericas (MinMax)": [
            "diasfilaespera", "idade", "distancia_km", "quant_bus",
            "censo_favela_comunidade", "censo_taxa_alfabetizacao",
            "censo_renda_media_winsor",
        ],
        "target_encoding": [
            "cid_3dig", "especialidademae", "censo_distrito_sanitario",
            "cid_especialidade",
        ],
        "onehot": [
            "corregulador", "possuiunidadeindicada", "censo_coordenadoria_saude",
        ],
        "smotetomek": {
            "under_sampling_strategy": 0.5,
            "smote_sampling_strategy": 0.8,
        },
    },
    "colunas_removidas": cols_para_remover,
    "versoes": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "imblearn": imblearn.__version__,
    },
    "arquivos_saida": {
        "01_validacao_folds.csv": "métricas por combinação de hiperparâmetros e fold",
        "02_validacao_resumo.csv": "média/DP por combinação, ranking e seleção",
        "03_teste_resultados.csv": "métricas finais no teste (melhor combinação)",
        "04_teste_probabilidades.parquet": "y_test e probabilidades previstas",
        "05_modelos_finais.joblib": "pipelines ajustados em treino+validação",
        "06_comparacao_validacao_teste.csv": "validação x teste",
    },
}
with open(DIR_SAIDA / "00_config_execucao.json", "w", encoding="utf-8") as f:
  json.dump(config, f, ensure_ascii=False, indent=2, default=str)

print(f"Resultados gravados em: {DIR_SAIDA.resolve()}")
