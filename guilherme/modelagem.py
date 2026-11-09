import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Seed global para reprodutibilidade
SEED = 42
np.random.seed(SEED)


df = pd.read_parquet("dados/processados/gercon_poa_pronto.parquet", engine='pyarrow')

df.info()
df.isna().sum().sort_values(ascending=False)

# Preparação Manual

# Remover colunas irrelevantes
df.drop(['idpaciente'], axis=1, inplace=True)

# Transformar target em binário
df['situacaoagenda'] = df['situacaoagenda'].map({'REALIZADA': 1, 'FALTANTE': 0})

#Separar em features e target
X = df.drop('situacaoagenda', axis=1)
y = df['situacaoagenda']

# Identificar colunas categóricas e numéricas
cat_cols = X.select_dtypes(include=['category', 'str']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Shape do dataset: {df.shape}")
print(f"Colunas numéricas: {list(num_cols)}")
print(f"Colunas categóricas: {list(cat_cols)}")


# Divisão treino/teste estratificada
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Treino: {X_train_raw.shape[0]} amostras | Teste: {X_test_raw.shape[0]} amostras")
print(f"Distribuição no treino:\n{y_train.value_counts(normalize=True).round(3)}")

def make_base_preprocessor():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('numerical', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols)
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

pipeline_baseline = Pipeline([
    ('data_transform', make_base_preprocessor()),
    ('classifier', KNeighborsClassifier(n_neighbors=11))
])

scores_nosampling = cross_val_score(pipeline_baseline, X_train_raw, y_train,
                                    scoring='f1', cv=cv, n_jobs=-1)
print(f"[Sem re-amostragem] F1 médio: {scores_nosampling.mean():.4f} ± {scores_nosampling.std():.4f}")
