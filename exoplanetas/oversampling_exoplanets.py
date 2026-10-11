import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN


# 1. CARGAR DATASET
df = pd.read_csv(
    "/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv"
)

# 2. TARGET Y FEATURES
y = df["habitable_zone_flag"]

# Variables predictoras
X = df.drop(columns=["habitable_zone_flag"])

# División estratificada 70/30

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.30,

    stratify=y,

    random_state=42

)

print("Train:", X_train.shape)
print("Test :", X_test.shape)
print("\nDistribución entrenamiento:")
print(y_train.value_counts())
print("\nDistribución prueba:")
print(y_test.value_counts())

# KNN IMPUTER

# Variables numéricas

numeric_features = X_train.select_dtypes(

    include=["int64", "float64"]

).columns

# Variables categóricas

categorical_features = X_train.select_dtypes(

    include=["object", "bool", "category"]

).columns

# PREPROCESAMIENTO

numeric_transformer = Pipeline(

    steps=[

        ("imputer", KNNImputer(n_neighbors=5)),

        ("scaler", StandardScaler())

    ]

)

categorical_transformer = Pipeline(

    steps=[

        (

            "encoder",

            OneHotEncoder(

                handle_unknown="ignore"

            )

        )

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_transformer,

            numeric_features

        ),

        (

            "cat",

            categorical_transformer,

            categorical_features

        )

    ]

)

# AJUSTAR SOLO CON TRAIN

X_train = preprocessor.fit_transform(X_train)

# TRANSFORMAR TEST

X_test = preprocessor.transform(X_test)

# VERIFICACIÓN

print("\n===================================")
print("Train:", X_train.shape)
print("Test :", X_test.shape)
print("===================================")

###############################################################
# RANDOM OVERSAMPLING
###############################################################

ros = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

X_train_ros, y_train_ros = ros.fit_resample(
    X_train,
    y_train
)


# VERIFICACIÓN

print("\n===================================")
print("Antes de Random Oversampling")
print(y_train.value_counts())

print("\nDespués de Random Oversampling")
print(y_train_ros.value_counts())

print("\nDimensiones")

print("X_train:", X_train.shape)
print("X_train_ros:", X_train_ros.shape)

print("X_test:", X_test.shape)
print("===================================")

###############################################################
# SMOTE
###############################################################

print("Antes de SMOTE")
print(y_train.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nDespués de SMOTE")
print(y_train_smote.value_counts())

print("\nDimensiones")
print("X_train:", X_train.shape)
print("X_train_smote:", X_train_smote.shape)
print("X_test:", X_test.shape)
print("===================================")

###############################################################
# BORDERLINE-SMOTE
###############################################################

print("Antes de Borderline-SMOTE")
print(y_train.value_counts())

borderline = BorderlineSMOTE(
    random_state=42,
    kind="borderline-1"
)

X_train_border, y_train_border = borderline.fit_resample(
    X_train,
    y_train
)

print("\nDespués de Borderline-SMOTE")
print(y_train_border.value_counts())

print("\nDimensiones")
print("X_train:", X_train.shape)
print("X_train_border:", X_train_border.shape)
print("X_test:", X_test.shape)
print("===================================")


###############################################################
# ADASYN
###############################################################

print("Antes de ADASYN")
print(y_train.value_counts())

adasyn = ADASYN(
    random_state=42
)

X_train_adasyn, y_train_adasyn = adasyn.fit_resample(
    X_train,
    y_train
)

print("\nDespués de ADASYN")
print(y_train_adasyn.value_counts())

print("\nDimensiones")
print("X_train:", X_train.shape)
print("X_train_adasyn:", X_train_adasyn.shape)
print("X_test:", X_test.shape)
print("===================================")

# MODELOS DE CLASIFICACIÓN

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

}

# EVALUACIÓN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

results = []

for name, model in models.items():

    print(f"\nEntrenando {name}...")

    model.fit(
        X_train_adasyn,
        y_train_adasyn
    )

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)

    results.append({

        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred
            ),

        "F1":
            f1_score(
                y_test,
                y_pred
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                y_prob
            )

    })

results = pd.DataFrame(results)

print("\n======================================")
print(results)
print("======================================")