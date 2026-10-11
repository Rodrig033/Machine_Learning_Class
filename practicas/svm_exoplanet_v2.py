import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. CARGAR DATASET

df = pd.read_csv(
    "/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv"
)

# 2. TARGET Y FEATURES

y = df["habitable_zone_flag"]

X = df.drop("habitable_zone_flag", axis=1)

# Solo columnas numéricas
X = X.select_dtypes(include=[np.number])

# Limpiar NaN
X = X.fillna(X.mean())

# 3. KERNELS

kernels = {
    "Linear": "linear",
    "Polynomial": "poly",
    "RBF": "rbf",
    "Sigmoid": "sigmoid"
}


# 4. LISTAS PARA DATAFRAMES

split_details = []
split_summary = []

cv_details = []
cv_summary = []

# 5. SPLIT 70/30

print("\n==============================")
print(" SPLIT 70/30 ")
print("==============================")

for kernel_name, kernel_type in kernels.items():

    print(f"\n########## {kernel_name} ##########")

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for rs in range(6):

        # Split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=rs
        )

        # Modelo

        model = Pipeline([
            ("scaler", StandardScaler()),

            ("svc", SVC(
                kernel=kernel_type,
                C=0.5,
                class_weight="balanced",
                random_state=42
            ))
        ])

        # Entrenamiento

        model.fit(X_train, y_train)

        # Predicción

        y_pred = model.predict(X_test)

        # Métricas

        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="binary"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="binary"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="binary"
        )

        # Guardar para estadísticas

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Guardar detalle

        split_details.append({
            "Kernel": kernel_name,
            "RandomState": rs,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

    # RESUMEN SPLIT

    metrics_dict = {
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1-score": f1s
    }

    for metric_name, values in metrics_dict.items():

        split_summary.append({
            "Kernel": kernel_name,
            "Metric": metric_name,
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Max": np.max(values),
            "Min": np.min(values),
            "Std": np.std(values)
        })

# 6. STRATIFIED 10-FOLD CV

print("\n==============================")
print(" STRATIFIED 10-FOLD ")
print("==============================")

for kernel_name, kernel_type in kernels.items():

    print(f"\n########## {kernel_name} ##########")

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    fold = 1

    for train_idx, test_idx in skf.split(X, y):

        # Fold split

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Modelo

        model = Pipeline([
            ("scaler", StandardScaler()),

            ("svc", SVC(
                kernel=kernel_type,
                C=1,
                class_weight="balanced",
                random_state=42
            ))
        ])

        # Entrenamiento

        model.fit(X_train, y_train)

        # Predicción

        y_pred = model.predict(X_test)

        # Métricas

        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="binary"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="binary"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="binary"
        )

        # Guardar métricas

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Guardar detalle

        cv_details.append({
            "Kernel": kernel_name,
            "Fold": fold,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

        fold += 1

    # RESUMEN CV

    metrics_dict = {
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1-score": f1s
    }

    for metric_name, values in metrics_dict.items():

        cv_summary.append({
            "Kernel": kernel_name,
            "Metric": metric_name,
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Max": np.max(values),
            "Min": np.min(values),
            "Std": np.std(values)
        })

# 7. DATAFRAMES

df_split_details = pd.DataFrame(split_details)

df_split_summary = pd.DataFrame(split_summary)

df_cv_details = pd.DataFrame(cv_details)

df_cv_summary = pd.DataFrame(cv_summary)

# 8. EXPORTAR A EXCEL

output_file = "results.xlsx"
df_split_details = df_split_details.round(4)
df_split_summary = df_split_summary.round(4)
df_cv_details = df_cv_details.round(4)
df_cv_summary = df_cv_summary.round(4)

with pd.ExcelWriter(output_file) as writer:

    df_split_details.to_excel(
        writer,
        sheet_name="Split_Details",
        index=False
    )

    df_split_summary.to_excel(
        writer,
        sheet_name="Split_Summary",
        index=False
    )

    df_cv_details.to_excel(
        writer,
        sheet_name="CV_Details",
        index=False
    )

    df_cv_summary.to_excel(
        writer,
        sheet_name="CV_Summary",
        index=False
    )

# 9. MENSAJE FINAL

print("\n===================================")
print(" ARCHIVO EXCEL GENERADO ")
print("===================================")

print(f"\nArchivo guardado como: {output_file}")