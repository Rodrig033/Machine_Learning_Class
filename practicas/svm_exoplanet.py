import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Cargar dataset
df = pd.read_csv(
    "/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv"
)

# Target y features
y = df["habitable_zone_flag"]
X = df.drop("habitable_zone_flag", axis=1)

# Solo columnas numéricas
X = X.select_dtypes(include=[np.number])

# Limpiar NaN
X = X.fillna(X.mean())

# Kernels
kernels = {
    "Linear": "linear",
    "Polynomial": "poly",
    "RBF": "rbf",
    "Sigmoid": "sigmoid"
}

# Split 70/30
print(" SPLIT 70/30 ")

split_results = []
best_split_f1 = 0
best_split_model = None
best_split_y_test = None
best_split_kernel = None

best_f1_split = 0
best_cm_split = None
best_kernel_split = None

best_cv_model = None
best_cv_X_test = None
best_cv_y_test = None

for kernel_name, kernel_type in kernels.items():

    print(f"\n########## {kernel_name} ##########")

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for rs in range(6):

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=rs
        )

        model = Pipeline([
            ("scaler", StandardScaler()),

            ("svc", SVC(
                kernel=kernel_type,
                C=2,
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

        # Mostrar
        print(f"\nRandom State: {rs}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        # Mejor matriz de confusión
        if f1 > best_split_f1:

            best_split_f1 = f1
            best_split_model = model

            best_split_X_test = X_test
            best_split_y_test = y_test

            best_split_kernel = kernel_name

            best_cm_split = confusion_matrix(y_test, y_pred)


    # Estadísticas

    print("\n----- Estadísticas -----")

    print(f"Accuracy Mean : {np.mean(accuracies):.4f}")
    print(f"Accuracy Median : {np.median(accuracies):.4f}")
    print(f"Accuracy Std : {np.std(accuracies):.4f}")
    print(f"Max    : {np.max(accuracies):.4f}")
    print(f"Min    : {np.min(accuracies):.4f}")

    print(f"F1 Mean : {np.mean(f1s):.4f}")
    print(f"F1 Median : {np.median(f1s):.4f}")
    print(f"Max    : {np.max(f1s):.4f}")
    print(f"Min    : {np.min(f1s):.4f}")
    print(f"F1 Std : {np.std(f1s):.4f}")

    print(f"Precision Mean : {np.mean(precisions):.4f}")
    print(f"Precision Median : {np.median(precisions):.4f}")
    print(f"Max    : {np.max(precisions):.4f}")
    print(f"Min    : {np.min(precisions):.4f}")
    print(f"Precision Std : {np.std(precisions):.4f}")

    print(f"Recall Mean : {np.mean(recalls):.4f}")
    print(f"Recall Median : {np.median(recalls):.4f}")
    print(f"Max    : {np.max(recalls):.4f}")
    print(f"Min    : {np.min(recalls):.4f}")
    print(f"Recall Std : {np.std(recalls):.4f}")


    split_results.append([
        kernel_name,
        np.mean(accuracies),
        np.median(accuracies),
        np.std(accuracies),
        np.mean(f1s),
        np.median(f1s),
        np.std(f1s)
    ])


# 5. CV 10-FOLD


print("\n==============================")
print(" STRATIFIED 10-FOLD ")
print("==============================")

best_f1_cv = 0
best_cm_cv = None
best_kernel_cv = None

for kernel_name, kernel_type in kernels.items():

    print(f"\n########## {kernel_name} ##########")

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    fold = 1

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),

            ("svc", SVC(
                kernel=kernel_type,
                C=1,
                class_weight="balanced",
                random_state=42
            ))
        ])

        # Entrenar
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

        print(f"\nFold {fold}")

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        # Mejor matriz de confusión
        if f1 > best_f1_cv:

            best_f1_cv = f1

            best_cm_cv = confusion_matrix(
                y_test,
                y_pred
            )

            best_kernel_cv = kernel_name

            best_cv_model = model
            best_cv_X_test = X_test
            best_cv_y_test = y_test
    
    # Promedios CV

    print("\n----- Promedios -----")

    print(f"Accuracy : {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall   : {np.mean(recalls):.4f}")
    print(f"F1-score : {np.mean(f1s):.4f}")

    print("\n----- Mediana -----")
    print(f"Accuracy : {np.median(accuracies):.4f}")
    print(f"Precision: {np.median(precisions):.4f}")
    print(f"Recall   : {np.median(recalls):.4f}")
    print(f"F1-score : {np.median(f1s):.4f}")

    print("\n----- Max -----")
    print(f"Accuracy : {np.max(accuracies):.4f}")
    print(f"Precision: {np.max(precisions):.4f}")
    print(f"Recall   : {np.max(recalls):.4f}")
    print(f"F1-score : {np.max(f1s):.4f}")

    print("\n----- Min -----")
    print(f"Accuracy : {np.min(accuracies):.4f}")
    print(f"Precision: {np.min(precisions):.4f}")
    print(f"Recall   : {np.min(recalls):.4f}")
    print(f"F1-score : {np.min(f1s):.4f}")

    print("\n----- Std -----")
    print(f"Accuracy : {np.std(accuracies):.4f}")
    print(f"Precision: {np.std(precisions):.4f}")
    print(f"Recall   : {np.std(recalls):.4f}")
    print(f"F1-score : {np.std(f1s):.4f}")

# Mejores matrices de confusión
print("\n==============================")
print(" MEJOR MATRIZ SPLIT ")
print("==============================")

print(f"Kernel: {best_split_kernel}")
print(best_cm_split)

disp = ConfusionMatrixDisplay(
    confusion_matrix=best_cm_split,
    display_labels=["No Habitable", "Habitable"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title(f"Best Split Confusion Matrix - {best_split_kernel}")
plt.savefig("best_split_confusion_matrix.png")
plt.show()

print("\n==============================")
print(" MEJOR MATRIZ CV ")
print("==============================")

print(f"Kernel: {best_kernel_cv}")
print(best_cm_cv)

disp = ConfusionMatrixDisplay(
    confusion_matrix=best_cm_cv,
    display_labels=["No Habitable", "Habitable"]
)

disp.plot(cmap="Greens", values_format="d")

plt.title(f"Best CV Confusion Matrix - {best_kernel_cv}")
plt.savefig("best_cv_confusion_matrix.png")
plt.show()
