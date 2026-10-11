import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from yellowbrick.classifier import ConfusionMatrix

# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n===================================")
print(" DATASET INFORMATION ")
print("===================================")

df = pd.read_csv(
    "/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv"
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# 2. TARGET
# =========================================================

y = df["habitable_zone_flag"]

X = df.drop(
    "habitable_zone_flag",
    axis=1
)

# =========================================================
# 3. ENCODE CATEGORICAL FEATURES
# =========================================================

categorical_cols = X.select_dtypes(
    include=["object", "bool"]
).columns

label_encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

    label_encoders[col] = le

# =========================================================
# 4. DATA CLEANING
# =========================================================

print("\n===================================")
print(" DATA CLEANING ")
print("===================================")

imputer = SimpleImputer(strategy="mean")

X_imputed = imputer.fit_transform(X)

# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

print("\n===================================")
print(" FEATURE ENGINEERING ")
print("===================================")

X_fe = pd.DataFrame(
    X_imputed,
    columns=X.columns
)

# Nueva característica:
# Relación masa/radio

X_fe["mass_radius_ratio"] = (
    X_fe["planet_mass_earth"] /
    (X_fe["planet_radius_earth"] + 1e-5)
)

# Nueva característica:
# Temperatura relativa

X_fe["temp_ratio"] = (
    X_fe["equilibrium_temp_k"] /
    (X_fe["star_temp_k"] + 1e-5)
)

# =========================================================
# 6. SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_fe)

# =========================================================
# 7. PCA
# =========================================================

print("\n===================================")
print(" PCA ")
print("===================================")

# Conserva 95% de la varianza

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("\nOriginal Shape:")
print(X_scaled.shape)

print("\nReduced Shape:")
print(X_pca.shape)

print("\nExplained Variance:")
print(np.sum(pca.explained_variance_ratio_))

# =========================================================
# 8. t-SNE (SOLO VISUALIZACIÓN)
# =========================================================

print("\n===================================")
print(" t-SNE VISUALIZATION ")
print("===================================")

# Tomar muestra para acelerar
sample_size = 1500

indices = np.random.choice(
    len(X_pca),
    sample_size,
    replace=False
)

X_sample = X_pca[indices]
y_sample = y.iloc[indices]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_sample)

# =========================================================
# 9. t-SNE PLOT
# =========================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_sample,
    alpha=0.7
)

plt.title("t-SNE Visualization")

plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.colorbar(scatter)

plt.show()

# =========================================================
# 10. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# =========================================================
# 11. SVM MODEL
# =========================================================

print("\n===================================")
print(" TRAINING SVM ")
print("===================================")

model = SVC(
    kernel="linear",
    C=1,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 12. PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# 13. METRICS
# =========================================================

acc = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n===================================")
print(" FINAL METRICS ")
print("===================================")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\n===================================")
print(" CLASSIFICATION REPORT ")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===================================")
print(" CONFUSION MATRIX ")
print("===================================")

print(cm)

visualizer = ConfusionMatrix(
    model,
    classes=["No Habitable", "Habitable"],
    cmap="Blues",
    percent=True
)

visualizer.score(X_test, y_test)

visualizer.show(
    outpath="improved_confusion_matrix.png"
)

# =========================================================
# 16. STRATIFIED 10-FOLD
# =========================================================

print("\n===================================")
print(" STRATIFIED 10-FOLD ")
print("===================================")

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

for train_idx, test_idx in skf.split(X_pca, y):

    X_train = X_pca[train_idx]
    X_test = X_pca[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = SVC(
        kernel="linear",
        C=1,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    accuracies.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"\nFold {fold}")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    fold += 1

# =========================================================
# 17. FINAL CV METRICS
# =========================================================

print("\n===================================")
print(" CV FINAL RESULTS ")
print("===================================")

print(f"Accuracy Mean : {np.mean(accuracies):.4f}")
print(f"Precision Mean: {np.mean(precisions):.4f}")
print(f"Recall Mean   : {np.mean(recalls):.4f}")
print(f"F1 Mean       : {np.mean(f1s):.4f}")

print("\n===================================")
print(" STANDARD DEVIATION ")
print("===================================")

print(f"Accuracy Std : {np.std(accuracies):.4f}")
print(f"Precision Std: {np.std(precisions):.4f}")
print(f"Recall Std   : {np.std(recalls):.4f}")
print(f"F1 Std       : {np.std(f1s):.4f}")