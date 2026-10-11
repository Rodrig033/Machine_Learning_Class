import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# CARGA DE DATOS
df = pd.read_csv("neo.csv")

print("\n===================================")
print(" DATASET INFORMATION ")
print("===================================\n")

print(df.head())
print("\nShape:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# SELECCIÓN DE VARIABLES

# Ignoramos: 
features = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude"
]

X = df[features]

# EDA

print("\n===================================")
print(" DESCRIPTIVE STATISTICS ")
print("===================================\n")

print(X.describe())

# CORRELACIÓN

print("\n===================================")
print(" CORRELATION MATRIX ")
print("===================================\n")

corr = X.corr()

print(corr)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.show()

# HISTOGRAMAS

print("\n===================================")
print(" HISTOGRAMS ")
print("===================================\n")

X.hist(
    figsize=(12,8),
    bins=30
)

plt.tight_layout()

plt.show()

# ESCALADO

print("\n===================================")
print(" STANDARDIZATION ")
print("===================================\n")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# PCA EXPLORATORIO

print("\n===================================")
print(" PCA ANALYSIS ")
print("===================================\n")

pca_full = PCA()

pca_full.fit(X_scaled)

cum_var = np.cumsum(
    pca_full.explained_variance_ratio_
)

for i, v in enumerate(cum_var, start=1):
    print(
        f"Component {i}: {v:.4f}"
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(1,len(cum_var)+1),
    cum_var,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    color="red"
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.grid()
plt.show()

# PCA FINAL

pca = PCA(
    n_components=0.95
)

X_pca = pca.fit_transform(
    X_scaled
)

print("\nOriginal Shape:")
print(X_scaled.shape)
print("\nReduced Shape:")
print(X_pca.shape)
print("\nExplained Variance:")
print(
    pca.explained_variance_ratio_.sum()
)

# MÉTODO DEL CODO

print("\n===================================")
print(" ELBOW METHOD ")
print("===================================\n")

inertias = []

for k in range(2,11):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    km.fit(X_pca)
    inertias.append(
        km.inertia_
    )

plt.figure(figsize=(8,5))
plt.plot(
    range(2,11),
    inertias,
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()

# SILHOUETTE SCORE

print("\n===================================")
print(" SILHOUETTE SCORES ")
print("===================================\n")

scores = []

for k in range(2,11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = km.fit_predict(
        X_pca
    )

    score = silhouette_score(
        X_pca,
        labels
    )

    scores.append(score)

    print(
        f"K={k} -> {score:.4f}"
    )

best_k = range(2,11)[
    np.argmax(scores)
]

print(
    f"\nBest K according to silhouette: {best_k}"
)

# KMEANS FINAL

print("\n===================================")
print(" FINAL KMEANS ")
print("===================================\n")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(
    X_pca
)

df["cluster"] = clusters

# VISUALIZACIÓN 2D

pca_vis = PCA(
    n_components=2
)

X_vis = pca_vis.fit_transform(
    X_scaled
)

centroids_vis = pca_vis.transform(
    pca.inverse_transform(
        kmeans.cluster_centers_
    )
)

plt.figure(figsize=(10,7))

plt.scatter(
    X_vis[:,0],
    X_vis[:,1],
    c=clusters,
    alpha=0.5
)

plt.scatter(
    centroids_vis[:,0],
    centroids_vis[:,1],
    marker="X",
    s=400
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    f"K-Means Clustering (K={best_k})"
)

plt.show()

# PERFILAMIENTO DE CLUSTERS

print("\n===================================")
print(" CLUSTER PROFILES ")
print("===================================\n")

profile = (
    df
    .groupby("cluster")[features]
    .mean()
)

print(profile)

# DISTRIBUCIÓN DE CLUSTERS

print("\n===================================")
print(" CLUSTER DISTRIBUTION ")
print("===================================\n")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)

# ANÁLISIS DE HAZARDOUS

print("\n===================================")
print(" HAZARDOUS ANALYSIS ")
print("===================================\n")

hazard_profile = pd.crosstab(
    df["cluster"],
    df["hazardous"],
    normalize="index"
) * 100

print(
    hazard_profile.round(2)
)

# FINAL VISUALIZATION

plt.figure(figsize=(10,7))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis",
    s=10,
    alpha=0.5
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    c="red",
    marker="X",
    s=300,
    label="Centroides"
)

plt.title("Clusters de Asteroides")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()