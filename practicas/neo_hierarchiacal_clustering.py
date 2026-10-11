import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix

)
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

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

df_sample = df.sample(
    n = 2000,
    random_state= 0
)

# SELECCIÓN DE VARIABLES

# Ignoramos: 
features = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude"
]

X = df_sample[features]

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

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

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
print("\n===================================")
print(" FINAL PCA ")
print("===================================\n")

pca = PCA(
    n_components=0.95
)

X_pca = pca.fit_transform(X_scaled)
print("Original Shape:")
print(X_scaled.shape)
print("\nReduced Shape:")
print(X_pca.shape)
print("\nExplained Variance:")
print(np.sum(pca.explained_variance_ratio_))

# Dendrograma 

print("\n===================================")
print(" DENDROGRAM ")
print("===================================\n")

Z = linkage(X_pca, method="ward")

plt.figure(figsize=(14,6))

dendrogram(
    Z,
    truncate_mode="level",
    p=5
)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Selección del número del clusters
n_clusters = 2

print("\n===================================")
print(" HIERARCHICAL CLUSTERING ")
print("===================================\n")

hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward" # Produce grupos similares a K-Means
)

clusters = hc.fit_predict(X_pca)

""" Clustering metrics silhouette, Davies-Bouldin, Calinski-Harabasz"""

print("\n===================================")
print(" CLUSTERING METRICS ")
print("===================================\n")

# Silhouette
sil_score = silhouette_score(
    X_pca,
    clusters
)

# Davies-Bouldin
dbi = davies_bouldin_score(
    X_pca,
    clusters
)

# Calinski-Harabasz
chi = calinski_harabasz_score(
    X_pca,
    clusters
)

# Inertia

print(f"Silhouette Score      : {sil_score:.4f}")
print(f"Davies-Bouldin Index  : {dbi:.4f}")
print(f"Calinski-Harabasz     : {chi:.4f}")


# Visualización de clusters
print("\n===================================")
print(" VISUALIZATION ")
print("===================================\n")

pca_vis = PCA(n_components=2)

X_vis = pca_vis.fit_transform(
    X_scaled
)

plt.figure(figsize=(10,7))

plt.scatter(
    X_vis[:,0],
    X_vis[:,1],
    c=clusters,
    cmap="viridis",
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Hierarchical Clustering")
plt.colorbar(label="Cluster")
plt.show()

# Perfil de los clusters
df_clusters = X.copy()
df_clusters["cluster"] = clusters

print("\n===================================")
print(" CLUSTER PROFILES ")
print("===================================\n")

profiles = df_clusters.groupby("cluster").mean()
print(profiles)

# Distribución de clusters
print("\n===================================")
print(" CLUSTER DISTRIBUTION ")
print("===================================\n")

print(
    df_clusters["cluster"]
    .value_counts()
)

# Inertia

print("\n===================================")
print(" INERTIA (KMEANS REFERENCE)")
print("===================================\n")

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=0,
    n_init=10
)

kmeans.fit(X_pca)

print(f"Inertia: {kmeans.inertia_:.4f}")

# Comparar si los clusters coinciden con el dataset original:
df_clusters["hazardous"] = df_sample["hazardous"].values

# Tabla cruzada:
tabla_cruzada = pd.crosstab(
    df_clusters["cluster"],
    df_clusters["hazardous"],
    normalize="index"
)*100

print(tabla_cruzada)

# Procedimiento para etiquetar automáticamente un nuevo dato (2D)


# Métricas de aprendizaje supervisado (Evaluación final)
centroid_2D = []
    