import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargar el dataset:
planets = pd.read_csv("nasa_exoplanets.csv")

# Solo datos numéricos:
numeric_data = planets.select_dtypes(include=[np.number])

# Imputar valores (nunca hacer PCA sin limpiar datos):
numeric_data = numeric_data.fillna(numeric_data.mean())

print(planets.head())

# Dimensiones:
print("Planets: ",planets.shape)

print("\n----------- Tipos de datos ---------------------")
planets.info()
print("\n------------------------------------------------")

# Valores nulos
print(planets.isnull().sum())

# Estadística descriptiva:
print(planets.describe())

# Distribución de variables númericas:
planets.hist(figsize=(15, 10))
plt.show()


# Matriz de correlación
plt.figure(figsize=(12, 8))

sns.heatmap(planets.corr(numeric_only=True),
            annot= True,
            cmap= "coolwarm")

plt.title("Matriz de correlación")
plt.show()

# Analizar valores nulos visualmente
plt.figure(figsize=(12, 6))
sns.heatmap(planets.isnull(), cbar=False)
plt.title("Mapa de Valores Nulos")
plt.show()

# Matriz de correlación (numéricas)
plt.figure(figsize=(14, 10))

sns.heatmap(numeric_data.corr(numeric_only=True),
            annot= True,
            center= 0,
            cmap= "coolwarm")

plt.title("Matriz de correlación")
plt.show()

# PCA 
scaler = StandardScaler() # Escalar los datos antes 
X_scaled = scaler.fit_transform(numeric_data)


pca = PCA()
pca.fit(X_scaled)

# Varianza explicada 
explained_variance = pca.explained_variance_ratio_

print("Varianza explicada por componente: ")
print(explained_variance)

# Gráfica de varianza acumulada:
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance), marker= 'o') 
plt.xlabel("Número de componentes")
plt.ylabel("Varianza acumulada")
plt.title("PCA - Varianza acumulada")
plt.grid()
plt.show()

pca_95 = PCA(n_components=0.95)
reduced_data = pca_95.fit_transform(X_scaled)

print("Componentes seleccionados:", pca_95.n_components_)
print("Nueva dimensión:", reduced_data.shape)

# Visualización de Exoplanetas en 2D:
pca_2 = PCA(n_components=2)
data_2d = pca_2.fit_transform(X_scaled)

print("Forma reducida: ", data_2d.shape)

plt.figure(figsize=(10, 7))
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D - Exoplanetas")

plt.grid()
plt.show()

# Método del codo:
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_2d)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.grid()
plt.show()

# Aplicar KMeans:
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_2d)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, clusters, test_size=0.2, random_state=42
)

# K-NN (K-Nearest Neighbor):
def knn_predicts (X_train, y_train, x_new, K=3): 
    distancias = np.sqrt(np.sum((X_train - x_new)**2, axis=1))
    k_idx = np.argsort(distancias)[:K]
    k_labels = y_train[k_idx]
    values, counts = np.unique(k_labels, return_counts=True)

    return values[np.argmax(counts)]

def knn_predicts_batch(X_train, y_train, X_test, K):
    predictions = []

    for i in X_test:
        pred = knn_predicts(X_train, y_train, i, K)
        predictions.append(pred)

    return np.array(predictions)

plt.figure(figsize=(10, 7))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cluster de Exoplanetas (KMeans + PCA)")

plt.grid()
plt.show()

# Análisis de cada cluster:
numeric_data["cluster"] = clusters
print(numeric_data.groupby("cluster").mean())

# Extraer las variables más importantes:
for i, comp in enumerate(pca.components_):
    print(f"\nComponente {i+1}")
    sorted_vars = sorted(zip(numeric_data.columns, comp), 
                         key=lambda x: abs(x[1]), reverse=True)
    
    for var, val in sorted_vars[:3]:
        print(f"{var}: {val:.3f}")

# Gráfico de barras para explicar los componentes
plt.bar(range(len(explained_variance)), explained_variance)
plt.xlabel("Componentes")
plt.ylabel("Varianza explicada")
plt.title("Varianza explicada por componente")
plt.show()

# Prueba:
prediction = knn_predicts(X_train, y_train, X_test[0], K=3)

y_pred = knn_predicts_batch(X_train, y_train, X_test, K = 3)
accuracy = np.mean(y_pred == y_test)
print("Accuracy: ", accuracy)

# Probar con distintos valores para K:
for k in [1, 3, 5, 7, 9, 11]:
    y_pred = knn_predicts_batch(X_train, y_train, X_test, K = k)
    accuracy = np.mean(y_pred == y_test)
    print(f"K={k}, Accuracy: {accuracy}")

k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    y_pred = knn_predicts_batch(X_train, y_train, X_test, K = k)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

plt.figure()
plt.plot(k_values, accuracies, marker='o')

plt.title("K vs accuracy")
plt.xlabel("Número de vecinos K")
plt.ylabel("Accuracy")

plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred)


plt.figure()
plt.imshow(cm)

plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.colorbar()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Discretización