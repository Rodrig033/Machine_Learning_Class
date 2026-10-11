import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

print(df.head())

# Datos númericos 
numeric_data = df.select_dtypes(include=[np.number])


# Características iniciales
print(df.info())

# Estadística descriptiva
print(df.describe()) 

# Shape
print(df.shape)

# Valores nulos
print(df.isnull().sum())

# Distribución de variables númericas:
df.hist(figsize=(15, 10))
plt.show()


# Matriz de correlación
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(numeric_only=True),
            annot= True,
            cmap= "coolwarm")

plt.title("Matriz de correlación")
plt.show()

# Al parcer las variables con mayor correlación son flex acidity y ph (-0.68)

# PCA 
scaler = StandardScaler() # Escalar los datos antes 
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA()
pca.fit(scaled_data)

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

# PCA con 2 componentes
pca_2 = PCA(n_components=2)
principal_components = pca_2.fit_transform(scaled_data)

# Loadings (importancia de variables)
loadings = pd.DataFrame(
    pca_2.components_.T,
    columns=['PC1', 'PC2'],
    index=numeric_data.columns
)

print(loadings)

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],
            principal_components[:,1],
            alpha=0.5)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - 2 Componentes")
plt.grid()
plt.show()

# Gráfico de barras para explicar los componentes
plt.bar(range(len(explained_variance)), explained_variance)
plt.xlabel("Componentes")
plt.ylabel("Varianza explicada")
plt.title("Varianza explicada por componente")
plt.show()