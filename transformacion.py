import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
plt.style.use("ggplot")


iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].replace({0: "setosa", 1: "versicolor", 2: "virginica"})

x = df[iris.feature_names]
pca = PCA(n_components= 2)
x_pca = pca.fit_transform(x)

# Almacenar datos en la matriz
df["PC1"] = x_pca[:,0]
df["PC2"] = x_pca[:,1]

# Separar clases
setosa_points = df[df["species_name"] == "setosa"]
virginica_points = df[df["species_name"] == "virginica"]
versicolor_points = df[df["species_name"] == "versicolor"]

plt.scatter(setosa_points["PC1"], setosa_points["PC2"], label = "Setosa")
plt.scatter(virginica_points["PC1"], setosa_points["PC2"], label = "Virginica")
plt.scatter(versicolor_points["PC1"], setosa_points["PC2"], label = "Versicolor")


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Transformación")
plt.legend()
plt.show()

# Varianza explicada
print(pca.explained_variance_ratio_)

loadings = pd.DataFrame(pca.components_, columns= iris.feature_names, index= ["PC1", "PC2"])
print(loadings)

# Single Value Decomposition

# ¿Cuál es la varianza explicada para cada una?
print(np.sum(loadings.loc["PC1"].values**2))
print(np.sum(loadings.loc["PC2"].values**2))    
