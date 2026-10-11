import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer


# Cargar el dataset:
planets = pd.read_csv("nasa_exoplanets.csv")

# Solo datos numéricos:
numeric_data = planets.select_dtypes(include=[np.number])

# Imputar valores (nunca hacer PCA sin limpiar datos):
numeric_data = numeric_data.fillna(numeric_data.mean())

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

# Aplicar KMeans:
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_2d)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, clusters, test_size=0.2, random_state=42
)

# Distribución de variables númericas:
planets.hist(figsize=(15, 10))
plt.show()


discretizer = KBinsDiscretizer(
    n_bins=5,
    encode='ordinal',
    strategy='quantile'  
)

X_binned = discretizer.fit_transform(X_scaled)

#  Dividir los datos Train / Test:
X_train, X_test, y_train, y_test = train_test_split(
    X_binned, clusters, test_size=0.2, random_state= 42
)

def entrenar_naive_bayes(X, y):
    clases = np.unique(y)
    n_features = X.shape[1]
    
    prob_clases = {}
    prob_condicional = {}
    
    for c in clases:
        X_c = X[y == c]
        prob_clases[c] = len(X_c) / len(X)
        
        prob_condicional[c] = {}
        
        for j in range(n_features):
            valores, counts = np.unique(X_c[:, j], return_counts=True)
            
            total = len(X_c)
            probs = {}
            
            n_bins = 10
            for v, count in zip(valores, counts):
                probs[v] = (count + 1) / (total + n_bins)  # Laplace
            
            prob_condicional[c][j] = probs
    
    return prob_clases, prob_condicional


import math

def predecir_naive_bayes(x, prob_clases, prob_condicional):
    resultados = {}
    
    for c in prob_clases:
        log_prob = math.log(prob_clases[c])
        
        for j, valor in enumerate(x):
            if valor in prob_condicional[c][j]:
                log_prob += math.log(prob_condicional[c][j][valor])
            else:
                log_prob += math.log(1e-6)
        
        resultados[c] = log_prob
    
    return max(resultados, key=resultados.get)

def predecir_batch(X, prob_clases, prob_condicional):
    predictions = []

    for x in X:
        pred = predecir_naive_bayes(x, prob_clases, prob_condicional)
        predictions.append(pred)

    return np.array(predictions)

prob_clases, prob_condicional = entrenar_naive_bayes(X_train, y_train)
y_pred = predecir_batch(X_test, prob_clases, prob_condicional)

# Matriz de confusión:
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

print("------ Métricas de precisión ------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)