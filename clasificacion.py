from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = load_wine()
X = data.data
y = data.target


# Escalar lo datos:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos (Train / Test):
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size= 0.2, random_state= 42
)

# K-NN (K-Nearest Neighbor):
def knn_predicts (X_train, y_train, x_new, K=3): 
    distancias = np.sqrt(np.sum((X_train - x_new)**2, axis=1))
    k_idx = np.argsort(distancias)[:K]
    k_labels = y_train[k_idx]
    values, counts = np.unique(k_labels, return_counts=True)

    return values[np.argmax(counts)]


# Prueba:
prediction = knn_predicts(X_train, y_train, X_test[0], K=3)
print("Predicción: ", prediction)
print("Valor real: ", y_test[0])

def knn_predicts_batch(X_train, y_train, X_test, K):
    predictions = []

    for i in X_test:
        pred = knn_predicts(X_train, y_train, i, K)
        predictions.append(pred)

    return np.array(predictions)

y_pred = knn_predicts_batch(X_train, y_train, X_test, K = 3)
accuracy = np.mean(y_pred == y_test)
print("Accuracy: ", accuracy)

# Probar con distintos valores para K:
for k in [1, 3, 5, 7, 9, 11]:
    y_pred = knn_predicts_batch(X_train, y_train, X_test, K = k)
    accuracy = np.mean(y_pred == y_test)
    print(f"K={k}, Accuracy: {accuracy}")

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
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