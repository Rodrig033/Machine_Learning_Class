import numpy as np
from sklearn.datasets import make_classification # Para creación de datos sinteticos
from sklearn.preprocessing import StandardScaler

class PracticalMultiADALINE:
    def __init__(self, n_outputs, learning_rate=0.01, n_epochs=1000):
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # One-hot encoding si y es vector de etiquetas
        if len(y.shape) == 1:
            y_onehot = np.eye(self.n_outputs)[y]
        else:
            y_onehot = y

        # Inicializar parámetros
        self.weights = np.random.randn(self.n_outputs, n_features) * 0.01
        self.biases = np.zeros(self.n_outputs)

        # Entrenamiento
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Forward pass
                linear_outputs = np.dot(self.weights, X[i]) + self.biases
                # En esta sección debes llamar la función de activación

                # Backward pass
                errors = y_onehot[i] - linear_outputs

                # Actualizar
                for j in range(self.n_outputs):
                    self.weights[j] += self.lr * errors[j] * X[i]
                    self.biases[j] += self.lr * errors[j]

        return self

    def predict(self, X):
        linear_outputs = np.dot(X, self.weights.T) + self.biases
        return np.argmax(linear_outputs, axis=1)

# Ejemplo con datos sintéticos
X, y = make_classification(n_samples=100, n_features=2, n_classes=3,
                          n_redundant=0, n_clusters_per_class=1,
                          random_state=42)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
model = PracticalMultiADALINE(n_outputs=3, learning_rate=0.01, n_epochs=1000)
model.fit(X_scaled, y)

# Evaluar
predictions = model.predict(X_scaled)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.3f}")

print("Pesos finales:")
print(model.weights)
print("Sesgos finales:")
print(model.biases)

from sklearn.datasets import load_iris
import pandas as pd
# Cargar dataset iris
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)
feature_names = iris.feature_names
target_names = iris.target_names
iris_df = pd.DataFrame(data=X, columns=feature_names)
iris_df['species'] = y
print("First 5 rows of the Iris DataFrame:")
print(iris_df.head())
print("\nFeature names:", feature_names)
print("Target names:", target_names)