import numpy as np
import matplotlib.pyplot as plt

class MADALINE:
    def __init__(self, n_inputs, n_adalines, learning_rate = 0.1):
        self.n_inputs = n_inputs
        self.n_adalines = n_adalines
        self.learning_rate = learning_rate

        # Pesos de cada ADALINE
        self.weights = np.random.uniform(-0.5, 0.5, (n_adalines, n_inputs))

        # Bias de cada ADALINE
        self.bias = np.zeros(n_adalines)

        # Historial de entrenamiento
        self.error_history = []
        self.epochs_history = []

    @staticmethod
    def bipolar_step(x):
        return np.where(x >= 0, 1, -1)

    def forward(self, X):
        # Salida lineal de cada ADALINE
        z = X @ self.weights.T + self.bias

        # Salida bipolar
        h = self.bipolar_step(z)

        return z, h

    def fit(self, X, y, epochs=100):
        self.error_history = []
        self.epochs_history = []

    def predict(self, X):
        z, h = self.forward(X)

        # Votación mayoritaría
        votes = np.sum(h, axis=1)

        return self.bipolar_step(votes)

    def calculate_error(self, y_true, y_pred):
        return np.mean(y_true - y_pred) ** 2

    def show_weights(self):

        print("\nPesos de la ADALINE")

        for i in range(self.n_adalines):
            print(
                f"ADALINE {i + 1}: " 
                f"w = {self.weights[i]}, "
                f"b = {self.bias[i]}: "
            )

    def plot_error(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.error_history,
            linewidth=2
        )

        plt.xlabel("Época")
        plt.ylabel("Error")
        plt.title("Entrenamiento MADALINE")

        plt.grid(True)

        plt.show()

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            epoch_error = 0

            for x_i, y_i in zip(X, y):

                x = X[x_i]
                target = y[y_i]

                # 1. Obtener salida
                z = np.dot(self.weights, x) + self.bias
                epoch_error = y - y_i

                # 2. Comprobar si es correcta
                if y != target:
                    if target == 1:
                        idx = np.argmin(np.abs(z))
                        self.weights[idx] += (self.learning_rate * epoch_error * x_i)
                        self.bias[idx] += self.learning_rate * np.mean(epoch_error)

                    elif target == -1:
                        # Salida positiva
                        for j in range(len(z)):
                            if z[j] == 1:
                                self.weights[j] += self.learning_rate * (-1 - z[j])
                                self.bias[j] += self.learning_rate * (-1 - z[j])
                                
            self.error_history.append(epoch_error)


X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

y = np.array([
    -1,
     1,
     1,
    -1
])


madaline = MADALINE(
    n_inputs=2,
    n_adalines=2,
    learning_rate=0.1
)

madaline.train(X, y, epochs=100)
madaline.show_weights()

z, h = madaline.forward(X)

print("\nSalidas lineales:")
print(z)

print("\nSalidas bipolares:")
print(h)

print("\nPredicciones:")
print(madaline.predict(X))

print("\nObjetivos:")
print(y)

madaline.plot_error()