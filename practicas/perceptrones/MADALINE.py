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

    def predict(self, X):
        z, h = self.forward(X)

        # Votación mayoritaría
        votes = np.sum(h, axis=1)

        return self.bipolar_step(votes)

    def calculate_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

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
            self.epochs_history,
            self.error_history,
            linewidth = 2
        )

        plt.xlabel("Época")
        plt.ylabel("Error")
        plt.title("Entrenamiento MADALINE")

        plt.grid(True)

        plt.show()

    def plot_decision_boundary(self, X, y, margin = 1.0, resolution = 1000):
        if self.n_inputs != 2:
            raise ValueError("Solo se puede graficar con 2 entradas.")

        # Límite del plano
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 0].min() - margin, X[:, 1].max() + margin    

        # Malla de puntos
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )    

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(7,6))

        plt.contourf(
            xx, yy, Z,
            levels=[-2, 0, 2],
            colors=["tab:red", "tab:blue"],
            alpha = 0.2
        )

        xs = np.linspace(x_min, x_max, 100)
        for i in range(self.n_adalines):
            w1, w2 = self.weights[i]
            b = self.bias[i]

            if abs(w2) > 1e-12:
                plt.plot(xs, - (w1 * xs + b) / w2, "--", linewidth = 1.5, label=f"ADALINE{i + 1}")
            elif abs(w1 > 1e-12):
                plt.axvline(-b / w1, linestyle="--", linewidth=1.5, label=f"ADALINE{i +1}")

        # Patrones de entrenamiento
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:blue", s=120,
                    edgecolors="black", marker="o", label="Clase +1")
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c="tab:red", s=120,
                    edgecolors="black", marker="s", label="Clase -1")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.title("Frontera de decisión MADALINE")
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self, X, y, epochs=100):

        self.error_history = []
        self.epochs_history = []

        for epoch in range(epochs):

            for x, target in zip(X, y):

                # 1. Obtener salida lineal de cada ADALINE
                z = np.dot(self.weights, x) + self.bias

                # 2. Calcular la salida bipolar
                h = self.bipolar_step(z)

                # 3. Salida de la red
                output = self.bipolar_step(np.sum(h))

                """ y ≠ ŷ """

                # 2. Comprobar si es correcta
                if output != target:
                    if target == 1:

                        candidates = np.where(z < 0)[0]
                        idx = candidates[np.argmin(np.abs(z[candidates]))]

                        # Error de esta ADALINE
                        error = target - z[idx]

                        # Actualizó los pesos y el bias
                        self.weights[idx] += self.learning_rate * error * x
                        self.bias[idx] += self.learning_rate * error

                    elif target == -1:
                        # Salida positiva
                        for j in range(len(z)):
                            if z[j] == 1:
                                error = target - z[j]
                                self.weights[j] += self.learning_rate * error * x
                                self.bias[j] += self.learning_rate * error

            epoch_error = self.calculate_error(y, self.predict(X))
            self.error_history.append(epoch_error)
            self.epochs_history.append(epoch + 1)

            if epoch_error == 0:
                print(f"La red convergió en {epochs + 1} épocas.")
                break
        else:
            print(f"La red no convergió en {epochs} épocas.")



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
madaline.plot_decision_boundary(X, y)