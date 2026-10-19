import numpy as np
import pandas as pd
from itertools import product
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MADALINE - Una capa .9111 de accuracy

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ACTIVATIONS = {
    "lineal":   (lambda z: z,
                 lambda z: np.ones_like(z)),
    "sigmoide": (_sigmoid,
                 lambda z: _sigmoid(z) * (1 - _sigmoid(z))),
    "tanh":     (np.tanh,
                 lambda z: 1 - np.tanh(z) ** 2),
    "relu":     (lambda z: np.maximum(0, z),
                 lambda z: (z > 0).astype(float)),
}

LOSSES = {
    "MSE": (lambda t, a: np.mean((t - a) ** 2),
            lambda a, t: a - t),
    "MAE": (lambda t, a: np.mean(np.abs(t - a)),
            lambda a, t: np.sign(a - t)),
}


class PracticalMultiADALINE:
    def __init__(self, n_outputs, learning_rate=0.01, n_epochs=1000,
                 activation="lineal", loss="MSE", seed=None):
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.loss = loss
        self.f, self.df = ACTIVATIONS[activation]
        self.loss_fn, self.loss_grad = LOSSES[loss]
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.biases = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # One-hot encoding si y es vector de etiquetas
        if len(y.shape) == 1:
            y_onehot = np.eye(self.n_outputs)[y]
        else:
            y_onehot = y

        # Inicializar parámetros
        self.weights = self.rng.standard_normal((self.n_outputs, n_features)) * 0.01
        self.biases = np.zeros(self.n_outputs)
        self.loss_history = []

        with np.errstate(all="ignore"):
            for epoch in range(self.n_epochs):
                # Barajar el orden de los patrones en cada época
                for i in self.rng.permutation(n_samples):
                    # Forward pass
                    z = np.dot(self.weights, X[i]) + self.biases
                    a = self.f(z)                      # <- función de activación

                    # Backward pass: gradiente de la pérdida respecto a z
                    delta = self.loss_grad(a, y_onehot[i]) * self.df(z)

                    # Actualizar todas las salidas a la vez (descenso de gradiente)
                    self.weights -= self.lr * np.outer(delta, X[i])
                    self.biases -= self.lr * delta

                # Pérdida de la época sobre todo el conjunto de entrenamiento
                A = self.f(X @ self.weights.T + self.biases)
                self.loss_history.append(self.loss_fn(y_onehot, A))

        return self

    def predict(self, X):
        with np.errstate(all="ignore"):
            A = self.f(X @ self.weights.T + self.biases)
        return np.argmax(A, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def prepare_data(random_state=42, test_size=0.30):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=test_size,
        stratify=iris.target,          # mantiene 1/3 de cada especie en ambos conjuntos
        random_state=random_state,
    )
    scaler = StandardScaler().fit(X_train)   # el escalador solo ve el entrenamiento
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


if __name__ == "__main__":
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    print("Primeras filas del DataFrame de Iris:")
    print(iris_df.head())

    X_train, X_test, y_train, y_test = prepare_data(random_state=42)
    print(f"\nEntrenamiento: {len(X_train)} patrones | Prueba: {len(X_test)} patrones")

    # Configuración original 
    base = PracticalMultiADALINE(3, learning_rate=0.01, n_epochs=1000, seed=0)
    base.fit(X_train, y_train)
    print("\n== Configuración original (lineal, MSE, lr=0.01, 1000 épocas) ==")
    print(f"Accuracy train: {base.accuracy(X_train, y_train):.3f}")
    print(f"Accuracy test : {base.accuracy(X_test, y_test):.3f}")

    # Búsqueda de hiperparámetros 
    rows = []
    for act, loss, lr, ep in product(ACTIVATIONS, LOSSES,
                                     [0.001, 0.01, 0.1], [100, 500, 1000]):
        m = PracticalMultiADALINE(3, lr, ep, act, loss, seed=0).fit(X_train, y_train)
        rows.append(dict(activacion=act, perdida=loss, lr=lr, epocas=ep,
                         acc_train=m.accuracy(X_train, y_train),
                         acc_test=m.accuracy(X_test, y_test),
                         perdida_final=m.loss_history[-1]))

    df = pd.DataFrame(rows).sort_values(["acc_test", "acc_train"], ascending=False)
    print("\n== Mejores 10 configuraciones (de 72) ==")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = df.iloc[0]
    accs = []
    for rs in range(20):
        Xtr, Xte, ytr, yte = prepare_data(random_state=rs)
        m = PracticalMultiADALINE(3, best.lr, int(best.epocas),
                                  best.activacion, best.perdida, seed=0).fit(Xtr, ytr)
        accs.append(m.accuracy(Xte, yte))
    print(f"\nMejor configuración en 20 particiones distintas: "
          f"media = {np.mean(accs):.3f}, mín = {np.min(accs):.3f}, máx = {np.max(accs):.3f}, "
          f"100% en {sum(a == 1.0 for a in accs)}/20")