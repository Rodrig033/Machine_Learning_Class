import numpy as np
import pandas as pd
from itertools import product
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)          # estabilidad numérica
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


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
    # Entropía cruzada: siempre va con softmax en la salida.
    # El gradiente combinado softmax + CE respecto a z es simplemente (a - t).
    "CE":  (lambda t, a: -np.mean(np.sum(t * np.log(np.clip(a, 1e-12, 1)), axis=-1)),
            None),
}


class PracticalMLP:
    """Perceptrón multicapa: entrada -> capa oculta -> salida."""

    def __init__(self, n_outputs, n_hidden=8, learning_rate=0.01, n_epochs=500,
                 hidden_activation="tanh", output_activation="lineal",
                 loss="CE", seed=None):
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.loss = loss
        self.f_h, self.df_h = ACTIVATIONS[hidden_activation]
        if loss == "CE":
            self.output_activation = "softmax"
            self.f_o, self.df_o = _softmax, None
        else:
            self.output_activation = output_activation
            self.f_o, self.df_o = ACTIVATIONS[output_activation]
        self.loss_fn, self.loss_grad = LOSSES[loss]
        self.rng = np.random.default_rng(seed)
        self.loss_history = []

    def _init_layer(self, n_in, n_out, activation):
        # He para ReLU, Xavier/Glorot para el resto: evita gradientes que
        # se desvanecen o explotan al apilar capas
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        return self.rng.standard_normal((n_out, n_in)) * scale, np.zeros(n_out)

    def _forward(self, X):
        z1 = X @ self.W1.T + self.b1
        h = self.f_h(z1)
        z2 = h @ self.W2.T + self.b2
        return z1, h, z2, self.f_o(z2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_onehot = np.eye(self.n_outputs)[y] if y.ndim == 1 else y

        self.W1, self.b1 = self._init_layer(n_features, self.n_hidden, self.hidden_activation)
        self.W2, self.b2 = self._init_layer(self.n_hidden, self.n_outputs, self.output_activation)
        self.loss_history = []

        with np.errstate(all="ignore"):
            for epoch in range(self.n_epochs):
                for i in self.rng.permutation(n_samples):
                    x, t = X[i], y_onehot[i]

                    # Forward pass
                    z1 = self.W1 @ x + self.b1
                    h = self.f_h(z1)                    # capa oculta
                    z2 = self.W2 @ h + self.b2
                    a = self.f_o(z2)                    # capa de salida

                    # Backward pass (retropropagación)
                    if self.loss == "CE":
                        delta2 = a - t
                    else:
                        delta2 = self.loss_grad(a, t) * self.df_o(z2)
                    delta1 = (self.W2.T @ delta2) * self.df_h(z1)

                    # Actualización por descenso de gradiente
                    self.W2 -= self.lr * np.outer(delta2, h)
                    self.b2 -= self.lr * delta2
                    self.W1 -= self.lr * np.outer(delta1, x)
                    self.b1 -= self.lr * delta1

                A = self._forward(X)[-1]
                self.loss_history.append(self.loss_fn(y_onehot, A))

        return self

    def predict(self, X):
        with np.errstate(all="ignore"):
            return np.argmax(self._forward(X)[-1], axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def prepare_data(random_state=42, test_size=0.30):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=test_size,
        stratify=iris.target,
        random_state=random_state,
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(random_state=42)
    print(f"Entrenamiento: {len(X_train)} patrones | Prueba: {len(X_test)} patrones")

    # Configuración de referencia 
    base = PracticalMLP(3, n_hidden=8, learning_rate=0.01, n_epochs=500,
                        hidden_activation="tanh", loss="CE", seed=0).fit(X_train, y_train)
    print("\n== MLP de referencia (8 ocultas tanh, softmax + CE, lr=0.01, 500 épocas) ==")
    print(f"Accuracy train: {base.accuracy(X_train, y_train):.3f}")
    print(f"Accuracy test : {base.accuracy(X_test, y_test):.3f}")

    # Búsqueda de hiperparámetros
    rows = []
    grid = product([4, 8, 16], ["tanh", "relu", "sigmoide"], ["CE", "MSE"],
                   [0.01, 0.1], [50, 50])
    for nh, act, loss, lr, ep in grid:
        out_act = "sigmoide" if loss == "MSE" else "lineal"   # ignorado con CE
        m = PracticalMLP(3, nh, lr, ep, act, out_act, loss, seed=0).fit(X_train, y_train)
        rows.append(dict(ocultas=nh, activacion=act, perdida=loss, lr=lr, epocas=ep,
                         acc_train=m.accuracy(X_train, y_train),
                         acc_test=m.accuracy(X_test, y_test),
                         perdida_final=m.loss_history[-1]))

    df = pd.DataFrame(rows).sort_values(["acc_test", "acc_train"], ascending=False)
    print(f"\n== Mejores 10 configuraciones (de {len(df)}) ==")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Robustez en 20 particiones 
    best = df.iloc[0]
    out_act = "sigmoide" if best.perdida == "MSE" else "lineal"
    accs = []
    for rs in range(20):
        Xtr, Xte, ytr, yte = prepare_data(random_state=rs)
        m = PracticalMLP(3, int(best.ocultas), best.lr, int(best.epocas),
                         best.activacion, out_act, best.perdida, seed=0).fit(Xtr, ytr)
        accs.append(m.accuracy(Xte, yte))
    print(f"\nMejor configuración en 20 particiones distintas: "
          f"media = {np.mean(accs):.3f}, mín = {np.min(accs):.3f}, máx = {np.max(accs):.3f}, "
          f"100% en {sum(a == 1.0 for a in accs)}/20")