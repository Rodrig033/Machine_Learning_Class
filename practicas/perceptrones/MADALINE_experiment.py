import numpy as np
import pandas as pd


# Funciones de activación

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ACTIVATIONS = {
    "lineal": dict(
        f=lambda z: z,
        df=lambda z: np.ones_like(z, dtype=float),
        low=-1, high=1, thr=0.0),
    "tanh": dict(
        f=np.tanh,
        df=lambda z: 1 - np.tanh(z) ** 2,
        low=-1, high=1, thr=0.0),
    "sigmoide": dict(
        f=_sigmoid,
        df=lambda z: _sigmoid(z) * (1 - _sigmoid(z)),
        low=0, high=1, thr=0.5),
    "relu": dict(
        f=lambda z: np.maximum(0, z),
        df=lambda z: (np.asarray(z) > 0).astype(float),
        low=0, high=1, thr=0.5),
    "leaky_relu": dict(
        f=lambda z: np.where(z > 0, z, 0.01 * z),
        df=lambda z: np.where(z > 0, 1.0, 0.01),
        low=-1, high=1, thr=0.0),
}

# Pérdidas

LOSSES = {
    "MSE": dict(
        value=lambda y, s: np.mean((y - s) ** 2),
        grad=lambda a, t: a - t),
    "MAE": dict(
        value=lambda y, s: np.mean(np.abs(y - s)),
        grad=lambda a, t: np.sign(a - t)),
}


class MADALINE:
    def __init__(self, n_inputs, n_adalines=3, learning_rate=0.1,
                 activation="lineal", loss="MSE", init="aleatoria", seed=None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation}")
        if loss not in LOSSES:
            raise ValueError(f"Pérdida desconocida: {loss}")

        self.n_inputs = n_inputs
        self.n_adalines = n_adalines
        self.learning_rate = learning_rate
        self.act = ACTIVATIONS[activation]
        self.loss_fn = LOSSES[loss]
        self.rng = np.random.default_rng(seed)

        if init == "aleatoria":
            self.weights = self.rng.uniform(-0.5, 0.5, (n_adalines, n_inputs))
        elif init == "ceros":
            self.weights = np.zeros((n_adalines, n_inputs))
        else:
            raise ValueError(f"Inicialización desconocida: {init}")
        self.bias = np.zeros(n_adalines)

        self.loss_history = []
        self.acc_history = []
        self.epochs_history = []
        self.converged = False
        self.converged_epoch = None

    @staticmethod
    def bipolar_step(x):
        return np.where(x >= 0, 1, -1)

    # inferencia
    def forward(self, X):
        z = X @ self.weights.T + self.bias
        a = self.act["f"](z)
        h = np.where(a >= self.act["thr"], 1, -1)   # voto bipolar
        return z, a, h

    def predict(self, X):
        _, _, h = self.forward(X)
        return self.bipolar_step(np.sum(h, axis=1))

    def soft_output(self, X):
        _, a, _ = self.forward(X)
        lo, hi = self.act["low"], self.act["high"]
        a_bip = -1 + 2 * (a - lo) / (hi - lo)
        return a_bip.mean(axis=1)

    def loss(self, X, y):
        return self.loss_fn["value"](y, self.soft_output(X))

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    # entrenamiento
    def _update(self, j, z_j, a_j, t, x):
        g = self.loss_fn["grad"](a_j, t) * self.act["df"](np.asarray(z_j))
        self.weights[j] -= self.learning_rate * g * x
        self.bias[j] -= self.learning_rate * g

    def train(self, X, y, epochs=100, early_stop=True):
        self.loss_history, self.acc_history, self.epochs_history = [], [], []
        self.converged, self.converged_epoch = False, None
        lo, hi, thr = self.act["low"], self.act["high"], self.act["thr"]

        with np.errstate(all="ignore"):
            for epoch in range(1, epochs + 1):
                for i in self.rng.permutation(len(X)):
                    x, target = X[i], y[i]
                    z, a, h = (v[0] for v in self.forward(x.reshape(1, -1)))
                    output = self.bipolar_step(np.sum(h))

                    if output == target:
                        continue

                    t = hi if target == 1 else lo

                    if target == 1:
                        # una sola ADALINE: la que vota -1 y está más cerca del umbral
                        cands = np.where(h == -1)[0]
                        units = [cands[np.argmin(np.abs(a[cands] - thr))]]
                    else:
                        # todas las que votan +1
                        units = np.where(h == 1)[0]

                    for j in units:
                        self._update(j, z[j], a[j], t, x)

                acc = self.accuracy(X, y)
                self.loss_history.append(self.loss(X, y))
                self.acc_history.append(acc)
                self.epochs_history.append(epoch)

                if acc == 1.0 and self.converged_epoch is None:
                    self.converged_epoch = epoch
                if early_stop and acc == 1.0:
                    break

        self.converged = self.acc_history[-1] == 1.0



# Datos: XOR con ruido gaussiano

def make_xor(n_per_corner=20, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    labels = np.array([-1, 1, 1, -1])
    X = np.repeat(corners, n_per_corner, axis=0)
    X = X + rng.normal(0, noise, X.shape)
    y = np.repeat(labels, n_per_corner)
    return X, y


# Experimentos

BASE = dict(n_adalines=3, learning_rate=0.1, activation="lineal",
            loss="MSE", init="aleatoria")
BASE_EPOCHS = 500


def evaluate_config(data, cfg, epochs, early_stop, n_seeds):
    X_tr, y_tr, X_te, y_te = data
    tr, te, ls, conv, conv_ep = [], [], [], [], []

    for s in range(n_seeds):
        m = MADALINE(n_inputs=2, seed=s, **cfg)
        m.train(X_tr, y_tr, epochs=epochs, early_stop=early_stop)
        tr.append(m.accuracy(X_tr, y_tr))
        te.append(m.accuracy(X_te, y_te))
        ls.append(m.loss_history[-1])
        conv.append(m.converged)
        if m.converged_epoch is not None:
            conv_ep.append(m.converged_epoch)

    k = sum(conv)
    estado = "Sí" if k == n_seeds else ("No" if k == 0 else "Parcial")
    return {
        "Acc train": f"{np.mean(tr) * 100:.1f}%",
        "Acc test": f"{np.mean(te) * 100:.1f}%",
        "Pérdida final": f"{np.nanmean(ls):.4f}" if not np.all(np.isnan(ls)) else "nan",
        "¿Converge?": f"{estado} ({k}/{n_seeds})",
        "Época conv.": f"{np.mean(conv_ep):.0f}" if conv_ep else "—",
    }


def run_experiment(title, param, values, data, n_seeds=5):
    rows = [] 
    for v in values:
        cfg, epochs, early = dict(BASE), BASE_EPOCHS, True
        if param == "epochs":
            epochs, early = v, False      # sin parada temprana para que las épocas importen
        else:
            cfg[param] = v
        row = {"Experimento": title, "Valor": str(v)}
        row.update(evaluate_config(data, cfg, epochs, early, n_seeds))
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print(df.drop(columns="Experimento").to_string(index=False))
    return df


if __name__ == "__main__":
    X_train, y_train = make_xor(20, 0.25, seed=0)
    X_test, y_test = make_xor(20, 0.25, seed=1)
    data = (X_train, y_train, X_test, y_test)

    experiments = [
        ("1. Tasa de aprendizaje", "learning_rate", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]),
        ("2. Función de activación", "activation", list(ACTIVATIONS)),
        ("3. Pérdida", "loss", list(LOSSES)),
        ("4. Inicialización", "init", ["aleatoria", "ceros"]),
        ("5. Épocas", "epochs", [100, 500, 1000, 3000]),
    ]

    reports = [run_experiment(*e, data) for e in experiments]
    full = pd.concat(reports, ignore_index=True)
    full.to_csv("reporte_madaline.csv", index=False)
    print("\nReporte guardado en reporte_madaline.csv")