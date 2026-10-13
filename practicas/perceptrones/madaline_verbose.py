import numpy as np


def fmt(v, d=4):
    """Formatea un vector con signo y d decimales."""
    return np.array2string(
        np.asarray(v, dtype=float),
        formatter={"float_kind": lambda x: f"{x:+.{d}f}"},
        separator=", ",
    )


def sgn(v):
    return f"{int(v):+d}"


class MADALINE:
    def __init__(self, n_inputs, n_adalines, learning_rate=0.1, seed=None):
        self.n_inputs = n_inputs
        self.n_adalines = n_adalines
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-0.5, 0.5, (n_adalines, n_inputs))
        self.bias = np.zeros(n_adalines)

        self.error_history = []
        self.epochs_history = []

    @staticmethod
    def bipolar_step(x):
        return np.where(x >= 0, 1, -1)

    def forward(self, X):
        z = X @ self.weights.T + self.bias
        h = self.bipolar_step(z)
        return z, h

    def predict(self, X):
        _, h = self.forward(X)
        return self.bipolar_step(np.sum(h, axis=1))

    def calculate_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # ------------------------------------------------------------------
    # Utilidades de impresión
    # ------------------------------------------------------------------
    def print_parameters(self, title):
        print(f"\n{title}")
        for j in range(self.n_adalines):
            print(f"  ADALINE {j + 1}: w = {fmt(self.weights[j])}, b = {self.bias[j]:+.4f}")

    def _linear_expr(self, j, x):
        terms = " + ".join(
            f"({w:+.4f})({xi:+g})" for w, xi in zip(self.weights[j], x)
        )
        return f"{terms} + ({self.bias[j]:+.4f})"

    # ------------------------------------------------------------------
    # Entrenamiento con explicación matemática
    # ------------------------------------------------------------------
    def train_verbose(self, X, y, epochs=3):
        self.error_history = []
        self.epochs_history = []
        lr = self.learning_rate
        N = len(X)

        print("=" * 70)
        print("ENTRENAMIENTO MADALINE (Madaline Rule I)")
        print("=" * 70)
        print(f"Tasa de aprendizaje η = {lr}")
        print(f"Entradas = {self.n_inputs}, ADALINEs = {self.n_adalines}, épocas máx. = {epochs}")
        self.print_parameters("Parámetros iniciales:")

        for epoch in range(1, epochs + 1):
            print("\n" + "#" * 70)
            print(f"ÉPOCA {epoch}")
            print("#" * 70)

            n_updates = 0

            for p, (x, target) in enumerate(zip(X, y), start=1):
                print(f"\n--- Época {epoch} | Patrón {p}/{N} ---")
                print(f"x = {x},  objetivo t = {sgn(target)}")

                # Paso 1: salida lineal
                z = np.dot(self.weights, x) + self.bias
                print("\nPaso 1) Salida lineal de cada ADALINE:  z_j = w_j · x + b_j")
                for j in range(self.n_adalines):
                    print(f"  z{j + 1} = {self._linear_expr(j, x)} = {z[j]:+.4f}")

                # Paso 2: salida bipolar
                h = self.bipolar_step(z)
                print("\nPaso 2) Voto de cada ADALINE:  h_j = +1 si z_j ≥ 0, -1 si z_j < 0")
                for j in range(self.n_adalines):
                    cmp = "≥" if z[j] >= 0 else "<"
                    print(f"  h{j + 1}: z{j + 1} = {z[j]:+.4f} {cmp} 0  →  h{j + 1} = {sgn(h[j])}")

                # Paso 3: salida de la red
                total = int(np.sum(h))
                output = int(self.bipolar_step(total))
                suma = " + ".join(f"({sgn(v)})" for v in h)
                print("\nPaso 3) Votación mayoritaria:  ŷ = sgn(Σ h_j)")
                print(f"  Σh = {suma} = {total:+d}  →  ŷ = {sgn(output)}", end="")
                print("   (un empate, Σh = 0, cuenta como +1)" if total == 0 else "")

                # Paso 4: comparación
                print("\nPaso 4) Comparación:")
                if output == target:
                    print(f"  ŷ = {sgn(output)} = t = {sgn(target)}  →  CORRECTO, no se modifica nada.")
                    continue

                print(f"  ŷ = {sgn(output)} ≠ t = {sgn(target)}  →  ERROR, hay que corregir.")
                n_updates += 1

                # Paso 5: selección de ADALINEs a corregir
                if target == 1:
                    candidates = np.where(z < 0)[0]
                    idx = candidates[np.argmin(np.abs(z[candidates]))]
                    units = [idx]
                    print("\nPaso 5) La red dijo -1 y debía decir +1.")
                    print("  Se corrige UNA sola ADALINE: entre las que votan -1,")
                    print("  la de menor |z| (la más fácil de voltear).")
                    for j in candidates:
                        print(f"    ADALINE {j + 1}: |z| = {abs(z[j]):.4f}")
                    print(f"  → se elige la ADALINE {idx + 1}.")
                else:
                    units = list(np.where(z >= 0)[0])
                    print("\nPaso 5) La red dijo +1 y debía decir -1.")
                    print("  Se corrigen TODAS las ADALINEs que votan +1.")
                    print(f"  → ADALINEs a corregir: {[int(j) + 1 for j in units]}")

                # Paso 6: regla delta
                print("\nPaso 6) Regla delta:  e_j = t - z_j,  Δw_j = η · e_j · x,  Δb_j = η · e_j")
                for j in units:
                    e = target - z[j]
                    dw = lr * e * x
                    db = lr * e
                    w_old, b_old = self.weights[j].copy(), self.bias[j]

                    self.weights[j] = w_old + dw
                    self.bias[j] = b_old + db

                    print(f"\n  ADALINE {j + 1}:")
                    print(f"    e  = t - z = ({sgn(target)}) - ({z[j]:+.4f}) = {e:+.4f}")
                    print(f"    Δw = η·e·x = {lr}·({e:+.4f})·{x} = {fmt(dw)}")
                    print(f"    Δb = η·e   = {lr}·({e:+.4f}) = {db:+.4f}")
                    print(f"    w_nuevo = {fmt(w_old)} + {fmt(dw)} = {fmt(self.weights[j])}")
                    print(f"    b_nuevo = ({b_old:+.4f}) + ({db:+.4f}) = {self.bias[j]:+.4f}")

            # --------------------------------------------------------------
            # Resumen de la época
            # --------------------------------------------------------------
            print("\n" + "-" * 70)
            print(f"RESUMEN DE LA ÉPOCA {epoch}")
            print("-" * 70)
            print(f"Patrones que provocaron actualización: {n_updates} de {N}")
            self.print_parameters("Parámetros al final de la época:")

            y_pred = self.predict(X)
            diff = y - y_pred
            sq = diff ** 2
            epoch_error = self.calculate_error(y, y_pred)

            print("\nEvaluación de toda la red con los nuevos parámetros:")
            print(f"  y       = {y}")
            print(f"  ŷ       = {y_pred}")
            print(f"  y - ŷ   = {diff}")
            print(f"  (y-ŷ)²  = {sq}")
            print(f"  Error = (1/N)·Σ(y-ŷ)² = {int(sq.sum())}/{N} = {epoch_error:.4f}")
            print(f"  Patrones mal clasificados: {int(np.sum(y != y_pred))}")

            self.error_history.append(epoch_error)
            self.epochs_history.append(epoch)

            if epoch_error == 0:
                print(f"\n>>> La red convergió en la época {epoch}: error = 0.")
                break
        else:
            print(f"\n>>> La red no convergió en {epochs} épocas (error final = {epoch_error:.4f}).")

    def print_decision_lines(self):
        print("\nRectas de decisión de cada ADALINE:  w1·x1 + w2·x2 + b = 0")
        for j in range(self.n_adalines):
            w1, w2 = self.weights[j]
            print(f"  ADALINE {j + 1}: ({w1:+.4f})·x1 + ({w2:+.4f})·x2 + ({self.bias[j]:+.4f}) = 0")


if __name__ == "__main__":
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ])
    y = np.array([-1, 1, 1, -1])

    madaline = MADALINE(n_inputs=2, n_adalines=2, learning_rate=0.1, seed=0)
    madaline.train_verbose(X, y, epochs=3)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    madaline.print_parameters("Parámetros finales:")
    madaline.print_decision_lines()
    print(f"\nPredicciones: {madaline.predict(X)}")
    print(f"Objetivos:    {y}")