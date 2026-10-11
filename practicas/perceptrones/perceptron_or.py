import numpy as np
import matplotlib.pyplot as plt

import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas esperadas (predicciones)
y = np.array([0, 1, 1, 1])

# Parámetros iniciales
w = np.array([0.2, 0.1])   # pesos iniciales
b = -0.3                    # bias inicial
eta = 0.1                 # learning rate
epochs = 10                # iteraciones

# Funición de activación
def step_function(z):
    if z >= 0:
        return 1
    return 0

# Entrenamiento
for epoch in range(epochs):
    print(f"\n===== Iteración {epoch+1} =====")
    total_error = 0

    # Gráfica

    # Separar puntos por clase
    class0 = X[y == 0]
    class1 = X[y == 1]

    plt.figure(figsize=(6,6))

    # Clase 0
    plt.scatter(
        class0[:,0],
        class0[:,1],
        label="Clase 0",
        s=100
    )

    # Clase 1
    plt.scatter(
        class1[:,0],
        class1[:,1],
        label="Clase 1",
        s=100
    )

    # Frontera de decisión

    x_values = np.linspace(-0.5, 1.5, 100)

    # w1*x1 + w2*x2 + b = 0
    # despejamos x2

    if w[1] != 0:
        y_values = -(w[0] * x_values + b) / w[1]

        plt.plot(
            x_values,
            y_values,
            label="Frontera decisión"
        )

    # Configuración
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Perceptrón OR")

    plt.grid(True)
    plt.legend()

    plt.show()    

    for i in range(len(X)):

        x_i = X[i]

        # Forward
        z = np.dot(w, x_i) + b
        y_pred = step_function(z)

        # Error
        error = y[i] - y_pred

        total_error += abs(error)

        # Backward
        w = w + eta * error * x_i
        b = b + eta * error

        print(f"\nEntrada: {x_i}")
        print(f"Salida esperada: {y[i]}")
        print(f"Predicción: {y_pred}")
        print(f"Error: {error}")

        print(f"Pesos nuevos: {w}")
        print(f"Bias nuevo: {b}")

    # Si no hay error termina
    if total_error == 0:
        print("\nEntrenamiento completado.")
        break

# Resultados finales
print("\n===== RESULTADOS FINALES =====")
print("Pesos:", w)
print("Bias:", b)

