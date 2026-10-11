import numpy as np
import matplotlib.pyplot as plt
 

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# Aquí creamos UNA sola red
# Parámetros iniciales
eta = 1.0        # learning rate
epochs = 10000                # iteraciones

np.random.seed(42)

# Entrada (capa oculta)
W1 = np.random.randn(2, 2) * 0.5

b1 = np.zeros(2)

W2 = np.random.randn(2, 1) * 0.5

b2 = np.zeros(1)

# Funición de activación (sigmoide)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivada de la sigmoide
def sigmoid_derivative(a):
    return a * (1 - a)

for epoch in range(epochs):

    total_error = 0.0
    loss_history = []

    for i in range(len(X)):

        x_i = X[i]

        # 1. FORWARD
        z1 = np.dot(x_i, W1) + b1

        # Aplicamos la función de activación:
        h = sigmoid(z1)

        z2 = np.dot(h, W2) + b2
        y_pred = sigmoid(z2)

        # 2. ERROR / LOSS
        error = y_pred - y[i]

        loss = 0.5 * error**2
        total_error += loss
        

        # 3. BACKPROPAGATION
        delta2 = error * sigmoid_derivative(y_pred)
        dW2 = np.outer(h, delta2)
        db2 = delta2 # Bias de salida


        """Los delta nos indican cuánto debe corregirse una neurona durante el aprendizaje"""

        delta1 = (W2.flatten() * delta2) * sigmoid_derivative(h)
        dW1 = np.outer(x_i, delta1)
        db1 = delta1

        # 4. UPDATE
        W2 -= eta * dW2
        b2 -= eta * db2

        W1 -= eta * dW1
        b1 -= eta * db1

        loss_history.append(total_error)
    
    # revisar progreso de la época

    print(f"\nEpoch {epoch} - Loss: {total_error}")
    print("Resultados finales:")

    for i in range(len(X)):

        x_i = X[i]

        z1 = np.dot(x_i, W1) + b1
        h = sigmoid(z1)

        z2 = np.dot(h, W2) + b2
        y_pred = sigmoid(z2)

        print(
            f"Entrada: {x_i} | "
            f"Esperado: {y[i]} | "
            f"Predicción: {y_pred[0]:.4f}"
        )

print("\nParámetros finales:")
print("W1:")
print(W1)

print("b1:")
print(b1)

print("W2:")
print(W2)

print("b2:")
print(b2)

plt.plot(loss_history)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Entrenamiento de la MLP para XOR")
plt.grid(True)
plt.show()