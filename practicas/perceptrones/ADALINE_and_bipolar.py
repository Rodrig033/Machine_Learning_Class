import numpy as np
import matplotlib.pyplot as plt

# Compuerta AND bipolar

X = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])

# Salidas esperadas (predicciones)
y = np.array([-1, -1, -1, 1])

# Parámetros iniciales
w = np.array([0, 0])
b = 0.0
eta = 0.1
epochs = 100


# Funcion de activación
def funcion_activacion_lineal(z):
    if z >= 0:
        return 1 
    elif z < 0:
        return -1
    return 0

# Función de perdida por época:
def perdida_por_epoca(lost):
    l = lost / 4
    print("La pérdida es: ", l)
    return l
        