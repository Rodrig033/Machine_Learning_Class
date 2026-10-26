import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras import layers
from keras.layers import Dense

celsius = np.array([-40, -10, 0, 15, 8, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# units: número de neuronas (dimensiones de salida) de la capa

"""
modelo = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(units=1)
])
"""

oculta1 = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(units=3)
])

oculta2 = keras.Sequential([
    keras.layers.Dense(units=3)
])

salida = keras.Sequential([
    keras.layers.Dense(units=1)
])

modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

print("¡Modelo entrenado!")

plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("¡Hagamos una predicción!")
X_test = np.array([[100.0]])
resultado = modelo.predict(X_test)
print(f"¡El resultado para 100°C es {resultado[0][0]} fahrenheit!")

print("Variables internas del modelo")
print(modelo.get_weights())