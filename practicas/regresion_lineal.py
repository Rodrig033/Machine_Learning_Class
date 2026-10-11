import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Almacenar los datos en un diccionario:
data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)

# Escalado del número de equipos afectados
df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype(int)

# Escalado del coste
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype(int)
print(df.head(10))

# Representación gráfica del conjunto de datos
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()

# Construcción del modelo
lin_reg = LinearRegression()

# Entrenar nuestro modelo
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)

# Parámetro theta 0
lin_reg.intercept_

# Parámetro theta 1
lin_reg.coef_

# Predicción para el elemento máximo y mínimo del conjunto de datos de entrenamiento
X_min_max = np.array([[df['n_equipos_afectados'].min()], [df['n_equipos_afectados'].max()]])
y_train_pred = lin_reg.predict(X_min_max)

# Representación gráfica de la hipotesís generada
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()

# Predicción de nuevos ejemplos
x_new = np.array([[1200]])

# Predicción del coste que tendría el incidente
coste = lin_reg.predict(x_new)

print("El coste del incidente sería:", int(coste[0]), "€")

plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.title("Nuevos datos")
plt.show()