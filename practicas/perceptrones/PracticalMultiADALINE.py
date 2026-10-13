import numpy as np
from sklearn.datasets import make_classification # Para creación de datos sinteticos
from sklearn.preprocessing import StandardScaler

class PracticaMADALINE:
    def __init__(self, n_outputs, tasa_aprendizaje=0.01, n_epocas=1000):
        self.n_outputs = n_outputs
        self.lr = tasa_aprendizaje
        self.n_epocas = n_epocas
        self.pesos = None
        self.sesgos = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # One-hot encoding si y es vector de etiquetas
        if len(y.shape) == 1:
            y_estimada = np.eye(self.n_outputs)[y]
        else:
            y_estimada = y
        
        # Inicializar parámetros
        self.pesos = np.random.randn(self.n_outputs, n_features) * 0.01
        self.sesgos = np.zeros(self.n_outputs)
        
        # Entrenamiento
        for epoch in range(self.n_epocas): # por cada época 
            for i in range(n_samples): # por cada muestra
                # Propagación hacia adelante Forward pass
                z = np.dot(self.pesos, X[i]) + self.sesgos
                # En esta sección debes llamar la función de activación
                # Realmente, para ADALINE, la función de activación es lineal, por lo que no necesitamos aplicar ninguna función adicional aquí.                
                error = y_estimada[i] - z # Para ADALINE, el error es la diferencia entre la salida deseada y la salida lineal.
                # Para este experimento, vas a explorar cambiando la función de activación.
                #
                #
                #
                # Actualizar los pesos y sesgos.
                for j in range(self.n_outputs):
                    self.pesos[j] += self.lr * error[j] * X[i]
                    self.sesgos[j] += self.lr * error[j]
        
        return self
    
    def predict(self, X):
        salida = np.dot(X, self.pesos.T) + self.sesgos
        return np.argmax(salida, axis=1)

# Creación de datos sintéticos usando la biblioteca sklearn
X, y = make_classification(n_samples=100, n_features=4, n_classes=3,
                          n_redundant=0, n_clusters_per_class=1,
                          random_state=42)    

# Escalar datos
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Entrenar modelo
modelo = PracticaMADALINE(n_outputs=3, tasa_aprendizaje=0.01, n_epocas=1000)
modelo.fit(X_scaled, y)

# Evaluar
predictions = modelo.predict(X_scaled)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.3f}")

print("Pesos finales:")
print(modelo.pesos)
print("Sesgos finales:")
print(modelo.sesgos)