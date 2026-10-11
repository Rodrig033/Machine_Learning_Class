import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Dataset manual 
data = {
    "Edad": ["<=30", "<=30", "31-40", ">40", ">40", ">40", "31-40", "<=30", "<=30", ">40", "<=30", "31-40", "31-40", ">40"],
    "Ingreso": ["alto", "alto", "alto", "medio", "bajo", "bajo", "bajo", "medio", "bajo", "medio", "medio", "medio", "alto", "medio"],
    "Estudiante": ["no", "no", "no", "no", "sí", "sí", "sí", "no", "sí", "sí", "sí", "no", "sí", "no"],
    "Crédito": ["justo", "excelente", "justo", "justo", "justo", "excelente", "excelente", "justo", "justo", "justo", "excelente", "excelente", "justo", "excelente"],
    "Clase": ["no", "no", "sí", "sí", "sí", "no", "sí", "no", "sí", "sí", "sí", "sí", "sí", "no"]
}

df = pd.DataFrame(data)

# Codificar variables categóricas
le = LabelEncoder()

df_encoded = df.apply(le.fit_transform)

# Separar X y y
X = df_encoded.drop("Clase", axis=1)
y = df_encoded["Clase"]

# Modelo ID3-like
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Visualización
plt.figure(figsize=(12,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["no", "sí"],
    filled=True
)
plt.show()