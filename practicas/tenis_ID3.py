import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("/Users/farid/Documents/Machine_Learning/tenis.csv")
print(df.head())

le = LabelEncoder()

for columna in df.columns:
    df[columna] = le.fit_transform(
        df[columna]
    )

X = df.drop(columns=["jugar-tenis"])
atributos = X.columns
target = df["jugar-tenis"]

modelo = DecisionTreeClassifier(
    criterion = "entropy",  # parecido a ID3
    random_state = 0
)

# Entrenar
modelo.fit(X, target)

def entropia(columna):
    valores = Counter(columna)
    total = len(columna)
    entropy = 0

    for frecuencia in valores.values():
        p = frecuencia / total
        entropy -= p * math.log2(p)

    return entropy

def calcular_ganancia(df, atributo, target):
    entropia_total = entropia(target)
    valores = df[atributo].unique()
    entropia_ponderada = 0
    total = len(df)


    for valor in valores:
        # Subdataset
        subset = df[df[atributo] == valor]
        # Columna target
        subset_target = subset["jugar-tenis"]
        # Calcular peso
        peso = len(subset) / total
        # Calcular la entropía del subset
        entropia_subset = entropia(subset_target)
        # Entropía ponderada
        entropia_ponderada += peso * entropia_subset
    
    # Ganancia IG(S,A)= H(S)-\sum \frac{|S_v|}{|S|}H(S_v)
    ganancia = entropia_total - entropia_ponderada

    return ganancia

print("\nGANANCIA DE INFORMACIÓN\n")

for atributo in atributos:

    ganancia = calcular_ganancia(
        df,
        atributo,
        target
    )

    print(f"{atributo}: {ganancia:.4f}")


# Visualizar árbol:

plt.figure(figsize=(12,8))

plot_tree(
    modelo,
    feature_names = X.columns,
    class_names = ["No", "Sí"],
    filled = True
)

plt.show()