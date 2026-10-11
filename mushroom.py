import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

data = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises",
    "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color",
    "population", "habitat"
]

df = pd.read_csv(url, names=data)
df = df.replace("?", pd.NA)
df = df.dropna()

print(df.head())

columnas = df.columns
X = df.drop("class", axis=1)
y = df["class"]

# Dividir los datos (Train / Test):
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Contar frecuencias:
def contar_frecuencias(lista):
    frecuencias = {}
    for item in lista:
        frecuencias[item] = frecuencias.get(item, 0) + 1
    return frecuencias

def calcular_frecuencias_condicionales(df, target_col):
    frecuencias = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    conteo_clases = df[target_col].value_counts().to_dict()
    
    for _, row in df.iterrows():
        clase = row[target_col]
        
        for col in df.columns:
            if col == target_col:
                continue
                
            valor = row[col]
            frecuencias[clase][col][valor] += 1
    
    return frecuencias, conteo_clases

def calcular_probabilidades(df, target_col):
    probabilidades = {}
    conteo_clases = df[target_col].value_counts().to_dict()
    total_datos = len(df)

    for clase in conteo_clases:
        probabilidades[clase] = {}
        df_clase = df[df[target_col] == clase]
        
        for col in df.columns:
            if col == target_col:
                continue
            
            probabilidades[clase][col] = {}
            
            valores_unicos = df[col].unique()
            total_clase = len(df_clase)
            
            for valor in valores_unicos:
                conteo = len(df_clase[df_clase[col] == valor])
                
                prob = (conteo + 1) / (total_clase + len(valores_unicos))
                
                probabilidades[clase][col][valor] = prob

    return probabilidades, conteo_clases

probabilidades, conteo_clases = calcular_probabilidades(df, "class")

def predecir(instancia, probabilidades, conteo_clases):
    resultados = {}
    total_datos = sum(conteo_clases.values())
    
    for clase in probabilidades:
        log_prob = math.log(conteo_clases[clase] / total_datos)
        
        for col, valor in instancia.items():
            if col == "class":
                continue
            
            if valor in probabilidades[clase][col]:
                log_prob += math.log(probabilidades[clase][col][valor])
            else:
                log_prob += math.log(1e-6)
        
        resultados[clase] = log_prob
    
    return max(resultados, key=resultados.get)
        
test = df.iloc[0].to_dict()
pred = predecir(test, probabilidades, conteo_clases)

print("Predicción:", pred)
print("Real:", test["class"])

correctos = 0

for _, row in df.iterrows():
    instancia = row.to_dict()
    pred = predecir(instancia, probabilidades, conteo_clases)
    
    if pred == instancia["class"]:
        correctos += 1

accuracy = correctos / len(df)

y_true = []
y_pred = []

for _, row in test_df.iterrows():
    instancia = row.to_dict()
    
    pred = predecir(instancia, probabilidades, conteo_clases)
    
    y_pred.append(pred)
    y_true.append(instancia["class"])
    
print("Accuracy: ", accuracy)


# Matriz de confusión:
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.show()

# Precision
precision = precision_score(y_true, y_pred, pos_label='p')
print("Precision: ", precision)

# Recall
recall = recall_score(y_true, y_pred, pos_label='p')
print("Recall: ", recall)

# F1-Score
f1_score = f1_score(y_true, y_pred, pos_label='p')
print("F1-score: ", f1_score)