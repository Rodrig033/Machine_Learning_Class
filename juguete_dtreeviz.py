import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from dtreeviz import model as dtreeviz_model

# Dataset manual 
data = {
    "Edad": ["<=30", "<=30", "31-40", ">40", ">40", ">40", "31-40", "<=30", "<=30", ">40", "<=30", "31-40", "31-40", ">40"],
    "Ingreso": ["alto", "alto", "alto", "medio", "bajo", "bajo", "bajo", "medio", "bajo", "medio", "medio", "medio", "alto", "medio"],
    "Estudiante": ["no", "no", "no", "no", "sí", "sí", "sí", "no", "sí", "sí", "sí", "no", "sí", "no"],
    "Crédito": ["justo", "excelente", "justo", "justo", "justo", "excelente", "excelente", "justo", "justo", "justo", "excelente", "excelente", "justo", "excelente"],
    "Clase": ["no", "no", "sí", "sí", "sí", "no", "sí", "no", "sí", "sí", "sí", "sí", "sí", "no"]
}

df = pd.DataFrame(data)

# Codificar
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X = df_encoded.drop("Clase", axis=1)
y = df_encoded["Clase"]

# Modelo
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)


viz_model = dtreeviz_model(
    model,
    X,
    y,
    feature_names=X.columns,
    target_name="Clase",
    class_names=["no", "sí"]
)

viz = viz_model.view()
viz.show()