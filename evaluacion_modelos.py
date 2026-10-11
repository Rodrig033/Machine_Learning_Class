import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from dtreeviz import model as dtreeviz_model
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv")

# Target (usa tu columna real)
y = (df["habitable_zone_flag"])
X = df.drop("equilibrium_temp_k", axis=1)
X = X.select_dtypes(include=[np.number])

# Limpiar NaN
X = X.fillna(X.mean())


# Modelos 
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))

    ]),
    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(criterion="entropy"))
    ])

}

# Cross Validation mostrar los resultados para cada split
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    
    print(f"\n{name}")
    print("Scores:", scores)
    print("Mean:", scores.mean())

# K-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nModelo: {name}")

    fold = 1
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        print(f"Fold {fold}: {acc}")
        fold += 1

    print(f"Promedio: {np.mean(scores)}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("------")

# Leave-one-out
loo = LeaveOneOut()
print(" Leave-one-out ")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=loo)
    print(f"\nName: {name} (LOO)")
    print(f"Mean: {scores.mean()}")
    
