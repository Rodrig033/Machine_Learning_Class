import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("/Users/farid/Documents/Machine_Learning/nasa_exoplanet_intelligence.csv")

# Variables continuas
features = [
    "disc_year",
    "orbital_period_days",
    "planet_radius_earth",
    "planet_mass_earth",
    "equilibrium_temp_k",
    "orbital_eccentricity",
    "semi_major_axis_au",
    "star_temp_k",
    "star_radius_sun",
    "star_mass_sun",
    "star_age_gyr",
    "star_surface_gravity",
    "star_metallicity",
    "dist_from_earth_pc",
    "star_vmag",
    "ra",
    "dec"
]

X = df[features]
y = df["habitable_zone_flag"]

# Imputación
imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# División Entrenamiento / Prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Configuración a comparar
configs = {

    "MLP_1": MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    ),

    "MLP_2": MLPClassifier(
        hidden_layer_sizes=(64,32),
        activation="relu",
        solver="adam",
        max_iter=700,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    ),

    "MLP_3": MLPClassifier(
        hidden_layer_sizes=(128,64,32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0005,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    ),

    "MLP_4": MLPClassifier(
        hidden_layer_sizes=(64,32),
        activation="tanh",
        solver="adam",
        max_iter=700,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
}

results = []
best_model = None
best_f1 = 0

for name, mlp in configs.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        acc,
        prec,
        rec,
        f1
    ])

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model = mlp
        best_name = name

# Tabla de resumen
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\n")
print(results_df)

# Mejor modelo
print("\nMEJOR MODELO")
print(best_name)
print("F1 =", best_f1)

# Gráfica de pérdida
plt.figure(figsize=(10,5))

plt.plot(
    best_model.loss_curve_
)

plt.title(
    f"Training Loss - {best_name}"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Error de validación
validation_error = (
    1 -
    np.array(
        best_model.validation_scores_
    )
)

plt.figure(figsize=(10,5))

plt.plot(
    validation_error
)

plt.title(
    f"Validation Error - {best_name}"
)

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.grid()
plt.show()

# Curva combinada
plt.figure(figsize=(10,5))

plt.plot(
    best_model.loss_curve_,
    label="Training Loss"
)

plt.plot(
    validation_error,
    label="Validation Error"
)

plt.title(
    "Training vs Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()