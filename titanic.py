import pandas as pd
import matplotlib.pyplot as plt # Graficar 
from collections import Counter


df = pd.read_csv("titanic_test.csv")

# Primeros 5 filas
print(df.head())

# Estadística descriptiva
print(df.describe()) 

df2 = pd.read_csv("gender_submission.csv")

print("Sobrevivientes: ",df2["Survived"].sum())
print(df.info())
print(df.value_counts())
print(df["Pclass"].value_counts())
print(df["Sex"].value_counts())
print((df["Pclass"] == 1).value_counts())
print(df["Pclass"] == 1)

# Imprimir el ID de los pasajeros
""" En este print es como si hubiera una 
    condición y así poder recuperar el ID """


# ¿Sobrevivieron más los pasajeros de primera clase?
id2_1class = print(df[df["Pclass"] == 1]["PassengerId"])
id2_2class = print(df[df["Pclass"] == 2]["PassengerId"])
id2_3class = print(df[df["Pclass"] == 3]["PassengerId"])

# Sobrevivieron más los de tercera clase.
print(" --- Primera clase ---")
c1ass_surv = df2[df2["PassengerId"] == id2_1class]
ids_first_class = df[df["Pclass"] == 1]["PassengerId"]
sur_first_class = (df2[df2["PassengerId"].isin(ids_first_class)]["Survived"] == 1).sum()

ids_second_class = df[df["Pclass"] == 2]["PassengerId"]
sur_second_class = (df2[df2["PassengerId"].isin(ids_second_class)]["Survived"] == 1).sum()

ids_third_class = df[df["Pclass"] == 3]["PassengerId"]
sur_third_class = (df2[df2["PassengerId"].isin(ids_third_class)]["Survived"] == 1).sum()

print(c1ass_surv)
print("---- Sobrevivientes ----")
print("Primera clase: ", sur_first_class)
print("Segunda clase: ", sur_second_class)
print("Tercera clase: ",sur_third_class)

plt.figure()
# Grafico de barras: 
plt.bar(["1st class ", "2nd class", "3rd class"], [sur_first_class, sur_second_class, sur_third_class])
plt.title("Titanic survivors")
plt.xlabel("Clases")
plt.ylabel("Número de pasajeros")
plt.show()

# Número de pasajeros por clase:
ps_first_class = (df["Pclass"] == 1).sum()
ps_second_class = (df["Pclass"] == 2).sum()
ps_third_class = (df["Pclass"] == 3).sum()

plt.figure()
plt.bar(["1st class ", "2nd class", "3rd class"], [ps_first_class, ps_second_class, ps_third_class], color = 'g')
plt.title("Titanic passenger")
plt.xlabel("Clases")
plt.ylabel("Número de pasajeros")
plt.show()

# Sobrevivientes según el sexo
sb_masculino_sex = (df["Sex"] == "male").sum()
sb_femenino_sex = (df["Sex"] == "female").sum()

plt.figure()
plt.bar(["Male", "Female"], [sb_masculino_sex, sb_femenino_sex])
plt.title("Titanic gender")
plt.xlabel("Sex")
plt.ylabel("Número de pasajeros")
plt.show()

# Sobrevivencia por rango de edad

""" 
- Analizar las edades
- Proponer rango de edades
- ¿Qué hacer con las Nan? 
"""
edades = df["Age"].value_counts()

# Rellenamos los Nan con la media
df["Age"] = df["Age"].fillna(df["Age"].mean())

bins = [0, 12, 18, 35, 60, 100]
labels = ["Niños", "Adolescentes", "Jóvenes Adultos", "Adultos", "Adultos Mayores"]

df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
conteo = df["AgeGroup"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(conteo.index, conteo.values)
plt.title("Supervivencia por rango de edad")
plt.xlabel("Rango de edad")
plt.ylabel("Cantidad de personas")
plt.xticks(rotation=45)
plt.show()

# Voy a mezclar los dataframes para facilitar las consultas:
df_merged = df.merge(df2[["PassengerId", "Survived"]], on="PassengerId")


# Tarifas
plt.hist(df_merged["Fare"], bins=20)
plt.title("Distribución de tarifas")
plt.xlabel("Fare")
plt.ylabel("Frecuencia")
plt.show()

# Distribución de edad
plt.hist(df_merged["Age"].dropna(), bins=20)
plt.title("Distribución de edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Supervivencia por puerto de embarco
df_merged.groupby("Embarked")["Survived"].mean().plot(kind="bar")
plt.title("Supervivencia por puerto de embarque")
plt.show()