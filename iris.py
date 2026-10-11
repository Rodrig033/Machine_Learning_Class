import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# Crea un data frame
df = pd.DataFrame(iris.data, columns= iris.feature_names) 
df["species"] = iris.target
df["species_name"] = df["species"].replace({0: "Setosa", 2:"Virginica", 1: "Versicolor"})

setosa_points = df[df["species_name"] == "Setosa"]
virginica_points = df[df["species_name"] == "Virginica"]
versicolor_points = df[df["species_name"] == "Versicolor"]

col = "petal width (cm)"
plt.bar(["Setosa", "Virginica", "Versicolor"],[setosa_points[col].mean(), virginica_points[col].mean(), versicolor_points[col].mean()])
plt.title(col + "mean by class")
plt.xlabel("species")
plt.ylabel(col + "mean")
plt.show()

col = "petal width (cm)"
plt.hist(setosa_points[col], bins=10, color="blue", label="Setosa")
plt.hist(virginica_points[col], bins=10, color="red", label="Virginica")
plt.hist(versicolor_points[col], bins=10, color="green", label="Versicolor")
plt.title(col + "Histogram by specie")
plt.legend()
plt.xlabel(col)
plt.ylabel("frequency")
plt.show()

col = "sepal width (cm)"
plt.boxplot([setosa_points[col], virginica_points[col], versicolor_points[col]]
            , labels= ["Setosa", "Virginica", "Versicolor"])
plt.title(col + "Comparassion")
plt.ylabel(col)
plt.show()

x_col = "petal length (cm)"
y_col = "sepal length (cm)"

plt.scatter(setosa_points[x_col], setosa_points[y_col], color = "red", label = "Setosa")
plt.title(x_col + " vs " + y_col)
plt.ylabel(y_col)
plt.xlabel(x_col)
plt.show()