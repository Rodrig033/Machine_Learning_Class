import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.style.use("ggplot")
sns.set_theme()

# Datasets
books = pd.read_csv("Books.csv")
user = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

print(books.head())
print(user.head())
print(ratings.head())

# Dimensiones
print("Books: ", books.shape)
print("Users: ", user.shape)
print("Rating: ", ratings.shape)

# Tipos de datos
print("\n----------- Tipos de datos -----------")
books.info()
user.info()
ratings.info()


# Valores nulos
print("------------------------")
print(books.isnull().sum())
print("------------------------")
print(user.isnull().sum())
print("------------------------")
print(ratings.isnull().sum())

# Libros más calificados
top_rated_books = (
    ratings.groupby("ISBN").size().sort_values(ascending=False).head(10).reset_index(name="Rating-Count"))

top_books_info = books.merge(top_rated_books, on= "ISBN")
top_books_info["Book-Title"] = top_books_info["Book-Title"].str.slice(0, 40)

print(top_books_info[["Book-Title", "Book-Author"]])


plt.figure(figsize=(10, 6))
plt.bar(top_books_info["Book-Title"], top_books_info["Rating-Count"])
plt.xlabel("Libro")
plt.ylabel("Número de Ratings")
plt.title("Libros más calificados")
plt.xticks(rotation=65)
plt.tight_layout()
plt.show()

# Usuarios que más califican
top_users = (
    ratings.groupby("User-ID").size().sort_values(ascending=False).head(10))

plt.figure()
top_users.plot(kind="bar", color = "Green")
plt.xlabel("User ID")
plt.ylabel("Número de ratings")
plt.title("Usuarios que más califican")
plt.tight_layout()
plt.show()

# Autores más populares
author_popularity = (
    books.merge(ratings, on= "ISBN").groupby("Book-Author").size().sort_values(ascending=False).head(10))

print(author_popularity)

plt.figure()
author_popularity.plot(kind="barh", color= "Orange")
plt.xlabel("Número de ratings")
plt.xticks(rotation=35)
plt.ylabel("Autor")
plt.title("Autores más populares")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Edad de los usuarios
users_clean = user[(user["Age"] > 5) & (user["Age"] < 90)]

plt.figure()
users_clean["Age"].hist(bins=20, color = 'red')

plt.xlabel("Edad")
plt.ylabel("Número de usuarios")
plt.title("Distribución de edades")
plt.show()

# Años de publicación
""" Convertir columna en número """
books["Year-Of-Publication"] = pd.to_numeric(
    books["Year-Of-Publication"],
    errors="coerce"
)

books_clean = books[books["Year-Of-Publication"] < 2025]

plt.figure()
books_clean["Year-Of-Publication"].hist(bins=30)

plt.xlabel("Año")
plt.ylabel("Número de libros")
plt.title("Distribución por año de publicación")
plt.tight_layout()
plt.show()

# Año con más publicación


# Paises o ciudades con más usuarios
user["Country"] = user["Location"].apply(lambda x: x.split(",")[-1].strip())
top_countries = user["Country"].value_counts().head(10)


plt.figure()
top_countries.plot(kind="bar", color = "Brown")
plt.xlabel("País")
plt.xticks(rotation=35)
plt.ylabel("Número de usuarios")
plt.title("Top países de usuarios")
plt.tight_layout()
plt.show()

# HeatMap Usuario-Libro
""" Filtra los usuarios que están activos """
active_user = ratings["User-ID"].value_counts()
active_user = active_user[active_user > 200].index

popular_books = ratings["ISBN"]

""" Verificamos los libros con muchos ratings """
popular_books = ratings["ISBN"].value_counts()
popular_books = popular_books[popular_books > 100].index

filtered_ratings = ratings[
    (ratings["User-ID"].isin(active_user)) &
    (ratings["ISBN"].isin(popular_books))
]

""" Crea la matriz """
matrix = filtered_ratings.pivot_table(index = "User-ID", columns = "ISBN", values = "Book-Rating")
matrix = matrix.fillna(0)

plt.figure()
plt.imshow(matrix, aspect="auto")
plt.xlabel("Libros")
plt.ylabel("Usuarios")
plt.title("HeatMap Usuario-Libro")
plt.colorbar()
plt.show()


print("\nUsuarios activos:", matrix.shape[0])
print("Libros populares:", matrix.shape[1])


# Clustering de Usuarios lectores (K-means)
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix)

pca = PCA(n_components=20)
matrix_pca = pca.fit_transform(matrix_scaled)

kmeans = KMeans(n_clusters= 5, random_state=42)

cluster = kmeans.fit_predict(matrix_pca)

matrix["Cluster"] = cluster

# PCA
pca_vis = PCA(n_components= 2)
reduced = pca_vis.fit_transform(matrix_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(reduced[:,0], reduced[:,1], c= cluster, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters de usuarios lectores")
plt.legend(*scatter.legend_elements(), title = "Cluster")
plt.show()


print("\nVarianza explicada por PCA:")
print(pca.explained_variance_ratio_)
print("Varianza total explicada:", pca.explained_variance_ratio_.sum())

# Visualizar los treinta mejores libros en el clustering
top_books = ratings["ISBN"].value_counts().head(30).index

heatmap_data = ratings[
    ratings["ISBN"].isin(top_books)
].pivot_table(
    index= "User-ID",
    columns = "ISBN",
    values= "Book-Rating"
).fillna(0)


plt.figure(figsize=(10,8))

plt.imshow(heatmap_data.values, aspect="auto")

plt.xlabel("Libros")
plt.ylabel("Usuarios")
plt.title("Mapa de calor Usuario-Libro")

plt.colorbar(label="Rating")

plt.show()

ordered_matrix = matrix.sort_values("Cluster")
heatmap_matrix = ordered_matrix.drop("Cluster", axis=1)


plt.figure(figsize=(12,8))
sns.heatmap(heatmap_matrix.iloc[:200, :50], cmap= "viridis", xticklabels= False, yticklabels=False)
plt.title("HeatMap de usuarios agrupados por clustering")
plt.xlabel("Libros")
plt.ylabel("Usuarios ordenados por Cluster")
plt.show()

# Perfil de cada cluster
cluster_profiles = matrix.groupby("Cluster").mean()
print(cluster_profiles)

# Libros que definen cada cluster
for c in cluster_profiles.index:
    print(f"\nCluster {c}")
    print(cluster_profiles.loc[c].sort_values(ascending = False).head(5))
    top_isbn = cluster_profiles.loc[c] \
        .sort_values(ascending=False) \
        .head(5) \
        .index

    top_books_cluster = books[
        books["ISBN"].isin(top_isbn)
    ][["Book-Title", "Book-Author"]]

    print(top_books_cluster)

# Distribución de usuarios por Cluster
matrix["Cluster"].value_counts().plot(kind="bar")

plt.title("Distribución de usuarios por cluster")
plt.xlabel("Cluster")
plt.ylabel("Número de usuarios")

plt.show()

print("\nResumen de clusters:")
for c in matrix["Cluster"].unique():
    n_users = (matrix["Cluster"] == c).sum()
    print(f"Cluster {c}: {n_users} usuarios")
