import networkx as nx
import matplotlib.pyplot as plt

# Crear grafo dirigido
G = nx.DiGraph()

# Estados (0 a 18)
estados = list(range(19))

# Alfabeto (productos)
productos = [1, 2, 3, 5]

# Agregar nodos
G.add_nodes_from(estados)

# Generar transiciones
for estado in estados:
    for p in productos:
        nuevo_estado = estado + p
        
        if nuevo_estado <= 18:
            G.add_edge(estado, nuevo_estado, label=str(p))

# Dibujar grafo
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700)

# Etiquetas de aristas (pesos)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()