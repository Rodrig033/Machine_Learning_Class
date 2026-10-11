import networkx as nx
import matplotlib.pyplot as plt

# Crear grafo dirigido
G = nx.DiGraph()

# Estados
estados = [
    "q0\nInicio",
    "q1\nFundamentos",
    "q2\nModelado",
    "q3\nBúsqueda",
    "q4\nAgentes",
    "q5\nMulti-Agente",
    "q6\nSimulación",
    "q7\nIA",
    "q8\nProyecto",
    "q9\nOptimización",
    "qf\nEmpleo"
]

# Agregar nodos
G.add_nodes_from(estados)

# Transiciones con etiquetas
transiciones = [
    ("q0\nInicio", "q1\nFundamentos", "Programación"),
    ("q1\nFundamentos", "q2\nModelado", "Grafos"),
    ("q2\nModelado", "q3\nBúsqueda", "Algoritmos"),
    ("q3\nBúsqueda", "q4\nAgentes", "Concepto de agente"),
    ("q4\nAgentes", "q5\nMulti-Agente", "Interacción"),
    ("q5\nMulti-Agente", "q6\nSimulación", "Proyectos"),
    ("q6\nSimulación", "q7\nIA", "Reinforcement Learning"),
    ("q7\nIA", "q8\nProyecto", "Integración"),
    ("q8\nProyecto", "q9\nOptimización", "Escalabilidad"),
    ("q9\nOptimización", "qf\nEmpleo", "Aplicación laboral")
]

# Agregar aristas
for origen, destino, etiqueta in transiciones:
    G.add_edge(origen, destino, label=etiqueta)

# Layout horizontal (clave para que se vea bien)
pos = {estado: (i, 0) for i, estado in enumerate(estados)}

# Colores (resaltar inicio y final)
node_colors = []
for n in estados:
    if "q0" in n:
        node_colors.append("lightgreen")
    elif "qf" in n:
        node_colors.append("red")
    else:
        node_colors.append("lightblue")

# Dibujar nodos
nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors)

# Dibujar etiquetas de aristas
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Camino hacia Multi-Agent Systems (Modelo de Estados)")
plt.show()