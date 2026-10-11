import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Estados (ejemplo)
G.add_node(("I","I","I","I"))
G.add_node(("D","I","D","I"))

# Transición
G.add_edge(("I","I","I","I"), ("D","I","D","I"))

# Dibujar
nx.draw(G, with_labels=True)
plt.show()