import networkx as nx
import matplotlib.pyplot as plt
import imageio
import numpy as np
import random
from io import BytesIO
from PIL import Image

# Parâmetros
num_nodes = 100
num_leaders = 10
iterations = 100

# Criar grafo aleatório
G = nx.erdos_renyi_graph(num_nodes, 0.05)

# Inicializar atributos dos nós
for node in G.nodes():
    G.nodes[node]['belief'] = random.choice([0, 1])
    G.nodes[node]['leader'] = False
    G.nodes[node]['fixed'] = False

# Selecionar líderes (maior grau)
leaders = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_leaders]
for i, (node, _) in enumerate(leaders):
    G.nodes[node]['leader'] = True
    G.nodes[node]['fixed'] = True
    G.nodes[node]['belief'] = 1 if i < num_leaders // 2 else 0  # metade crentes, metade céticos

# Função de atualização (Metrópolis simplificado)
def metropolis_step(G):
    for node in G.nodes():
        if G.nodes[node]['fixed']:
            continue
        current = G.nodes[node]['belief']
        proposed = 1 - current
        neighbors = list(G.neighbors(node))
        influence_current = sum(G.nodes[n]['belief'] == current for n in neighbors)
        influence_proposed = sum(G.nodes[n]['belief'] == proposed for n in neighbors)
        delta_E = influence_current - influence_proposed
        if delta_E <= 0 or random.random() < np.exp(-delta_E):
            G.nodes[node]['belief'] = proposed

# Layout fixo
pos = nx.spring_layout(G, seed=42)
frames = []

# Gerar quadros
for _ in range(iterations):
    metropolis_step(G)
    colors = []
    for node in G.nodes():
        if G.nodes[node]['leader']:
            colors.append('darkred' if G.nodes[node]['belief'] == 1 else 'darkblue')
        else:
            colors.append('lightcoral' if G.nodes[node]['belief'] == 1 else 'lightblue')
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, ax=ax)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    frames.append(image.copy())
    buf.close()
    plt.close(fig)

# Salvar como GIF
frames[0].save("fake_news_propagation.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
print("GIF salvo como 'fake_news_propagation.gif'")
