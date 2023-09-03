import networkx as nx
from matplotlib.lines import Line2D
import enum
from matplotlib import pyplot as plt
import torch
from typing import NoReturn

torch.set_printoptions(profile="full")


def plot_internal_graph(graph: nx.Graph, node_types: enum.IntEnum) -> NoReturn:
    # TUM colors
    blue = (0, 0.4, 0.74)
    ivory = (0.85, 0.84, 0.8)
    orange = (0.89, 0.45, 0.13)
    green = (0.64, 0.68, 0)
    purple = (0.41, 0.03, 0.35)
    turquoise = (0, 0.47, 0.54)
    dark_green = (0, 0.49, 0.19)
    yellow = (1, 0.86, 0)
    red = (0.77, 0.03, 0.11)
    dark_red = (0.61, 0.05, 0.09)
    light_blue = (0.39216, 0.62745, 0.78431)
    dark_blue = (0.00000, 0.32157, 0.57647)
    tum_colors = [blue, ivory, orange, green, purple, turquoise, dark_green, yellow, red, dark_red, light_blue, dark_blue]
    font = {'size': 24}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (10, 10)
    node_color = []
    node_size = []
    for node_id, node_data in graph.nodes(data=True):
        for idx, node_type in enumerate(node_types):
            if node_type == node_data['ntype']:
                node_color.append(tum_colors[idx])
                node_size.append(3000)

    edge_colors = []
    edge_widths = []
    for edge in graph.edges:
        edge_colors.append('grey')
        edge_widths.append(1)

    nx.draw_kamada_kawai(graph, with_labels=True, font_color='white', width=edge_widths, edge_color=edge_colors,
                         node_size=node_size, node_color=node_color)
    legend_elements = []
    for idx, node_type in enumerate(node_types):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=node_type, markerfacecolor=tum_colors[idx], markersize=20))

    plt.legend(handles=legend_elements)
    plt.savefig('/tmp/plot.pdf')
    plt.show()
