# BIBLIOTECAS
from __future__ import annotations
import networkx as nx
import time
import sys
from memory_profiler import memory_usage
from functools import partial
import heapq
import math
from math import inf, ceil

# ALGORITMOS PARA O PROBLEMA DO CAIXEIRO VIAJANTE

# 1. Branch-And-Bound

# 1.1. Definição da classe Node que será usada no algoritmo

class Node:
    # Construtor
    def __init__(self, boundary, level, cost, sol, weights,):
        self.boundary = boundary
        self.level = level
        self.cost = cost
        self.sol = sol
        self.weights = weights
    
    # Comparador de custos entre nós
    def __lt__(self, other) -> bool:
        return self.cost < other.cost

# 1.2. Calcular limite inferior da instância TSP e retornar uma lista de tuplas que contém os pesos mínimos de cada um dos nós do problema 
def initial_bound(graph):
    n = graph.number_of_nodes()
    boundary = 0
    min_weights = []
    for i in range(n):
        smallest = (inf, inf, False)
        for j in range(n):
            if i == j:
                continue
            weight = graph[i][j]["weight"]
            if weight < smallest[0]:
                smallest = (weight, smallest[0], False)
            elif weight < smallest[1]:
                smallest = (smallest[0], weight, False)
        min_weights.append(smallest)
        boundary += smallest[0] + smallest[1]
    return ceil(boundary) / 2, min_weights

# 1.3. Atualizar o limite de um nó após um vértice do grafo ser adicionado no conjunto solução
def update_bound(graph, v, node):
    u = node.sol[-1]
    weight = graph[u][v]["weight"]
    min_weights = node.weights.copy()
    min_weights[u], increment = update_bound_helper(weight, min_weights[u])
    min_weights[v], other_increment = update_bound_helper(weight, min_weights[v])
    increment += other_increment
    boundary = node.boundary + ceil(increment / 2)
    return boundary, min_weights

# 1.4. Função auxiliar para encontrar a aresta de peso mínimo e o incremento de peso
def update_bound_helper(weight, index_min):
    increment = 0
    if weight > index_min[0]:
        if not index_min[2]:
            increment += weight - index_min[1]
            index_min = (index_min[0], weight, True)
        else:
            increment += weight - index_min[0]
    return index_min, increment

# 1.5. Retorna o custo ótimo para o problema do Caixeiro Viajante usando Branch-and-Bound
def branch_and_bound(graph):
    n = graph.number_of_nodes()
    boundary, min_weights = initial_bound(graph)
    root = Node(boundary, 1, 0, [0], min_weights)
    queue = [root]
    heapq.heapify(queue)
    best = inf
    while len(queue) != 0:
        node = queue.pop(0)
        if node.level > n:
            if best > node.cost:
                best = node.cost
        elif node.boundary < best:
            if node.level < n:
                for k in range(1, n):
                    new_node = branch_and_bound_helper(graph, k, node, best)
                    if new_node is not None:
                        heapq.heappush(queue, new_node)
            else:
                cycle_node = branch_and_bound_helper(graph, 0, node, best)
                if cycle_node is not None:
                    heapq.heappush(queue, cycle_node)
    return best

def branch_and_bound_helper(graph, k, node, best):
    if (k not in node.sol or k == 0) and graph.has_edge(node.sol[-1], k):
        new_bound, new_min_weights = update_bound(graph, k, node)
        if new_bound < best:
            new_weight = graph[node.sol[-1]][k]["weight"]
            new_node = Node(new_bound, node.level + 1, node.cost + new_weight, node.sol + [k],new_min_weights,)
            return new_node
    return None

# 2. Twice-Around-The-Tree

def twice_around_the_tree(graph):
    mst = nx.minimum_spanning_tree(graph)
    cycle = list(nx.dfs_preorder_nodes(mst, 0))
    cycle.append(0)
    cost = 0
    for i in range(0, len(cycle) - 1):
        cost += graph[cycle[i]][cycle[i + 1]]["weight"]
    return cost

# 3. Christofides

def christofides(graph):
    mst = nx.minimum_spanning_tree(graph)
    odd_degree_vertices = [node for (node, val) in mst.degree if val % 2 == 1]
    odd_graph = nx.induced_subgraph(graph, odd_degree_vertices)
    matching = nx.min_weight_matching(odd_graph)
    eulerian_multigraph = nx.MultiGraph()
    eulerian_multigraph.add_edges_from(mst.edges)
    eulerian_multigraph.add_edges_from(matching)
    edges = list(nx.eulerian_circuit(eulerian_multigraph, 0))
    cycle = [0]
    for _, v in edges:
        if v in cycle:
            continue
        cycle.append(v)
    cycle.append(0)
    cost = 0
    for i in range(0, len(cycle) - 1):
        cost += graph[cycle[i]][cycle[i + 1]]["weight"]
    return cost

# LEITURA DO DATASET E MONTAGEM DO GRAFO

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = {}
    node_coord_section = False

    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
            continue
        elif line.startswith("EOF"):
            break

        if node_coord_section:
            parts = line.split()
            node = int(parts[0]) - 1
            x = float(parts[1])
            y = float(parts[2])
            coordinates[node] = (x, y)

    return coordinates

def create_complete_graph(coordinates):
    graph = nx.Graph()

    for node in coordinates:
        graph.add_node(node, pos=coordinates[node])

    for node1 in coordinates:
        for node2 in coordinates:
            if node1 != node2:
                distance = euclidean_distance(coordinates[node1], coordinates[node2])
                graph.add_edge(node1, node2, weight=distance)

    return graph

#EXECUÇÃO DOS ALGORITMOS

if __name__ == "__main__":
    file_path = sys.argv[1]
    coordinates = read_tsp_file(file_path)
    graph = create_complete_graph(coordinates)

    begin = time.time()

    memory_twice_around_the_tree, cost_twice_around_the_tree = memory_usage(partial(twice_around_the_tree, graph), interval=1.0, max_usage=True, retval=True)

    end = time.time()

    total_time_tree = end - begin

    begin = time.time()

    memory_christofides, cost_christofides = memory_usage(partial(christofides, graph), interval=1.0, max_usage=True, retval=True)

    end = time.time()

    total_time_chris = end - begin

    print("Twice-Around-The-Tree")
    print("Tempo: ", total_time_tree)
    print("Memória: ", memory_twice_around_the_tree)
    print("Valor: ", cost_twice_around_the_tree)

    print("Christofides")
    print("Tempo: ", total_time_chris)
    print("Memória: ", memory_christofides)
    print("Valor: ", cost_christofides)

    # Usado para testar o branch-and-bound - todas as instâncias superam os 30 minutos de execução
    #cost_branch_and_bound = branch_and_bound(graph)
    #print("Branch-And-Bound")
    #print("Valor: ", cost_branch_and_bound)