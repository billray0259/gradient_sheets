import gspread
import re
from torch import Tensor, optim
import torch.functional as F
import numpy as np
import copy
import random

def get_children(target_cell_name, formula_sheet):
    r, c = gspread.utils.a1_to_rowcol(target_cell_name)
    r -= 1
    c -= 1

    item = formula_sheet[r][c]
    if type(item) == str and item.startswith('='):
        expression = item[1:].upper()
        param_names = re.findall(r'[A-Z]+\d+', expression)
        children = [get_children(cell_name, formula_sheet) for cell_name in param_names]
        return {target_cell_name: children}
    else:
        return {target_cell_name: []}


def find_edges(formula_sheet):
    edges = set()
    for r, row in enumerate(formula_sheet):
        for c, item in enumerate(row):
            if type(item) == str and item.startswith('='):
                expression = item[1:].upper()
                param_names = re.findall(r'[A-Z]+\d+', expression)
                for cell_name in param_names:
                    edges.add((cell_name, gspread.utils.rowcol_to_a1(r+1, c+1)))
    return list(edges)


def get_labels(cell_names, formula_sheet):

    labels = []
    for cell_name in cell_names:
        r, c = gspread.utils.a1_to_rowcol(cell_name)
        r -= 1
        c -= 2
        labels.append(formula_sheet[r][c])
    return labels


def get_node_locations_v1(edges: list[tuple[str, str]]):

    # here's the plan
    # Display the nodes in layers
    # Nodes with no inputs are in the first layer
    # Nodes with inputs in the first layer are in the second layer and so on
    # The layers are spaced evenly
    # The nodes in each layer are positioned to minimize a cost function that we'll define later

    cell_names = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    cell_names = sorted(cell_names)


    # make an adjacency matrix
    N = len(cell_names)
    adjacency_matrix = np.zeros((N, N))
    for edge in edges:
        i = cell_names.index(edge[0])
        j = cell_names.index(edge[1])
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    
    adjacency_matrix = Tensor(adjacency_matrix).triu()

    # first we need to figure out which layer each node is in
    layer_idx = {}
    current_layer = set([edge[0] for edge in edges]) - set([edge[1] for edge in edges]) # only nodes with no inputs
    
    i = 0
    while current_layer:
        for node in current_layer:
            layer_idx[node] = i
        current_layer = {edge[1] for edge in edges if edge[0] in current_layer} # nodes following the current layer
        i += 1
    
    for node in current_layer:
        layer_idx[node] = i

    # make an adjacency matrix where nodes are in the same layer are connected
    layer_adjacency_matrix = np.zeros((N, N))
    for i, cell_name in enumerate(cell_names):
        for j, cell_name2 in enumerate(cell_names):
            if layer_idx[cell_name] == layer_idx[cell_name2]:
                layer_adjacency_matrix[i, j] = 1
                layer_adjacency_matrix[j, i] = 1
    

    layer_adjacency_matrix = Tensor(layer_adjacency_matrix).triu()
    

    heights = Tensor(np.arange(len(cell_names)).reshape((-1, 1))) # the tunable parameters
    heights.requires_grad = True

    n_steps = 10000

    # child_loss_weight = np.linspace(10, 2, n_steps)
    # # neighbor_loss_weight = np.linspace(0, 1, n_steps)
    # neighbor_loss_weight = np.linspace(1, 30, n_steps)
    # # boundary_loss_weight = 10 * np.ones(n_steps)
    # boundary_loss_weight =  np.linspace(1, 1, n_steps)

    noise = np.linspace(1, 0, n_steps)

    child_loss_weight = np.ones(n_steps) * 0.25
    neighbor_loss_weight = np.ones(n_steps) * 20
    boundary_loss_weight = np.ones(n_steps) * 0.5

    def loss_fn(child_loss_weight, neighbor_loss_weight, boundary_loss_weight, noise):
        # add some noise to the coefficients
        child_loss_weight *= np.exp(np.random.normal(0, noise))
        neighbor_loss_weight *= np.exp(np.random.normal(0, noise))
        boundary_loss_weight *= np.exp(np.random.normal(0, noise))

        # pull the furthest node towards zero
        boundary_loss = boundary_loss_weight * heights.abs().max() 

        distances = (heights - heights.T).abs()

        # pull the node furthest away from its child towards it
        where_adj_nonzero = adjacency_matrix.nonzero()
        child_loss = child_loss_weight * (distances * adjacency_matrix)[where_adj_nonzero].max()
        
        # push the node closest to the node in the same layer away from it
        where_layer_adj_nonzero = layer_adjacency_matrix.nonzero()
        neighbor_loss = -neighbor_loss_weight * ((distances+0.1)**0.5 * layer_adjacency_matrix)[where_layer_adj_nonzero].mean()

        

        
        loss = child_loss + neighbor_loss + boundary_loss

        print(child_loss.item(), neighbor_loss.item(), boundary_loss.item())

        return loss
    
    # now we need to minimize the loss
    # we'll use gradient descent
    optimizer = optim.Adam([heights], lr=0.01)
    for cl, nl, bl, n in zip(child_loss_weight, neighbor_loss_weight, boundary_loss_weight, noise):
        optimizer.zero_grad()
        loss = loss_fn(cl, nl, bl, n)
        loss.backward()
        optimizer.step()
    
    # now we have the heights of each node
    # we can use them to position the nodes
    # we'll use the heights to determine the y position of each node
    # we'll use the layer index to determine the x position of each node

    node_locations = {}
    for i, cell_name in enumerate(cell_names):
        node_locations[cell_name] = (layer_idx[cell_name], heights[i, 0].item())
        # node_locations[cell_name] = (layer_idx[cell_name], i)

    
    print(node_locations)
    
    return node_locations



def count_crossings(edges: list[tuple[str, str]], node_locations: dict[str, tuple[float, float]]):
    crossings = 0
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges):
            if i >= j: # only count each edge once
                continue

            if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]: # don't count edges that share a node
                continue

            A = node_locations[edge1[0]]
            B = node_locations[edge1[1]] 

            C = node_locations[edge2[0]] 
            D = node_locations[edge2[1]]

            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
            crossings += ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    return crossings


def min_edge_node_distances(edges: list[tuple[str, str]], node_locations: dict[str, tuple[float, float]]):
    edge_node_distances = {}
    for node in node_locations:
        for edge in edges:
            A = node_locations[edge[0]]
            B = node_locations[edge[1]] 

            if node == edge[0] or node == edge[1]:
                continue

            slope = (B[1] - A[1]) / (B[0] - A[0])
            intercept = A[1] - slope * A[0]
            line = lambda x: slope * x + intercept
            min_x = min(A[0], B[0])
            max_x = max(A[0], B[0])

            if not (min_x < node_locations[node][0] < max_x):
                continue
        
            distance = abs(line(node_locations[node][0]) - node_locations[node][1])

            if node not in edge_node_distances:
                edge_node_distances[node] = distance
            elif distance < edge_node_distances[node]:
                edge_node_distances[node] = distance
            
    return edge_node_distances


            


def measure_squared_edge_length(edges: list[tuple[str, str]], node_locations: dict[str, tuple[float, float]]):
    total_length = 0
    for edge in edges:
        A = node_locations[edge[0]]
        B = node_locations[edge[1]] 

        total_length += ((A[0] - B[0])**2 + (A[1] - B[1])**2)
    
    return total_length


def cost_fn(edges: list[tuple[str, str]], node_locations: dict[str, tuple[float, float]]):
    n_crossings = count_crossings(edges, node_locations)
    sq_edge_length = measure_squared_edge_length(edges, node_locations)
    min_edge_node_distance = min(list(min_edge_node_distances(edges, node_locations).values()))

    if min_edge_node_distance == 0:
        return float('inf')


    return n_crossings - min_edge_node_distance * 0.1 + sq_edge_length * 0.1
 


def get_node_locations_v2(edges: list[tuple[str, str]]):
    cell_names = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    cell_names = sorted(cell_names)

    # first we need to figure out which layer each node is in
    layer_idx = {} # maps cell name to layer index
    current_layer = set([edge[0] for edge in edges]) - set([edge[1] for edge in edges]) # only nodes with no inputs
    
    i = 0
    while current_layer:
        for node in current_layer:
            layer_idx[node] = i
        current_layer = {edge[1] for edge in edges if edge[0] in current_layer} # nodes following the current layer
        i += 1
    
    for node in current_layer:
        layer_idx[node] = i


    best_layers = []
    layer_counts = []
    for i in range(max(layer_idx.values())+1):
        best_layers.append([node for node in layer_idx if layer_idx[node] == i])
        layer_counts.append(len(best_layers[-1]))

    # pad the layers with None so that they are all the same length
    max_layer_length = max(layer_counts)
    for i in range(len(best_layers)):
        best_layers[i] += [None] * (max_layer_length - layer_counts[i])
    
    for layer in best_layers:
        random.shuffle(layer)

    best_node_locations = node_locations_from_layers(best_layers)
    min_cost = cost_fn(edges, best_node_locations)
    for i in range(100):
        done = True
        for layer_config in iter_swaps(best_layers):
            node_locations = node_locations_from_layers(layer_config)
            cost = cost_fn(edges, node_locations)
            if cost < min_cost:
                min_cost = cost
                best_node_locations = node_locations
                best_layers = layer_config
                done = False
                break
        if done:
            break
        print(i, min_cost)

    
    return best_node_locations



def node_locations_from_layers(layers):
    node_locations = {}

    max_layer_height = max([len(layer) for layer in layers]) - 1

    for i, layer in enumerate(layers):
        if len(layer) == 1:
            node_locations[layer[0]] = (i, max_layer_height / 2)
            continue

        for j, node in enumerate(layer):
            if node is None:
                continue
            # y_offset = 0.25 if i % 2 == 1 else -0.25
            y_offset = 0
            node_locations[node] = (i, j + y_offset)
        
    return node_locations


def iter_swaps(layers):
    # iterate over different configurations of the layers where each node is swapped with a node in the same layer
    # this is a generator function

    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            for k, other_node in enumerate(layer):
                if j >= k: # only swap each node once
                    continue
                new_layers = copy.deepcopy(layers)
                new_layers[i][j] = other_node
                new_layers[i][k] = node
                yield new_layers
