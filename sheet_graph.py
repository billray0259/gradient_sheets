import gspread
import re
from torch import Tensor, optim
import torch.functional as F
import numpy as np

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


def get_node_locations(edges: list[tuple[str, str]]):

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

    




    