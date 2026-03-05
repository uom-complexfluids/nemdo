from models.preproc import load_gnn
import torch
from models.preproc import calc_moments_torch
from scipy.spatial import cKDTree
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
import numpy as np
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gnn_weights(coordinates, h, total_nodes, nodes_in_domain):

    # size of the stencil
    num_neighbours = 35
    tree = cKDTree(coordinates)
    # Below we detail the number of neighbours for each model
    # nemdo_x and nemdo_lap: 35
    # Below are the two mo
    # nemdo_1: 10 (computes x-derivative weights)
    # nemdo_2: 15 (computes x-derivative weights)
    model_x, _ = load_gnn(model_class='x_and_lap',
                          path='models/trained_model/nemdo_x.pth') # model for x derivative
    model_laplace, _ = load_gnn(model_class='x_and_lap',
                                path='models/trained_model/nemdo_lap.pth')  # model for laplace


    ref_node_ls = []
    neigh_coor_dict = {}
    h_dict = {}
    neigh_xy_dist = {}

    batch_size = min(nodes_in_domain, 4096) # limiting the batch size for inference

    num_edges = 2 * (num_neighbours - 1)
    edge_index = torch.zeros((2, num_edges * batch_size), dtype=torch.long)
    batch = torch.zeros(num_neighbours * batch_size, dtype=torch.long)
    x = torch.zeros((num_neighbours * batch_size, 2), dtype=torch.float32)
    h_total = np.zeros(batch_size)

    idx_correct = torch.arange(1, num_neighbours)

    data_loader = []
    b = -1

    # Manually building graphs
    for ref_x, ref_y in coordinates:
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        b += 1
        n = b * num_neighbours

        ref_node = (ref_x, ref_y)
        ref_node_ls.append(ref_node)
        neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes_kdtree(coordinates,
                                                                                  ref_node,
                                                                                  2 * h,
                                                                                  tree,
                                                                                  max_neighbors=num_neighbours)

        neigh_xy_dist[ref_node] = neigh_xy_d

        # Computing the batches
        batch[b * num_neighbours : (b + 1) * num_neighbours] = b

        # Computing the edge indexes
        edge_index[0, b * num_edges : b * num_edges + num_edges // 2] = n + idx_correct
        edge_index[0, b * num_edges + num_edges // 2: b * num_edges + num_edges] = n

        edge_index[1, b * num_edges : b * num_edges + num_edges // 2] = n
        edge_index[1, b * num_edges + num_edges // 2: b * num_edges + num_edges] = n + idx_correct

        # stencil length
        h_total[b] = neigh_r_d[-1]

        # node features
        x[b * num_neighbours: (b + 1) * num_neighbours, :] = torch.from_numpy(neigh_xy_d / neigh_r_d[-1])

        if b == batch_size - 1:
            data_loader.append([x, edge_index, batch, h_total])
            b = -1
            edge_index = torch.zeros((2, num_edges * batch_size), dtype=torch.long)
            batch = torch.zeros(num_neighbours * batch_size, dtype=torch.long)
            x = torch.zeros((num_neighbours * batch_size, 2), dtype=torch.float32)
            h_total = torch.zeros(batch_size, dtype=torch.float32)


    if b > -1:
        x = x[:(b + 1) * num_neighbours, :]
        edge_index = edge_index[:, :b * num_edges + num_edges]
        batch = batch[:(b + 1) * num_neighbours]
        h_total = h_total[:b + 1]
        data_loader.append([x, edge_index, batch, h_total])


    weights_x = []
    weights_laplace = []

    model_x.eval()
    model_laplace.eval()

    with torch.no_grad():
        for b in tqdm(data_loader, desc="Predicting GNN Weights for " + str(total_nodes), ncols=100):

            x          = b[0]
            edge_index = b[1]
            batch      = b[2]
            h          = b[3]

            out_x = model_x(x,
                            edge_index,
                            batch)

            out_laplace = model_laplace(x,
                                        edge_index,
                                        batch)

            # operations below are not optimised
            pred_reshape_x = out_x.view((int(batch[-1]) + 1, -1))
            weights_x.extend(pred_reshape_x.cpu().numpy() / h[:, None])

            pred_reshape_laplace = out_laplace.view((int(batch[-1]) + 1, -1))
            weights_laplace.extend(pred_reshape_laplace.cpu().numpy() / (h[:, None] ** 2))


    weights_x_dict = {}
    weights_laplace_dict = {}
    for key, x, lap in zip(ref_node_ls, weights_x, weights_laplace):
        weights_x_dict[key] = x
        weights_laplace_dict[key] = lap

    return weights_x_dict, weights_laplace_dict, neigh_coor_dict, h_dict, neigh_xy_dist

