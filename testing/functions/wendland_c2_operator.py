import numpy as np
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
import math


def wendland_c2_sph(neighbours_r, h):
    q = neighbours_r/h

    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >0 and <=2")

    w_ji = (7/(math.pi*h**2) ) * (1 - q)**4 * (1 + 4 * q)
    return w_ji


def wendland_c2_deriv(neighbours_r, neigh_xy_d, h, s, deriv):
    if deriv.lower() not in ['dx', 'dy']:
        raise ValueError("deriv must be either 'dx' or 'dy'")

    q = neighbours_r / h
    if (q < 0).any() or (q > 2).any():
        raise ValueError("r must be >= 0 and <=2")

    deriv = deriv.lower()
    if deriv == 'dx':
        dist = neigh_xy_d[:, 0]
    else:
        dist = neigh_xy_d[:, 1]



    w_ji = 7 * dist * (1 - q/2) ** 3 * (-5 * q) / (4 * np.pi * h ** 3) * s ** 2

    w_ji[1:] /=  neighbours_r[1:]

    return w_ji


def wendlandc2_weights(coordinates, h, total_nodes, s):
    tree = cKDTree(coordinates)

    neigh_r_dict    = {}
    neigh_coor_dict = {}
    neigh_xy_dist   = {}
    weights_x       = {}
    weights_y       = {}
    weights_laplace = {}

    support_radius = 2 * h


    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Wendland Weights for " + str(total_nodes) + ", ", ncols=100):
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue
        ref_node = (ref_x, ref_y)
        (neigh_r_d,
         neigh_xy_d,
         neigh_coor_dict[ref_node]) = neighbour_nodes_kdtree(coordinates,
                                                             ref_node,
                                                             support_radius,
                                                             tree,
                                                             max_neighbors=100)

        neigh_xy_dist[ref_node] = neigh_xy_d
        neigh_r_dict[ref_node] = neigh_r_d
        weights_x[ref_node] = -wendland_c2_deriv(neigh_r_d, neigh_xy_d, h, s, 'dx')
        weights_y[ref_node] = -wendland_c2_deriv(neigh_r_d, neigh_xy_d, h, s, 'dy')
        weights_laplace[ref_node] = 2 * (neigh_xy_d[1:, 0] * weights_x[ref_node][1:] +
                                         neigh_xy_d[1:, 1] * weights_y[ref_node][1:]) / neigh_r_d[1:] ** 2
        weights_laplace[ref_node] = np.concatenate(([0], weights_laplace[ref_node]))


    return weights_x, weights_y, weights_laplace, neigh_coor_dict, neigh_xy_dist