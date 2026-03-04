import numpy as np
from tqdm import tqdm
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree

def quintic_spline(neighbours_r, h):
    norm = neighbours_r / h
    w = []

    for s in norm:
        if 0 <= s < 1:
            ww = (3 - s) ** 5 - 6 * (2 - s) ** 5 + 15 * (1 - s) ** 5
        elif 1 <= s < 2:
            ww = (3 - s) ** 5 - 6 * (2 - s) ** 5
        elif 2 <= s < 3:
            ww = (3 - s) ** 5
        else:
            ww = 0.0
        w.append(ww)

    w = (7/(478*np.pi*h**2))*np.array(w)

    return w

def quintic_spline_deriv(neighbours_r, neigh_xy_d, h, s, deriv):
    q_array = neighbours_r / h
    w = []

    if deriv == 'dx':
        xy_div_r = np.zeros_like(neighbours_r)
        mask = neighbours_r > 0
        xy_div_r[mask] = neigh_xy_d[mask, 0] / neighbours_r[mask]
    elif deriv == 'dy':
        xy_div_r = np.zeros_like(neighbours_r)
        mask = neighbours_r > 0
        xy_div_r[mask] = neigh_xy_d[mask, 1] / neighbours_r[mask]
    else:
        raise ValueError('deriv must be dx or dy')

    for i in range(q_array.shape[0]):
        q = q_array[i]
        if 0 <= q < 1:
            ww = -5*(3-q)**4 + 30*(2-q)**4 - 75*(1-q)**4
        elif 1 <= q < 2:
            ww = -5*(3-q)**4 + 30*(2-q)**4
        elif 2 <= q < 3:
            ww = -5*(3-q)**4
        else:
            ww = 0

        ww = (xy_div_r[i] / h) * ww
        w.append(ww)

    w = (7/(478*np.pi*h**2))*np.array(w) * s ** 2
    return w


def qspline_weights(coordinates, h, total_nodes, s):
    tree = cKDTree(coordinates)

    neigh_r_dict    = {}
    neigh_coor_dict = {}
    neigh_xy_dist   = {}
    weights_x       = {}
    weights_y       = {}
    weights_laplace = {}
    support_radius = 3 * h


    for ref_x, ref_y in tqdm(coordinates, desc="Calculating Quintinc Spline Weights for " + str(total_nodes) + ", ", ncols=100):
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

        weights_x[ref_node] = -quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, s, 'dx')

        weights_y[ref_node] = -quintic_spline_deriv(neigh_r_d, neigh_xy_d, h, s, 'dy')
        weights_laplace[ref_node] =   (2 * (neigh_xy_d[1:, 0] * weights_x[ref_node][1:] +
                                            neigh_xy_d[1:, 1] * weights_y[ref_node][1:]) /  neigh_r_d[1:] ** 2)
        weights_laplace[ref_node] = np.concatenate(([0], weights_laplace[ref_node]))


    return weights_x, weights_y, weights_laplace, neigh_coor_dict, neigh_xy_dist
