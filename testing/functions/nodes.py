import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial import cKDTree
import random


def random_matrix(seed: float, shape: tuple, s: float) -> NDArray:
    random.seed(seed)
    return np.array([[random.uniform(-s*0.25, s*0.25) for _ in range(shape[1])] for _ in range(shape[0])])



def calc_h(s: float, kernel: int | str) -> float:
    # This determines the smoothing scale

    if isinstance(kernel, int):
        if   kernel <= 2: h = 1.5 * s
        elif kernel <= 4: h = 1.9 * s
        elif kernel <= 6: h = 2.3 * s
        elif kernel <= 8: h = 2.7 * s
        return h
    elif isinstance(kernel, str):
        if kernel == 'wc2': h = 1.5 * s
        elif kernel == 'q_s': h = 1.5 * s
        elif kernel == 'models': h = 2.3 * s
        return h



def create_nodes(total_nodes: int, s: float, h: float) -> Tuple[NDArray, int]:
    """
    :param total_nodes: is a scalar that states the number of nodes we would like to have inside the domain
    :param s: is a scalar that determines the average distance between points inside the comp. stencil
    :return:
    coordinates: is a numpy array that contains all coordinates of nodes, the first and second column contain the x and
    y coordinates, respectively. It is possible to access the ith node with "coordinates[i]"
    """
    delta = 1.0 / (total_nodes - 1)  # Determine the spacing delta between the points in the original domain
    two_times_influence_radius = 2 * h
    n = int(two_times_influence_radius / delta)  # Calculate the number of points to be added on each side

    # change this to only create nodes to the specific size of h
    x = np.linspace(-0.5 - two_times_influence_radius,
                    0.5 + two_times_influence_radius, total_nodes + n * 2)  # Creates x coordinates with boundary
    y = np.linspace(-0.5 - two_times_influence_radius,
                    0.5 + two_times_influence_radius, total_nodes + n * 2)  # Creates y coordinates with boundary

    X, Y = np.meshgrid(x, y)  # Create a 2D grid of x and y coordinates

    # Perturb the coordinates
    shift_x = random_matrix(1, X.shape, s)
    shift_y = random_matrix(2, Y.shape, s)
    X = X + shift_x
    Y = Y + shift_y

    # Stack the perturbed coordinates
    coordinates = np.column_stack((X.ravel(), Y.ravel()))
    coordinates = np.around(coordinates, 15)

    nodes_in_domain = 0
    for ref_x, ref_y in coordinates:
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue
        nodes_in_domain += 1

    return coordinates, nodes_in_domain


def neighbour_nodes_kdtree(coordinates, ref_node, h, tree, max_neighbors=1000):
    # Query the tree for points within a radius of h from the reference node
    indices = tree.query_ball_point(ref_node, h)
    max_neighbors = min(max_neighbors, len(indices))

    all_distances = np.sqrt(np.sum((coordinates[indices] - ref_node) ** 2, axis=1))
    sorted_indices = np.argsort(all_distances)[:max_neighbors]
    indices = np.array(indices)[sorted_indices]

    # Extract neighbor coordinates based on the filtered/sorted indices
    neigh_coor = coordinates[indices]

    # Calculate displacements and distances
    displacements = neigh_coor - ref_node
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    return distances, displacements, neigh_coor
