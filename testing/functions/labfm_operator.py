from functions.plot import plot_kernel
import numpy as np
import math
from functions.nodes import neighbour_nodes_kdtree
from scipy.spatial import cKDTree
from tqdm import tqdm
import time


def monomial_power(polynomial):
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_scaling_vector(monomial_exponent, h):

    scaling_m = np.zeros((len(monomial_exponent), 1))
    index = 0
    for exp_x, exp_y in monomial_exponent:
        exp_h = -(exp_x + exp_y)
        scaling_m[index] = h ** exp_h
        index = index + 1
    return scaling_m


def calc_monomial(neigh_xy_d, mon_power):
    monomial = []
    for index, (x_dist, y_dist) in enumerate(neigh_xy_d):
        row = []
        for power_x, power_y in mon_power:
            temp_variable = (x_dist ** power_x * y_dist ** power_y) / (
                        math.factorial(power_x) * math.factorial(power_y))
            row.append(temp_variable)
        monomial.append(row)
    monomial = np.array(monomial)
    return monomial.T


def gaussian_rbf(neighbours_r, h):

    q = neighbours_r / h

    w_ji = (9 / math.pi) * math.exp(-9 * (q ** 2))

    return w_ji


def wendland_rbf_c6(neighbours_r, h):

    q = neighbours_r/h

    w_ji = (78/(28 * math.pi))*(1-q/2) ** 8 * (4*q**3 + 6.35 * q ** 2 + 4 * q + 1)
    return float(w_ji)


def wendland_rbf_c2(neighbours_r, h):
    q = neighbours_r/h
    if q > 2:
        raise ValueError('q cannot be larger than 2')

    w_ji = (7/(4 * math.pi)) * (2 * q + 1) * (1 - .5*q)**4
    return float(w_ji)


def calc_hp(exp_a, dist_xy, h):

    z = dist_xy / (h * (2 ** .5))

    if exp_a == 0:
        h = 1

    elif exp_a == 1:
        h = 2 * z

    elif exp_a == 2:
        h = 4 * z ** 2 - 2

    elif exp_a == 3:
        h = 8 * z ** 3 - 12 * z

    elif exp_a == 4:
        h = 16 * z ** 4 - 48 * z ** 2 + 12

    elif exp_a == 5:
        h = 32 * z ** 5 - 160 * z ** 3 + 120 * z

    elif exp_a == 6:
        h = 64 * z ** 6 - 480 * z ** 4 + 720 * z ** 2 - 120

    elif exp_a == 7:
        h = 128 * z ** 7 - 1344 * z ** 5 + 3360 * z ** 3 - 1680 * z

    elif exp_a == 8:
        h = 256 * z ** 8 - 3584 * z ** 6 + 13440 * z ** 4 - 13440 * z ** 2 + 1680

    elif exp_a == 9:
        h = 512 * z ** 9 - 9216 * z ** 7 + 48384 * z ** 5 - 80640 * z ** 3 + 30240 * z

    elif exp_a == 10:
        h = 1024 * z ** 10 - 23040 * z ** 8 + 161280 * z ** 6 - 403200 * z ** 4 + 302400 * z ** 2 - 30240

    else:
        raise ValueError("Invalid value for polynomial. The polynomial must be a polynomial that can be handled by "
                         "the simulation. Try lowering the value of the polynomial")

    return h


def calc_abf(neigh_r, neigh_xy, m_power, h):  # Needs to be written
    """

    :param m_power:
    :param neigh_xy:
    :param neigh_r:
    :param h:
    :return:
    """
    basis_func = []

    for index, (x_dist, y_dist) in enumerate(neigh_xy):
        row = []
        rbf = wendland_rbf_c2(neigh_r[index], h)
        for power_a, power_b in m_power:
            h_a = calc_hp(power_a, x_dist, h)
            h_b = calc_hp(power_b, y_dist, h)
            two_power = 2 ** (power_a + power_b)
            temp_variable = (rbf / (two_power ** .5)) * h_a * h_b
            row.append(temp_variable)
        basis_func.append(row)
    return np.array(basis_func)


def calc_m(basis_func, monomial):
    m_matrix = np.zeros((basis_func.shape[1], monomial.shape[0]))
    for j in range(len(basis_func)):
        m_matrix = m_matrix + np.outer(monomial[:,j], basis_func[j])
    return m_matrix


def pointing_v(polynomial, d):

    if d not in ["x", "y", "Laplace"]:
        raise ValueError("The valid_string argument must be 'x', 'y', or 'Laplace'")

    n = int((polynomial ** 2 + 3 * polynomial) / 2)
    cd = np.zeros((n, 1))
    if d == 'x':
        cd[0] = 1
    elif d == 'y':
        cd[1] = 1
    elif d == 'Laplace':
        cd[2] = 1
        cd[4] = 1
    return cd


def calc_weights(coordinates, polynomial, h, total_nodes):

    weights_x = {}
    weights_y = {}
    weights_laplace = {}
    neigh_coor_dict = {}

    monomial_exponent = monomial_power(polynomial)
    scaling_vector = calc_scaling_vector(monomial_exponent, h)
    cd_x = pointing_v(polynomial, 'x')
    cd_x = cd_x * scaling_vector
    cd_y = pointing_v(polynomial, 'y')
    cd_y = cd_y * scaling_vector
    cd_laplace = pointing_v(polynomial, 'Laplace')
    cd_laplace = cd_laplace * scaling_vector
    tree = cKDTree(coordinates)
    neigh_xy_dist = {}


    for ref_x, ref_y in tqdm(coordinates, desc="Calculating LABFM_Weights for " + str(total_nodes) + ", " + str(polynomial), ncols=100):
        if ref_x > 0.5 or ref_x < -0.5 or ref_y > 0.5 or ref_y < -0.5: continue

        ref_node            = (ref_x, ref_y)
        neigh_r_d, neigh_xy_d, neigh_coor_dict[ref_node] = neighbour_nodes_kdtree(coordinates, ref_node, 2*h, tree, max_neighbors=10)
        neigh_xy_dist[ref_node] = neigh_xy_d
        monomial            = calc_monomial(neigh_xy_d, monomial_exponent) * scaling_vector
        basis_func          = calc_abf(neigh_r_d, neigh_xy_d, monomial_exponent, h)
        m_matrix            = calc_m(basis_func, monomial)
        psi_x               = np.linalg.solve(m_matrix, cd_x)
        psi_y               = np.linalg.solve(m_matrix, cd_y)
        psi_laplace         = np.linalg.solve(m_matrix, cd_laplace)
        node_weight_x       = basis_func @ psi_x
        node_weight_y       = basis_func @ psi_y
        node_weight_laplace = basis_func @ psi_laplace
        weights_x[ref_node] = node_weight_x[:,0]
        weights_y[ref_node] = node_weight_y[:,0]
        weights_laplace[ref_node] = node_weight_laplace[:,0]

    return weights_x, weights_y, weights_laplace, neigh_coor_dict, neigh_xy_dist
