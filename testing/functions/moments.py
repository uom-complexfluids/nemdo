import numpy as np
from functions.labfm_operator import monomial_power, calc_monomial
from torch import Tensor


def calc_moments(neigh_xy, w_x, w_l, h, polynomial=2):
    monomial_exponent = monomial_power(polynomial)

    moments_x, moments_l = [], []


    for node_i, neigh_list in neigh_xy.items():
        w_x_i = w_x[tuple(node_i)]
        w_l_i = w_l[tuple(node_i)]
        x_y_distances = neigh_list / h
        monomials = calc_monomial(x_y_distances, monomial_exponent)
        moments_x.append(monomials @ (w_x_i * h))
        moments_l.append(monomials @ (w_l_i * h ** 2))

    moments_x = np.array(moments_x)
    moments_l = np.array(moments_l)

    target_x = np.array([1, 0, 0, 0, 0])
    target_l = np.array([0, 0, 1, 0, 1])

    error_x = moments_x - target_x
    error_l = moments_l - target_l

    # Define a formatter for 4 significant figures in scientific notation
    sci_fmt = lambda x: f"{x:.2e}"

    # Uncomment below to check the moments of the operators
    # The "polynomial" variable controls up to which order the Taylor monomials are expanded
    #print('X Moments error', np.array2string(np.mean(abs(error_x), axis=0), formatter={'float_kind': sci_fmt}))
    #print('X Moments std  ', np.array2string(np.std(error_x, axis=0), formatter={'float_kind': sci_fmt}))
    #print('Lap Moments error', np.array2string(np.mean(abs(error_l), axis=0), formatter={'float_kind': sci_fmt}))
    #print('Lap Moments std  ', np.array2string(np.std(error_l, axis=0), formatter={'float_kind': sci_fmt}))

    return moments_x, moments_l