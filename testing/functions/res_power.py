import numpy as np
from typing import Tuple
from numpy import ndarray
from numpy.typing import NDArray
from tqdm import tqdm


def resolving_power_real(w_x: dict,
                     w_l: dict,
                     w_y: dict,
                    s: float,
                    sph_bool: bool,
                    neigh_dist_xy: dict,
                    neigh_r: dict,
                    n_samples: int,
                    rho: None | dict = None,
                    neigh_coor: None | dict = None,
                    ) ->  Tuple[NDArray, NDArray]:

    pi = np.pi
    k_ny = pi / s

    k = np.arange(start=1, stop=int(k_ny), step=2, dtype=np.float32)

    # Sampling nodes from domain to do resolving power analysis
    rng = np.random.default_rng(42)
    ls = list(neigh_dist_xy.keys())
    ls = np.array(ls)
    n_samples = min(n_samples, len(w_x))
    coor_idx = rng.choice(len(ls), size=n_samples, replace=False)
    coor_to_test = ls[coor_idx]

    x_res = []
    lap_res = []
    # looping over each sample in k_x
    for i in tqdm(range(len(k)), desc="Processing k_x"):
        l2_x = 0
        l2_l = 0

        # loops over each node that will be used to compute the resolving power
        for c_node in coor_to_test:
            n_dist = neigh_dist_xy[tuple(c_node)] # obtains the x and y distances of all neighbours of the node
            gsum = 0
            lsum = 0
            # loops over each neighbour of a central node
            for j, neigh in enumerate(n_dist):
                fji = np.sin(k[i] * neigh[0])

                if sph_bool:
                    neigh_coor_s = neigh_coor[tuple(c_node)][j]
                    rho_neigh = rho[tuple(neigh_coor_s)]
                    gsum = gsum + (fji * w_x[tuple(c_node)][j] / rho_neigh)

                    # Moris operator
                    fji = 0.5 - 0.5 * np.cos(k[i] * neigh[0]) * np.cos(k[i] * neigh[1])
                    term_x = neigh[0] * w_x[tuple(c_node)][j]
                    term_y = neigh[1] * w_y[tuple(c_node)][j]
                    denominator = rho_neigh * (neigh_r[tuple(c_node)][i] ** 2)
                    den_mask = denominator > 0
                    lsum = lsum + (2 * (term_x + term_y) / denominator) * fji if den_mask else 0
                else:
                    gsum = gsum + (fji * w_x[tuple(c_node)][j])
                    fji = 0.5 - 0.5 * np.cos(k[i] * neigh[0]) * np.cos(k[i] * neigh[1])
                    lsum = lsum + fji * w_l[tuple(c_node)][j]

            #
            tmp = gsum / k_ny
            l2_x = l2_x + tmp * tmp

            tmp = lsum / k_ny ** 2
            l2_l = l2_l + tmp * tmp


        # below some data structuring
        x_res.append([(l2_x / n_samples) ** .5, k[i] / k_ny])
        lap_res.append([(l2_l / n_samples) ** .5, k[i] / k_ny])

    x_res = np.array(x_res)
    lap_res = np.array(lap_res)

    return x_res, lap_res