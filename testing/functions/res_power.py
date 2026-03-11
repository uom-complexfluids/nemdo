import numpy as np
from typing import Tuple
from numpy import ndarray
from numpy.typing import NDArray
from torch_geometric.loader.ibmb_loader import topk_ppr_matrix
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

    # Angle of the coming wave can be changed here
    theta = 0

    k = np.arange(start=1, stop=int(k_ny), step=2, dtype=np.float32)

    # Sampling nodes from domain to do resolving power analysis
    rng = np.random.default_rng(42)
    ls = list(neigh_dist_xy.keys())
    ls = np.array(ls)
    n_samples = min(500, len(w_x))
    coor_idx = rng.choice(len(ls), size=n_samples, replace=False)
    coor_to_test = ls[coor_idx]

    x_res = []
    lap_res = []

    # looping over each sample in k_x
    for i in tqdm(range(len(k)), desc="Processing Modal Response"):
        l2_x_re = 0
        l2_l_re = 0

        l2_x_im = 0
        l2_l_im = 0


        # loops over each node that will be used to compute the resolving power
        for c_node in coor_to_test:
            n_dist = neigh_dist_xy[tuple(c_node)] # obtains the x and y distances of all neighbours of the node

            # real parts
            x_real_sum = 0
            lap_real_sum = 0

            x_im_sum = 0
            lap_im_sum = 0

            # loops over each neighbour of a central node
            for j, neigh in enumerate(n_dist):
                # Real part of derivative
                fji = np.sin(k[i] * np.cos(theta * pi) * neigh[0] + k[i] * np.sin(theta * pi) * neigh[1])
                x_real_sum = x_real_sum + fji * w_x[tuple(c_node)][j]

                # Imaginary part of derivative
                fji = 1 - np.cos(k[i] * np.cos(theta * pi) * neigh[0] + k[i] * np.sin(theta * pi) * neigh[1])
                x_im_sum = x_im_sum + fji * w_x[tuple(c_node)][j]

                # Real part of Laplacian
                fji = 1 - np.cos(k[i] * np.cos(theta * pi) * (neigh[0] + neigh[1]))
                lap_real_sum = lap_real_sum + fji * w_l[tuple(c_node)][j]

                # Imaginary part of Laplacian
                fji = np.sin(k[i] * np.cos(theta * pi) * (neigh[0] + neigh[1]))
                lap_im_sum = lap_im_sum - fji * w_l[tuple(c_node)][j]

            tmp = x_real_sum / (1 / s * pi)
            l2_x_re = l2_x_re + tmp * tmp

            tmp = lap_real_sum / ((1 / s * pi) ** 2)
            l2_l_re = l2_l_re + tmp * tmp

            tmp = x_im_sum / (1 / s * pi)
            l2_x_im = l2_x_im + tmp * tmp

            tmp = lap_im_sum / ((1 / s * pi) ** 2)
            l2_l_im = l2_l_im + tmp * tmp

        l2_x_re = np.sqrt(l2_x_re / n_samples)
        l2_l_re = np.sqrt(l2_l_re / n_samples)

        l2_x_im = np.sqrt(l2_x_im / n_samples)
        l2_l_im = np.sqrt(l2_l_im / n_samples)

        # below some data structuring
        x_res.append([l2_x_re, l2_x_im, k[i] / (1 / s * pi)])
        lap_res.append([l2_l_re / 2, l2_l_im / 2, (k[i] / (1 / s * pi)) ** 2])

    x_res_np = np.array(x_res)
    x_res_np= x_res_np
    lap_res_np = np.array(lap_res)


    return x_res_np, lap_res_np