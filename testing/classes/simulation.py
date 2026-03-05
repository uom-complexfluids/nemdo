from functions.nodes import create_nodes, calc_h
from functions.qspline_operator import qspline_weights
from functions.wendland_c2_operator import wendlandc2_weights
from functions.labfm_operator import calc_weights
from functions.gnn_operator import gnn_weights
from functions.p_test_function import test_function, dif_analytical, laplace_phi, dif_do, calc_l2


class AbstractBaseClass:
    def __init__(self, total_nodes, kernel):
        # global variables used for all approximations
        self.s              = 1.0 / (total_nodes - 1)
        self.h              = calc_h(self.s, kernel)
        (self.coordinates,
         self.nodes_in_domain)  = create_nodes(total_nodes, self.s, self.h)
        self.total_nodes    = total_nodes


    def polynomial_test_function_method(self):
        # computes the true values of the surface and its differential fields
        self.surface_value = test_function(self.coordinates)
        self.dtdx_true     = dif_analytical(self.coordinates, 'dtdx')
        self.laplace_true  = laplace_phi(self.coordinates)


    def approx_diff_op(self):
        # computes the discrete differential fields
        self.dtdx_approx    = dif_do(self.x, self.surface_value, self._neigh_coor)
        self.laplace_approx = dif_do(self.laplace, self.surface_value, self._neigh_coor)


    def calc_l2(self):
        # computes relative L2 error
        self.dx_l2       = calc_l2(self.dtdx_approx, self.dtdx_true)
        self.laplace_l2  = calc_l2(self.laplace_approx, self.laplace_true)



class LABFM(AbstractBaseClass):
    def __init__(self, polynomial, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        self.polynomial = polynomial
        super().__init__(total_nodes, self.polynomial)
        (self.x,
         self.y,
         self.laplace,
         self._neigh_coor,
         self._neigh_xy) = calc_weights(self.coordinates, self.polynomial, self.h, self.total_nodes)
        self.polynomial_test_function_method()
        self.approx_diff_op()
        self.calc_l2()


class GNN(AbstractBaseClass):
    def __init__(self, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        super().__init__(total_nodes, 'models')
        (self.x,
         self.laplace,
         self._neigh_coor,
         self.node_h,
         self._neigh_xy) = gnn_weights(self.coordinates, self.h, self.total_nodes, self.nodes_in_domain)
        self.polynomial_test_function_method()
        self.approx_diff_op()
        self.calc_l2()


class WLandC2(AbstractBaseClass):
    def __init__(self, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        super().__init__(total_nodes, 'wc2')
        (self.x,
         self.y,
         self.laplace,
         self._neigh_coor,
         self._neigh_xy) = wendlandc2_weights(self.coordinates, self.h, self.total_nodes, self.s)
        self.polynomial_test_function_method()
        self.approx_diff_op()
        self.calc_l2()


class QSPline(AbstractBaseClass):
    def __init__(self, total_nodes):
        self.s = 1.0 / (total_nodes - 1)
        super().__init__(total_nodes, 'q_s')
        (self.x,
         self.y,
         self.laplace,
         self._neigh_coor,
         self._neigh_xy) = qspline_weights(self.coordinates, self.h, self.total_nodes, self.s)
        self.polynomial_test_function_method()
        self.approx_diff_op()
        self.calc_l2()



def run(total_nodes_list, kernel_list):
    result = {}
    for total_nodes, k in zip(total_nodes_list, kernel_list):

        args = (total_nodes,)
        if k in [2, 3, 4, 6, 8]: kernel, args = LABFM, (k, total_nodes)
        elif k == 'q_s': kernel = QSPline
        elif k == 'wc2': kernel = WLandC2
        elif k == 'models': kernel = GNN
        else: raise ValueError(" kernel must be either polynomial order for labfm (2, 3, 4, 6, 8), or one of "
                               "the following kernels 'wc2', 'q_s', 'models'")

        sim = kernel(*args)
        result[(total_nodes, k)] = sim
    return result
