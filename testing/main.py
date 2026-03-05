from classes.simulation import run
from functions.plot import plot_convergence, plot_resolving_p, plot_stability_multi
import pickle as pk

plot_ls = [True,
           True,
           True]

bool_plot_stability   = plot_ls[0]
bool_plot_convergence = plot_ls[1]
bool_plot_resolving_p = plot_ls[2]

# Kernel options:
# Quintic Spline: 'q_s'
# Wendland C2:    'wc2'
# GNN:            'models'
# LABFM:          [2,4,6,8] -> each integer represent a different order of approximation

if __name__ == '__main__':
    total_nodes_list = [10, 20, 50, 100] * 4
    kernel_list =  ['models'] * 4 + [2] * 4 + ['q_s'] * 4 + ['wc2'] * 4
    results = run(total_nodes_list, kernel_list)

# Plot stability of operator
if bool_plot_stability:
    plot_stability_multi(results,
                         diff_operator='dx',
                         save=True,
                         filename='dx_spectrum.pdf')

if bool_plot_convergence:
    plot_convergence(results,
                     'dx',
                     size=20,
                     save=True,
                     show_legend=True)


if bool_plot_resolving_p:
    plot_resolving_p(results,
                          use_inset=True,
                          zoom_y=True,
                          save=True,
                          use_inset_x=True)