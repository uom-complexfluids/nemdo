import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, ScalarFormatter
from tqdm import tqdm


def plot_kernel(features, labels=None, alpha=0.6, size = 2):
    # flatten all arrays consistently
    x = features[:, :, 0].flatten()
    y = features[:, :, 1].flatten()
    if labels is not None:
        c = labels.flatten()

    plt.figure(figsize=(6, 6))
    if labels is not None:
        sc = plt.scatter(x, y, c=c, cmap='viridis', s=size, alpha=alpha)
    else:
        sc = plt.scatter(x, y, cmap='viridis', s=size, alpha=alpha)
    plt.xlabel('x distance')
    plt.ylabel('y distance')
    plt.title('Neighbour offsets coloured by target')
    plt.axis('equal')
    plt.colorbar(sc, label='Target value')
    plt.show()

mpl.rcParams.update({
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "axes.unicode_minus": False,
})

def plot_stability_multi(
    results: dict,
    diff_operator: str,
    save: bool = False,
    filename: str = "spectrum.pdf",
    column: str = "single",
    show_axes0: bool = True,
    point_size: float = 8.0,
    alpha: float = 0.65,
    resolutions: set[int] | None = None,
    kernels: set[str | int] | None = None,
    labels: dict[str | int, str] | None = None,
    legend: bool = True,
) -> None:

    if diff_operator not in ["dx", "laplace"]:
        raise ValueError("diff_operator must be one of: 'dx', 'laplace'")

    if labels is None:
        labels = {
            2: r"LABFM$_{p=2}$",
            4: r"LABFM ($p=4$)",
            6: r"LABFM ($p=6$)",
            8: r"LABFM ($p=8$)",
            "q_s": r"Quintic spline",
            "wc2": r"Wendland C2",
            "models": r"NeMDO$_{p=2}$(ours)",
        }

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)
    fig, ax = plt.subplots(figsize=figsize)

    def kernel_priority_laplace(k):
        # bottom → top: LABFM, NeMDO, WC2, Q_S
        if isinstance(k, int):  # LABFM orders
            return (0, k)
        if k == "models":
            return (1, 0)
        if k == "wc2":
            return (2, 0)
        if k == "q_s":
            return (3, 0)
        return (4, 0)

    def kernel_priority_nonlaplace(k):
        # bottom → top: WC2, LABFM, Q_S, NeMDO
        if k == "wc2":
            return (0, 0)
        if isinstance(k, int):
            return (1, k)
        if k == "q_s":
            return (2, 0)
        if k == "models":
            return (3, 0)
        return (4, 0)

    priority = kernel_priority_laplace if diff_operator == "laplace" else kernel_priority_nonlaplace

    items = sorted(
        results.items(),
        key=lambda kv: (priority(kv[0][1]), kv[0][0])
    )


    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "models": "tab:orange",
    }
    legend_done: set[str] = set()

    print("Computing eigenvalues (overlay plot)")

    for (resolution, kernel), attrs in tqdm(
            items,
            desc="Stability spectra",
            unit="kernel",
            leave=False,
    ):
        if resolutions is not None and resolution not in resolutions:
            continue
        if kernels is not None and kernel not in kernels:
            continue

        if diff_operator == "dx":
            weights = attrs.x
        else:
            weights = attrs.laplace

        h = attrs.h
        coor = attrs.coordinates
        neigh_coor = attrs._neigh_coor

        n = len(coor)
        A = np.zeros((n, n))
        coord_to_idx = {tuple(x): i for i, x in enumerate(coor)}
        #if kernel == 'models':
            #h = attrs.node_h
        if isinstance(h, dict):
            for i in range(len(coor)):
                loc = coor[i]
                if tuple(loc) not in weights.keys(): continue

                for j in range(neigh_coor[tuple(loc)].shape[0]):
                    n_j = neigh_coor[tuple(loc)][j]
                    neigh_idx = coord_to_idx[tuple(n_j)]
                    A[i, neigh_idx] = np.array(weights[tuple(loc)][j] * h[tuple(loc)])

                A[i, i] = 0
                A[i, i] = - np.sum(A[i, :])
        else:
            for i in range(len(coor)):
                loc = coor[i]
                if tuple(loc) not in weights.keys(): continue

                for j in range(neigh_coor[tuple(loc)].shape[0]):
                    n_j = neigh_coor[tuple(loc)][j]
                    neigh_idx = coord_to_idx[tuple(n_j)]
                    A[i, neigh_idx] = np.array(weights[tuple(loc)][j] * h)

                A[i, i] = 0
                A[i, i] = - np.sum(A[i, :])

        vals = np.linalg.eigvals(A)

        base_label = labels.get(kernel, str(kernel))
        this_label = f"{base_label}"
        plot_label = this_label if this_label not in legend_done else None
        if plot_label:
            legend_done.add(this_label)

        ax.scatter(
            vals.real,
            vals.imag,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            label=plot_label,
            zorder=3,
            color=colors[kernel]
        )

    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, which="major", linewidth=0.4, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.25)
    ax.minorticks_on()

    if show_axes0:
        ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="black", linewidth=0.9, zorder=1)

    if legend:
        ax.legend(fontsize=7, frameon=False, loc="best")

    ax.set_xlabel(r"$\Re(\mu)$", fontsize=9, labelpad=2)
    ax.set_ylabel(r"$\Im(\mu)$", fontsize=9, labelpad=2)

    fig.tight_layout()

    if save:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)
    else:
        plt.show()


def plot_resolving_p(
    results,
    save=False,
    filename_prefix="resolving_power",
    column="single",
    show_legend=False,
    use_inset_lap=True,
    zoom_y=False,
    zoom_pad=1.05,
    inset_loc="upper left",
    inset_size=("50%", "45%"),
    inset_xlim_lap=(0, 0.0035),
    inset_ylim_lap=(1e-8, 1e-4),

    use_inset_x=False,
    inset_loc_x="upper left",
    inset_size_x=("50%", "45%"),
    inset_xlim_x=(0.01, 0.1),
    inset_ylim_x=(1e-6,1e-2),
):
    from functions.res_power import resolving_power_real

    if use_inset_lap or use_inset_x:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "models": r"NeMDO$_{p=2}$(ours)",
    }

    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "models": "tab:orange",
    }

    figsize = (4.2, 2.6) if column == "single" else (6.75, 2.8)

    fig_x_re, ax_x_re = plt.subplots(figsize=figsize)
    fig_lap_re, ax_lap_re = plt.subplots(figsize=figsize)

    fig_x_im, ax_x_im = plt.subplots(figsize=figsize) if show_legend is False else plt.subplots(figsize=(5.5, 2.6))
    fig_lap_im, ax_lap_im = plt.subplots(figsize=figsize) if show_legend is False else plt.subplots(figsize=(5.5, 2.6))


    k_hat_ref = None
    x_curves = {}
    x_curves_im = {}
    lap_curves = {}
    lap_curves_im = {}


    for (resolution, kernel) in results.keys():
        attrs = results[(resolution, kernel)]
        sph_bool = False

        res_power_x, res_power_lap = resolving_power_real(
            w_x=attrs.x,
            w_l=attrs.laplace if not sph_bool else None,
            w_y=attrs.y if sph_bool else None,
            s=attrs.s,
            sph_bool=sph_bool,
            neigh_dist_xy=attrs._neigh_xy,
            neigh_r=attrs._neigh_r if sph_bool else None,
            rho=attrs.rho if sph_bool else None,
            neigh_coor=attrs._neigh_coor if sph_bool else None,
            n_samples=int(1e10),
        )

        k_hat_x   = res_power_x[:, -1]
        re_keff_x = res_power_x[:, 0]
        im_keff_x = res_power_x[:, 1]

        k_hat_l = res_power_lap[:, -1]
        re_lap_eff = res_power_lap[:, 0] #** 2
        im_lap_eff = res_power_lap[:, 1] #** 2
        lap_x = np.sqrt(k_hat_l)

        # ensures the spectral line is not drawn multiple times if multiple kernels are being plotted
        if k_hat_ref is None:
            # drawing the spectral lines on the plot
            k_hat_ref = True

            ax_x_re.plot(
                k_hat_x, k_hat_x,
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

            ax_lap_re.plot(
                k_hat_l, k_hat_l ** 2,
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

        # plots the resolving power for the mesh-free operators for the derivative
        ax_x_re.plot(
            k_hat_x, re_keff_x,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        x_curves[kernel] = (k_hat_x, abs(re_keff_x - k_hat_x))

        # plots the resolving power for the mesh-free operators for the Laplacian
        ax_lap_re.plot(
            lap_x, re_lap_eff,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        lap_curves[kernel] = (k_hat_l, abs(re_lap_eff - k_hat_l))

        ax_x_im.plot(
            k_hat_x, im_keff_x,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        x_curves_im[kernel] = (k_hat_x, im_keff_x)

        ax_lap_im.plot(
            lap_x, im_lap_eff,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        lap_curves_im[kernel] = (lap_x, im_lap_eff)


    ax_x_re.set_xlabel(r"$\hat{k}$", fontsize=9)
    ax_x_re.set_ylabel(r"$\Re\{\hat{k}_{\mathrm{eff}}\}$", fontsize=9)
    ax_x_re.tick_params(labelsize=8)
    ax_x_re.spines["top"].set_visible(False)
    ax_x_re.spines["right"].set_visible(False)
    ax_x_re.grid(True, linewidth=0.35, alpha=0.2, color='gray')

    ax_lap_re.set_xlabel(r"$\hat{q}$", fontsize=9)
    ax_lap_re.set_ylabel(r"$\Re\{\hat{q}_{\mathrm{eff}}^2\}$", fontsize=9)
    ax_lap_re.tick_params(labelsize=8)
    ax_lap_re.spines["top"].set_visible(False)
    ax_lap_re.spines["right"].set_visible(False)
    ax_lap_re.grid(True, linewidth=0.35, alpha=0.2, color='gray')

    ax_x_im.set_xlabel(r"$\hat{k}$", fontsize=9)
    ax_x_im.set_ylabel(r"$\Im\{\hat{k}_{\mathrm{eff}}\}$", fontsize=9)
    ax_x_im.tick_params(labelsize=8)
    ax_x_im.spines["top"].set_visible(False)
    ax_x_im.spines["right"].set_visible(False)
    ax_x_im.grid(True, linewidth=0.35, alpha=0.2, color='gray')

    ax_lap_im.set_xlabel(r"$\hat{q}$", fontsize=9)
    ax_lap_im.set_ylabel(r"$\Im\{\hat{q}_{\mathrm{eff}}^2\}$", fontsize=9)
    ax_lap_im.tick_params(labelsize=8)
    ax_lap_im.spines["top"].set_visible(False)
    ax_lap_im.spines["right"].set_visible(False)
    ax_lap_im.grid(True, linewidth=0.35, alpha=0.2, color='gray')

    if zoom_y and len(lap_curves) > 0:
        x_threshold = 0.8

        high_x_vals = []
        all_vals = []

        for (x_l, y_l) in lap_curves.values():
            all_vals.append(y_l[y_l > 0.0])

            mask = (x_l >= x_threshold) & (y_l > 0.0)
            if np.any(mask):
                high_x_vals.append(y_l[mask])

        all_vals = np.concatenate(all_vals)
        high_x_vals = np.concatenate(high_x_vals)

        ymin = np.min(high_x_vals)
        ymax = np.max(all_vals)

        #ax_lap_re.set_yscale("log")
        #ax_lap_re.set_ylim(ymin, zoom_pad * ymax)

    if use_inset_lap and len(lap_curves) > 0 and (k_hat_ref is not None):
        axins = inset_axes(
            ax_lap_re,
            bbox_to_anchor=(0.15, 0.0, 1, 1),  # (x, y, width, height) - adjust the 0.1
            bbox_transform=ax_lap_re.transAxes,
            width=inset_size[0],
            height=inset_size[1],
            loc=inset_loc,
            borderpad=0.8,
        )

        for kernel, (x_l, y_l) in lap_curves.items():
            axins.plot(x_l, y_l, linewidth=1.2, color=colors.get(kernel, "k"))

        #axins.plot(k_hat_ref, k_hat_ref**2, linestyle="--", linewidth=1.0, color="black")

        axins.set_xlim(*inset_xlim_lap)
        axins.set_ylim(*inset_ylim_lap)

        axins.set_ylabel(r"$|\Re\{\hat{q}^2_{eff}\} - \hat{q}^2|$", fontsize=8)
        axins.set_yscale("log")
        #axins.set_xscale("log")

        axins.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axins.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axins.xaxis.get_offset_text().set_fontsize(7)

        axins.tick_params(labelsize=7)
        axins.grid(True, linewidth=0.25, alpha=0.2, color='gray')


        axins_lap_im = inset_axes(
            ax_lap_im,
            bbox_to_anchor=(0, 0.0, 1, 1),  # (x, y, width, height) - adjust the 0.1
            bbox_transform=ax_lap_im.transAxes,
            width=inset_size[0],
            height=inset_size[1],
            loc='upper left',
            borderpad=1.6,
        )


        for kernel, (x_l, y_l) in lap_curves_im.items():
            axins_lap_im.plot(x_l, y_l, linewidth=1.2, color=colors.get(kernel, "k"))

        inset_lap_im_x_axis = (0, 0.3)
        inset_lap_im_y_axis = (0, 0.003)

        axins_lap_im.set_xlim(*inset_lap_im_x_axis)
        axins_lap_im.set_ylim(*inset_lap_im_y_axis)

        axins_lap_im.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axins_lap_im.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axins_lap_im.yaxis.get_offset_text().set_fontsize(7)

        axins_lap_im.tick_params(labelsize=7)
        axins_lap_im.grid(True, linewidth=0.25, alpha=0.2, color='gray')



        #mark_inset(ax_lap_re, axins, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    if use_inset_x and len(x_curves) > 0 and (k_hat_ref is not None):
        axins_x = inset_axes(
            ax_x_re,
            bbox_to_anchor=(0.15, 0.0, 1, 1),  # (x, y, width, height) - adjust the 0.1
            bbox_transform=ax_x_re.transAxes,
            width=inset_size_x[0],
            height=inset_size_x[1],
            loc=inset_loc_x,
            borderpad=0.8,
        )
        #axins_x.set_yscale("log")

        for kernel, (xx, yy) in x_curves.items():
            axins_x.plot(xx, yy, linewidth=1.2, color=colors.get(kernel, "k"))

        # Reference in inset
        #axins_x.plot(k_hat_ref, k_hat_ref, linestyle="--", linewidth=1.0, color="black")

        axins_x.set_xlim(*inset_xlim_x)
        axins_x.set_ylim(*inset_ylim_x)

        axins_x.set_ylabel(r"$|\Re\{\hat{k}_{eff}\} - \hat{k}|$", fontsize=8)
        axins_x.set_yscale("log")
        #axins_x.set_xscale("log")

        axins_x.tick_params(labelsize=7)
        axins_x.grid(True, linewidth=0.25, alpha=0.2, color='gray')

        axins_x_im = inset_axes(
            ax_x_im,
            bbox_to_anchor=(0, 0, 1, 1),  # (x, y, width, height) - adjust the 0.1
            bbox_transform=ax_x_im.transAxes,
            width=inset_size[0],
            height=inset_size[1],
            loc='upper left',
            borderpad=1.6,
        )


        for kernel, (x_l, y_l) in x_curves_im.items():
            axins_x_im.plot(x_l, y_l, linewidth=1.2, color=colors.get(kernel, "k"))

        inset_lap_im_x_axis = (0, 0.3)
        inset_lap_im_y_axis = (0, 0.003)

        axins_x_im.set_xlim(*inset_lap_im_x_axis)
        axins_x_im.set_ylim(*inset_lap_im_y_axis)

        axins_x_im.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axins_x_im.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axins_x_im.yaxis.get_offset_text().set_fontsize(7)

        axins_x_im.tick_params(labelsize=7)
        axins_x_im.grid(True, linewidth=0.25, alpha=0.2, color='gray')

        #mark_inset(ax_x_re, axins_x, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    if show_legend:
        legend_args = {'fontsize': 7.5, 'frameon': False, 'bbox_to_anchor': (1, 1), 'loc': 'upper left'}

        #ax_x_re.legend(**legend_args)
        #ax_lap_re.legend(**legend_args)
        ax_x_im.legend(**legend_args)
        ax_lap_im.legend(**legend_args)

    fig_x_re.tight_layout()
    fig_lap_re.tight_layout()
    fig_x_im.tight_layout()
    fig_lap_im.tight_layout()

    if save:
        fig_x_re.savefig(f"{filename_prefix}_real_dx.pdf", bbox_inches="tight")
        fig_lap_re.savefig(f"{filename_prefix}_real_laplace.pdf", bbox_inches="tight")
        fig_x_im.savefig(f"{filename_prefix}_im_dx.pdf", bbox_inches="tight")
        fig_lap_im.savefig(f"{filename_prefix}_im_laplace.pdf", bbox_inches="tight")
        plt.close(fig_x_re)
        plt.close(fig_lap_re)
        plt.close(fig_x_im)
        plt.close(fig_lap_im)
    else:
        plt.show()


def plot_convergence(
    results,
    derivative="dx",
    size=14,
    save=False,
    filename="convergence.pdf",
    column="single",
    rasterize_points=False,
    show_legend=True,
):

    allowed = [2, 4, 6, 8, "q_s", "wc2", "models"]
    poly_data = {}

    for k, v in results.items():
        poly_degree = k[1]
        if poly_degree not in allowed:
            continue

        s_value = 1.0 / float(k[0])
        l2_value = getattr(v, f"{derivative}_l2")

        if poly_degree not in poly_data:
            poly_data[poly_degree] = {"s": [], "l2": []}
        poly_data[poly_degree]["s"].append(s_value)
        poly_data[poly_degree]["l2"].append(float(np.asarray(l2_value).reshape(-1)[0]))

    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "models": "tab:orange",
    }
    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "models": r"NeMDO$_{p=2}$(ours)",
    }

    # --- Figure size (ICML)
    if column == "single":
        figsize = (3.25, 2.4)
    else:
        figsize = (6.75, 2.6)

    fig, ax = plt.subplots(figsize=figsize)

    order = [2, 4, 6, 8, "q_s", "wc2", "models"]
    for key in order:
        if key not in poly_data:
            continue

        s = np.asarray(poly_data[key]["s"], dtype=float)
        l2 = np.asarray(poly_data[key]["l2"], dtype=float)

        idx = np.argsort(s)
        s, l2 = s[idx], l2[idx]

        ax.plot(
            s, l2,
            color=colors.get(key, "k"),
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label=labels.get(key, str(key)),
            rasterized=rasterize_points,
        )

        ax.scatter(
            s, l2,
            color=colors.get(key, "k"),
            s=size,
            zorder=3,
            rasterized=rasterize_points,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$s/H$", fontsize=9)
    ax.set_ylabel(r"$L_2$ norm", fontsize=9)
    ax.tick_params(labelsize=8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(which="minor", colors="0.55")

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)

    if show_legend:
        ax.legend(
            fontsize=7.5,
            frameon=False,
            ncol=1,
            loc="best",
            handlelength=1.8,
            borderaxespad=0.2,
        )


    fig.tight_layout()

    if save:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)
    else:
        plt.show()

