import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
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
            "models": r"GNN$_{p=2}$(ours)",
        }

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)
    fig, ax = plt.subplots(figsize=figsize)

    # ---- explicit plotting order (bottom → top) ----
    def kernel_priority_laplace(k):
        # bottom → top: LABFM, GNN, WC2, Q_S
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
        # bottom → top: WC2, LABFM, Q_S, GNN
        if k == "wc2":
            return (0, 0)
        if isinstance(k, int):  # LABFM orders
            return (1, k)
        if k == "q_s":
            return (2, 0)
        if k == "models":
            return (3, 0)
        return (4, 0)

    priority = kernel_priority_laplace if diff_operator == "laplace" else kernel_priority_nonlaplace

    items = sorted(
        results.items(),
        key=lambda kv: (priority(kv[0][1]), kv[0][0])  # (kernel-priority, resolution)
    )

    # Get default matplotlib color cycle
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
            # if diff_operator == 'laplace': h = h ** 2
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

    # ---- axes styling ----
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # grid below everything
    ax.grid(True, which="major", linewidth=0.4, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.25)
    ax.minorticks_on()

    # ---- black zero axes on top ----
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
    show_legend=True,
    use_inset=True,                 # Laplacian inset
    zoom_y=False,                   # Laplacian log-zoom feature
    zoom_pad=1.05,
    inset_loc="upper right",
    inset_size=("35%", "25%"),
    inset_xlim=(0.1, 0.3),
    inset_ylim=(0.0, 0.1),

    use_inset_x=False,              # NEW: Gradient inset
    inset_loc_x="upper right",       # NEW
    inset_size_x=("25%", "25%"),    # NEW
    inset_xlim_x=(0.05, 0.25),        # NEW
    inset_ylim_x=(0.05, 0.25),        # NEW
):
    from functions.res_power import resolving_power_real

    if use_inset or use_inset_x:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "models": r"GNN$_{p=2}$(ours)",
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

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)

    fig_x, ax_x = plt.subplots(figsize=figsize)
    fig_lap, ax_lap = plt.subplots(figsize=figsize)

    k_hat_ref = None
    x_curves = {}     # NEW: store gradient curves for inset
    lap_curves = {}

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

        # ---- X derivative modal response
        k_hat_x = res_power_x[:, 1]
        re_keff_x = res_power_x[:, 0]
        idx = np.argsort(k_hat_x)
        k_hat_x = k_hat_x[idx]
        re_keff_x = re_keff_x[idx]

        # ---- Laplacian modal response
        k_hat_l = res_power_lap[:, 1]
        re_lap_eff = res_power_lap[:, 0]
        idx2 = np.argsort(k_hat_l)
        k_hat_l = k_hat_l[idx2]
        re_lap_eff = re_lap_eff[idx2]

        # Reference curves (plotted once)
        if k_hat_ref is None:
            k_hat_ref = k_hat_x

            ax_x.plot(
                k_hat_ref, k_hat_ref,
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

            ax_lap.plot(
                k_hat_ref, (k_hat_ref ** 2),
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

        # Plot lines only
        ax_x.plot(
            k_hat_x, re_keff_x,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        x_curves[kernel] = (k_hat_x, re_keff_x)   # NEW

        x_l = k_hat_l# ** 2
        y_l = re_lap_eff #** 2

        ax_lap.plot(
            x_l, y_l,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        lap_curves[kernel] = (x_l, y_l)

    # --- Formatting: X
    ax_x.set_xlabel(r"$\omega / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_x.set_ylabel(r"$\omega_{\mathrm{eff}} / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_x.tick_params(labelsize=8)
    ax_x.spines["top"].set_visible(False)
    ax_x.spines["right"].set_visible(False)
    ax_x.grid(True, linewidth=0.35)

    # --- Formatting: Laplacian
    ax_lap.set_xlabel(r"$\omega / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_lap.set_ylabel(r"$(\omega_{\mathrm{eff}} / \omega_{\mathrm{Ny}})^2$", fontsize=9)
    ax_lap.tick_params(labelsize=8)
    ax_lap.spines["top"].set_visible(False)
    ax_lap.spines["right"].set_visible(False)
    ax_lap.grid(True, linewidth=0.35)

    # --- Laplacian zoom feature (unchanged)
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

        #ax_lap.set_yscale("log")
        #ax_lap.set_ylim(ymin, zoom_pad * ymax)

    # --- Laplacian inset (unchanged)
    if use_inset and len(lap_curves) > 0 and (k_hat_ref is not None):
        axins = inset_axes(
            ax_lap,
            width=inset_size[0],
            height=inset_size[1],
            loc=inset_loc,
            borderpad=0.8,
        )

        for kernel, (x_l, y_l) in lap_curves.items():
            axins.plot(x_l, y_l, linewidth=1.2, color=colors.get(kernel, "k"))

        axins.plot(k_hat_ref, k_hat_ref**2, linestyle="--", linewidth=1.0, color="black")

        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)

        axins.tick_params(labelsize=7)
        axins.grid(True, linewidth=0.25)

        mark_inset(ax_lap, axins, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    # --- NEW: Gradient inset
    if use_inset_x and len(x_curves) > 0 and (k_hat_ref is not None):
        axins_x = inset_axes(
            ax_x,
            width=inset_size_x[0],
            height=inset_size_x[1],
            loc=inset_loc_x,
            borderpad=0.8,
        )
        #axins_x.set_yscale("log")

        for kernel, (xx, yy) in x_curves.items():
            axins_x.plot(xx, yy, linewidth=1.2, color=colors.get(kernel, "k"))

        # Reference in inset
        axins_x.plot(k_hat_ref, k_hat_ref, linestyle="--", linewidth=1.0, color="black")

        axins_x.set_xlim(*inset_xlim_x)
        axins_x.set_ylim(*inset_ylim_x)

        axins_x.tick_params(labelsize=7)
        axins_x.grid(True, linewidth=0.25)

        mark_inset(ax_x, axins_x, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    if show_legend:
        ax_x.legend(fontsize=7.5, frameon=False)
        ax_lap.legend(fontsize=7.5, frameon=False)

    fig_x.tight_layout()
    fig_lap.tight_layout()

    if save:
        fig_x.savefig(f"{filename_prefix}_real_dx.pdf", bbox_inches="tight")
        fig_lap.savefig(f"{filename_prefix}_real_laplace.pdf", bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_lap)
    else:
        plt.show()


def plot_convergence(
    results,
    derivative="dx",
    size=14,
    save=False,
    filename="convergence.pdf",
    column="single",            # "single" or "double"
    rasterize_points=False,     # True if many points
    show_legend=True,           # <--- ADD THIS
):

    # --- Collect data
    allowed = [2, 4, 6, 8, "q_s", "wc2", "models"]
    poly_data = {}

    for k, v in results.items():
        # k assumed to be (h_or_inv?, degree) as in your code
        poly_degree = k[1]
        if poly_degree not in allowed:
            continue

        s_value = 1.0 / float(k[0])  # keep your convention
        l2_value = getattr(v, f"{derivative}_l2")

        if poly_degree not in poly_data:
            poly_data[poly_degree] = {"s": [], "l2": []}
        poly_data[poly_degree]["s"].append(s_value)
        poly_data[poly_degree]["l2"].append(float(np.asarray(l2_value).reshape(-1)[0]))

    # --- Styling maps (no hard requirement to use these exact colors; keep if you want)
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
        "models": r"GNN$_{p=2}$(ours)",
    }

    # --- Figure size (ICML)
    if column == "single":
        figsize = (3.25, 2.4)
    else:
        figsize = (6.75, 2.6)

    fig, ax = plt.subplots(figsize=figsize)

    # --- Plot each method/degree (sorted)
    # Prefer a deterministic order in legend
    order = [2, 4, 6, 8, "q_s", "wc2", "models"]
    for key in order:
        if key not in poly_data:
            continue

        s = np.asarray(poly_data[key]["s"], dtype=float)
        l2 = np.asarray(poly_data[key]["l2"], dtype=float)

        # sort so lines connect correctly
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

        # Optional: emphasize scatter separately (often unnecessary since markers exist)
        ax.scatter(
            s, l2,
            color=colors.get(key, "k"),
            s=size,
            zorder=3,
            rasterized=rasterize_points,
        )

    # --- Axes / scales
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$s/H$", fontsize=9)
    ax.set_ylabel(r"$L_2$ norm", fontsize=9)  # clearer than "L2 norm" in a paper
    ax.tick_params(labelsize=8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Log ticks: major + minor, mathtext formatting
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(which="minor", colors="0.55")

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # --- Grid: light and unobtrusive
    ax.grid(True, which="major", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.8)

    # --- Legend (optional)
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

