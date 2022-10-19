import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

# avoid type-3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

#############################################################################################################################
#                                                       Plots
#############################################################################################################################

plot_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "black",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_n_shapley(
    n_shapley_values,
    max_degree=None,
    axis=None,
    feature_names=None,
    rotation=70,
    legend=True,
    fig_kwargs={"figsize": (6, 6)},
    barwidth=0.5,
):
    """Generate the plots in the paper.

    Args:
        n_shapley_values (nshap.nShapleyValues): The $n$-Shapley Values that we want to plot.
        max_degree (int, optional): Plots all effect of order larger than max_degree with a single color. Defaults to None.
        axis (optional): Matplotlib axis on which to plot. Defaults to None.
        feature_names (_type_, optional): Used to label the x-axis. Defaults to None.
        rotation (int, optional): Rotation for x-axis labels. Defaults to 70.
        legend (bool, optional): Plot legend. Defaults to True.
        fig_kwargs (dict, optional): fig_kwargs, handed down to matplotlib figure. Defaults to {"figsize": (6, 6)}.
        barwidth (float, optional): Widht of the bars. Defaults to 0.5.

    Returns:
        Matplotlib axis: The plot axis.
    """
    if max_degree == 1:
        n_shapley_values = n_shapley_values.shapley_values()
    num_features = n_shapley_values.dim
    vmax, vmin = 0, 0
    ax = axis
    if axis is None:
        _, ax = plt.subplots(**fig_kwargs)
    if max_degree is None or max_degree >= n_shapley_values.n:
        max_degree = n_shapley_values.n
    ax.axhline(y=0, color="black", linestyle="-")  # line at 0
    for i_feature in range(num_features):
        bmin, bmax = 0, 0
        v = n_shapley_values[(i_feature,)]
        ax.bar(
            x=i_feature,
            height=v,
            width=barwidth,
            bottom=0,
            align="center",
            label=f"Feature {i_feature}",
            color=plot_colors[0],
        )
        bmin = min(bmin, v)
        bmax = max(bmax, v)
        # higher-order effects, up to max_degree
        for n_k in range(2, n_shapley_values.n + 1):
            v_pos = np.sum(
                [
                    n_shapley_values[k] / len(k)
                    for k in n_shapley_values.keys()
                    if (len(k) == n_k and i_feature in k and n_shapley_values[k] > 0)
                ]
            )
            v_neg = np.sum(
                [
                    n_shapley_values[k] / len(k)
                    for k in n_shapley_values.keys()
                    if (len(k) == n_k and i_feature in k and n_shapley_values[k] < 0)
                ]
            )
            # 'max_degree or higher'
            if n_k == max_degree:
                v_pos = np.sum(
                    [
                        n_shapley_values[k] / len(k)
                        for k in n_shapley_values.keys()
                        if (
                            len(k) >= n_k and i_feature in k and n_shapley_values[k] > 0
                        )
                    ]
                )
                v_neg = np.sum(
                    [
                        n_shapley_values[k] / len(k)
                        for k in n_shapley_values.keys()
                        if (
                            len(k) >= n_k and i_feature in k and n_shapley_values[k] < 0
                        )
                    ]
                )
            if v_pos > 0:
                ax.bar(
                    x=i_feature,
                    height=v_pos,
                    width=barwidth,
                    bottom=bmax,
                    align="center",
                    color=plot_colors[n_k - 1],
                )
                bmax = bmax + v_pos
            if v_neg < 0:
                ax.bar(
                    x=i_feature,
                    height=v_neg,
                    width=barwidth,
                    bottom=bmin,
                    align="center",
                    color=plot_colors[n_k - 1],
                )
                bmin = bmin + v_neg
            if n_k == max_degree:  # no higher orders
                break
        vmin = min(vmin, bmin)
        vmax = max(vmax, bmax)
    # axes
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
    ax.set_ylim([1.1 * vmin, 1.1 * vmax])
    ax.set_xticks(np.arange(num_features))
    ax.set_xticklabels(feature_names, rotation=rotation)
    # legend with custom labels
    color_patches = [mpatches.Patch(color=color) for color in plot_colors]
    lables = ["Main"]
    if n_shapley_values.n > 1:
        lables.append("2nd order")
    if n_shapley_values.n > 2 and max_degree > 2:
        if max_degree == 3:
            lables.append(f"3rd-{n_shapley_values.n}th order")
        else:
            lables.append(f"3rd order")
            for i_degree in range(4, n_shapley_values.n + 1):
                if i_degree == max_degree and max_degree < num_features:
                    lables.append(f"{i_degree}-{n_shapley_values.n}th order")
                    break
                else:
                    if i_degree == n_shapley_values.n:
                        lables.append(f"{i_degree}th order")
                    else:
                        lables.append(f"{i_degree}th")
    ax.legend(
        color_patches,
        lables,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        handletextpad=0.5,
        handlelength=1,
        handleheight=1,
    )
    ax.get_legend().set_visible(legend)
    if axis is None:
        plt.show()
    return ax
