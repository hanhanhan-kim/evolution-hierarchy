import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib import font_manager
from pathlib import Path

# Optional local house style. design/ is gitignored, so a public checkout renders
# in plain matplotlib; figure code should use the colour cycle (C0, C1, ...)
# rather than named colours so it looks right either way.
_DESIGN = Path(__file__).parent / "design"
if (_DESIGN / "figures.mplstyle").exists():
    for _font in (_DESIGN / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(_font))
    plt.style.use(_DESIGN / "figures.mplstyle")


def plot_layer_evolution(layer_stats, layers_to_plot=None, figsize=(8, 6), save=False):
    """
    Plot evolution of statistics across generations for multiple layers.

    Parameters
    ----------
    layer_stats : dict
        Dictionary containing 'mean', 'q1s', 'q3s' arrays of shape (n_layers, n_generations)
    layers_to_plot : list, optional
        List of layer indices to plot. If None, plots all layers
    figsize : tuple, optional
        Figure size
    """
    n_layers, n_generations = layer_stats["mean"].shape
    if layers_to_plot is None:
        layers_to_plot = range(n_layers)

    plt.figure(figsize=figsize)

    # Use different colors for different layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))
    # make the last color "tab:orange" in terms of four numbers
    colors[-1] = (1.0, 0.4980392156862745, 0.0, 1.0)

    for layer_idx, color in zip(layers_to_plot, colors):
        # Plot mean line
        plt.plot(
            range(n_generations),
            layer_stats["mean"][layer_idx],
            color=color,
            label=f"Layer {layer_idx}",
        )

        # Plot shaded area between +/- 1 standard deviations
        plt.fill_between(
            range(n_generations),
            layer_stats["mean"][layer_idx] - 1 * layer_stats["stdev"][layer_idx],
            layer_stats["mean"][layer_idx] + 1 * layer_stats["stdev"][layer_idx],
            color=color,
            alpha=0.2,
        )

    # Remove borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Remove grid
    plt.grid(False)

    # Increase font sizes
    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Optimality", fontsize=18)
    plt.tick_params(axis="both", labelsize=16)

    # Remove legend border and adjust position
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig("output/layerwise_evolution.pdf", format="pdf", dpi=300)

    plt.show()


def plot_optimality_by_layer_at_last_generation(
    optimality_by_layer, figsize=(8, 6), save=False
):
    """
    Plot a line graph where the x-axis is the number of layers, and the y-axis is the
    optimality at the last generation for the top layer.

    Parameters
    ----------
    optimality_by_layer : dict
        Dictionary where keys are number of layers (int), and values are optimality matrices
        of shape (n_runs, n_generations).
    figsize : tuple
        Size of the overall figure.
    save : bool
        Whether to save the plot.
    """
    layer_counts = sorted(optimality_by_layer.keys(), key=lambda x: int(x))
    last_generation_optimality = []

    for layer in layer_counts:
        optimality_matrix = optimality_by_layer[
            layer
        ]  # shape: (n_layers, n_generations)
        # Extract the last layer's value at the last generation
        last_generation_optimality.append(optimality_matrix[-1, -1])

    # Plot the results
    plt.figure(figsize=figsize)
    plt.plot(
        layer_counts,
        last_generation_optimality,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Last Generation Optimality",
    )

    # Add labels, legend, and clean up the plot
    plt.xlabel("Number of Layers", fontsize=14)
    plt.ylabel("Optimality (Last Generation)", fontsize=14)
    plt.title("Optimality by Layer at Last Generation", fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(
            "output/optimality_by_layer_last_generation.pdf", format="pdf", dpi=300
        )

    plt.show()


def plot_stacked_ancestry_grid(
    ancestry_matrices,
    n_cols=3,
    figsize=(15, 8),
    titles=None,
    colormap="tab10",
    top_n_colored=10,
    save=False,
):
    """
    ancestry_matrices: list of ancestry matrices (each of shape n_ancestors x n_generations)
    n_cols: number of columns in the grid
    figsize: overall figure size
    titles: optional list of titles for each subplot
    """
    n_plots = len(ancestry_matrices)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, ancestry_matrix in enumerate(ancestry_matrices):
        ax = axes[i]

        n_ancestors, n_generations = ancestry_matrix.shape
        generations = range(n_generations)

        # Sort ancestors by total contribution (area under the curve)
        total_contribution = ancestry_matrix.sum(axis=1)
        sorted_indices = np.argsort(-total_contribution)
        sorted_ancestry = ancestry_matrix[sorted_indices]

        # Create a colormap with unique colors for top N contributors
        cmap = get_cmap(colormap)
        colors = [None] * n_ancestors
        for j in range(n_ancestors):
            if j < top_n_colored:
                colors[j] = cmap(j / top_n_colored)  # assign distinct color
            else:
                colors[j] = (0.7, 0.7, 0.7, 0.3)  # gray-ish, semi-transparent

        ax.stackplot(generations, sorted_ancestry, colors=colors, alpha=0.95)
        ax.set_title(titles[i] if titles and i < len(titles) else f"Run {i + 1}")
        ax.set_xlabel("generation")
        ax.set_ylabel("proportion")

        ax.grid(True, linestyle="--", alpha=0.4)

    # Turn off any unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Ancestry composition over time (across runs)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig("output/ancestry_over_generation.jpeg", format="jpeg", dpi=300)
    plt.show()


def plot_fitness_violin_by_layer(
    fitness_by_layer, generations_to_plot, figsize=(10, 6), save=False
):
    """
    Plot violin plots of normalized fitness across different layer depths for specified generations.

    Parameters
    ----------
    fitness_by_layer : dict
        Dictionary where keys are number of layers (int), and values are fitness matrices
        of shape (population_size, n_generations).
    generations_to_plot : list of int
        List of generation indices to plot.
    figsize : tuple
        Size of the overall figure.
    save : bool
        Whether to save the plot.
    """

    layer_counts = sorted(fitness_by_layer.keys(), key=lambda x: int(x))
    n_generations = len(generations_to_plot)

    _, axes = plt.subplots(1, n_generations, figsize=figsize, sharey=True)

    if n_generations == 1:
        axes = [axes]

    for i, gen in enumerate(generations_to_plot):
        data_for_violin = []
        positions = []

        for j, layer in enumerate(layer_counts):
            fitness_runs = fitness_by_layer[
                layer
            ]  # shape: (n_runs, population_size, n_generations)

            # Collect all normalized fitness values for this layer and generation
            all_normalized = []

            for run_fitness in fitness_runs:
                fitness_gen = run_fitness[:, gen]  # (population_size,)
                normalized = [np.ptp(fitness_gen)]  # peak-to-peak (max - min)
                all_normalized.extend(normalized)  # flatten across runs

            data_for_violin.append(all_normalized)
            positions.append(j)

        axes[i].violinplot(
            data_for_violin, positions=positions, showmeans=False, showmedians=True
        )
        axes[i].set_title(f"Generation {gen}", fontsize=14)
        axes[i].set_xlabel("number of layers", fontsize=12)
        axes[i].set_xticks(range(len(layer_counts)))
        axes[i].set_xticklabels([str(layer) for layer in layer_counts])
        if i == 0:
            axes[i].set_ylabel("max fitness - min fitness", fontsize=12)

    plt.tight_layout()

    if save:
        plt.savefig("output/fitness_violin_by_layer.pdf", format="pdf", dpi=300)

    plt.show()


def plot_median_fitness_by_generation(
    fitness_by_layer, figsize=(7, 4), threshold=0.9, save=False
):
    """
    Plot a line graph of the median of np.ptp(fitness_gen) across all simulations
    for each layer over all generations.

    Parameters
    ----------
    fitness_by_layer : dict
        Dictionary where keys are number of layers (int), and values are fitness matrices
        of shape (population_size, n_generations).
    figsize : tuple
        Size of the overall figure.
    save : bool
        Whether to save the plot.
    """
    layer_counts = sorted(fitness_by_layer.keys(), key=lambda x: int(x))
    n_generations = next(iter(fitness_by_layer.values())).shape[2]
    generation_achieve_threshold = [0] * len(layer_counts)

    plt.figure(figsize=figsize)

    for idx, layer in enumerate(layer_counts):
        fitness_runs = fitness_by_layer[
            layer
        ]  # shape: (n_runs, population_size, n_generations)
        fitness_runs = (
            fitness_runs * fitness_runs.shape[1]
        )  # Multiply by population size
        means = []

        for gen in range(n_generations):
            all_min = [np.min(run_fitness[:, gen]) for run_fitness in fitness_runs]
            means.append(np.mean(all_min))

        plt.plot(range(n_generations), means, label=f"{layer} layers", linewidth=2)

        try:
            generation_achieve_threshold[idx] = np.where(np.array(means) >= threshold)[
                0
            ][0]
        except IndexError:
            generation_achieve_threshold[idx] = -1  # or handle as appropriate

    # Axis settings
    plt.xlabel("Generations", fontsize=16)
    plt.ylabel("Mean (min fitness)", fontsize=16)

    # Legend settings
    plt.legend(fontsize=11)

    # Remove grid and top/right spines
    ax = plt.gca()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ticks font size
    ax.tick_params(axis="both", which="major", labelsize=13)

    # plot horizontal line at y=0.9
    # plt.axhline(y=0.9, color="black", linestyle="--", linewidth=1)

    # ax.set_ylim((0.88, 0.91))
    # ax.set_xlim((12, 80))

    plt.tight_layout()

    if save:
        plt.savefig("output/mean_fitness_by_generation.pdf", format="pdf", dpi=300)

    plt.show()
    return layer_counts, generation_achieve_threshold


def plot_generation_to_optimality(
    layer_counts, generation_achieve_threshold, save=False
):
    # plot the generation to achieves the threshold against the layer
    plt.figure(figsize=(5, 3))
    plt.plot(
        layer_counts,
        generation_achieve_threshold,
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    plt.xlabel("Total number of layers", fontsize=14)
    plt.ylabel("Avg. generation \n to convergence", fontsize=14)

    # remove top and right spines, remove grid
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(False)
    plt.tight_layout()

    if save:
        plt.savefig("output/generation_to_optimality.pdf", format="pdf", dpi=300)

    plt.show()


def plot_half_max_heatmap(df, figsize=(10, 8), save=False):
    """
    Create a heatmap visualization of the DataFrame showing first indices to reach half maximum fitness.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with max_depth as index and population_size as columns
    figsize : tuple
        Figure size (width, height)
    save : bool
        Whether to save the plot
    """
    # Convert DataFrame values to numeric type
    df_numeric = df.astype(float)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using imshow
    im = ax.imshow(df_numeric.values, cmap="YlOrRd")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Generation")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add labels
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Max Depth")

    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(
                j,
                i,
                f"{int(df_numeric.iloc[i, j])}",
                ha="center",
                va="center",
                color="black",
            )

    # Adjust layout
    plt.tight_layout()

    if save:
        plt.savefig(
            "output/half_max_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight"
        )

    plt.show()


def plot_fitness_grid(fitness_dict, figsize=(15, 5), save=False):
    """
    Create a grid of line plots showing fitness evolution for different population sizes at each max_depth.

    Parameters
    ----------
    fitness_dict : dict
        Nested dictionary where outer keys are max_depth (str), inner keys are population_size (str),
        and values are numpy arrays of fitness values.
    figsize : tuple
        Figure size (width, height)
    save : bool
        Whether to save the plot
    """
    n_max_depths = len(fitness_dict)
    fig, axes = plt.subplots(1, n_max_depths, figsize=figsize, sharey=True)

    # If only one max_depth, axes won't be an array
    if n_max_depths == 1:
        axes = [axes]

    # Get all population sizes (should be same for all max_depths)
    population_sizes = list(next(iter(fitness_dict.values())).keys())

    # Create a plot for each max_depth
    for ax, (max_depth, pop_dict) in zip(axes, fitness_dict.items()):
        # Plot a line for each population size
        for pop_size in population_sizes:
            fitness_array = pop_dict[pop_size][-1]  # Get the fitness array
            generations = np.arange(len(fitness_array))
            ax.plot(generations, fitness_array, label=f"Pop Size {pop_size}")

        # Customize the subplot
        ax.set_title(f"Max Depth = {max_depth}", fontsize=16)
        ax.set_xlabel("Generation", fontsize=16)
        if ax == axes[0]:  # Only for the first subplot
            ax.set_ylabel("Fitness", fontsize=16)
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.5), fontsize=8)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save:
        plt.savefig(
            "output/fitness_grid.pdf", format="pdf", dpi=300, bbox_inches="tight"
        )

    plt.show()


def plot_fitness_grid_by_population(fitness_dict, figsize=(15, 5), save=False):
    """
    Create a grid of line plots showing fitness evolution for different max depths at each population size.

    Parameters
    ----------
    fitness_dict : dict
        Nested dictionary where outer keys are max_depth (str), inner keys are population_size (str),
        and values are numpy arrays of fitness values.
    figsize : tuple
        Figure size (width, height)
    save : bool
        Whether to save the plot
    """
    # Get all population sizes (should be same for all max_depths)
    population_sizes = list(next(iter(fitness_dict.values())).keys())
    n_pop_sizes = len(population_sizes)

    fig, axes = plt.subplots(1, n_pop_sizes, figsize=figsize, sharey=True)

    # If only one population size, axes won't be an array
    if n_pop_sizes == 1:
        axes = [axes]

    # Create a plot for each population size
    for ax, pop_size in zip(axes, population_sizes):
        # Plot a line for each max_depth
        for max_depth, pop_dict in fitness_dict.items():
            fitness_array = pop_dict[pop_size][-1]  # Get the fitness array
            generations = np.arange(len(fitness_array))
            ax.plot(generations, fitness_array, label=f"Depth {max_depth}")

        # Customize the subplot
        ax.set_title(f"Population Size = {pop_size}")
        ax.set_xlabel("Generation")
        if ax == axes[0]:  # Only for the first subplot
            ax.set_ylabel("Fitness")
        ax.grid(True, alpha=0.3)

        # Add legend into the line plot
        ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.5))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save:
        plt.savefig(
            "output/fitness_grid_by_population.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
