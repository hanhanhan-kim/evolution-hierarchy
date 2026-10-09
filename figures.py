"""Regenerate the paper's data figures. Run: python figures.py

Styling comes from plotting.py (plain matplotlib unless a local style is present).
"""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np

import plotting  # noqa: F401  (applies the figure style on import)
from evolution import parallel_run_evolution
from robustness import measure_directional_robustness

LAYERS = [1, 2, 4, 6]
THRESHOLD = 0.65  # 0.7 sat on the 1-layer, low-population plateau (~0.69), so crossing it was noise
LOW_POP, HIGH_POP = 5, 75  # HIGH_POP reconstructed: the original run's size was not recorded


COLUMN_WIDTH = 3.2  # inches, one column of the two-column paper


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def run_fig5(population_size, seed, n_runs=32 * 20, n_generations=200):
    """Fig 5 simulations (parameters from Jerry's scratch/hpc main.py). Returns
    {layers: median fitness per generation} and generations to reach THRESHOLD."""
    np.random.seed(seed)  # parallel_run_evolution draws per-run seeds from this
    medians, gens = {}, {}
    for depth in LAYERS:
        res = parallel_run_evolution(
            n_runs, n_generations=n_generations, population_size=population_size,
            mutation_std=0.2, mutation_rate=1, eval_fraction=1, max_depth=depth,
            dim=20, normalize=False, fix_initial_pop_distance=True,
        )
        med = np.median(res["fitness"], axis=(0, 1))
        medians[depth] = med
        hit = np.flatnonzero(med > THRESHOLD)
        gens[depth] = int(hit[0]) if hit.size else None
    return medians, gens


def sensitivity_by_depth(path="paper/fig_model_b.pdf", max_depth=50, step=5):
    """Fig 4B: sensitivity at each layer to a random perturbation of layer 0."""
    medians, q1s, q3s = measure_directional_robustness(
        max_depth, n_models=25, use_sigmoid=False, rank_fraction=1.0
    )
    depths = np.arange(1, max_depth, step)
    idx = depths - 1

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    panel_letter(ax, "B")
    ax.fill_between(depths, q1s[idx], q3s[idx], color="C0", alpha=0.15, linewidth=0)
    ax.plot(depths, medians[idx], color="C0")
    ax.text(depths[-1], q3s[idx][-1] + 0.03, "interquartile range", color="C0",
            fontsize=7, family="monospace", ha="right", va="bottom")
    ax.text(depths[0] + 1.5, medians[idx][0], "median", color="C0", fontsize=7, family="monospace", va="bottom")
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Sensitivity to perturbation\n(cosine distance)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max_depth)
    fig.savefig(path)
    return fig


def fitness_and_convergence(path_b="paper/fig_sim_b.pdf",
                            path_c="paper/fig_sim_c.pdf", seed=0):
    """Fig 5B (median fitness over generations, low population) and
    Fig 5C (convergence rate, low vs high population)."""
    low_med, low_gens = run_fig5(LOW_POP, seed)
    _, high_gens = run_fig5(HIGH_POP, seed + 1)

    n_gen = len(low_med[LAYERS[0]])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.15))
    panel_letter(ax, "B")
    shade = dict(zip(LAYERS, plt.get_cmap()(np.linspace(0.3, 1, len(LAYERS)))))
    ends = {}
    for i, depth in enumerate(LAYERS):
        y = low_med[depth]
        ax.plot(y, color=shade[depth], linewidth=1.4)
        ends[depth] = y[-20:].mean()
    # Direct labels, nudged apart so they clear each other and the threshold line.
    placed = []
    for depth in sorted(LAYERS, key=lambda d: ends[d]):
        y = ends[depth]
        while any(abs(y - q) < 0.022 for q in placed + [THRESHOLD]):
            y -= 0.004
        placed.append(y)
        ax.text(n_gen + 3, y, f"{depth} layer{'s' * (depth > 1)}",
                color=shade[depth], fontsize=7, family="monospace", va="center")
    ax.axhline(THRESHOLD, color="0.45", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(n_gen + 3, THRESHOLD, "threshold", color="0.45", fontsize=7, family="monospace", va="center")
    ax.set_xlim(0, n_gen)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Median fitness")
    fig.savefig(path_b)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.15))
    panel_letter(ax, "C")
    for i, (label, gens) in enumerate([("high population", high_gens), ("low population", low_gens)]):
        xs = [d for d in LAYERS if gens[d]]
        ys = [1 / gens[d] for d in xs]
        ax.plot(xs, ys, marker="s", markersize=4, color=f"C{i}", linestyle="-" if i == 0 else (0, (4, 2)))
        for d in (d for d in LAYERS if gens[d] is None):
            # Never reached the threshold: the rate is below 1 / n_generations.
            ax.plot(d, 1 / 200, marker="v", markerfacecolor="none", color=f"C{i}")
            ax.text(d + 0.15, 1 / 200, "not reached\nin 200 gen.", color=f"C{i}", fontsize=6, family="monospace", va="center")
        ax.text(xs[-1] + 0.15, ys[-1], label, color=f"C{i}", fontsize=7, family="monospace", va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(LAYERS)
    ax.set_xlim(0.5, 7.8)
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Convergence rate\n(1 / generations to threshold)")
    fig.savefig(path_c)
    print("generations to threshold  low:", low_gens, " high:", high_gens)


def drift_map(path="paper/fig_framework.pdf", max_depth=46):
    """Fig 3: where selection can see a change, by its depth below the phenotype
    and the strength of selection on the phenotype relative to drift (N_e s).

    A change m layers below the phenotype reaches it attenuated by S(m)/S(1),
    with S the model's sensitivity curve (Fig 4B). It is visible to selection
    when N_e s * S(m)/S(1) > 1, and effectively neutral otherwise.
    """
    medians, _, _ = measure_directional_robustness(
        max_depth, n_models=25, use_sigmoid=False, rank_fraction=1.0
    )
    m = np.arange(1, max_depth + 1)
    attenuation = np.minimum.accumulate(medians / medians[0])  # monotone: smooths model noise
    boundary = 1 / attenuation  # N_e s needed for selection to see a change at depth m
    ymin, ymax = 0.5, 100

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    ax.fill_between(m, boundary, ymax, color="C0", alpha=0.12, linewidth=0)
    light = 0.35 * np.array(to_rgba("C1")[:3]) + 0.65  # the drift colour, mixed with white
    with plt.rc_context({"hatch.linewidth": 0.6}):
        ax.fill_between(m, ymin, boundary, facecolor="none", edgecolor=tuple(light),
                        hatch="////", linewidth=0)
    ax.plot(m, boundary, color="0.25", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(max_depth, 1)  # deep layers on the left, the phenotype on the right
    ax.grid(False)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([max_depth, 1])
    ax.set_xticklabels(["molecules", "phenotype"])
    ax.set_xlabel("Layer at which a change occurs")
    ax.set_ylabel("Selection on the phenotype\nrelative to drift ($N_e s$)")
    ax.text(16, 42, "selection holds the phenotype", color="C0", fontsize=7, family="monospace",
            fontweight="semibold", ha="center", va="center")
    ax.text(16, 28, "insect eye, rod photon detection", color="C0", fontsize=6.5, family="monospace",
            ha="center", va="center")
    ax.text(45, 1.3, "changes are effectively neutral", color="C1", fontsize=7, family="monospace",
            fontweight="semibold", ha="left", va="center")
    ax.text(45, 1.12, "channel conductances,\nreceptor genes", color="C1", fontsize=6.5, va="top", family="monospace",
            ha="left")
    ax.annotate("smaller $N_e$", xy=(3, 2.2), xytext=(3, 12), color="0.35", fontsize=7, family="monospace",
                ha="center", arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    fig.savefig(path)
    return fig


if __name__ == "__main__":
    sensitivity_by_depth()
    fitness_and_convergence()
    drift_map()
