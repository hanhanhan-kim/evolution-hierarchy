"""Regenerate the paper's data figures. Run: python figures.py

Styling comes from plotting.py (plain matplotlib unless a local style is present).
"""

import matplotlib.pyplot as plt
import numpy as np

import plotting  # noqa: F401  (applies the figure style on import)
from robustness import measure_directional_robustness


def sensitivity_by_depth(path="output/sensitivity_by_depth.pdf", max_depth=50, step=5):
    """Fig 4B: sensitivity at each layer to a random perturbation of layer 0."""
    medians, q1s, q3s = measure_directional_robustness(
        max_depth, n_models=25, use_sigmoid=False, rank_fraction=1.0
    )
    depths = np.arange(1, max_depth, step)
    idx = depths - 1

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.fill_between(depths, q1s[idx], q3s[idx], color="C0", alpha=0.15, linewidth=0)
    ax.plot(depths, medians[idx], color="C0")
    ax.text(depths[-1], q3s[idx][-1] + 0.03, "interquartile range", color="C0",
            fontsize=7, ha="right", va="bottom")
    ax.text(depths[0] + 1.5, medians[idx][0], "median", color="C0", fontsize=7, va="bottom")
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Sensitivity to perturbation\n(cosine distance)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max_depth)
    fig.savefig(path)
    return fig


if __name__ == "__main__":
    sensitivity_by_depth()
