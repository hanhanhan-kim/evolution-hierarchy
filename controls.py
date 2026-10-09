"""Controls for dimensional collapse (Linear FRN-2284). Run: python controls.py

Products of random Gaussian matrices lose effective dimensions with depth, so a
deeper organism has a lower-dimensional phenotype. Is that, rather than layering,
what speeds convergence (Fig 4) and buffers perturbation (Fig 3B)?

Arms, all with the Figure 4 settings (dim 20, mutation 0.2, 640 runs):
  gaussian    L random Gaussian layers (the paper's model)
  orthogonal  L random orthogonal layers: depth without collapse
  matched     one layer with the singular spectrum of an L-deep Gaussian product
  rank        one layer projecting onto k dimensions (flat spectrum)
  tanh g      L nonlinear layers, tanh(W x) with W ~ N(0, g^2/d)

The evolution loop is evolution.run_evolution, batched over runs; check()
compares the two.
"""

import numpy as np

D, MUT_STD, THRESHOLD = 20, 0.2, 0.65


def haar(rng, shape):
    q, r = np.linalg.qr(rng.standard_normal(shape))
    return q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[..., None, :]


def make_arm(kind, param, runs, rng):
    """Return (maps, act, scale): maps is a list of (runs, D, D) arrays, the
    genotype is multiplied by scale before the first map."""
    gauss = lambda: rng.standard_normal((runs, D, D))
    if kind == "gaussian":
        return [gauss() for _ in range(param)], None, 1.0
    if kind == "orthogonal":
        return [haar(rng, (runs, D, D)) for _ in range(param)], None, 1.0
    if kind == "matched":
        P = gauss()
        for _ in range(param - 1):
            P = gauss() @ P
        s = np.linalg.svd(P, compute_uv=False)
        return [haar(rng, (runs, D, D)) * s[:, None, :] @ haar(rng, (runs, D, D))], None, 1.0
    if kind == "rank":
        s = (np.arange(D) < param).astype(float)
        return [haar(rng, (runs, D, D)) * s @ haar(rng, (runs, D, D))], None, 1.0
    if kind.startswith("tanh"):
        g = float(kind[4:])
        # genotype on the sphere of radius sqrt(D), so preactivations are O(g)
        return [gauss() * g / np.sqrt(D) for _ in range(param)], np.tanh, np.sqrt(D)
    raise ValueError(kind)


def forward(X, maps, act, scale, upto=None):
    """X: (runs, n, D). Returns the list of layer outputs, genotype first."""
    out = [X * scale]
    for M in maps[:upto]:
        Y = out[-1] @ M.swapaxes(-1, -2)
        out.append(act(Y) if act else Y)
    return out


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def cos(Y, y):
    return np.einsum("rnd,rd->rn", unit(Y), unit(y))


def effective_dims(maps, act, scale, rng, n=4000):
    """Participation ratio of the second moment of phenotype directions under
    random genotypes: how many directions the phenotype actually spans."""
    X = unit(rng.standard_normal((maps[0].shape[0], n, D)))
    U = unit(forward(X, maps, act, scale)[-1])
    lam = np.linalg.eigvalsh(np.einsum("rni,rnj->rij", U, U) / n)
    return float(np.median(lam.sum(-1) ** 2 / (lam**2).sum(-1)))


def initial_population(maps, act, scale, y_opt, pop, rng):
    """Genotypes whose phenotype is orthogonal to the optimum (Fig 4's start)."""
    runs = y_opt.shape[0]
    if act is None:  # exact, as in evolution.run_evolution
        A = forward(np.broadcast_to(np.eye(D), (runs, D, D)), maps, act, scale)[-1]
        v = np.einsum("rij,rj->ri", A, y_opt)
        Z = rng.standard_normal((runs, pop, D))
        Z -= np.einsum("rnd,rd->rn", Z, v)[..., None] * v[:, None] / (v * v).sum(-1)[:, None, None]
        return unit(Z)
    # nonlinear: keep the candidates closest to orthogonal
    Z = unit(rng.standard_normal((runs, 4000, D)))
    c = np.abs(cos(forward(Z, maps, act, scale)[-1], y_opt))
    return np.take_along_axis(Z, np.argsort(c, axis=1)[:, :pop, None], axis=1)


def evolve(kind, param, pop, runs=640, gens=1000, seed=0):
    """Median fitness per generation, as in Fig 4B, plus effective dimensions."""
    rng = np.random.default_rng(seed)
    maps, act, scale = make_arm(kind, param, runs, rng)
    x_opt = unit(rng.standard_normal((runs, 1, D)))
    y_opt = forward(x_opt, maps, act, scale)[-1][:, 0]
    X = initial_population(maps, act, scale, y_opt, pop, rng)
    med = np.empty(gens)
    for g in range(gens):
        w = (cos(forward(X, maps, act, scale)[-1], y_opt) + 1) / 2 + 1e-6
        med[g] = np.median(w)
        cdf = np.cumsum(w, axis=1)
        u = rng.random((runs, pop, 1)) * cdf[:, -1:, None]
        parents = (u > cdf[:, None, :]).sum(-1)
        X = np.take_along_axis(X, parents[..., None], axis=1)
        X = unit(X + rng.normal(0, MUT_STD / np.sqrt(D), X.shape))
    return med, effective_dims(maps, act, scale, rng)


def sensitivity(kind, depth=50, runs=25, n=2000, noise=0.5, seed=42):
    """Fig 3B with 1 - cos (robustness.py uses 1 - |cos|, which hides sign flips)."""
    rng = np.random.default_rng(seed)
    maps, act, scale = make_arm(kind, depth, runs, rng)
    X = unit(rng.standard_normal((runs, n, D)))
    Xp = X + noise * rng.standard_normal(X.shape)
    a, b = forward(X, maps, act, scale), forward(Xp, maps, act, scale)
    return np.array([np.median(1 - (unit(p) * unit(q)).sum(-1)) for p, q in zip(a[1:], b[1:])])


def gens_to(med):
    hit = np.flatnonzero(med > THRESHOLD)
    return int(hit[0]) if hit.size else None


def check():
    """The batched loop matches evolution.run_evolution on the paper's model."""
    import contextlib, io
    from evolution import parallel_run_evolution
    np.random.seed(1)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = parallel_run_evolution(128, n_generations=150, population_size=5, mutation_std=MUT_STD,
                                     mutation_rate=1, eval_fraction=1, max_depth=4, dim=D,
                                     fix_initial_pop_distance=True)
    ref = np.median(ref["fitness"], axis=(0, 1))
    ours, _ = evolve("gaussian", 4, 5, runs=128, gens=150)
    assert abs(ref[:5].mean() - 0.5) < 0.02 and abs(ours[:5].mean() - 0.5) < 0.02, (ref[:5], ours[:5])
    assert abs(ref[-50:].mean() - ours[-50:].mean()) < 0.03, (ref[-50:].mean(), ours[-50:].mean())
    o = make_arm("orthogonal", 3, 2, np.random.default_rng(0))[0]
    assert np.allclose(o[0] @ o[0].swapaxes(-1, -2), np.eye(D))
    print("check ok: plateau", round(ref[-50:].mean(), 3), "vs", round(ours[-50:].mean(), 3))


ARMS = (
    [("gaussian", L) for L in (1, 2, 4, 6)]
    + [("orthogonal", L) for L in (1, 2, 4, 6)]
    + [("matched", L) for L in (2, 4, 6)]
    + [("rank", k) for k in (20, 14, 10, 7, 5, 3, 2)]
    + [(t, L) for t in ("tanh1", "tanh3") for L in (1, 2, 4, 6)]
)


def main(pops=(5, 75)):
    import json
    rows = []
    for pop in pops:
        for kind, p in ARMS:
            med, dims = evolve(kind, p, pop)
            rows.append(dict(kind=kind, param=p, pop=pop, dims=round(dims, 2), gens=gens_to(med),
                             plateau=round(float(med[-200:].mean()), 4), median=med.round(4).tolist()))
            print(f"pop {pop:3d}  {kind:10s} {p:2d}  dims {dims:5.2f}  gens {gens_to(med)}  plateau {rows[-1]['plateau']}")
    sens = {k: sensitivity(k).round(4).tolist() for k in ("gaussian", "orthogonal", "tanh1", "tanh3")}
    with open("output/controls.json", "w") as f:
        json.dump(dict(rows=rows, sensitivity=sens), f)


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(path="output/controls.pdf"):
    """Dimensional-collapse controls, using only the saved simulation results."""
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    import plotting  # noqa: F401  (applies the figure style on import)

    with open("output/controls.json") as f:
        data = json.load(f)
    kinds = list(dict.fromkeys(k for k, _ in ARMS))
    colors = {k: f"C{i}" for i, k in enumerate(kinds)}
    text = dict(fontsize=6, family="monospace")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7, 2.3), layout="constrained")

    for kind in kinds:
        color = colors[kind]
        for pop in (75, 5):
            rows = sorted((r for r in data["rows"] if r["kind"] == kind and r["pop"] == pop),
                          key=lambda r: r["param"])
            if kind in data["sensitivity"]:
                # Leave gaps for censored observations rather than join them as measurements.
                b.plot([r["param"] for r in rows],
                       [r["gens"] if r["gens"] is not None else np.nan for r in rows],
                       color=color, linestyle="-" if pop == 5 else "--", linewidth=1)
            for r in rows:
                censored = r["gens"] is None
                marker = dict(marker="v" if censored else "o", markersize=3.5,
                              markerfacecolor="none" if censored or pop == 75 else color,
                              color=color, linestyle="none", markeredgewidth=0.7)
                y = 1000 if censored else r["gens"]
                a.plot(r["dims"], y, alpha=1 if pop == 5 else 0.25, **marker)
                if kind in data["sensitivity"]:
                    b.plot(r["param"], y, **marker)

    # Direct labels with short leaders where the dimension controls overlap.
    positions = {"gaussian": (4, 230), "orthogonal": (8, 1000), "matched": (1, 85),
                 "rank": (16, 85), "tanh1": (8, 450), "tanh3": (14, 270)}
    for kind, xytext in positions.items():
        row = next(r for r in data["rows"] if r["kind"] == kind and r["pop"] == 5
                   and r["param"] == {"rank": 14, "matched": 2}.get(kind, 1))
        a.annotate(kind, (row["dims"], 1000 if row["gens"] is None else row["gens"]),
                   xytext=xytext, color=colors[kind], va="center", **text,
                   arrowprops=dict(arrowstyle="-", color=colors[kind], lw=0.5))

    for kind, values in data["sensitivity"].items():
        row = next(r for r in data["rows"] if r["kind"] == kind and r["pop"] == 5 and r["param"] == 6)
        b.annotate(kind, (6, 1000 if row["gens"] is None else row["gens"]),
                   xytext=(5, {"gaussian": -14, "tanh1": 10}.get(kind, 0)),
                   textcoords="offset points", color=colors[kind], va="center", **text,
                   arrowprops=dict(arrowstyle="-", color=colors[kind], lw=0.5))
        c.plot(np.arange(1, len(values) + 1), values, color=colors[kind], linewidth=1)
        c.text(52, values[-1], kind, color=colors[kind], va="center", **text)

    for ax, letter in zip((a, b, c), "ABC"):
        panel_letter(ax, letter)
    for ax in (a, b):
        ax.set_yscale("log")
        ax.set_ylim(3, 1700)
        ax.set_ylabel("Generations to threshold")
    a.set(xlabel="Effective dimensions", xlim=(0, 22), xticks=[0, 10, 20])
    b.set(xlabel="Number of layers", xlim=(0.7, 10.5), xticks=[1, 2, 4, 6])
    c.set(xlabel="Layer depth", ylabel="Sensitivity (1 − cos)", xlim=(0, 80),
          ylim=(0, 1.1), xticks=[1, 25, 50])
    a.text(0.02, 0.98, "faint/open: pop 75", transform=a.transAxes, va="top", **text)
    b.text(0.03, 0.80, "solid: pop 5\ndashed: pop 75\n$\\triangledown$: not reached",
           transform=b.transAxes, va="top", **text)
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=200)
    return fig


if __name__ == "__main__":
    import sys
    check() if "check" in sys.argv else figure() if "figure" in sys.argv else main()
