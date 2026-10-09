"""Developmental systems drift with evolving genotypes and developmental maps.

Run: python drift.py [check | quick | figure] [--tanh] [--seed SEED] [--workers N]
The default is the full 10-process grid; quick keeps B=2000 but uses L=6,
K=8 and T=3000. Both write output/drift.json (including parameters and timings).
States and maps use cosine divergence; omega is measured at the first recorded
neutral x0 divergence >= 0.25, or left undefined if that threshold is not reached.
Stiff drift uses each run's own fork Jacobian and normalized consensus genomes.
Its isotropic reference rank/genome_size is a reference, not a fitted baseline.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from controls import haar

OUTPUT = Path(__file__).resolve().parent / "output"


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-300)


def normalize(g, d):
    """Normalize x0 and every map in place; leading axes are arbitrary."""
    g[..., :d] = unit(g[..., :d])
    maps = g[..., d:].reshape(*g.shape[:-1], -1, d * d)
    maps[:] = unit(maps) * np.sqrt(d)
    return g


def forward(g, d, tanh=False):
    x = g[..., :d] * (np.sqrt(d) if tanh else 1)
    states = [x]
    maps = g[..., d:].reshape(*g.shape[:-1], -1, d, d)
    for l in range(maps.shape[-3]):
        x = np.einsum("...ij,...j->...i", maps[..., l, :, :], x)
        if tanh:
            x = np.tanh(x)
        states.append(x)
    return states


def fitness(z, optimum, s):
    distance = np.clip(1 - (unit(z) * optimum).sum(-1), 0, 2)
    return np.exp(-s * distance)


def mutate(g, d, u, sigma, rng):
    """Exact independent Bernoulli sites: binomial count, uniform distinct sites.

    Only draw normals at mutated sites and renormalize touched genome blocks;
    all untouched blocks already have the required norm.
    """
    count = rng.binomial(g.size, u)
    sites = rng.choice(g.size, count, replace=False)
    g.reshape(-1)[sites] += rng.normal(0, sigma, count)
    genomes = g.reshape(-1, g.shape[-1])
    row, site = np.divmod(sites, g.shape[-1])
    xs = np.unique(row[site < d])
    genomes[xs, :d] = unit(genomes[xs, :d])
    L = (g.shape[-1] - d) // (d * d)
    is_map = site >= d
    blocks = np.unique(row[is_map] * L + (site[is_map] - d) // (d * d))
    rows, layers = np.divmod(blocks, L)
    maps = genomes[:, d:].reshape(-1, L, d * d)
    maps[rows, layers] = unit(maps[rows, layers]) * np.sqrt(d)
    return g


def step(g, optimum, s, d, u, sigma, tanh, rng):
    K, N = g.shape[:2]
    w = fitness(forward(g, d, tanh)[-1], optimum, s)
    cdf = np.cumsum(w, axis=1)
    draws = rng.random((K, N, 1)) * cdf[:, -1:, None]
    parents = (draws > cdf[:, None, :]).sum(-1)
    g = g[np.arange(K)[:, None], parents]
    return mutate(g, d, u, sigma, rng)


def consensus(g, d):
    return normalize(g.mean(axis=-2), d)


def stiff_space(g, d, tanh=False, eps=1e-5):
    """Central finite differences of unit(z), in ambient genome coordinates."""
    delta = eps * np.eye(g.size)
    plus = unit(forward(g + delta, d, tanh)[-1])
    minus = unit(forward(g - delta, d, tanh)[-1])
    jacobian = ((plus - minus) / (2 * eps)).T
    _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    return vh[singular > 1e-6 * singular[0]], singular


def divergence(directions):
    """Mean 1-cos over distinct lineage pairs, without a K-by-K array."""
    K = len(directions)
    total = directions.sum(axis=0)
    return np.clip((K * K - (total * total).sum(-1)) / (K * (K - 1)), 0, 2)


def measure(g, optimum, s, d, tanh):
    states = forward(g, d, tanh)
    directions = unit(np.stack(states, axis=-2))  # K, N, L+1, d
    means = unit(directions.mean(axis=1))
    within = np.clip(1 - (directions * means[:, None]).sum(-1), 0, 2)
    maps = g[..., d:].reshape(*g.shape[:2], -1, d * d)
    return dict(state_divergence=divergence(means).tolist(),
                map_divergence=divergence(unit(maps.mean(axis=1))).tolist(),
                within_variation=within.mean(axis=(0, 1)).tolist(),
                mean_fitness=float(fitness(states[-1], optimum, s).mean()))


def run(L=6, N=100, s=100, init="gaussian", B=2000, K=32, T=20000,
        every=100, d=10, u=1e-3, sigma=0.1, seed=0, tanh=False, on_record=None):
    if init not in ("gaussian", "orthogonal"):
        raise ValueError(init)
    if min(L, N, d, every) < 1 or K < 2 or min(B, T) < 0:
        raise ValueError("Require L,N,d,every >= 1, K >= 2 and B,T >= 0")
    if not 0 <= u <= 1 or sigma < 0 or s < 0:
        raise ValueError("Require 0 <= u <= 1 and sigma,s >= 0")
    params = dict(L=L, N=N, s=s, init=init, B=B, K=K, T=T, every=every,
                  d=d, u=u, sigma=sigma, seed=seed, tanh=tanh)
    start = perf_counter()
    # Separate streams make ancestry/mutation draws match across selection arms.
    init_seed, burn_seed, fork_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(init_seed)
    x = unit(rng.standard_normal(d))
    maps = (haar(rng, (L, d, d)) if init == "orthogonal"
            else rng.normal(0, 1 / np.sqrt(d), (L, d, d)))
    ancestor = normalize(np.concatenate([x, maps.ravel()]), d)
    optimum = unit(forward(ancestor, d, tanh)[-1])
    g = np.broadcast_to(ancestor, (1, N, ancestor.size)).copy()
    rng = np.random.default_rng(burn_seed)
    for _ in range(B):
        g = step(g, optimum, s, d, u, sigma, tanh, rng)
    burn_seconds = perf_counter() - start
    fork = consensus(g, d)[0]
    basis, singular = stiff_space(fork, d, tanh)
    g = np.repeat(g, K, axis=0)
    rng = np.random.default_rng(fork_seed)
    records = []
    evolve_start = perf_counter()
    for t in range(T + 1):
        if t % every == 0 or t == T:
            record = dict(t=t, **measure(g, optimum, s, d, tanh))
            if t <= 3000:
                displacement = consensus(g, d) - fork
                norm2 = (displacement**2).sum(-1)
                projected2 = (displacement @ basis.T)**2
                fractions = np.divide(projected2.sum(-1), norm2,
                                      out=np.full(K, np.nan), where=norm2 > 1e-24)
                record["stiff_by_lineage"] = [float(f) if np.isfinite(f) else None for f in fractions]
                valid = fractions[np.isfinite(fractions)]
                record["stiff_fraction"] = float(valid.mean()) if valid.size else None
            if on_record is not None:
                record["hybrids"] = on_record(t, g, optimum)
            records.append(record)
        if t < T:
            g = step(g, optimum, s, d, u, sigma, tanh, rng)
    fork_seconds = perf_counter() - evolve_start
    return dict(**params, ancestor=ancestor.tolist(), optimum=optimum.tolist(),
                fork_genome=fork.tolist(), stiff_basis=basis.tolist(),
                singular_values=singular.tolist(), stiff_dimension=len(basis),
                genome_size=ancestor.size, neutral_expectation=len(basis) / ancestor.size,
                records=records, burn_seconds=burn_seconds, fork_seconds=fork_seconds,
                seconds=perf_counter() - start)


def add_omega(rows):
    for row in rows:
        neutral = next(r for r in rows if r["s"] == 0 and
                       all(r[k] == row[k] for k in ("L", "N", "init", "seed", "tanh")))
        hit = next((i for i, r in enumerate(neutral["records"])
                    if r["state_divergence"][0] >= 0.25), None)
        row["t_star"] = neutral["records"][hit]["t"] if hit is not None else None
        row["omega"] = None
        if hit is not None:
            a = np.array(row["records"][hit]["state_divergence"])
            b = np.array(neutral["records"][hit]["state_divergence"])
            row["omega"] = [float(x / y) if y > 0 else None for x, y in zip(a, b)]


def summary(rows):
    print("init        L   N   s    t*  omega_0 ... omega_L                         stiff@1000  stiff@3000   rank/G")
    for r in rows:
        omega = " ".join(f"{v:.3f}" if v is not None else "NA" for v in r["omega"]) if r["omega"] is not None else "NA (neutral x0 < 0.25)"
        stiff = [next((p.get("stiff_fraction") for p in r["records"] if p["t"] == t), None) for t in (1000, 3000)]
        stiff = "  ".join(f"{v:10.4f}" if v is not None else f"{'NA':>10}" for v in stiff)
        print(f"{r['init']:10s} {r['L']:2d} {r['N']:3d} {r['s']:3d} {str(r['t_star']):>5s}  {omega:47s} {stiff}  {r['neutral_expectation']:.4f}")


def worker(params):
    return run(**params)


def main(quick=False, seed=0, tanh=False, workers=10):
    from tqdm import tqdm
    configs = [dict(L=L, N=N, s=s, init="gaussian")
               for L in ((6,) if quick else (1, 3, 6)) for N in (10, 100) for s in (0, 10, 100)]
    configs += [dict(L=6, N=100, s=s, init="orthogonal") for s in (0, 100)]
    for config in configs:
        config.update(K=8 if quick else 32, T=3000 if quick else 20000,
                      seed=seed, tanh=tanh)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(worker, configs), total=len(configs), desc="drift configs"))
    add_omega(rows)
    OUTPUT.mkdir(exist_ok=True)
    payload = dict(quick=quick, elapsed_seconds=perf_counter() - start, rows=rows)
    (OUTPUT / "drift.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary(rows)
    if quick:
        r = next(r for r in rows if r["N"] == 100 and r["s"] == 100 and r["init"] == "gaussian")
        estimate = r["burn_seconds"] + r["fork_seconds"] * (32 / r["K"]) * (20000 / r["T"])
        print(f"Quick grid: {payload['elapsed_seconds']:.1f}s; estimated full L=6,N=100,s=100 config: {estimate / 60:.1f} min (linear K*T extrapolation).")


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # plotting.py still imports this alias, removed in matplotlib 3.11.
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401 (applies the local figure style)

    data = json.loads((OUTPUT / "drift.json").read_text())
    rows = data["rows"]
    selected = next(r for r in rows if (r["L"], r["N"], r["s"], r["init"]) == (6, 100, 100, "gaussian"))
    neutral = next(r for r in rows if (r["L"], r["N"], r["s"], r["init"]) == (6, 100, 0, "gaussian"))
    text_style = dict(fontsize=6, family="monospace", va="center")

    def end_labels(ax, items, gap=0.065):
        # Place labels in axes coordinates, with short connectors to the data.
        low, high = ax.get_ylim()
        placed = -gap
        ordered = sorted(items, key=lambda item: item[1])
        for i, (x, y, label, color) in enumerate(ordered):
            pos = np.clip((y - low) / (high - low), 0.02, 0.98 - gap * (len(items) - i - 1))
            pos = max(pos, placed + gap)
            placed = pos
            ax.annotate(label, xy=(x, y), xytext=(1.02, pos), textcoords="axes fraction",
                        color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.4),
                        annotation_clip=False, **text_style)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.895, bottom=0.22, top=0.85, wspace=0.95)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        t = np.array([r["t"] for r in selected["records"]])
        divergence_y = np.array([r["state_divergence"] for r in selected["records"]])
        labels = []
        for l, color in enumerate(plt.get_cmap()(np.linspace(0.15, 0.9, 7))):
            ax.plot(t, divergence_y[:, l], color=color)
            labels.append((t[-1], divergence_y[-1, l], f"x{l}", color))
        y = [r["state_divergence"][-1] for r in neutral["records"]]
        ax.plot(t, y, color="0.5", linestyle="--")
        labels.append((t[-1], y[-1], "neutral x6", "0.5"))
        ax.set(xlabel="Generation after fork", ylabel="State divergence", ylim=(0, max(max(y), divergence_y.max()) * 1.08))
        end_labels(ax, labels)

        ax = axes[1]
        labels = []
        missing = 0
        for i, r in enumerate(r for r in rows if r["L"] == 6 and r["s"] > 0):
            if r["omega"] is None:
                missing += 1
                continue
            color = f"C{i}"
            ax.plot(range(7), r["omega"], marker=".", color=color,
                    linestyle="--" if r["init"] == "orthogonal" else "-")
            label = "orthogonal" if r["init"] == "orthogonal" else f"{r['N']},{r['s']}"
            labels.append((6, r["omega"][-1], label, color))
        ax.set(xlabel="Layer (0 = genotype)", ylabel=r"$\omega_l$", xticks=[0, 3, 6], ylim=(0, None))
        ax.set_title("labels: N,s", fontsize=6, family="monospace")
        if missing:
            ax.text(0.02, 0.95, f"{missing} curves: t* not reached", transform=ax.transAxes,
                    fontsize=5.5, family="monospace", va="top")
        end_labels(ax, labels)

        ax = axes[2]
        labels = []
        for r, color, label in ((selected, "C0", "s=100"), (neutral, "C1", "neutral")):
            points = [p for p in r["records"] if p["t"] <= 3000 and p.get("stiff_fraction") is not None]
            ts, ys = [p["t"] for p in points], [p["stiff_fraction"] for p in points]
            ax.plot(ts, ys, color=color)
            labels.append((ts[-1], ys[-1], label, color))
        reference = selected["neutral_expectation"]
        ax.axhline(reference, color="C2", linestyle=":")
        labels.append((3000, reference, "rank/G", "C2"))
        if neutral["neutral_expectation"] != reference:
            ax.axhline(neutral["neutral_expectation"], color="C3", linestyle=":")
            labels.append((3000, neutral["neutral_expectation"], "neutral rank/G", "C3"))
        ax.set(xlabel="Generation after fork", ylabel="Stiff fraction", xlim=(0, 3000), ylim=(0, None))
        end_labels(ax, labels)
        if data["quick"]:
            fig.suptitle("Quick grid: K=8, T=3000", fontsize=7, family="monospace", y=0.99)
        fig.savefig(OUTPUT / "drift.pdf")
        fig.savefig(OUTPUT / "drift.png", dpi=200)
        plt.close(fig)


def check():
    start = perf_counter()
    args = dict(L=3, N=40, K=6, B=300, T=600, every=100, d=6, seed=7)
    selected, neutral = run(s=100, **args), run(s=0, **args)
    for r in (selected, neutral):
        assert np.allclose(r["records"][0]["state_divergence"], 0, atol=1e-12)
        assert np.allclose(r["records"][0]["map_divergence"], 0, atol=1e-12)
        assert r["stiff_dimension"] <= args["d"] - 1, r["singular_values"]
        assert r["records"][0]["stiff_fraction"] is None
    a, b = [r["records"][-1]["state_divergence"][-1] for r in (selected, neutral)]
    assert a < 0.25 * b, (a, b)
    assert selected["records"][0]["mean_fitness"] > 0.9
    rng = np.random.default_rng(2)
    g = normalize(rng.normal(size=(2, 4, 21)), 3)
    untouched = g.copy()
    assert np.array_equal(mutate(g, 3, 0, 0.1, rng), untouched)
    mutate(g, 3, 1, 0.1, rng)
    assert np.allclose(np.linalg.norm(g[..., :3], axis=-1), 1)
    assert np.allclose(np.linalg.norm(g[..., 3:].reshape(2, 4, 2, 9), axis=-1), np.sqrt(3))
    directions = unit(rng.normal(size=(6, 4, 3)))
    pairs = [1 - (directions[i] * directions[j]).sum(-1)
             for i in range(6) for j in range(i)]
    assert np.allclose(divergence(directions), np.mean(pairs, axis=0))
    # Exercise both threshold outcomes without changing the scientific cutoff.
    base = dict(L=1, N=2, init="gaussian", seed=0, tanh=False)
    fixture = [dict(**base, s=s, records=[dict(t=0, state_divergence=[0, 0]),
                                        dict(t=100, state_divergence=v)])
               for s, v in ((0, [0.3, 0.6]), (100, [0.15, 0.06]))]
    add_omega(fixture)
    assert fixture[1]["t_star"] == 100 and np.allclose(fixture[1]["omega"], [0.5, 0.1])
    fixture[0]["records"][1]["state_divergence"][0] = 0.2
    add_omega(fixture)
    assert fixture[1]["t_star"] is None and fixture[1]["omega"] is None
    orth = run(L=2, N=4, K=2, d=3, B=2, T=2, init="orthogonal", tanh=True)
    maps = np.array(orth["ancestor"])[3:].reshape(2, 3, 3)
    assert np.allclose(maps @ maps.swapaxes(-1, -2), np.eye(3))
    assert orth["stiff_dimension"] <= 2
    print(f"check ok ({perf_counter() - start:.1f}s): top divergence {a:.4g} selected / {b:.4g} neutral; "
          f"fork fitness {selected['records'][0]['mean_fitness']:.4f}; rank {selected['stiff_dimension']} <= {args['d'] - 1}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "quick", "figure", "run"), default="run")
    parser.add_argument("--tanh", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "check":
        check()
    elif args.command == "figure":
        figure()
    else:
        main(quick=args.command == "quick", seed=args.seed, tanh=args.tanh, workers=args.workers)
