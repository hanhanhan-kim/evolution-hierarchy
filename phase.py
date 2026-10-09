"""Experiment C: single-site DFEs, Kimura drift predictions and selection visibility.

Run: python phase.py [check | run | figure | summary] [--workers N] [--seed SEED]
Read experiment A's drift.json and, when present, drift_tanh.json. Sample 20000
mutations per block of each selected fork, retaining summaries only in phase.json.
Measured omega uses neutral x0's first divergence >= 0.25, as in drift.add_omega.
Phase grids hold N=100 fixed and vary selection strength, using the same mutants.
At an exact optimum first-order effects vanish: effects are mostly negative and
quadratic. A's post-burn-in consensus forks need not be exactly at that optimum;
keep their stored optima and report wild fitness and beneficial fractions.
Background mutations, linkage and population spread are ignored, so disagreement
with A's divergence ratios is informative rather than a bug.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from drift import add_omega, fitness, forward, mutate, normalize, unit

OUTPUT = Path(__file__).resolve().parent / "output"
QUANTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
KEYS = ("L", "N", "init", "seed", "tanh")
CAVEAT = ("At an exact fitness optimum, first-order effects vanish and s_mut is "
          "mostly negative and quadratic in mutation size. Stored fork genomes are "
          "post-burn-in consensuses, not necessarily exact optima; their stored "
          "optima are preserved. Background mutations, linkage and population "
          "spread are ignored; mismatches with divergence are informative, not bugs.")


def fixation(s, N):
    """Haploid Kimura u(s,N), stable at zero and for strong negative selection."""
    s = np.asarray(s, dtype=float)
    a = 2 * np.abs(s)
    ratio = np.divide(-np.expm1(-a), -np.expm1(-N * a),
                      out=np.full_like(a, 1 / N), where=a != 0)
    return ratio * np.exp(-2 * (N - 1) * np.maximum(-s, 0))


def block_slice(b, d):
    return slice(0, d) if b == 0 else slice(d + (b - 1) * d * d, d + b * d * d)


def sample_mutations(wild, d, b, n, sigma, rng):
    """One uniform site per genome, then normalize only its block, as in mutate."""
    g = np.broadcast_to(wild, (n, wild.size)).copy()
    block = block_slice(b, d)
    sites = rng.integers(block.start, block.stop, n)
    g[np.arange(n), sites] += rng.normal(0, sigma, n)
    g[:, block] = unit(g[:, block]) * (1 if b == 0 else np.sqrt(d))
    return g


def effects(z, wild_z, optimum, s):
    return fitness(z, optimum, s) / fitness(wild_z, optimum, s) - 1


def measured(row, rows):
    """x0 state and map divergence ratios at the same neutral t* as A."""
    neutral = next(r for r in rows if r["s"] == 0 and
                   all(r[k] == row[k] for k in KEYS))
    t = row["t_star"]
    if t is None:
        return [None] * (row["L"] + 1)
    selected = next(p for p in row["records"] if p["t"] == t)
    baseline = next(p for p in neutral["records"] if p["t"] == t)
    a = [selected["state_divergence"][0], *selected["map_divergence"]]
    b = [baseline["state_divergence"][0], *baseline["map_divergence"]]
    return [float(x / y) if y > 0 else None for x, y in zip(a, b)]


def worker(job):
    row, observed, source, seed, n = job
    rng = np.random.default_rng(seed)
    d, N, s = row["d"], row["N"], row["s"]
    wild, optimum = np.array(row["fork_genome"]), np.array(row["optimum"])
    wild_z = forward(wild, d, row["tanh"])[-1]
    wild_distance = float(-np.log(fitness(wild_z, optimum, 1)))
    reference = (row["L"], N, s, row["init"]) == (6, 100, 100, "gaussian")
    ns = np.logspace(-1, 7, 90)
    blocks, neutral_grid, omega_grid = [], [], []
    for b in range(row["L"] + 1):
        samples, distances = [], []
        for start in range(0, n, 2048):
            g = sample_mutations(wild, d, b, min(2048, n - start), row["sigma"], rng)
            z = forward(g, d, row["tanh"])[-1]
            samples.append(effects(z, wild_z, optimum, s))
            if reference:
                distances.append(-np.log(fitness(z, optimum, 1)))
        sm = np.concatenate(samples)
        pred = float(np.mean(N * fixation(sm, N)))
        obs = observed[b]
        blocks.append(dict(block="x0" if b == 0 else f"M{b}", layer=b,
                           s_mut_quantiles=np.quantile(sm, QUANTILES).tolist(),
                           median_abs_s_mut=float(np.median(np.abs(sm))),
                           beneficial_fraction=float(np.mean(sm > 0)),
                           deleterious_fraction=float(np.mean(sm < 0)),
                           neutral_fraction=float(np.mean(np.abs(N * sm) < 1)),
                           omega_pred=pred, omega_measured=obs,
                           ratio_pred_measured=pred / obs if obs is not None and obs > 0 else None))
        if reference:
            # Exact exp(-s*distance) fitness ratio, not a linear rescaling of s_mut.
            delta = np.concatenate(distances) - wild_distance
            grid_sm = np.expm1(-(ns[:, None] / N) * delta[None, :])
            neutral_grid.append(np.mean(np.abs(N * grid_sm) < 1, axis=1))
            omega_grid.append(np.mean(N * fixation(grid_sm, N), axis=1))
    result = {k: row[k] for k in (*KEYS, "s", "d", "sigma", "t_star")}
    result.update(source=source, sample_seed=seed, n=n,
                  wild_fitness=float(fitness(wild_z, optimum, s)),
                  wild_distance=wild_distance, blocks=blocks)
    phase = None
    if reference:
        phase = dict(source=source, tanh=row["tanh"], L=row["L"], N=N,
                     reference_s=s, init=row["init"], seed=row["seed"],
                     blocks=[b["block"] for b in blocks], Ns=ns.tolist(),
                     selection_strength=(ns / N).tolist(),
                     neutral_fraction=np.array(neutral_grid).T.tolist(),
                     omega_pred=np.array(omega_grid).T.tolist(),
                     median_abs_s_mut=[b["median_abs_s_mut"] for b in blocks])
    return result, phase


def run(workers=4, seed=0):
    from tqdm import tqdm

    start = perf_counter()
    jobs = []
    for name in ("drift.json", "drift_tanh.json"):
        path = OUTPUT / name
        if not path.exists():
            if name == "drift.json":
                raise FileNotFoundError(path)
            continue
        rows = json.loads(path.read_text())["rows"]
        add_omega(rows)
        for row in rows:
            if row["s"] > 0:
                jobs.append((row, measured(row, rows), name, None, 20000))
    seeds = np.random.SeedSequence(seed).spawn(len(jobs))
    jobs = [(r, obs, src, int(ss.generate_state(1)[0]), n)
            for (r, obs, src, _, n), ss in zip(jobs, seeds)]
    if workers == 1:
        results = list(tqdm(map(worker, jobs), total=len(jobs), desc="phase configs"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(worker, jobs), total=len(jobs), desc="phase configs"))
    payload = dict(seed=seed, n=20000, quantile_levels=QUANTILES, caveat=CAVEAT,
                   measured_definition="x0: state divergence; M_l: map divergence; selected/neutral at t_star",
                   phase_axes=["Ns", "block"], elapsed_seconds=perf_counter() - start,
                   rows=[r for r, _ in results], phases=[p for _, p in results if p is not None])
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "phase.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary(payload)
    return payload


def summary(data=None):
    if data is None:
        data = json.loads((OUTPUT / "phase.json").read_text())
    print("model  init        L   N   s block    predicted    measured   pred/meas  median|s_mut|")
    for r in data["rows"]:
        for b in r["blocks"]:
            vals = [b[k] for k in ("omega_pred", "omega_measured", "ratio_pred_measured", "median_abs_s_mut")]
            values = " ".join(f"{v:12.5g}" if v is not None else f"{'NA':>12}" for v in vals)
            model = "tanh" if r["tanh"] else "linear"
            print(f"{model:6s} {r['init']:10s} {r['L']:2d} {r['N']:3d} {r['s']:3g} {b['block']:>5s} {values}")
    ws = [r["wild_fitness"] for r in data["rows"]]
    bs = [b["beneficial_fraction"] for r in data["rows"] for b in r["blocks"]]
    print(f"\nWild fitness: {min(ws):.4f}–{max(ws):.4f}; beneficial fractions: {min(bs):.1%}–{max(bs):.1%}.")
    print(data["caveat"])
    print(f"{len(data['rows'])} configs, {data['n']} mutations/block, {data['elapsed_seconds']:.1f}s.")


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

    data = json.loads((OUTPUT / "phase.json").read_text())
    phase = next(p for p in data["phases"] if not p["tanh"])
    text_style = dict(fontsize=5.5, family="monospace")
    colors = plt.get_cmap()(np.linspace(0.1, 0.9, 7))
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.975, bottom=0.23, top=0.84, wspace=0.85)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        ns = np.array(phase["Ns"])
        fraction = np.array(phase["neutral_fraction"])
        edges = np.r_[ns[0], np.sqrt(ns[:-1] * ns[1:]), ns[-1]]
        mesh = ax.pcolormesh(np.arange(8) - 0.5, edges, fraction, vmin=0, vmax=1,
                             shading="flat", rasterized=True)
        ax.set(yscale="log", ylim=(0.1, 1e7), xlabel="Genome block", ylabel=r"$N\,s$",
               xticks=range(7), xticklabels=phase["blocks"])
        levels = [level for level in (0.1, 0.5, 0.9) if fraction.min() < level < fraction.max()]
        if levels:
            contour = ax.contour(np.arange(7), ns, fraction, levels=levels, colors=["C1"], linewidths=0.8)
            ax.clabel(contour, fmt="%.1f", fontsize=5, inline=True)
        ax.plot([0.94, 1], [1e4, 1e4], transform=ax.get_yaxis_transform(), color="white", lw=0.8)
        ax.annotate("A, B runs", (0.94, 1e4), xycoords=ax.get_yaxis_transform(),
                    xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                    color="white", **text_style)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.06, pad=0.04, ticks=[0, 0.5, 1])
        bar.set_label("Fraction neutral", fontsize=6, labelpad=2)
        ax.set_title("linear; N=100, L=6", **text_style)

        ax = axes[1]
        points = []
        for r in data["rows"]:
            for b in r["blocks"]:
                x, y = b["omega_measured"], b["omega_pred"]
                if x is not None and x > 0 and y > 0:
                    ax.scatter(x, y, s=12, marker="^" if r["tanh"] else "o",
                               color=colors[b["layer"]], linewidths=0.25, alpha=0.7)
                    points.append((r, b, x, y))
        low = min(min(x, y) for _, _, x, y in points) / 1.4
        high = max(max(x, y) for _, _, x, y in points) * 1.4
        ax.plot([low, high], [low, high], color="C7", linestyle=":", zorder=0)
        ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
               xlabel=r"Measured $\omega$", ylabel=r"Predicted $\omega$")
        ax.set_title("o linear; ^ tanh; colour: layer", **text_style)
        for r, b, x, y in points:
            if (r["L"], r["N"], r["s"], r["init"]) == (6, 100, 100, "gaussian") and b["layer"] in (0, 6):
                label = ("tanh " if r["tanh"] else "lin ") + b["block"]
                offset = (5, 13 if r["tanh"] else -13)
                ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                            color=colors[b["layer"]], arrowprops=dict(arrowstyle="-", lw=0.4,
                            color=colors[b["layer"]]), **text_style)

        ax = axes[2]
        for r in data["rows"]:
            if r["tanh"] or (r["N"], r["s"], r["init"]) != (100, 100, "gaussian"):
                continue
            y = [b["median_abs_s_mut"] for b in r["blocks"]]
            color = f"C{(1, 3, 6).index(r['L'])}"
            ax.plot(range(r["L"] + 1), y, marker=".", color=color)
            ax.annotate(f"L={r['L']}", (r["L"], y[-1]), xytext=(3, 4),
                        textcoords="offset points", color=color, **text_style)
        ax.set(yscale="log", xlabel="Genome block", ylabel=r"Median $|s_{\rm mut}|$",
               xticks=range(7), xticklabels=phase["blocks"], xlim=(-0.2, 7.4))
        ax.set_title("linear; N=100, s=100", **text_style)
        fig.savefig(OUTPUT / "phase.pdf")
        fig.savefig(OUTPUT / "phase.png", dpi=200)
        plt.close(fig)


def check():
    start = perf_counter()
    for N in (1, 10, 100):
        assert np.allclose(fixation(np.array([-1e-14, 0, 1e-14]), N), 1 / N, rtol=1e-10)
        s = np.array([-0.03, 0.02])
        assert np.allclose(fixation(s, N), (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * N * s)))
    assert np.isfinite(fixation(np.array([-1, 0, 1]), 100000)).all()
    assert fixation(-1, 100000) == 0
    rng = np.random.default_rng(7)
    d, L, n = 4, 3, 256
    wild = normalize(rng.normal(size=d + L * d * d), d)
    for b in range(L + 1):
        block = block_slice(b, d)
        # Replay the same site/delta draws through drift.mutate, forcing one site
        # per genome. Normalization can move all coordinates in the touched block.
        draw = np.random.default_rng(b)
        sites = draw.integers(block.start, block.stop, n)
        delta = draw.normal(0, 0.1, n)

        class FixedDraws:
            def binomial(self, size, u):
                return n

            def choice(self, size, count, replace=False):
                return np.arange(n) * wild.size + sites

            def normal(self, mean, sigma, count):
                return delta

        g = sample_mutations(wild, d, b, n, 0.1, np.random.default_rng(b))
        expected = mutate(np.tile(wild, (n, 1)), d, 0.1, 0.1, FixedDraws())
        assert np.array_equal(g, expected)
        changed = np.stack([np.any(g[:, block_slice(j, d)] != wild[block_slice(j, d)], axis=1)
                            for j in range(L + 1)], axis=1)
        assert np.all(changed.sum(axis=1) == 1) and changed[:, b].all()
        assert np.allclose(np.linalg.norm(g[:, block], axis=1), 1 if b == 0 else np.sqrt(d))
        zero = sample_mutations(wild, d, b, n, 0, rng)
        for tanh in (False, True):
            z = forward(wild, d, tanh)[-1]
            optimum = unit(z)
            assert np.allclose(effects(forward(zero, d, tanh)[-1], z, optimum, 100), 0, atol=1e-12)
            sm = effects(forward(g, d, tanh)[-1], z, optimum, 100)
            assert np.mean(sm <= 1e-12) > 0.95
            distances = -np.log(fitness(forward(g, d, tanh)[-1], optimum, 1))
            delta_distance = distances + np.log(fitness(z, optimum, 1))
            assert np.allclose(sm, np.expm1(-100 * delta_distance), atol=3e-14)
    base = dict(L=1, N=10, init="gaussian", seed=0, tanh=False)
    rows = [dict(**base, s=s, records=[dict(t=0, state_divergence=[0, 0], map_divergence=[0]),
                                      dict(t=100, state_divergence=v, map_divergence=m)])
            for s, v, m in ((0, [0.3, 0.6], [0.4]), (100, [0.15, 0.06], [0.1]))]
    add_omega(rows)
    assert np.allclose(measured(rows[1], rows), [0.5, 0.25])
    rows[0]["records"][1]["state_divergence"][0] = 0.2
    add_omega(rows)
    assert measured(rows[1], rows) == [None, None]
    print(f"check ok ({perf_counter() - start:.2f}s): Kimura, exact mutation convention, "
          "normalization, optimum DFE, zero mutation, phase rescaling and measured omega")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "run", "figure", "summary"), default="run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.command == "run":
        run(workers=args.workers, seed=args.seed)
    elif args.command == "figure":
        figure()
    elif args.command == "summary":
        summary()
    else:
        check()
