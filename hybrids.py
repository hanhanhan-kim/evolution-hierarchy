"""Hybrid breakdown after developmental systems drift (experiment B).

Run: python hybrids.py [check | quick | run | summary | figure] [--seed SEED] [--workers N]
Reuses drift's burn-in and fork; assays use an independent random stream.
Each record samples P=16 distinct-lineage pairs (with replacement across pairs),
one individual per parent, and both reciprocal crosses. Within-lineage controls
cross distinct individuals from the first lineage of each pair, reciprocally.
Fitness and top-layer cosine distance are averaged over pairs and directions;
crossovers and controls retain k=0..L-1. F2 uses independent site masks.
The full nine-config grid uses ten workers. Quick keeps B=2000 but uses L=6,
K=8, T=4000. Both write compact output/hybrids.json; figure reads that file.
Snowball slopes fit log(mean hybrid distance - mean parent distance) against
log(t), using positive times with 1e-4 < delta distance < 0.3, without subtracting
the fork value. Fewer than two eligible times gives an NA slope. Bootstrap CIs
resample sampled lineage-pair identities across times, keeping reciprocals
together; these quantify pair sampling uncertainty, not independent runs.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np

import drift

OUTPUT = drift.OUTPUT


def crossover(a, b, k, d):
    """x0 and the first k maps from a; all remaining maps from b."""
    cut = d + k * d * d
    return np.concatenate((a[..., :cut], b[..., cut:]), axis=-1)


def recombine(a, b, d, rng):
    return drift.normalize(np.where(rng.random(a.shape) < 0.5, a, b), d)


def measure(g, optimum, s, d, rng, P=16, tanh=False):
    K, N, G = g.shape
    L = (G - d) // (d * d)
    i = rng.integers(K, size=P)
    j = (i + rng.integers(1, K, size=P)) % K
    ia, ib = rng.integers(N, size=(2, P))
    a, b = g[i, ia], g[j, ib]
    # Both reciprocal directions share the same sampled parental pair.
    left, right = np.concatenate((a, b)), np.concatenate((b, a))
    ic = (ia + rng.integers(1, N, size=P)) % N if N > 1 else ia
    c = g[i, ic]
    within_left, within_right = np.concatenate((a, c)), np.concatenate((c, a))
    crosses = np.stack([crossover(left, right, k, d) for k in range(L)])
    within = np.stack([crossover(within_left, within_right, k, d) for k in range(L)])

    def stats(genomes):
        z = drift.forward(genomes, d, tanh)[-1]
        distance = np.clip(1 - (drift.unit(z) * optimum).sum(-1), 0, 2)
        return dict(fitness=drift.fitness(z, optimum, s).mean(axis=-1).tolist(),
                    distance=distance.mean(axis=-1).tolist(),
                    distance_by_pair=((distance[..., :P] + distance[..., P:]) / 2).tolist())

    return dict(parent=stats(left), crossover=stats(crosses),
                F2=stats(recombine(left, right, d, rng)), within=stats(within),
                within_F2=stats(recombine(within_left, within_right, d, rng)),
                within_parent=stats(within_left),
                pair_ids=(np.minimum(i, j) * K + np.maximum(i, j)).tolist())


def series(row, kind, field="fitness"):
    return np.array([r["hybrids"][kind][field] for r in row["records"]])


def breakdown(row, kind):
    parent = series(row, "parent")
    hybrid = series(row, kind)
    return 1 - hybrid / (parent[:, None] if hybrid.ndim == 2 else parent)


def incompatibility(row, kind, field="distance"):
    """Paired excess distance; within-lineage assays use their own parents."""
    parent = series(row, "within_parent" if kind.startswith("within") else "parent", field)
    hybrid = series(row, kind, field)
    return hybrid - (parent[:, None] if hybrid.ndim > parent.ndim else parent)


def snowball(row, kind="F2", k=None, bootstrap=500):
    t = np.array([r["t"] for r in row["records"]])
    delta = incompatibility(row, kind)
    if k is not None:
        delta = delta[:, k]

    def fit(values):
        keep = (t > 0) & (values > 1e-4) & (values < 0.3)
        slope = float(np.polyfit(np.log(t[keep]), np.log(values[keep]), 1)[0]) if keep.sum() >= 2 else None
        return dict(slope=slope, points=int(keep.sum()), times=t[keep].tolist())

    result = fit(delta)
    result.update(ci95=None, bootstrap_samples=0)
    if result["slope"] is not None and bootstrap and all("pair_ids" in r["hybrids"] for r in row["records"]):
        pair_delta = incompatibility(row, kind, "distance_by_pair")
        if k is not None:
            pair_delta = pair_delta[:, k]
        ids = np.array([r["hybrids"]["pair_ids"] for r in row["records"]])
        pairs, inverse = np.unique(ids, return_inverse=True)
        rng = np.random.default_rng(np.random.SeedSequence(row.get("seed", 0), spawn_key=(4,)))
        counts = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)), size=bootstrap)
        weights = counts[:, inverse.reshape(ids.shape)]
        totals = weights.sum(axis=-1)
        means = np.divide((weights * pair_delta).sum(axis=-1), totals,
                          out=np.full(totals.shape, np.nan), where=totals > 0)
        slopes = [f["slope"] for values in means if (f := fit(values))["slope"] is not None]
        result["bootstrap_samples"] = len(slopes)
        if len(slopes) >= 2:
            result["ci95"] = np.percentile(slopes, [2.5, 97.5]).tolist()
    return result


def worker(params):
    rng = np.random.default_rng(np.random.SeedSequence(params.get("seed", 0), spawn_key=(3,)))

    def on_record(t, g, optimum):
        return measure(g, optimum, params["s"], params.get("d", 10), rng, tanh=params.get("tanh", False))

    result = drift.run(**params, on_record=on_record)
    keys = ("L", "N", "s", "init", "B", "K", "T", "every", "d", "u", "sigma",
            "seed", "tanh", "burn_seconds", "fork_seconds", "seconds")
    row = {k: result[k] for k in keys}
    row.update(P=16, records=[dict(t=r["t"], hybrids=r["hybrids"]) for r in result["records"]])
    row["snowball"] = snowball(row)
    row["crossover_snowball"] = [snowball(row, "crossover", k) for k in range(row["L"])]
    return row


def summary(rows):
    print("Delta d = mean hybrid distance - mean sampled parent distance; NA = not recorded / insufficient points.")
    print("Slopes fit 1e-4 < delta d < 0.3; n = eligible times; CIs = lineage-pair bootstrap 95% intervals.")
    max_layers = max(row["L"] for row in rows)
    print("init        L   N   s  F2@2000  F2@5000 F2@20000   F2 slope (n) [95% CI]" +
          "".join(f"  {'k=' + str(k) + ' slope (n) [95% CI]':>29}" for k in range(max_layers)))

    def fit_cell(fit):
        if fit["slope"] is None:
            return f"NA ({fit['points']})"
        ci = fit["ci95"]
        interval = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else ""
        return f"{fit['slope']:.3f} ({fit['points']}){interval}"

    for row in rows:
        f2 = incompatibility(row, "F2")
        values = []
        for t in (2000, 5000, 20000):
            index = next((i for i, r in enumerate(row["records"]) if r["t"] == t), None)
            values.append(f"{f2[index]:8.4f}" if index is not None else f"{'NA':>8}")
        fits = [snowball(row), *(snowball(row, "crossover", k) for k in range(row["L"]))]
        cells = "  ".join(f"{fit_cell(f):>29}" for f in fits)
        cells += "".join(f"  {'NA':>29}" for _ in range(max_layers - row["L"]))
        print(f"{row['init']:10s} {row['L']:2d} {row['N']:3d} {row['s']:3d} {' '.join(values)}  {cells}")


def main(quick=False, seed=0, workers=10):
    from tqdm import tqdm
    configs = [dict(L=L, N=N, s=s, init="gaussian")
               for L in ((6,) if quick else (3, 6)) for N in (10, 100) for s in (10, 100)]
    configs += [dict(L=6, N=100, s=100, init="orthogonal")]
    for config in configs:
        config.update(K=8 if quick else 32, T=4000 if quick else 20000, every=100, seed=seed)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(worker, configs), total=len(configs), desc="hybrid configs"))
    payload = dict(quick=quick, workers=workers, elapsed_seconds=perf_counter() - start, rows=rows)
    if quick:
        # Extrapolate a full wave; fewer than nine workers require multiple waves.
        estimate = max(r["burn_seconds"] + r["fork_seconds"] * (32 / r["K"]) * (20000 / r["T"])
                       for r in rows)
        payload["estimated_full_seconds"] = estimate
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "hybrids.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary(rows)
    if quick:
        print(f"Quick grid: {payload['elapsed_seconds']:.1f}s; estimated full grid: {estimate / 60:.1f} min "
              "(slowest config, linear K*T extrapolation; worker count, CPU contention and population scaling may increase this).")


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.ticker import FuncFormatter, NullLocator
    # plotting.py still imports this alias, removed in matplotlib 3.11.
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401 (applies the local figure style)

    data = json.loads((OUTPUT / "hybrids.json").read_text())
    rows = [r for r in data["rows"] if r["L"] == 6]
    selected = next(r for r in rows if (r["N"], r["s"], r["init"]) == (100, 100, "gaussian"))
    text_style = dict(fontsize=5.5, family="monospace", va="center")

    def end_labels(ax, items, gap=0.075):
        # Transform through the axes so spacing also works on log scales.
        items = sorted((ax.transAxes.inverted().transform(ax.transData.transform((x, y)))[1],
                        x, y, label, color) for x, y, label, color in items)
        placed = -gap
        for i, (pos, x, y, label, color) in enumerate(items):
            pos = max(placed + gap, np.clip(pos, 0.02, 0.98 - gap * (len(items) - i - 1)))
            placed = pos
            ax.annotate(label, xy=(x, y), xytext=(1.02, pos), textcoords="axes fraction",
                        color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.4),
                        annotation_clip=False, **text_style)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.84, bottom=0.22, top=0.84, wspace=1.05)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        t = np.array([r["t"] for r in selected["records"]])
        curves = [(series(selected, "parent"), "parents", "C0", "-"),
                  (series(selected, "F2"), "F2", "C1", "-"),
                  (series(selected, "within").mean(axis=1), "within k", "C2", "--"),
                  (series(selected, "within_F2"), "within F2", "0.5", "--")]
        curves += [(series(selected, "crossover")[:, k], f"k={k}", color, "-")
                   for k, color in enumerate(plt.get_cmap()(np.linspace(0.15, 0.9, 6)))]
        labels = []
        for y, label, color, style in curves:
            ax.plot(t, y, color=color, linestyle=style)
            labels.append((t[-1], y[-1], label, color))
        ax.set(xlabel="Generations since split", ylabel="Mean fitness", ylim=(0, 1.04))
        ax.set_title("N=100, s=100", fontsize=6, family="monospace")
        end_labels(ax, labels)

        ax = axes[1]
        labels = []
        for i, row in enumerate(rows):
            y, color = breakdown(row, "crossover")[-1], f"C{i}"
            label = "orthogonal" if row["init"] == "orthogonal" else f"{row['N']},{row['s']}"
            ax.plot(range(6), y, marker=".", color=color,
                    linestyle="--" if row["init"] == "orthogonal" else "-")
            labels.append((5, y[-1], label, color))
        ax.set(xlabel="Crossover layer k", ylabel="Breakdown at T", xticks=[0, 2, 5])
        ax.set_title(f"T={selected['T']}; labels: N,s", fontsize=6, family="monospace")
        end_labels(ax, labels)

        ax = axes[2]
        labels = []
        for i, row in enumerate(rows):
            t = np.array([r["t"] for r in row["records"]])
            delta = incompatibility(row, "F2")
            keep = (t > 0) & (delta > 0)
            y, ts, color = delta[keep], t[keep], f"C{i}"
            fit = snowball(row, bootstrap=0)["slope"]
            slope = f"{fit:.2f}" if fit is not None else "NA"
            label = "orth" if row["init"] == "orthogonal" else f"{row['N']},{row['s']}"
            if len(ts):
                ax.loglog(ts, y, color=color)
                labels.append((ts[-1], y[-1], f"{label}: {slope}", color))
        # Show the matched N=100,s=100 Gaussian control. Nonpositive excess
        # distances are undefined on a log axis; preserve gaps instead of clipping.
        t = np.array([r["t"] for r in selected["records"]])
        control = incompatibility(selected, "within_F2")
        positive = (t > 0) & (control > 0)
        ax.loglog(t[t > 0], np.where(control[t > 0] > 0, control[t > 0], np.nan),
                  color="0.5", linestyle="--")
        if positive.any():
            labels.append((t[positive][-1], control[positive][-1], "within F2 (100,100)", "0.5"))
        if labels:
            lo, hi = ax.get_ylim()
            xs = np.array([100, min(400, selected["T"])])
            for power in (1, 2):
                ys = lo * 1.4 * (xs / xs[0])**power
                ax.loglog(xs, ys, color="C0", alpha=0.2, linestyle=":")
                ax.text(xs[-1], ys[-1], f" {power}", color="C0", alpha=0.5, **text_style)
            ax.set_ylim(lo, max(hi, ys[-1] * 1.1))
        ax.set(xlabel="Generations since split", ylabel=r"$\Delta d_{F2}$")
        ax.set_xticks(sorted({100, 1000, selected["T"]}))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:g}k"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title("labels: N,s: slope", fontsize=6, family="monospace")
        end_labels(ax, labels)
        if data["quick"]:
            fig.suptitle("Quick grid: L=6, K=8, T=4000", fontsize=7, family="monospace", y=0.99)
        fig.savefig(OUTPUT / "hybrids.pdf")
        fig.savefig(OUTPUT / "hybrids.png", dpi=200)
        plt.close(fig)


def check():
    start = perf_counter()
    rng = np.random.default_rng(2)
    d, L = 6, 3
    a, b = drift.normalize(rng.normal(size=(2, 16, d + L * d * d)), d)
    for k in range(L):
        assert np.array_equal(crossover(a, a, k, d), a)
        hybrid = crossover(a, b, k, d)
        cut = d + k * d * d
        assert np.array_equal(hybrid[..., :cut], a[..., :cut])
        assert np.array_equal(hybrid[..., cut:], b[..., cut:])
    assert np.allclose(recombine(a, a, d, rng), a)
    hybrid = recombine(a, b, d, rng)
    assert np.allclose(np.linalg.norm(hybrid[..., :d], axis=-1), 1)
    assert np.allclose(np.linalg.norm(hybrid[..., d:].reshape(16, L, d * d), axis=-1), np.sqrt(d))
    args = dict(L=L, N=40, K=6, B=300, T=600, every=100, d=d, seed=7, s=100)
    plain = drift.run(**args)
    sampled = drift.run(**args, on_record=lambda t, g, optimum: measure(g, optimum, 100, d, rng))
    # The assay cannot change any scientific output of experiment A.
    for key in plain:
        if key not in ("records", "seconds", "burn_seconds", "fork_seconds"):
            assert plain[key] == sampled[key], key
    for p, q in zip(plain["records"], sampled["records"]):
        assert p == {k: v for k, v in q.items() if k != "hybrids"}
    fork = sampled["records"][0]["hybrids"]
    assert np.max(np.abs(np.array(fork["within"]["fitness"]) - fork["within_parent"]["fitness"])) < 0.05
    within_f2 = incompatibility(sampled, "within_F2")
    assert np.max(np.abs(within_f2[:2])) < 0.01, within_f2[:2]
    for kind in ("F2", "crossover", "within_F2", "within"):
        assert np.allclose(incompatibility(sampled, kind),
                           incompatibility(sampled, kind, "distance_by_pair").mean(axis=-1))
    f2 = breakdown(sampled, "F2")
    assert f2[-1] > f2[0], (f2[0], f2[-1])
    fixture = dict(records=[dict(t=t, hybrids=dict(parent=dict(distance=0.01),
                                                  F2=dict(distance=0.01 + 0.01 * t**2),
                                                  crossover=dict(distance=[0.01 + 0.01 * t, 0.01 + 0.01 * t**2])))
                           for t in (0, 0.01, 1, 2, 3, 10)])
    assert np.isclose(snowball(fixture)["slope"], 2)
    assert snowball(fixture)["times"] == [1, 2, 3]
    assert np.isclose(snowball(fixture, "crossover", 0)["slope"], 1)
    assert np.isclose(snowball(fixture, "crossover", 1)["slope"], 2)
    assert snowball(dict(records=fixture["records"][:1]))["slope"] is None
    # Identical pair trajectories give an exact CI; reciprocal means are the
    # bootstrap unit and paired parental distances are subtracted before fitting.
    for record in fixture["records"]:
        assay = record["hybrids"]
        assay["pair_ids"] = [1, 2, 3]
        for kind in ("parent", "F2", "crossover"):
            assay[kind]["distance_by_pair"] = np.repeat(np.array(assay[kind]["distance"])[..., None], 3, axis=-1).tolist()
    fit = snowball(fixture, bootstrap=32)
    assert fit["bootstrap_samples"] == 32 and np.allclose(fit["ci95"], [2, 2])
    assert np.allclose(snowball(fixture, "crossover", 0, bootstrap=32)["ci95"], [1, 1])
    print(f"check ok ({perf_counter() - start:.1f}s): F2 breakdown {f2[0]:.4f} at fork / {f2[-1]:.4f} at end; "
          f"within F2 delta d {within_f2[0]:.4g} at fork / {within_f2[1]:.4g} at first record; "
          "crossovers, normalization, within controls, callback invariance and distance snowball fits/bootstrap passed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "quick", "run", "summary", "figure"), default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "check":
        check()
    elif args.command == "figure":
        figure()
    elif args.command == "summary":
        summary(json.loads((OUTPUT / "hybrids.json").read_text())["rows"])
    else:
        main(quick=args.command == "quick", seed=args.seed, workers=args.workers)
