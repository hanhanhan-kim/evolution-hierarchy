"""Experiment H: can neutrally inserted developmental layers become entrenched?

Run: python layers.py [check | quick | figure | run] [--seed SEED] [--workers N]
Full: K=16, T=100000, N={10,100}, s={0,10,100}, plus deletion-only and
10x structural-mutation controls at N=100,s=100. Quick: K=4,T=10000,N=100.
Both write output/layers.json; figure writes layers.pdf and layers.png.

Each replicate starts from the same exactly optimal Gaussian ancestor as A,
without burn-in or a fork. Offspring inherit maps, depths and birth times, then
undergo sparse site mutation and independent insertion/deletion trials. The two
structural events occur in random order, avoiding an ordering bias at bounds.
Insertion is uniform over L+1 gaps (including before/after all active maps).
New identities first receive site mutations in the following generation.

Records retain depth histograms and replicate means, fitness, and age-binned
deletion costs from up to eight uniformly sampled individuals per replicate.
Costs are signed increases in clipped cosine distance, including beneficial
deletions, summarized by pooled medians and quartiles. They remain defined at
s=0. Records also retain the sampled fraction with fitness below 1e-6.
Deleting a sole layer is a diagnostic assay only, not a permitted evolutionary
event. Founders have birth
time zero, inactive slots -1. All-layer and inserted-only age summaries are kept
separately: founding Gaussian maps need not be dispensable at age zero. Panel B
includes founders; inserted-only summaries isolate the neutral additions.
Bins pool layer observations, not independent lineages;
repeated observations of inherited layers are not independent evidence.

Equal insertion/deletion rates imply symmetric interior neutral transitions,
not a constant mean of three: reflecting bounds eventually favor mean 6.5.
Fixation fractions are omitted (birth times are not unique event identifiers).
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from drift import fitness, normalize, unit

OUTPUT = Path(__file__).resolve().parent / "output"


def forward(g, depth, d, skip=None):
    """Apply only active maps; optionally omit one layer for the deletion assay."""
    x = g[..., :d]
    maps = g[..., d:].reshape(*g.shape[:-1], -1, d, d)
    for l in range(int(np.max(depth))):
        if l != skip:
            y = np.einsum("...ij,...j->...i", maps[..., l, :, :], x)
            x = np.where((depth > l)[..., None], y, x)
    return x


def insert(g, depth, birth, rows, positions, t, d):
    """Insert identities in distinct flattened population rows, in place."""
    maps = g[..., d:].reshape(-1, birth.shape[-1], d, d)
    born = birth.reshape(-1, birth.shape[-1])
    slots = np.arange(birth.shape[-1])[None, :]
    source = np.maximum(slots - (slots > positions[:, None]), 0)
    maps[rows] = maps[rows[:, None], source]
    born[rows] = born[rows[:, None], source]
    maps[rows, positions] = np.eye(d)
    born[rows, positions] = t
    depth.reshape(-1)[rows] += 1


def delete(g, depth, birth, rows, positions, d):
    """Delete active maps in distinct flattened rows and reset the last slot."""
    maps = g[..., d:].reshape(-1, birth.shape[-1], d, d)
    born = birth.reshape(-1, birth.shape[-1])
    slots = np.arange(birth.shape[-1])[None, :]
    source = np.minimum(slots + (slots >= positions[:, None]), birth.shape[-1] - 1)
    maps[rows] = maps[rows[:, None], source]
    born[rows] = born[rows[:, None], source]
    maps[rows, -1] = np.eye(d)
    born[rows, -1] = -1
    depth.reshape(-1)[rows] -= 1


def mutate(g, depth, d, u, sigma, rng):
    """Exact Bernoulli sites in the packed active blocks; normalize touched blocks."""
    lengths = d + depth.reshape(-1) * d * d
    ends = np.cumsum(lengths)
    count = rng.binomial(ends[-1], u)
    if not count:
        return g
    sites = rng.choice(ends[-1], count, replace=False)
    rows = np.searchsorted(ends, sites, side="right")
    sites -= ends[rows] - lengths[rows]
    genomes = g.reshape(-1, g.shape[-1])
    genomes[rows, sites] += rng.normal(0, sigma, count)
    xs = np.unique(rows[sites < d])
    genomes[xs, :d] = unit(genomes[xs, :d])
    Lmax = (g.shape[-1] - d) // (d * d)
    is_map = sites >= d
    blocks = np.unique(rows[is_map] * Lmax + (sites[is_map] - d) // (d * d))
    rows, layers = np.divmod(blocks, Lmax)
    maps = genomes[:, d:].reshape(-1, Lmax, d * d)
    maps[rows, layers] = unit(maps[rows, layers]) * np.sqrt(d)
    return g


def reproduce(g, depth, birth, parents):
    rows = np.arange(g.shape[0])[:, None]
    return g[rows, parents], depth[rows, parents], birth[rows, parents]


def step(g, depth, birth, optimum, s, d, u, sigma, mu_ins, mu_del, t, rng):
    K, N = depth.shape
    w = fitness(forward(g, depth, d), optimum, s)
    cdf = np.cumsum(w, axis=1)
    draws = rng.random((K, N, 1)) * cdf[:, -1:, None]
    parents = (draws > cdf[:, None, :]).sum(-1)
    g, depth, birth = reproduce(g, depth, birth, parents)
    mutate(g, depth, d, u, sigma, rng)
    trials = rng.random((2, K * N)) < np.array([mu_ins, mu_del])[:, None]
    # Half of simultaneous events insert first, half delete first.
    first = rng.integers(0, 2, K * N)
    flat = depth.reshape(-1)
    for order in (0, 1):
        for event in (0, 1):
            valid = flat < birth.shape[-1] if event == 0 else flat > 1
            rows = np.flatnonzero(trials[event] & ((first == event) == (order == 0)) & valid)
            if rows.size:
                positions = rng.integers(0, flat[rows] + (event == 0))
                if event == 0:
                    insert(g, depth, birth, rows, positions, t, d)
                else:
                    delete(g, depth, birth, rows, positions, d)
    return g, depth, birth


def distance(z, optimum):
    """Phenotypic distance, using exactly the clipping in drift.fitness."""
    return np.clip(1 - (unit(z) * optimum).sum(-1), 0, 2)


def empty_bins(edges):
    # Final two bins separately pool young/old observations across age bins.
    return {kind: dict(age_sum=np.zeros(len(edges) + 1),
                       costs=[[] for _ in range(len(edges) + 1)])
            for kind in ("all", "inserted", "founders")}


def bin_costs(totals, ages, costs, born, edges):
    bins = np.searchsorted(edges, ages, side="right") - 1
    masks = [bins == i for i in range(len(edges) - 1)] + [ages < 1000, ages > 20000]
    for kind, keep in (("all", np.ones(ages.size, dtype=bool)),
                       ("inserted", born > 0), ("founders", born == 0)):
        data = totals[kind]
        for i, mask in enumerate(masks):
            selected = keep & mask
            data["age_sum"][i] += ages[selected].sum()
            data["costs"][i].extend(costs[selected].tolist())


def pack_bins(totals):
    packed = {}
    for kind, data in totals.items():
        stats = []
        for costs in data["costs"]:
            q25, median, q75 = np.percentile(costs, [25, 50, 75]).tolist() if costs else (None,) * 3
            stats.append(dict(count=len(costs), median_delta_d=median,
                              q25_delta_d=q25, q75_delta_d=q75))
        mean_age = [float(age / stat["count"]) if stat["count"] else None
                    for age, stat in zip(data["age_sum"], stats)]
        packed[kind] = {key: [stat[key] for stat in stats[:-2]] for key in stats[0]}
        packed[kind].update(mean_age=mean_age[:-2], young=stats[-2], old=stats[-1])
    return packed


def measure(g, depth, birth, optimum, s, d, t, sample, edges, rng):
    K, N = depth.shape
    # Separate observation RNG: measurement frequency cannot change evolution.
    people = np.stack([rng.choice(N, min(sample, N), replace=False) for _ in range(K)])
    rows = np.arange(K)[:, None]
    gs, ds, bs = g[rows, people], depth[rows, people], birth[rows, people]
    z = forward(gs, ds, d)
    baseline = distance(z, optimum)
    w = fitness(z, optimum, s)
    totals = empty_bins(edges)
    for l in range(int(ds.max())):
        active = ds > l
        removed = distance(forward(gs, ds, d, skip=l), optimum)
        cost = removed[active] - baseline[active]
        born = bs[..., l][active]
        bin_costs(totals, t - born, cost, born, edges)
    histogram = np.stack([np.bincount(p, minlength=birth.shape[-1] + 1)[1:] for p in depth])
    ws = fitness(forward(g, depth, d), optimum, s).mean(axis=1)
    record = dict(t=t, mean_depth=float(depth.mean()), depth_by_replicate=depth.mean(axis=1).tolist(),
                  depth_counts=histogram.sum(axis=0).tolist(), depth_counts_by_replicate=histogram.tolist(),
                  mean_fitness=float(ws.mean()), fitness_by_replicate=ws.tolist(),
                  sampled_low_fitness_fraction=float(np.mean(w < 1e-6)),
                  entrenchment=pack_bins(totals))
    return record, totals


def run(N=100, s=100, K=16, T=100000, every=1000, d=10, L=3, Lmax=12,
        u=1e-3, sigma=0.1, mu_ins=1e-4, mu_del=1e-4, sample=8, seed=0, control="baseline"):
    if not (1 <= L <= Lmax) or min(N, K, d, every, sample) < 1 or T < 0:
        raise ValueError("Require 1 <= L <= Lmax, N,K,d,every,sample >= 1 and T >= 0")
    if any(not 0 <= p <= 1 for p in (u, mu_ins, mu_del)) or min(sigma, s) < 0:
        raise ValueError("Require mutation probabilities in [0,1] and sigma,s >= 0")
    params = dict(N=N, s=s, K=K, T=T, every=every, d=d, L=L, Lmax=Lmax, u=u, sigma=sigma,
                  mu_ins=mu_ins, mu_del=mu_del, sample=sample, seed=seed, control=control)
    start = perf_counter()
    init_seed, evolve_seed, sample_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(init_seed)
    x = unit(rng.standard_normal(d))
    maps = rng.normal(0, 1 / np.sqrt(d), (L, d, d))
    ancestor = normalize(np.concatenate([x, maps.ravel()]), d)
    ancestor = np.concatenate([ancestor, np.tile(np.eye(d).ravel(), Lmax - L)])
    optimum = unit(forward(ancestor, np.array(L), d))
    g = np.broadcast_to(ancestor, (K, N, ancestor.size)).copy()
    depth = np.full((K, N), L, dtype=np.int64)
    birth = np.full((K, N, Lmax), -1, dtype=np.int64)
    birth[..., :L] = 0
    rng, observation_rng = np.random.default_rng(evolve_seed), np.random.default_rng(sample_seed)
    edges = np.r_[0, np.geomspace(1, max(T + 1, 2), 19)]
    totals = empty_bins(edges)
    records = []
    evolve_seconds = measure_seconds = 0.
    for t in range(T + 1):
        if t % every == 0 or t == T:
            tick = perf_counter()
            record, bins = measure(g, depth, birth, optimum, s, d, t, sample, edges, observation_rng)
            records.append(record)
            for kind in totals:
                totals[kind]["age_sum"] += bins[kind]["age_sum"]
                for pooled, samples in zip(totals[kind]["costs"], bins[kind]["costs"]):
                    pooled.extend(samples)
            measure_seconds += perf_counter() - tick
        if t < T:
            tick = perf_counter()
            g, depth, birth = step(g, depth, birth, optimum, s, d, u, sigma, mu_ins, mu_del, t + 1, rng)
            evolve_seconds += perf_counter() - tick
    late = [r for r in records if r["t"] >= T / 2]
    slope = float(np.polyfit([r["t"] for r in late], [r["mean_depth"] for r in late], 1)[0]) if len(late) > 1 else None
    return dict(**params, optimum=optimum.tolist(), ancestor=ancestor[:d + L * d * d].tolist(),
                age_edges=edges.tolist(), records=records, entrenchment=pack_bins(totals),
                depth_slope=slope, evolve_seconds=evolve_seconds, measure_seconds=measure_seconds,
                seconds=perf_counter() - start)


def summary(rows):
    print("N    s  control        depth    slope/gen      young Δd       old Δd   inserted young/old Δd")
    fmt = lambda x: f"{x:.5g}" if x is not None else "NA"
    for r in rows:
        all_layers, inserted = r["entrenchment"]["all"], r["entrenchment"]["inserted"]
        costs = [fmt(all_layers[k]["median_delta_d"]) for k in ("young", "old")]
        new = "/".join(fmt(inserted[k]["median_delta_d"]) for k in ("young", "old"))
        print(f"{r['N']:3d} {r['s']:3d}  {r['control']:13s} {r['records'][-1]['mean_depth']:6.3f}  "
              f"{fmt(r['depth_slope']):>11s}  {costs[0]:>12s} {costs[1]:>12s}   {new}")
    print("Median Δd pools sampled layer observations: young <1000, old >20000 generations; NA = no observations.")


def worker(params):
    return run(**params)


def main(quick=False, seed=0, workers=32):
    from tqdm import tqdm
    configs = [dict(N=N, s=s) for N in ((100,) if quick else (10, 100)) for s in (0, 10, 100)]
    configs += [dict(N=100, s=100, mu_ins=0, control="deletion-only"),
                dict(N=100, s=100, mu_ins=1e-3, mu_del=1e-3, control="10x rates")]
    for config in configs:
        config.update(K=4 if quick else 16, T=10000 if quick else 100000, seed=seed)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as pool:
        rows = list(tqdm(pool.map(worker, configs), total=len(configs), desc="layers configs"))
    payload = dict(quick=quick, workers=workers, elapsed_seconds=perf_counter() - start, rows=rows)
    if quick:
        # One process per config: the full eight-config grid uses at most eight
        # of 32 workers. Its critical path is an N=100 config, not work / 32.
        estimate = max(r["seconds"] * (16 / r["K"]) * (100000 / r["T"]) for r in rows)
        payload["full_32_workers_estimate_seconds"] = estimate
        payload["runtime_estimate_method"] = "max quick config seconds * 4 K * 10 T; 8 config processes, no 32x speedup; assumes similar depths and hardware"
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "layers.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    summary(rows)
    if quick:
        print(f"Quick grid: {payload['elapsed_seconds']:.1f}s; full grid on 32 workers: ~{estimate / 60:.1f} min "
              "(linear K*T estimate; eight config processes; depth growth/hardware affect timing).")


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

    data = json.loads((OUTPUT / "layers.json").read_text())
    rows = data["rows"]

    def label(r):
        suffix = {"baseline": "", "deletion-only": " del", "10x rates": " 10x"}[r["control"]]
        return f"{r['N']},{r['s']}{suffix}"

    def end_labels(ax, items, gap=0.08):
        ax.get_ylim()  # Resolve autoscaling before transforming label positions.
        placed = -gap
        for i, (x, y, name, color) in enumerate(sorted(items, key=lambda item: item[1])):
            fraction = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))[1]
            pos = np.clip(fraction, 0.03, 0.97 - gap * (len(items) - i - 1))
            pos = max(pos, placed + gap)
            placed = pos
            ax.annotate(name, xy=(x, y), xytext=(1.02, pos), textcoords="axes fraction",
                        color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.4),
                        annotation_clip=False, fontsize=5, family="monospace", va="center")

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.23, top=0.79, wspace=1.15)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        depth_labels, cost_labels = [], []
        positive = [v for r in rows for key in ("median_delta_d", "q25_delta_d", "q75_delta_d")
                    for a, v in zip(r["entrenchment"]["all"]["mean_age"], r["entrenchment"]["all"][key])
                    if a is not None and a > 0 and v is not None and v > 0]
        cost_floor = min(positive) / 2 if positive else 1e-6
        for i, r in enumerate(rows):
            color, style = f"C{i}", "-" if r["control"] == "baseline" else "--"
            ts = [p["t"] for p in r["records"]]
            ys = [p["mean_depth"] for p in r["records"]]
            axes[0].plot(ts, ys, color=color, linestyle=style)
            depth_labels.append((ts[-1], ys[-1], label(r), color))
            bins = r["entrenchment"]["all"]
            points = [(a, c, lo, hi) for a, c, lo, hi in zip(
                bins["mean_age"], bins["median_delta_d"], bins["q25_delta_d"], bins["q75_delta_d"])
                if a is not None and a > 0]
            if points:
                xs, ys, lower, upper = np.array(points).T
                # Keep signed values in the data; nonpositive medians cannot be
                # shown on log axes. IQRs crossing zero extend to the plot floor.
                visible = ys > 0
                axes[1].plot(xs, np.where(visible, ys, np.nan), color=color,
                             linestyle="--" if r["s"] == 0 else "-")
                axes[1].fill_between(xs, np.maximum(lower, cost_floor),
                                     np.where(upper > 0, upper, np.nan), color=color, alpha=0.15)
                if visible.any():
                    cost_labels.append((xs[visible][-1], ys[visible][-1], label(r), color))
        axes[0].set(xlabel="Generation", ylabel="Mean depth")
        axes[0].set_title("labels: N,s; dashed: controls", fontsize=5.5, family="monospace")
        axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[1].set(xlabel="Layer age (generations)", ylabel=r"Median deletion $\Delta d$",
                    xscale="log", yscale="log", ylim=(cost_floor, None))
        axes[1].set_title("All layers; IQR; dashed: s=0\n<=0 omitted; IQR clipped at axis floor",
                          fontsize=5.5, family="monospace")
        end_labels(axes[0], depth_labels)
        end_labels(axes[1], cost_labels)
        for i, r in enumerate(rows):
            if r["N"] == 100 and r["s"] in (0, 100) and r["control"] == "baseline":
                xs = np.arange(1, r["Lmax"] + 1)
                ys = np.array(r["records"][-1]["depth_counts"]) / (r["K"] * r["N"])
                axes[2].stairs(ys, np.arange(0.5, r["Lmax"] + 1.5), color=f"C{i}")
                peak = int(np.argmax(ys))
                axes[2].annotate(f"s={r['s']}", xy=(xs[peak] + 0.5, ys[peak]),
                                 xytext=(3, -2), textcoords="offset points", color=f"C{i}",
                                 fontsize=5, family="monospace", va="top")
        axes[2].set(xlabel="Final depth", ylabel="Frequency", xticks=[1, 3, 6, 9, 12], ylim=(0, None))
        axes[2].set_title("N=100", fontsize=6, family="monospace")
        if data["quick"]:
            fig.suptitle("Quick grid: K=4, T=10000", fontsize=7, family="monospace", y=0.99)
        fig.savefig(OUTPUT / "layers.pdf")
        fig.savefig(OUTPUT / "layers.png", dpi=200)
        plt.close(fig)


def check():
    start = perf_counter()
    rng = np.random.default_rng(7)
    d, Lmax = 3, 5
    g = normalize(rng.normal(size=(2, 3, d + Lmax * d * d)), d)
    depth = np.array([[1, 2, 3], [3, 2, 1]])
    birth = np.full((2, 3, Lmax), -1)
    active = np.arange(Lmax) < depth[..., None]
    maps = g[..., d:].reshape(2, 3, Lmax, d, d)
    maps[~active] = np.eye(d)
    birth[active] = np.arange(active.sum())
    original = (g.copy(), depth.copy(), birth.copy())
    phenotype = forward(g, depth, d)
    optimum = unit(phenotype[0, 0])
    rows = np.arange(depth.size)
    # Test every insertion gap, including both endpoints, and exact restoration.
    for offset in range(4):
        positions = np.minimum(offset, depth.reshape(-1))
        insert(g, depth, birth, rows, positions, 25, d)
        assert np.array_equal(forward(g, depth, d), phenotype)
        for row, pos in zip(rows, positions):
            n = original[1].reshape(-1)[row]
            expected = np.insert(original[2].reshape(-1, Lmax)[row, :n], pos, 25)
            assert np.array_equal(birth.reshape(-1, Lmax)[row, :n + 1], expected)
            genome = g.reshape(-1, g.shape[-1])[row]
            layers = depth.reshape(-1)[row]
            delta = distance(forward(genome, layers, d, skip=pos), optimum) - distance(
                forward(genome, layers, d), optimum)
            assert np.isclose(delta, 0, rtol=0, atol=1e-15)
        record, _ = measure(g, depth, birth, optimum, 100, d, 25, 3, np.array([0, 1, 26]), rng)
        fresh = record["entrenchment"]["inserted"]
        assert fresh["count"][0] == depth.size
        for key in ("median_delta_d", "q25_delta_d", "q75_delta_d"):
            assert np.isclose(fresh[key][0], 0, rtol=0, atol=1e-15)
        delete(g, depth, birth, rows, positions, d)
        assert all(np.array_equal(a, b) for a, b in zip((g, depth, birth), original))
    # Deliberately corrupt inactive slots: forward must explicitly skip them.
    maps[~active] = 123
    assert np.array_equal(forward(g, depth, d), phenotype)
    maps[~active] = np.eye(d)
    parents = np.array([[2, 2, 0], [1, 0, 1]])
    children = reproduce(g, depth, birth, parents)
    for before, after in zip((g, depth, birth), children):
        assert np.array_equal(after, before[np.arange(2)[:, None], parents])
    mutate(g, depth, d, 0, 0.1, rng)
    assert np.array_equal(g, original[0])
    mutate(g, depth, d, 1, 0.1, rng)
    assert np.array_equal(maps[~active], np.broadcast_to(np.eye(d), maps[~active].shape))
    assert np.allclose(np.linalg.norm(g[..., :d], axis=-1), 1)
    assert np.allclose(np.linalg.norm(maps, axis=(-2, -1)), np.sqrt(d))
    for t in range(1, 31):
        g, depth, birth = step(g, depth, birth, optimum, 10, d, 0.1, 0.1, 1, 1, t, rng)
        assert np.all((depth >= 1) & (depth <= Lmax))
        active = np.arange(Lmax) < depth[..., None]
        assert np.all(birth[~active] == -1) and np.all(birth[active] >= 0)
        maps = g[..., d:].reshape(2, 3, Lmax, d, d)
        assert np.array_equal(maps[~active], np.broadcast_to(np.eye(d), maps[~active].shape))
    args = dict(K=2, N=6, d=3, Lmax=5, T=40, every=10, mu_ins=0.2, mu_del=0.2)
    neutral = run(s=0, **args)
    for bins in neutral["entrenchment"].values():
        assert all(c is None or -2 <= c <= 2 for c in bins["median_delta_d"])
    assert any(c is not None and c != 0 for c in neutral["entrenchment"]["all"]["median_delta_d"])
    assert all(r["sampled_low_fitness_fraction"] == 0 for r in neutral["records"])
    # A near-dead phenotype has finite signed Δd, independent of selection.
    dead = np.concatenate(([1., 0., 0.], -np.eye(d).ravel()))[None, None, :]
    assays = [measure(dead, np.ones((1, 1), dtype=int), np.zeros((1, 1, 1), dtype=int),
                      np.array([1., 0., 0.]), s, d, 1, 1, np.array([0, 2]), rng)[0]
              for s in (0, 100)]
    assert assays[0]["entrenchment"] == assays[1]["entrenchment"]
    assert assays[1]["entrenchment"]["all"]["median_delta_d"] == [-2.]
    assert [r["sampled_low_fitness_fraction"] for r in assays] == [0., 1.]
    # Pool samples, not per-record quantiles; test strict age thresholds too.
    edges = np.array([0, 1000, 20001, 30000])
    pooled = empty_bins(edges)
    for ages, costs, born in (([0], [-2.], [0]),
                              ([999, 1000, 20000, 20001], [-1., 0., 1., 2.], [1, 1, 1, 1])):
        bin_costs(pooled, np.array(ages), np.array(costs), np.array(born), edges)
    packed = pack_bins(pooled)
    assert packed["all"]["count"] == [2, 2, 1]
    assert packed["all"]["young"] == dict(count=2, median_delta_d=-1.5, q25_delta_d=-1.75, q75_delta_d=-1.25)
    assert packed["all"]["old"]["count"] == 1 and packed["all"]["old"]["median_delta_d"] == 2
    assert packed["inserted"]["young"]["median_delta_d"] == -1
    assert packed["founders"]["count"] == [1, 0, 0]
    selected = run(s=10, **args)
    assert np.isclose(selected["records"][0]["mean_fitness"], 1)
    # Observation frequency must not perturb the population's random stream.
    other = run(s=10, **{**args, "every": 40})
    for key in ("depth_counts_by_replicate", "fitness_by_replicate"):
        assert selected["records"][-1][key] == other["records"][-1][key]
    for ins, dele, expected in ((1, 0, 5), (0, 1, 1)):
        bounded = run(**{**args, "mu_ins": ins, "mu_del": dele})
        assert bounded["records"][-1]["mean_depth"] == expected
    json.dumps(selected, allow_nan=False)
    print(f"check ok ({perf_counter() - start:.1f}s): exact identity neutrality/restoration, inactive slots, "
          "birth inheritance, sparse mutation, depth bounds, signed distance costs, pooled quantiles, "
          "low-fitness fraction, independent observation RNG")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "quick", "figure", "run"), default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "check":
        check()
    elif args.command == "figure":
        figure()
    else:
        main(quick=args.command == "quick", seed=args.seed, workers=args.workers)
