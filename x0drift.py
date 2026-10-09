"""Experiment C2: epistasis, linkage, fork effects and substitution vs divergence.

Run: python x0drift.py [check | quick | run | figure | summary] [--workers N]
Full: five configs, four conditions, selected and condition-matched neutral arms.
Quick: linear L=1,6, s=100, K=8, T=4000; both retain B=2000. Conditions apply
throughout burn-in and evolution. DFE: 20000 single-site mutations per genome,
consensus plus 32 uniformly sampled fork individuals (without replacement when
N>=32). Diagnostics use 2000 fixed additive site/value mutations, every 500 gen.
Outputs contain summaries, x0 records and lag curves, never full populations/DFEs.
Near one means within a factor of two; verdicts are descriptive, not causal tests.
A 500-generation diagnostic cannot resolve turnover faster than 2N=200.
Allele IDs label direct site mutations, not coordinate changes from normalization.
Fixations are population-wide ID replacements, counted every post-fork generation.
Reversion-like means repeated fixation within W=1000, not a signed reversal.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

import drift
from drift import consensus, divergence, fitness, forward, mutate, normalize, unit
from phase import block_slice, effects, fixation, sample_mutations

OUTPUT = Path(__file__).resolve().parent / "output"
CONDITIONS = ("baseline", "frozen", "recombining", "both")
REVERSION_WINDOW = 1000


class Alleles:
    """Track x0/M1 ancestry without consuming evolutionary random draws."""

    def __init__(self, shape):
        self.ids = np.zeros(shape, dtype=np.int64)
        self.last_fixed = np.zeros((shape[0], shape[2]), dtype=np.int64)
        self.last_time = np.full_like(self.last_fixed, -10**15)
        self.counts = np.zeros_like(self.last_fixed)
        self.reversions = np.zeros_like(self.last_fixed)
        self.next_id = 1

    def tag(self, sites, width):
        rows, columns = np.divmod(sites, width)
        tracked = columns < self.ids.shape[-1]
        self.ids.reshape(-1, self.ids.shape[-1])[rows[tracked], columns[tracked]] = (
            self.next_id + np.flatnonzero(tracked))
        # Reserve IDs even for untracked map mutations; every mutation is unique.
        self.next_id += len(sites)

    def detect(self, t, count=True):
        first = self.ids[:, 0, :]
        events = np.all(self.ids == first[:, None, :], axis=1) & (first != self.last_fixed)
        repeated = events & (t - self.last_time <= REVERSION_WINDOW)
        if count:
            self.counts += events
            self.reversions += repeated
        self.last_fixed[events] = first[events]
        self.last_time[events] = t
        return events, repeated

    def fork(self, K):
        for name in ("ids", "last_fixed", "last_time", "counts", "reversions"):
            setattr(self, name, np.repeat(getattr(self, name), K, axis=0))

    def report(self, d, T):
        result = dict(window=REVERSION_WINDOW, previous_fixation_includes_burn_in=True)
        for name, block in (("x0", slice(0, d)), ("M1", slice(d, None))):
            counts, repeated = self.counts[:, block], self.reversions[:, block]
            total, rev = int(counts.sum()), int(repeated.sum())
            result[name] = dict(count=total, reversion_like_count=rev,
                                rate_per_site=total / (counts.size * T) if T else None,
                                reversion_like_fraction=rev / total if total else None,
                                counts_by_lineage_site=counts.tolist(),
                                reversion_like_by_lineage_site=repeated.tolist())
        return result


class TaggedMutationDraws:
    """Intercept the existing mutation kernel's site draw, leaving it unchanged."""

    def __init__(self, rng, alleles, width):
        self.rng, self.alleles, self.width = rng, alleles, width

    def binomial(self, *args):
        return self.rng.binomial(*args)

    def choice(self, *args, **kwargs):
        sites = self.rng.choice(*args, **kwargs)
        self.alleles.tag(sites, self.width)
        return sites

    def normal(self, *args):
        return self.rng.normal(*args)


def recombine(g, parents, d, rng, alleles=None):
    """Choose independently between the two sampled parents for every block."""
    K, N, G = g.shape
    L = (G - d) // (d * d)
    choices = rng.integers(0, 2, (K, N, L + 1))
    child = np.empty_like(g)
    child_ids = np.empty_like(alleles.ids) if alleles is not None else None
    for b in range(L + 1):
        parent = np.take_along_axis(parents, choices[..., b, None], axis=-1)[..., 0]
        block = block_slice(b, d)
        child[..., block] = g[np.arange(K)[:, None], parent, block]
        if alleles is not None and b <= 1:
            child_ids[..., block] = alleles.ids[np.arange(K)[:, None], parent, block]
    if alleles is not None:
        alleles.ids = child_ids
    return child


def mutate_x0(g, d, u, sigma, rng):
    """A's Bernoulli-site scheme restricted to x0; map blocks remain untouched."""
    x = g[..., :d].copy().reshape(-1, d)
    count = rng.binomial(x.size, u)
    sites = rng.choice(x.size, count, replace=False)
    x.reshape(-1)[sites] += rng.normal(0, sigma, count)
    rows = np.unique(sites // d)
    x[rows] = unit(x[rows])
    g[..., :d] = x.reshape(*g.shape[:-1], d)
    return g


def step(g, optimum, s, d, u, sigma, tanh, rng, condition, alleles=None):
    K, N = g.shape[:2]
    w = fitness(forward(g, d, tanh)[-1], optimum, s)
    cdf = np.cumsum(w, axis=1)
    if condition in ("recombining", "both"):
        draws = rng.random((K, N, 2, 1)) * cdf[:, -1:, None, None]
        parents = (draws > cdf[:, None, None, :]).sum(-1)
        g = recombine(g, parents, d, rng, alleles)
    else:
        draws = rng.random((K, N, 1)) * cdf[:, -1:, None]
        parents = (draws > cdf[:, None, :]).sum(-1)
        g = g[np.arange(K)[:, None], parents]
        if alleles is not None:
            alleles.ids = alleles.ids[np.arange(K)[:, None], parents]
    mutation = mutate_x0 if condition in ("frozen", "both") else mutate
    if alleles is not None:
        rng = TaggedMutationDraws(rng, alleles, d if mutation is mutate_x0 else g.shape[-1])
    return mutation(g, d, u, sigma, rng)


def prediction(wild, optimum, d, N, s, sigma, tanh, n, rng, block=0):
    """C's DFE sampling and mean N*u_fix, batched to bound memory."""
    wild_z = forward(wild, d, tanh)[-1]
    total = 0.0
    for start in range(0, n, 2048):
        mutants = sample_mutations(wild, d, block, min(2048, n - start), sigma, rng)
        sm = effects(forward(mutants, d, tanh)[-1], wild_z, optimum, s)
        total += (N * fixation(sm, N)).sum()
    return float(total / n)


def fork_predictions(g, optimum, d, N, s, sigma, tanh, n, samples, rng):
    wild = consensus(g, d)[0]
    indices = rng.choice(N, samples, replace=N < samples)
    genomes = g[0, indices]
    a = prediction(wild, optimum, d, N, s, sigma, tanh, n, rng)
    b = [prediction(x, optimum, d, N, s, sigma, tanh, n, rng) for x in genomes]
    return dict(consensus=a, population=float(np.mean(b)),
                consensus_M1=prediction(wild, optimum, d, N, s, sigma, tanh, n, rng, block=1),
                individual_predictions=b, sampled_indices=indices.tolist(),
                consensus_fitness=float(fitness(forward(wild, d, tanh)[-1], optimum, s)),
                population_fitness=float(fitness(forward(g, d, tanh)[-1], optimum, s).mean()),
                sampled_fitness=fitness(forward(genomes, d, tanh)[-1], optimum, s).tolist())


def neutral_manifold(wild, d, tanh, condition, eps=1e-5):
    """Jacobian on normalized genome blocks, with only evolving maps included."""
    delta = eps * np.eye(wild.size)
    plus = unit(forward(normalize(wild + delta, d), d, tanh)[-1])
    minus = unit(forward(normalize(wild - delta, d), d, tanh)[-1])
    z = unit(forward(wild, d, tanh)[-1])
    J = (np.eye(d) - np.outer(z, z)) @ ((plus - minus) / (2 * eps)).T
    J[:, :d] = J[:, :d] @ (np.eye(d) - np.outer(wild[:d], wild[:d]))
    maps = J[:, d:] if condition not in ("frozen", "both") else np.zeros((d, 0))
    full = np.concatenate([J[:, :d], maps], axis=1)
    tolerance = max(float(np.linalg.norm(full, ord=2)) * 1e-6, 1e-12)
    rank = lambda a: int(np.count_nonzero(np.linalg.svd(a, compute_uv=False) > tolerance))
    r_full, r_maps, r_x0 = rank(full), rank(maps), rank(J[:, :d])
    return dict(rank_full=r_full, rank_maps=r_maps, rank_x0=r_x0,
                x0_directions_need_compensation=r_full - r_maps,
                x0_tangent_dimension=d - 1,
                x0_compensable_or_intrinsically_neutral=d - 1 - (r_full - r_maps),
                rank_tolerance=tolerance)


def fixed_effects(wild, sites, values, optimum, d, s, tanh):
    """Replay fixed additive mutations, with the same block normalization as C."""
    mutants = np.broadcast_to(wild, (len(sites), wild.size)).copy()
    mutants[np.arange(len(sites)), sites] += values
    mutants[:, :d] = unit(mutants[:, :d])
    return effects(forward(mutants, d, tanh)[-1], forward(wild, d, tanh)[-1], optimum, s)


def lag_curves(times, sm, N):
    """Average within-pair Pearson correlations/Jaccards over all time origins."""
    sm = np.asarray(sm)
    neutral = np.abs(N * sm) < 1
    centered = sm - sm.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    curves = []
    for lag in range(len(times)):
        a, b = centered[:len(times) - lag], centered[lag:]
        denom = norms[:len(times) - lag] * norms[lag:]
        corr = np.divide((a * b).sum(-1), denom,
                         out=np.full(len(a), np.nan), where=denom > 0)
        x, y = neutral[:len(times) - lag], neutral[lag:]
        union = (x | y).sum(-1)
        jaccard = np.divide((x & y).sum(-1), union,
                            out=np.ones(len(x)), where=union > 0)
        valid = corr[np.isfinite(corr)]
        curves.append(dict(delta=int(times[lag] - times[0]), pairs=len(a),
                           correlation=float(valid.mean()) if len(valid) else None,
                           correlation_pairs=len(valid), jaccard=float(jaccard.mean())))
    # First crossing only; no exponential-decay assumption or extrapolation.
    crossings = {}
    for key in ("correlation", "jaccard"):
        hit = next((i for i, p in enumerate(curves) if p[key] is not None and p[key] <= 0.5), None)
        crossings[key] = ([curves[max(0, hit - 1)]["delta"], curves[hit]["delta"]]
                          if hit is not None else None)
    return dict(times=times, curves=curves, neutral_fraction=neutral.mean(axis=1).tolist(),
                half_crossing_intervals=crossings, neutral_fixation_generations=2 * N,
                empty_union_jaccard=1, lineage=0)


def run(L=6, N=100, s=100, condition="baseline", B=2000, K=32, T=20000,
        every=100, d=10, u=1e-3, sigma=0.1, seed=0, tanh=False,
        n_dfe=20000, samples=32, diagnostic_every=500, n_fixed=2000):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if min(L, N, d, every, n_dfe, samples, diagnostic_every, n_fixed) < 1 or K < 2 or min(B, T, seed) < 0:
        raise ValueError("Invalid sizes, times or seed")
    if not 0 <= u <= 1 or min(s, sigma) < 0:
        raise ValueError("Require 0 <= u <= 1 and s,sigma >= 0")
    params = dict(L=L, N=N, s=s, condition=condition, B=B, K=K, T=T, every=every,
                  d=d, u=u, sigma=sigma, seed=seed, tanh=tanh, init="gaussian",
                  n_dfe=n_dfe, samples=samples, diagnostic_every=diagnostic_every, n_fixed=n_fixed)
    start = perf_counter()
    # Exactly A's three evolutionary streams; measurements never consume them.
    init_seed, burn_seed, fork_seed = np.random.SeedSequence(seed).spawn(3)
    dfe_seed, diagnostic_seed = np.random.SeedSequence([seed, 2]).spawn(2)
    rng = np.random.default_rng(init_seed)
    x = unit(rng.standard_normal(d))
    maps = rng.normal(0, 1 / np.sqrt(d), (L, d, d))
    ancestor = normalize(np.concatenate([x, maps.ravel()]), d)
    optimum = unit(forward(ancestor, d, tanh)[-1])
    g = np.broadcast_to(ancestor, (1, N, ancestor.size)).copy()
    alleles = Alleles((1, N, d + d*d))
    rng = np.random.default_rng(burn_seed)
    for t in range(1 - B, 1):
        g = step(g, optimum, s, d, u, sigma, tanh, rng, condition, alleles)
        alleles.detect(t, count=False)
    burn_seconds = perf_counter() - start
    prediction_start = perf_counter()
    pred = (fork_predictions(g, optimum, d, N, s, sigma, tanh, n_dfe, samples,
                             np.random.default_rng(dfe_seed)) if s > 0 else None)
    manifold = neutral_manifold(consensus(g, d)[0], d, tanh, condition)
    prediction_seconds = perf_counter() - prediction_start
    diagnostic = L == 6 and s == 100 and not tanh and condition in ("baseline", "recombining")
    rng_diag = np.random.default_rng(diagnostic_seed)
    sites = rng_diag.integers(0, d, n_fixed)
    values = rng_diag.normal(0, sigma, n_fixed)
    g = np.repeat(g, K, axis=0)
    alleles.fork(K)
    rng = np.random.default_rng(fork_seed)
    records, times, sm = [], [], []
    fixation_generations = []
    diagnostic_seconds = 0.0
    evolve_start = perf_counter()
    for t in range(T + 1):
        if t % every == 0 or t == T:
            # Same layer-0 expression as drift.measure, including tanh scaling.
            directions = unit(g[..., :d] * (np.sqrt(d) if tanh else 1))
            maps = g[..., d:d+d*d]
            records.append(dict(t=t, x0_divergence=float(divergence(unit(directions.mean(axis=1)))),
                                M1_divergence=float(divergence(unit(maps.mean(axis=1))))))
        if diagnostic and t % diagnostic_every == 0:
            tick = perf_counter()
            times.append(t)
            sm.append(fixed_effects(consensus(g[0], d), sites, values, optimum, d, s, tanh))
            diagnostic_seconds += perf_counter() - tick
        if t < T:
            g = step(g, optimum, s, d, u, sigma, tanh, rng, condition, alleles)
            events, repeated = alleles.detect(t + 1)
            # Site-resolved totals plus per-generation block totals; no ID histories.
            fixation_generations.append([int(a[:, b].sum()) for a in (events, repeated)
                                         for b in (slice(0, d), slice(d, None))])
    fork_seconds = perf_counter() - evolve_start - diagnostic_seconds
    diag = lag_curves(times, sm, N) if diagnostic else None
    return dict(**params, records=records, predictions=pred, diagnostic=diag, manifold=manifold,
                substitutions=alleles.report(d, T), fixation_generations=fixation_generations,
                fixation_generation_columns=["x0", "M1", "x0_reversion_like", "M1_reversion_like"],
                burn_seconds=burn_seconds, fork_seconds=fork_seconds,
                prediction_seconds=prediction_seconds, diagnostic_seconds=diagnostic_seconds,
                seconds=perf_counter() - start)


def add_omega(rows):
    """A's neutral threshold with an explicit last-record fallback, per condition."""
    for row in rows:
        neutral = next(r for r in rows if r["s"] == 0 and all(
            r[k] == row[k] for k in ("L", "N", "seed", "tanh", "condition")))
        hit = next((p for p in neutral["records"] if p["x0_divergence"] >= 0.25), None)
        point = hit if hit is not None else neutral["records"][-1]
        selected = next(p for p in row["records"] if p["t"] == point["t"])
        a, b = selected["x0_divergence"], point["x0_divergence"]
        omega = a / b if b > 0 else None
        row.update(t_star=point["t"], threshold_fallback=hit is None,
                   selected_divergence=a, neutral_divergence=b, omega_x0=omega)
        ma, mb = selected["M1_divergence"], point["M1_divergence"]
        # Frozen maps have no mutational exposure: their ratios are undefined.
        evolving_maps = row["condition"] not in ("frozen", "both")
        row["omega_M1"] = ma / mb if evolving_maps and mb > 0 else None
        for block in ("x0", "M1"):
            a_rate = row["substitutions"][block]["rate_per_site"]
            b_rate = neutral["substitutions"][block]["rate_per_site"]
            row[f"substitution_ratio_{block}"] = (a_rate / b_rate if a_rate is not None
                                                  and b_rate is not None and b_rate > 0 else None)
        row["ratios"] = ({k: row["predictions"][k] / omega if omega is not None and omega > 0 else None
                          for k in ("consensus", "population")} if row["predictions"] else None)


def config_label(r):
    return f"{'tanh' if r['tanh'] else 'linear'} L={r['L']} s={r['s']:g}"


def verdict(rows):
    """Describe closures and partial reductions of absolute log prediction error."""
    by_condition = {r["condition"]: r for r in rows}
    ratios = {c: r["ratios"]["consensus"] for c, r in by_condition.items()}
    base = ratios["baseline"]
    pop = by_condition["baseline"]["ratios"]["population"]
    near = lambda x: x is not None and 0.5 <= x <= 2
    hypotheses = []
    if not near(base):
        if near(ratios["frozen"]) and not near(ratios["recombining"]):
            hypotheses.append("H1 (freezing closes the gap; recombination does not)")
        if near(ratios["recombining"]):
            hypotheses.append("H2 (recombination closes the gap)")
        if near(pop):
            hypotheses.append("H3 (population DFE closes the baseline gap)")
    result = "; ".join(hypotheses) if hypotheses else (
        "baseline already near 1; no resolved deficit" if near(base) else "H1-H3 do not close the gap")
    partial = []
    if base is not None and base > 0 and not near(base):
        for label, value in [("frozen", ratios["frozen"]), ("recombining", ratios["recombining"]),
                             ("both", ratios["both"]), ("population DFE", pop)]:
            if value is not None and value > 0:
                reduction = 1 - abs(np.log(value)) / abs(np.log(base))
                partial.append(f"{label} {reduction:+.0%}")
    h4 = []
    for c, r in by_condition.items():
        sub, div, pred = r["substitution_ratio_x0"], r["omega_x0"], r["predictions"]["consensus"]
        matches = sub is not None and sub > 0 and near(pred / sub)
        suppressed = sub is not None and div is not None and sub > 2 * div
        need = r["manifold"]["x0_directions_need_compensation"]
        status = ("substitution/divergence distinction supported" if matches and suppressed else
                  "Kimura matches substitutions, no divergence suppression" if matches else
                  "unresolved (no substitutions)" if sub is None or sub == 0 else
                  "Kimura/substitution mismatch")
        msub, mdiv = r["substitution_ratio_M1"], r["omega_M1"]
        map_match = (msub is not None and msub > 0 and mdiv is not None
                     and near(r["predictions"]["consensus_M1"] / msub) and near(mdiv / msub))
        map_status = "NA (frozen)" if c in ("frozen", "both") else "similar" if map_match else "mismatch"
        h4.append(f"{c}: {status}, uncompensated rank {need}/{r['d']-1}, M1 {map_status}")
    return (result + ("; log-gap reductions: " + ", ".join(partial) if partial else "")
            + "; H4 [" + "; ".join(h4) + "]")


def summary(data=None):
    if data is None:
        data = json.loads((OUTPUT / "x0drift.json").read_text())
    print("config                 condition       t*    div_x0     a_x0     b_x0    a/div    b/div   sub_x0     a_M1   sub_M1   div_M1   rev_x0   rev_M1   need/d-1 full/maps")
    selected = [r for r in data["rows"] if r["s"] > 0]
    for r in selected:
        values = [r["omega_x0"], r["predictions"]["consensus"], r["predictions"]["population"],
                  r["ratios"]["consensus"], r["ratios"]["population"],
                  r["substitution_ratio_x0"], r["predictions"]["consensus_M1"],
                  r["substitution_ratio_M1"], r["omega_M1"],
                  r["substitutions"]["x0"]["reversion_like_fraction"],
                  r["substitutions"]["M1"]["reversion_like_fraction"]]
        numbers = " ".join(f"{v:8.4f}" if v is not None else f"{'NA':>8}" for v in values)
        m = r["manifold"]
        print(f"{config_label(r):22s} {r['condition']:11s} {r['t_star']:5d}{'*' if r['threshold_fallback'] else ' '} {numbers}"
              f"     {m['x0_directions_need_compensation']}/{r['d']-1}       {m['rank_full']}/{m['rank_maps']}")
    print("* = neutral x0 never reached 0.25; used last record. Near 1 = [0.5, 2].")
    print("sub = selected/neutral fixation rate over all T post-fork generations; div = divergence ratio at t*.")
    print("rev = fraction of selected fixations within W=1000 of a prior fixation (including burn-in); not proof of reversal.")
    print("need = 'x0 directions that need compensation' = rank(J_full)-rank(J_maps), out of d-1:")
    print("  this is the residual rank that maps cannot compensate; zero means all x0 effects are compensable or neutral.")
    print("M1 ratios are NA for frozen maps (0/0); a_M1 remains the hypothetical single-mutation prediction.")
    for label in dict.fromkeys(config_label(r) for r in selected):
        group = [r for r in selected if config_label(r) == label]
        print(f"{label}: {verdict(group)}")
    for r in selected:
        diag = r["diagnostic"]
        if diag is None:
            continue
        print(f"Diagnostic {r['condition']}: 2N={2*r['N']}; first half-crossing intervals "
              f"{diag['half_crossing_intervals']} (None = not reached by {diag['times'][-1]}).")
        interval = diag["half_crossing_intervals"]["jaccard"]
        if interval is None or interval[0] > 2 * r["N"]:
            print("  Neutral-set half-turnover is slower than 2N at sampled lags; no support for rapid-turnover H1.")
        else:
            print("  Turnover relative to 2N is unresolved: first lag is 500 > 200; cannot establish fast-turnover H1.")
    print("Diagnostics average all time-origin pairs; they also reflect x0 changes, not just map epistasis.")
    print(f"{len(data['rows'])} runs; elapsed {data['elapsed_seconds']:.1f}s.")
    if data.get("full_runtime_estimate"):
        est = data["full_runtime_estimate"]
        print(f"Estimated full grid on 15 workers: {est['seconds']/60:.1f} min. {est['method']}")


def full_configs():
    return [dict(L=L, s=s, tanh=False) for L in (1, 6) for s in (10, 100)] + [dict(L=1, s=100, tanh=True)]


def estimate_runtime(rows, workers=15):
    durations = []
    for config in full_configs():
        for condition in CONDITIONS:
            for s in (config["s"], 0):
                r = next(r for r in rows if r["L"] == config["L"] and r["condition"] == condition
                         and (r["s"] > 0) == (s > 0))
                diagnostic = r["diagnostic_seconds"] * 5 if s == 100 and config["L"] == 6 else 0
                durations.append(r["burn_seconds"] + r["prediction_seconds"] + diagnostic +
                                 r["fork_seconds"] * (32 / r["K"]) * (20000 / r["T"]))
    lanes = np.zeros(workers)
    for duration in durations:
        lanes[np.argmin(lanes)] += duration
    return dict(workers=workers, seconds=float(lanes.max()), worker_seconds=float(sum(durations)),
                method="K*T extrapolation of quick timings with 40 jobs scheduled in grid order; "
                       "s=10 and tanh use linear s=100 proxies; contention/scaling may differ.")


def worker(params):
    return run(**params)


def main(quick=False, seed=0, workers=15):
    from tqdm import tqdm
    configs = [dict(L=L, s=100, tanh=False) for L in (1, 6)] if quick else full_configs()
    jobs = [dict(config, s=s, condition=condition, seed=seed, K=8 if quick else 32,
                 T=4000 if quick else 20000)
            for config in configs for condition in CONDITIONS for s in (config["s"], 0)]
    start = perf_counter()
    if workers == 1:
        rows = list(tqdm(map(worker, jobs), total=len(jobs), desc="C2 runs"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(tqdm(pool.map(worker, jobs), total=len(jobs), desc="C2 runs"))
    add_omega(rows)
    data = dict(quick=quick, seed=seed, workers=workers, elapsed_seconds=perf_counter() - start,
                near_one_interval=[0.5, 2], dfe_definition="mean N*Kimura(s_mut,N); 20000 mutations/genome",
                diagnostic_definition="fixed additive mutations; mean pairwise Pearson/Jaccard at each lag",
                substitution_definition="population-unanimous mutation IDs replacing the last fixed ID; "
                                        "counts/(K * sites * T); per-generation rows start at t=1",
                full_runtime_estimate=estimate_runtime(rows) if quick else None, rows=rows)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "x0drift.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    summary(data)
    return data


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

    data = json.loads((OUTPUT / "x0drift.json").read_text())
    selected = [r for r in data["rows"] if r["s"] > 0]
    labels = list(dict.fromkeys(config_label(r) for r in selected))
    colors = plt.get_cmap()(np.linspace(0.15, 0.85, len(labels)))
    style = dict(fontsize=5, family="monospace", va="center")

    def end_labels(ax, items):
        low, high = ax.get_ylim()
        scale = np.log if ax.get_yscale() == "log" else lambda x: x
        positions = [(scale(y) - scale(low)) / (scale(high) - scale(low)) for _, y, _, _ in items]
        placed = -0.13
        for j, i in enumerate(np.argsort(positions)):
            x, y, label, color = items[i]
            pos = max(placed + 0.13, min(positions[i], 0.95 - 0.13 * (len(items) - j - 1)))
            placed = pos
            ax.annotate(label, (x, y), xytext=(1.02, pos), textcoords="axes fraction",
                        color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.4),
                        annotation_clip=False, **style)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 5,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.88, bottom=0.30, top=0.82, wspace=1.3)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        for ax, method in zip(axes[:1], ("consensus",)):
            ax.set_yscale("log")
            ax.axhline(1, color="C7", ls=":", lw=0.7)
            items = []
            for label, color in zip(labels, colors):
                group = {r["condition"]: r for r in selected if config_label(r) == label}
                y = [group[c]["ratios"][method] for c in CONDITIONS]
                y = [v if v is not None and v > 0 else np.nan for v in y]
                ax.plot(range(4), y, color=color, marker=".")
                if np.isfinite(y[-1]):
                    r = group["both"]
                    short = f"{'T' if r['tanh'] else 'L'}{r['L']},{r['s']:g}"
                    items.append((3, y[-1], short, color))
            ax.set(xticks=range(4), xticklabels=CONDITIONS, ylabel="Predicted / measured", xlim=(-0.15, 3.15))
            ax.tick_params(axis="x", rotation=40)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
            ax.set_title("Fork consensus" if method == "consensus" else "32 fork individuals", fontsize=7)
            end_labels(ax, items)
        ax = axes[1]
        ax.set_yscale("log")
        ax.axhline(1, color="C7", ls=":", lw=0.7)
        quantities = (("Kimura", "o"), ("Substitution", "s"), ("Divergence", "^"))
        offsets = np.linspace(-0.22, 0.22, len(labels)) if len(labels) > 1 else [0]
        omitted = 0
        for label, color, offset in zip(labels, colors, offsets):
            group = {r["condition"]: r for r in selected if config_label(r) == label}
            for j, (name, marker) in enumerate(quantities):
                y = [[group[c]["predictions"]["consensus"], group[c]["substitution_ratio_x0"],
                      group[c]["omega_x0"]][j] for c in CONDITIONS]
                omitted += sum(v is None or v <= 0 for v in y)
                y = [v if v is not None and v > 0 else np.nan for v in y]
                ax.plot(np.arange(4) + offset + (j - 1) * 0.055, y, color=color,
                        marker=marker, markersize=3, linestyle="none",
                        markerfacecolor="none" if j == 0 else color)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="0.3", marker=marker, linestyle="none",
                                  markersize=3, label=name,
                                  markerfacecolor="none" if j == 0 else "0.3")
                           for j, (name, marker) in enumerate(quantities)],
                  loc="lower left", bbox_to_anchor=(0, 1.04), fontsize=4.5, ncol=3,
                  frameon=False, handletextpad=0.3, columnspacing=0.6, borderaxespad=0)
        ax.set(xticks=range(4), xticklabels=CONDITIONS, ylabel=r"$x_0$ ratio", xlim=(-0.4, 3.4))
        ax.tick_params(axis="x", rotation=40)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        if omitted:
            ax.text(0.02, 0.02, f"{omitted} zero/NA omitted", transform=ax.transAxes, fontsize=5)
        ax = axes[2]
        items = []
        for i, r in enumerate(r for r in selected if r["diagnostic"] is not None):
            curves = r["diagnostic"]["curves"]
            x, y = [p["delta"] for p in curves], [p["jaccard"] for p in curves]
            color = f"C{i}"
            ax.plot(x, y, color=color)
            items.append((x[-1], y[-1], r["condition"], color))
        N = next((r["N"] for r in selected if r["diagnostic"] is not None), 100)
        ax.axvline(2 * N, color="C7", ls=":", lw=0.7)
        ax.text(2 * N, 1.04, "2N", ha="center", **style)
        ax.set(xlabel=r"Lag $\Delta$ (generations)", ylabel="Neutral-set Jaccard", ylim=(0, 1.08))
        ax.set_title("L=6, s=100", fontsize=7)
        end_labels(ax, items)
        fig.suptitle(("Quick: K=8, T=4000; " if data["quick"] else "") + "labels: L=linear / T=tanh; depth,s",
                     fontsize=6, family="monospace", y=0.99)
        fig.savefig(OUTPUT / "x0drift.pdf")
        fig.savefig(OUTPUT / "x0drift.png", dpi=200)
        plt.close(fig)


def check():
    start = perf_counter()
    rng = np.random.default_rng(4)
    d, L = 3, 2
    g = normalize(rng.normal(size=(2, 8, d + L*d*d)), d)
    original = g.copy()
    mutate_x0(g, d, 1, 0.1, rng)
    assert np.array_equal(g[..., d:], original[..., d:])
    assert not np.array_equal(g[..., :d], original[..., :d])
    parents = rng.integers(0, 8, (2, 8, 2))
    children = recombine(g, parents, d, rng)
    for b in range(L + 1):
        block = block_slice(b, d)
        a = g[np.arange(2)[:, None], parents[..., 0], block]
        z = g[np.arange(2)[:, None], parents[..., 1], block]
        assert np.all(np.all(children[..., block] == a, axis=-1) | np.all(children[..., block] == z, axis=-1))
    # IDs inherit exactly the same parent's block, without extra random draws.
    tags = Alleles((2, 8, d + d*d))
    tags.ids[:] = np.arange(tags.ids.size).reshape(tags.ids.shape)
    encoded = np.zeros_like(g)
    encoded[..., :d+d*d] = tags.ids
    children = recombine(encoded, parents, d, rng, tags)
    assert np.array_equal(children[..., :d+d*d], tags.ids)
    # Only unanimous new IDs are fixations, and W includes its endpoint.
    tags = Alleles((2, 8, d + d*d))
    tags.ids[0, :, 0] = 1
    tags.ids[1, 0, 1] = 2
    events, repeated = tags.detect(1)
    assert events.sum() == 1 and events[0, 0] and not repeated.any()
    assert np.all((tags.ids == tags.ids[:, :1, :]) | ~events[:, None, :])
    assert not tags.detect(2)[0].any()
    tags.ids[0, :, 0] = 3
    assert tags.detect(1001)[1][0, 0]
    tags.ids[0, :, 0] = 4
    assert not tags.detect(2002)[1].any()
    assert tags.counts[0, 0] == 3 and tags.reversions[0, 0] == 1
    # Tag only directly mutated sites, even when normalization moves neighbors.
    tags = Alleles((1, 2, d + d*d))
    tags.tag(np.array([0, d, g.shape[-1] + 1, g.shape[-1] - 1]), g.shape[-1])
    assert sorted(tags.ids[tags.ids > 0]) == [1, 2, 3]
    assert tags.next_id == 5
    # Identical ancestral maps remain bitwise unchanged across both frozen modes.
    ancestor = original[0, 0]
    optimum = unit(forward(ancestor, d)[-1])
    for condition in ("frozen", "both"):
        frozen = np.broadcast_to(ancestor, original.shape).copy()
        for _ in range(12):
            frozen = step(frozen, optimum, 10, d, 0.5, 0.1, False, rng, condition)
        assert np.array_equal(frozen[..., d:], np.broadcast_to(ancestor[d:], frozen[..., d:].shape))
    args = dict(L=2, N=8, K=3, B=7, T=12, every=3, d=3, seed=7)
    for tanh in (False, True):
        for s in (0, 100):
            a = drift.run(s=s, tanh=tanh, **args)
            b = run(s=s, tanh=tanh, n_dfe=64, samples=4, **args)
            assert [p["t"] for p in a["records"]] == [p["t"] for p in b["records"]]
            assert np.array_equal([p["state_divergence"][0] for p in a["records"]],
                                  [p["x0_divergence"] for p in b["records"]])
            assert np.array_equal([p["map_divergence"][0] for p in a["records"]],
                                  [p["M1_divergence"] for p in b["records"]])
    for condition in CONDITIONS:
        zero = run(s=0, u=0, condition=condition, n_dfe=8, samples=2, **args)
        assert all(zero["substitutions"][b]["count"] == 0 for b in ("x0", "M1"))
        assert not np.any(zero["fixation_generations"])
        # Tracking must preserve every condition's evolutionary stream.
        tracked, untracked = original.copy(), original.copy()
        tags = Alleles((*tracked.shape[:2], d + d*d))
        r1, r2 = np.random.default_rng(23), np.random.default_rng(23)
        for t in range(20):
            tracked = step(tracked, optimum, 10, d, 0.1, 0.1, False, r1, condition, tags)
            untracked = step(untracked, optimum, 10, d, 0.1, 0.1, False, r2, condition)
            assert np.array_equal(tracked, untracked)
            events, _ = tags.detect(t + 1)
            assert np.all((tags.ids == tags.ids[:, :1, :]) | ~events[:, None, :])
    # Low N*u limits interference from recurrent mutation; many independent lines
    # keep this short neutral-rate check informative despite linked x0 sites.
    neutral = run(L=1, N=12, K=256, d=3, B=200, T=1600, every=400,
                  s=0, u=0.001, seed=31, n_dfe=8, samples=2)
    rate = neutral["substitutions"]["x0"]["rate_per_site"]
    assert 0.65 < rate / neutral["u"] < 1.35, rate
    generation_counts = np.asarray(neutral["fixation_generations"])
    assert generation_counts[:, 0].sum() == neutral["substitutions"]["x0"]["count"]
    assert generation_counts[:, 2].sum() == neutral["substitutions"]["x0"]["reversion_like_count"]
    # An invertible single evolving map already spans every phenotype direction.
    wild = normalize(np.r_[unit(np.ones(d)), np.eye(d).ravel()], d)
    for tanh in (False, True):
        for condition in CONDITIONS:
            m = neutral_manifold(wild, d, tanh, condition)
            assert m["rank_full"] == d - 1
            assert m["rank_maps"] == (0 if condition in ("frozen", "both") else d - 1)
            assert m["x0_directions_need_compensation"] == m["rank_full"] - m["rank_maps"]
    sites, values = rng.integers(0, d, 2000), rng.normal(0, 0.1, 2000)
    sm = fixed_effects(ancestor, sites, values, optimum, d, 100, False)
    assert np.array_equal(sm, fixed_effects(ancestor, sites, values, optimum, d, 100, False))
    curves = lag_curves([0, 500], [sm, sm], 100)
    assert np.isclose(curves["curves"][1]["correlation"], 1)
    assert curves["curves"][1]["jaccard"] == 1
    fixture = [dict(L=1, N=100, seed=0, tanh=False, condition=c, s=s, predictions=None,
                    substitutions={b: dict(rate_per_site=0 if c == "frozen" and b == "M1" else
                                           0.001 * (1 if s == 0 else 0.5)) for b in ("x0", "M1")},
                    records=[dict(t=0, x0_divergence=0, M1_divergence=0),
                             dict(t=100, x0_divergence=v, M1_divergence=v)])
               for c, v0 in (("baseline", 0.3), ("frozen", 0.2)) for s, v in ((0, v0), (100, v0/2))]
    add_omega(fixture)
    assert all(r["t_star"] == 100 for r in fixture)
    assert all(r["threshold_fallback"] == (r["condition"] == "frozen") for r in fixture)
    assert all(r["omega_x0"] == 0.5 for r in fixture if r["s"] > 0)
    assert all(r["substitution_ratio_x0"] == 0.5 for r in fixture if r["s"] > 0)
    assert all(r["substitution_ratio_M1"] is None and r["omega_M1"] is None
               for r in fixture if r["condition"] == "frozen")
    print(f"check ok ({perf_counter()-start:.2f}s): frozen maps, parent-block inheritance, "
          "exact A baseline records (selected/neutral, linear/tanh), fixed DFE replay, lag curves, t* fallback; "
          f"allele inheritance/fixation/window, u=0, neutral rate/u={rate/neutral['u']:.3f}, tangent ranks")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "quick", "run", "figure", "summary"), default="run")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.seed < 0:
        parser.error("Require workers >= 1 and seed >= 0")
    if args.command == "check":
        check()
    elif args.command == "figure":
        figure()
    elif args.command == "summary":
        summary()
    else:
        main(quick=args.command == "quick", seed=args.seed, workers=args.workers)
