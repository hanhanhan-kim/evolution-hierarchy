"""Experiment F3: equilibration, standing variation and interference in F2.

Run: python code3.py [check | quick | benchmark | run | summary | figure]
     [--config-chunk 2] [--individual-chunk 128] [--max-hours 4.5]
Uses code2's real van Hateren channel, normalized genomes, Bernoulli-site
mutation (u=.001, sigma=.1), fitness exp(s*(I/Imax-1)), batching and measures.
No downloads. quick writes code3-quick.*, run writes code3.*. run requires a
GPU; benchmark measures this device and projects from F2's measured shared
B300 rate, 2.75 million individual-generations/s. Preallocation defaults off.

Full grid: Ns={300,3000}, N={10,30,100,300,1000,3000}, K=8, L=3.
T=ceil(max(20000,20000/s)/1000)*1000, capped at 200000; records every 1000.
Clonal at both Ns, block and free recombination at Ns=300. ALL arms have both
random and optimum starts, so recombination also gets an equilibration test.
An optimum ancestor is the corresponding F2 random ancestor with its last map
left-multiplied by a Householder reflection taking unit(top) to env.best_z.
This preserves every block norm and all lower states; the top direction is the
best numerical representable code, not the projected CDF or a certified global
optimum. Each replicate starts clonal. Paired arms share F2 seeds; replicates
are independent. Recombination samples two fitness-weighted parents with
replacement, independently chooses each block/site with probability 1/2,
then applies F2 mutation. Free mosaics are renormalized before mutation to
retain the model's block constraints (raw site inheritance is tested first).

Consensus = block-normalized arithmetic mean genome. Genetic distance = mean
pairwise squared Euclidean distance, divided by genome norm squared 1+16L;
computed in O(NG), excluding self-pairs. Phenotype z is unit(top); its variance
is trace of the population covariance with denominator N. Consensus is a
coordinate-dependent diagnostic, not necessarily an attainable genotype.

Equilibration evidence requires both starts' tail means AND each start's
late-tail change to be equivalent within .01 efficiency, using conservative
Student-t 95% intervals over K replicate summaries, not over correlated records.
The tail is the last quarter. Passing is evidence at this time/tolerance, not
proof of stationarity. Failure leaves the equilibrium and selection thresholds
unresolved. A quick run NEVER establishes equilibrium or a mechanism.

Theory (phenomenological, no external data): delta=1-eta. Haploid WF diffusion
near a q-dimensional quadratic optimum has density exp(-2Ns*delta), giving
drift load q/(4Ns), q=15. Fit a/Ns as an effective-dimension alternative.
House-of-cards mutation-selection balance suggests delta=b*U/s=b*NU/Ns,
U=(16+256L)u=.784, assuming independent deleterious mutations and small load;
b is an effective damaging fraction. Small Gaussian mutational effects give
delta proportional to sqrt(U/s), with curvature/effect size absorbed in b.
Compare nonnegative fits of drift, mutation, drift+mutation, drift+Gaussian
by leave-one-N-out RMSE on equilibrated clonal cells. No universal law is
assumed; linkage, epistasis and large s can invalidate these approximations.
Invert the winning fit for 95/99% efficiency, marking extrapolation. Fitness
cost of a 1% information loss is 1-exp(-.01s), approximately .01s only for
small costs. N is census size here; replacing it by Ne requires calibration.

Gap diagnostics use N=10 versus 3000 at Ns=300: reduction of F2's gap after
long adaptation, consensus-minus-mean gap, and gap rescue by recombination.
These overlap and must NOT be added as causal percentages. Recombination
rescue supports interference but can also reflect disruption of epistasis.
Compact JSON stores replicate records, never genomes. Completed batches are
checkpointed; --resume continues matching configurations. No full run is
performed by check, quick or benchmark.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from time import perf_counter

import code2 as f2
import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as np
from scipy.optimize import nnls
from scipy.stats import t as student_t

OUTPUT = f2.OUTPUT
NS = (300, 3000)
POPULATIONS = (10, 30, 100, 300, 1000, 3000)
MODES = ("clonal", "block", "free")
METRICS = ("efficiency", "consensus", "best", "genetic_distance", "z_variance")
B300_RATE = 2_750_000
TOLERANCE = .01


def optimum(g, z):
    """Orthogonally rotate the final map; retain the random genetic background."""
    top = f2.unit(f2.forward(g)[..., -1, :])
    v = f2.unit(top - jnp.asarray(z))
    last = g[..., -256:].reshape(*g.shape[:-1], 16, 16)
    rotated = last - 2 * v[..., :, None] * jnp.einsum("...i,...ij->...j", v, last)[..., None, :]
    return g.at[..., -256:].set(rotated.reshape(*g.shape[:-1], 256))


def recombine(g, parents, key, mode):
    """Raw mosaics: every block/site is exactly inherited before normalization."""
    G = g.shape[-1]
    if mode == "block":
        blocks = jnp.concatenate((jnp.zeros(16, dtype=int), 1 + jnp.arange(G - 16) // 256))
        choices = random.bernoulli(key, .5, (len(g), 1 + (G - 16) // 256))[:, blocks]
    else:
        choices = random.bernoulli(key, .5, (len(g), G))
    return jnp.where(choices, g[parents[:, 1]], g[parents[:, 0]])


class Engine(f2.Engine):
    def __init__(self, env, individual_chunk=128, mode="clonal"):
        self.mode = mode
        super().__init__(env, individual_chunk)

    def step(self, carry, params, neutral=False):
        if self.mode == "clonal":
            return super().step(carry, params, neutral)
        g, info, keys = carry
        s, u = params

        def population(genomes, values, key, selection, mutation):
            next_key, parent_key, cross_key, mask_key, noise_key = random.split(key, 5)
            logw = selection * (values / self.env.I_max - 1)
            cdf = jnp.cumsum(jnp.exp(logw - logw.max()))
            draws = random.uniform(parent_key, (len(genomes), 2)) * cdf[-1]
            parents = jnp.minimum(jnp.searchsorted(cdf, draws), len(genomes) - 1)
            inherited = recombine(genomes, parents, cross_key, self.mode)
            if self.mode == "free":
                # Preserve unchanged blocks exactly, including a u=0 clone.
                normalized = f2.normalize(inherited)
                same_x = jnp.all(inherited[:, :16] == genomes[parents[:, 0], :16], -1, keepdims=True)
                maps = inherited[:, 16:].reshape(len(genomes), -1, 256)
                same_m = jnp.all(maps == genomes[parents[:, 0], 16:].reshape(maps.shape), -1, keepdims=True)
                inherited = jnp.concatenate((jnp.where(same_x, inherited[:, :16], normalized[:, :16]),
                    jnp.where(same_m, maps, normalized[:, 16:].reshape(maps.shape)).reshape(len(genomes), -1)), -1)
            mask = random.bernoulli(mask_key, mutation, inherited.shape)
            noise = random.normal(noise_key, inherited.shape)
            offspring = vmap(f2.mutate_one, in_axes=(0, 0, 0, None))(inherited, mask, noise, .1)
            return offspring, next_key

        batch = vmap(vmap(population, in_axes=(0, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0))
        offspring, keys = batch(g, info, keys, s, u)
        # Parent information cannot be cached across recombination.
        return (offspring, self.evaluate(offspring), keys), None

    def snapshot(self, g, info):
        result = super().snapshot(g, info)
        consensus = f2.normalize(g.mean(axis=2))
        z = f2.unit(f2.forward(g)[..., -1, :])
        centered = g - g.mean(axis=2, keepdims=True)
        N, G = g.shape[2:]
        norm2 = 1 + 16 * ((G - 16) // 256)
        result.update(consensus=self.evaluate(consensus[:, :, None])[:, :, 0] / self.env.I_max,
                      best=(info / self.env.I_max).max(axis=2),
                      genetic_distance=2 * (centered**2).sum(axis=(2, 3)) / (max(N - 1, 1) * norm2),
                      z_variance=((z - z.mean(axis=2, keepdims=True))**2).sum(-1).mean(axis=2))
        return result


def grid(seed=0, quick=False, generations=None):
    rows = []
    for N in ((4, 8) if quick else POPULATIONS):
        for Ns in NS:
            for mode in (MODES if Ns == 300 else ("clonal",)):
                for start in ("random", "optimum"):
                    T = min(200000, int(np.ceil(max(20000, 20000 * N / Ns) / 1000)) * 1000)
                    row = f2.config(N, Ns / N, K=2 if quick else 8,
                                    T=(generations or 20) if quick else T,
                                    every=5 if quick else 1000, seed=seed)
                    rows.append(dict(**row, mode=mode, start=start))
    return rows


def identity(row):
    return tuple(row[k] for k in ("N", "s", "L", "u", "K", "T", "every", "seed", "mode", "start"))


def batches(rows, chunk):
    for mode in MODES:
        yield from f2.groups([r for r in rows if r["mode"] == mode], chunk)


def execute(engine, rows):
    g, keys = f2.initial(rows)
    for i, row in enumerate(rows):
        if row["start"] == "optimum":
            g = g.at[i].set(optimum(g[i], engine.env.best_z))
    begin = perf_counter()
    final, records = engine.run(g, keys, jnp.array([r["s"] for r in rows]),
                               jnp.array([r["u"] for r in rows]), T=rows[0]["T"], every=rows[0]["every"])
    final.block_until_ready()
    seconds = perf_counter() - begin
    records = jax.device_get(records)
    result = []
    for i, row in enumerate(rows):
        times = np.unique(np.r_[np.arange(0, row["T"] + 1, row["every"]), row["T"]])
        keep = (*METRICS, "ks", "ks_best")
        rec = {k: np.asarray(records[k][:, i]) for k in keep}
        tail = times >= .75 * row["T"]
        values = rec["efficiency"][tail]
        middle = len(values) // 2
        change = values[middle:].mean(0) - values[:middle].mean(0) if middle else np.zeros(row["K"])
        result.append(dict(**row, times=times.tolist(),
            records={k: np.round(v, 7).tolist() for k, v in rec.items()},
            tail={k: v[tail].mean(0).tolist() for k, v in rec.items()},
            tail_change=change.tolist(), tail_records=int(tail.sum()),
            batch_seconds=seconds, batch_configs=len(rows)))
    return result, seconds


def interval(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1) / np.sqrt(len(values))))


def equivalent(values):
    stats = interval(values)
    return abs(stats["mean"]) + student_t.ppf(.975, len(values)-1) * stats["se"] <= TOLERANCE


def equilibria(rows, quick=False):
    cells = []
    for a in rows:
        if a["start"] != "random":
            continue
        b = next((b for b in rows if b["start"] == "optimum" and
                  all(a[k] == b[k] for k in ("N", "s", "mode"))), None)
        if b is None:
            continue
        difference = np.subtract(a["tail"]["efficiency"], b["tail"]["efficiency"])
        meets = equivalent(difference)
        stable = all(r["tail_records"] >= 4 and equivalent(r["tail_change"]) for r in (a, b))
        confirmed = bool(not quick and meets and stable and min(a["T"], b["T"]) >= 20000)
        # Pool starts only after convergence. Keep paired replicate summaries.
        tail = {k: ((np.array(a["tail"][k]) + b["tail"][k]) / 2 if confirmed else
                    np.array(a["tail"][k])).tolist() for k in METRICS}
        cells.append(dict(N=a["N"], Ns=a["N"] * a["s"], s=a["s"], mode=a["mode"],
                          confirmed=confirmed, starts_meet=bool(meets), stable=bool(stable),
                          start_difference=interval(difference), tail=tail,
                          stats={k: interval(v) for k, v in tail.items()}))
    return cells


def features(cells, model):
    Ns = np.array([r["Ns"] for r in cells])
    NU = np.array([r["N"] * .784 for r in cells])
    drift, mutation = 1 / Ns, NU / Ns
    return {"drift": drift[:, None], "mutation": mutation[:, None],
            "drift+mutation": np.column_stack((drift, mutation)),
            "drift+Gaussian": np.column_stack((drift, np.sqrt(mutation)))}[model]


def theory(cells):
    rows = [r for r in cells if r["mode"] == "clonal" and r["confirmed"]]
    if len({r["N"] for r in rows}) < 3 or len({r["Ns"] for r in rows}) < 2:
        return dict(status="unresolved: need equilibrated clonal cells at >=3 N and both Ns", thresholds=[])
    y = np.array([1 - r["stats"]["efficiency"]["mean"] for r in rows])
    fits = {}
    for model in ("drift", "mutation", "drift+mutation", "drift+Gaussian"):
        x = features(rows, model)
        beta = nnls(x, y)[0]
        prediction = np.empty(len(rows))
        for N in {r["N"] for r in rows}:
            test = np.array([r["N"] == N for r in rows])
            prediction[test] = x[test] @ nnls(x[~test], y[~test])[0]
        fits[model] = dict(coefficients=beta.tolist(), cv_rmse=float(np.sqrt(np.mean((y - prediction)**2))),
                           rmse=float(np.sqrt(np.mean((y - x @ beta)**2))))
    winner = min(fits, key=lambda k: fits[k]["cv_rmse"])
    adequate = fits[winner]["cv_rmse"] <= .02
    beta = fits[winner]["coefficients"]
    thresholds = []
    for N in POPULATIONS:
        for target in (.95, .99):
            delta, NU = 1 - target, N * .784
            if winner == "drift+Gaussian":
                a, b = beta
                # Positive root for y=1/sqrt(Ns), avoiding cancellation.
                root = 2 * delta / (b * np.sqrt(NU) + np.sqrt(b*b*NU + 4*a*delta)) if a or b else np.inf
                Ns = 1 / root**2
            else:
                numerator = beta[0] if winner == "drift" else beta[0] * NU if winner == "mutation" else beta[0] + beta[1] * NU
                Ns = numerator / delta
            thresholds.append(dict(N=N, target=target, Ns=float(Ns), s=float(Ns/N),
                cost_of_one_percent=float(-np.expm1(-.01 * Ns/N)),
                approximate_cost_times_N=float(.01 * Ns),
                extrapolated=bool(N not in {r["N"] for r in rows} or not min(r["Ns"] for r in rows) <= Ns <= max(r["Ns"] for r in rows))))
    return dict(status=("conditional equilibrium-model estimates; validate extrapolated thresholds" if adequate else
                        "no candidate fits within .02 held-out RMSE; thresholds are unsupported extrapolations"),
                adequate=adequate,
                cells=len(rows), winner=winner, fits=fits, thresholds=thresholds,
                fixed_q15_drift_rmse=float(np.sqrt(np.mean((y - 15 / (4 * np.array([r["Ns"] for r in rows])))**2))))


def baseline():
    """Read the saved F2 table; never substitute quick results for its gap."""
    path = OUTPUT / "code2-summary.txt"
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) > 7 and fields[:2] == ["vanhateren", "3"]:
                N, u, Ns = int(fields[2]), float(fields[4]), float(fields[5])
                if N in (10, 3000) and u == .001 and Ns == 300:
                    values[N] = float(fields[6])
    return values[10] - values[3000] if len(values) == 2 else None


def verdict(cells, quick):
    if quick:
        return dict(text="UNRESOLVED (quick smoke test): non-equilibrium, variation load and interference are untested.", effects={})
    ends = {(r["mode"], r["N"]): r for r in cells if r["Ns"] == 300 and r["N"] in (10, 3000)}
    if not all(("clonal", N) in ends for N in (10, 3000)):
        return dict(text="UNRESOLVED: endpoint cells are incomplete.", effects={})
    low, high = (ends["clonal", N] for N in (10, 3000))
    gap = low["stats"]["efficiency"]["mean"] - high["stats"]["efficiency"]["mean"]
    f2gap = baseline()
    effects = {}
    both = low["confirmed"] and high["confirmed"]
    if f2gap is not None:
        effects["non_equilibrium"] = dict(points=f2gap-gap, fraction_of_F2_gap=(f2gap-gap)/f2gap,
                                          supported=bool(both), note="F2 gap minus long-run clonal gap; independent runs, descriptive")
    load = lambda r: np.subtract(r["tail"]["consensus"], r["tail"]["efficiency"])
    effect = float(load(high).mean() - load(low).mean())
    se = float(np.hypot(interval(load(high))["se"], interval(load(low))["se"]))
    critical = float(student_t.ppf(.975, len(load(high))-1))
    effects["variation_load"] = dict(points=effect, se=se, fraction_of_F2_gap=effect/f2gap if f2gap else None,
        supported=bool(both and effect > critical*se), large_N_consensus=high["stats"]["consensus"]["mean"])
    for mode in MODES[1:]:
        if all((mode, N) in ends for N in (10, 3000)):
            a, b = (ends[mode, N] for N in (10, 3000))
            contrasts = [np.subtract(ends["clonal", N]["tail"]["efficiency"], ends[mode, N]["tail"]["efficiency"]) for N in (10, 3000)]
            rescue = float(contrasts[0].mean() - contrasts[1].mean())
            se = float(np.hypot(*(interval(x)["se"] for x in contrasts)))
            large_rescue = -float(contrasts[1].mean())
            effects[mode + "_interference"] = dict(points=rescue, se=se,
                fraction_of_F2_gap=rescue/f2gap if f2gap else None,
                supported=bool(both and a["confirmed"] and b["confirmed"] and rescue > critical*se
                               and large_rescue > critical*interval(contrasts[1])["se"] + TOLERANCE),
                large_N_rescue=large_rescue)
    supported = [k for k, v in effects.items() if v["supported"] and v["points"] > TOLERANCE]
    if not both:
        text = "UNRESOLVED equilibrium: starts do not meet or tails remain unstable; non-equilibrium is not excluded."
    elif supported:
        text = "Supported explanations: " + ", ".join(supported) + ". Largest descriptive gap reduction: " + max(supported, key=lambda k: effects[k]["points"]) + "."
    else:
        text = "No tested mechanism explains >.01 efficiency of the gap with the required convergence evidence."
    return dict(text=text, F2_gap=f2gap, clonal_tail_gap=gap, effects=effects,
                caution="Effects overlap; do not sum percentages. Recombination rescue is evidence consistent with interference, not a unique causal identification.")


def analysis(data):
    cells = equilibria(data["rows"], data["quick"])
    return dict(cells=cells, theory=theory(cells), verdict=verdict(cells, data["quick"]))


def summary(data):
    print("F3 " + ("QUICK SMOKE TEST — no equilibrium inference" if data["quick"] else "equilibration, load and interference"))
    print(f"Completed {len(data['rows'])}/{len(data['planned'])} arms; sigma_n={data['sigma_n']}; U=.784; K replicates, not time points, determine errors.")
    print("Tail = last quarter; equilibrium evidence = starts agree and tails stable within .01 including 95% replicate intervals.")
    print("N      Ns mode     mean    consensus best    genetic distance z variance  equilibrium")
    for c in data["analysis"]["cells"]:
        m = c["stats"]
        print(f"{c['N']:4d} {c['Ns']:7g} {c['mode']:7s} " + " ".join(f"{m[k]['mean']:.5f}" for k in METRICS) +
              f"  {'supported' if c['confirmed'] else 'unresolved'} (start gap={c['start_difference']['mean']:+.4f}, stable={c['stable']})")
    v = data["analysis"]["verdict"]
    print("VERDICT:", v["text"])
    for name, value in v["effects"].items():
        fraction = value.get("fraction_of_F2_gap")
        print(f"  {name}: {100*value['points']:+.2f} percentage points" +
              (f", {100*fraction:+.1f}% of F2 gap" if fraction is not None else "") +
              f"; {'supported' if value['supported'] else 'descriptive/unresolved'}")
    print(v.get("caution", "No full experiment has been run."))
    t = data["analysis"]["theory"]
    print("Theory:", t["status"])
    print("delta=1-eta; drift=q/(4Ns), q=15 (also fit a/Ns); mutation=b*NU/Ns; Gaussian=b*sqrt(NU/Ns). U=784u.")
    print("Assumptions: local quadratic haploid WF drift; independent deleterious house-of-cards mutations or small Gaussian effects. Coefficients fitted, not universal.")
    if t.get("fits"):
        print(f"Fixed q=15 drift RMSE={t['fixed_q15_drift_rmse']:.5f}; winning leave-one-N-out fit: {t['winner']}")
        for model, fit in t["fits"].items():
            print(f"  {model}: coefficients={fit['coefficients']}, held-out RMSE={fit['cv_rmse']:.5f}")
        print("Conditional thresholds: N target Ns s cost(1% information loss) [* = extrapolation]")
        for r in t["thresholds"]:
            print(f"  {r['N']:4d} {r['target']:.2f} {r['Ns']:.2f} {r['s']:.5g} {r['cost_of_one_percent']:.5g}" + (" *" if r["extrapolated"] else ""))
        print("The 30/Ne claim corresponds to Ns=3000 only under the small-cost approximation and N=Ne. Compare fitted Ns to 3000 above; an N-dependent threshold invalidates a universal 30/Ne coefficient.")
    else:
        print("95%/99% equilibrium thresholds and the '1% information loss costs >30/Ne' claim remain unresolved; do not infer them from this run.")
    if data.get("benchmark"):
        b = data["benchmark"]
        print(f"B300 projection: {b['F2_projected_hours']:.2f} h at measured F2 throughput; {b['budget_hours']:.2f} h scheduling budget; compilation/contention can change this.")


def save(data, stem, echo=True):
    data["analysis"] = analysis(data)
    stem.parent.mkdir(parents=True, exist_ok=True)
    target = stem.with_suffix(".json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(target)
    stream = StringIO()
    with redirect_stdout(stream):
        summary(data)
    stem.with_name(stem.name + "-summary.txt").write_text(stream.getvalue())
    if echo:
        print(stream.getvalue(), end="")


def benchmark(env, args):
    measurements = {}
    gpu = jax.default_backend() == "gpu"
    for mode in MODES:
        engine = Engine(env, args.individual_chunk, mode)
        rows = [dict(**f2.config(1000 if gpu else 32, 3, K=8 if gpu else 2,
                                T=100, every=100 if gpu else 20, seed=args.seed),
                     mode=mode, start=start) for start in ("random", "optimum")[:args.config_chunk]]
        _, cold = execute(engine, rows)
        _, warm = execute(engine, rows)
        count = sum(r["N"] * r["K"] * r["T"] for r in rows)
        measurements[mode] = dict(cold_seconds=cold, warm_seconds=warm, rate=count/warm,
                                  N=rows[0]["N"], K=rows[0]["K"], T=rows[0]["T"], configs=len(rows))
        print(f"{mode}: {count/warm:,.0f} individual-generations/s on {jax.devices()[0]}; cold={cold:.2f}s, warm={warm:.2f}s", flush=True)
    planned = grid(args.seed)
    work = {mode: sum(r["N"]*r["K"]*r["T"] for r in planned if r["mode"] == mode) for mode in MODES}
    historical = sum(work.values()) / B300_RATE / 3600
    current = sum(work[m] / measurements[m]["rate"] for m in MODES) / 3600
    budget = 1.25 * max(historical, current if gpu else historical)
    result = dict(backend=jax.default_backend(), device=str(jax.devices()[0]), pilots=measurements,
        individual_generations=work, F2_B300_rate=B300_RATE, F2_projected_hours=historical,
        same_device_hours=current, budget_hours=budget, margin=1.25, full_arms=len(planned),
        note="F2's measured shared B300 rate, not a CPU speedup guess. F3 recombination and records add overhead; GPU pilot gates run. All N retained within default budget.")
    print(f"Full grid: {sum(work.values()):,} individual-generations, {len(planned)} arms including both starts for recombination.")
    print(f"F2 shared-B300 projection {historical:.2f} h; 25% scheduling margin/current GPU pilot => {budget:.2f} h.")
    print(f"Same-device projection {current:.2f} h. No full run performed by benchmark.")
    return result


def main(args, quick=False):
    stem = args.output or OUTPUT / ("code3-quick" if quick else "code3")
    if not quick and jax.default_backend() != "gpu":
        raise RuntimeError("Full run requires GPU; use quick or benchmark on CPU")
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    estimate = None if quick else benchmark(env, args)
    if estimate and estimate["budget_hours"] > args.max_hours:
        # Small-N cells already have T=20000 under this schedule; omitting their
        # extensions cannot save work. Do not silently drop replicates or time.
        raise RuntimeError(f"Projected budget {estimate['budget_hours']:.2f} h exceeds --max-hours {args.max_hours}; "
                           "priority N={100,1000,3000} would not shorten this schedule. Rebenchmark when contention falls or set an explicit larger budget.")
    planned = grid(args.seed, quick, args.generations)
    data = dict(experiment="F3", quick=quick, sigma_n=env.sigma_n, planned=planned, rows=[],
                environment=env.params, calibration=env.calibration, benchmark=estimate,
                definitions=__doc__, elapsed_seconds=0.)
    if args.resume and stem.with_suffix(".json").exists():
        previous = json.loads(stem.with_suffix(".json").read_text())
        if previous["planned"] != planned or previous["environment"] != env.params or previous["quick"] != quick:
            raise ValueError("Resume configuration/environment differs from checkpoint")
        data = previous
    done = {identity(r) for r in data["rows"]}
    engines = {mode: Engine(env, args.individual_chunk, mode) for mode in MODES}
    for batch in batches([r for r in planned if identity(r) not in done], args.config_chunk):
        result, seconds = execute(engines[batch[0]["mode"]], batch)
        data["rows"].extend(result)
        data["elapsed_seconds"] += seconds
        save(data, stem, echo=False)
        print(f"{len(data['rows'])}/{len(planned)} arms: N={batch[0]['N']} s={batch[0]['s']:g} {batch[0]['mode']} T={batch[0]['T']} ({seconds:.2f}s)", flush=True)
    save(data, stem)
    figure(stem)


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(stem=OUTPUT / "code3"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    data = json.loads(stem.with_suffix(".json").read_text())
    cells, rows = data["analysis"]["cells"], data["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    style = dict(fontsize=5.5, family="monospace", va="center")

    def labels(ax, items):
        low, high = ax.get_ylim()
        placed = -.1
        for i, (x, y, text, color) in enumerate(sorted(items, key=lambda v: v[1])):
            pos = np.clip((y-low)/(high-low), .05, .94 - .13*(len(items)-i-1))
            pos = max(pos, placed + .13)
            ax.annotate(text, xy=(x, y), xycoords="data", xytext=(.60, pos), textcoords="axes fraction",
                        color=color, arrowprops=dict(arrowstyle="-", color=color, lw=.5), **style)
            placed = pos

    ax = axes[0]
    selected = [r for r in rows if r["mode"] == "clonal" and r["N"]*r["s"] == 300]
    N = max(r["N"] for r in selected)
    ends = []
    for i, start in enumerate(("random", "optimum")):
        r = next(r for r in selected if r["N"] == N and r["start"] == start)
        x, values = np.array(r["times"])/1000, np.array(r["records"]["efficiency"])
        y, se = values.mean(1), values.std(1, ddof=1)/np.sqrt(r["K"])
        ax.plot(x, y, color=f"C{i}", lw=1)
        ax.fill_between(x, y-se, y+se, color=f"C{i}", alpha=.15)
        ends.append((x[-1], y[-1], start, f"C{i}"))
    ax.set(xlabel="generation (×1000)", ylabel="efficiency", title=f"N={N}, Ns=300", ylim=(0, 1.03))
    labels(ax, ends)
    for ax, mode_panel in zip(axes[1:], (False, True)):
        ends = []
        lows = []
        curves = [(300, mode, "efficiency", f"C{i}", "-") for i, mode in enumerate(MODES)] if mode_panel else [
            (Ns, "clonal", metric, f"C{i}", "-" if metric == "efficiency" else "--")
            for i, Ns in enumerate(NS) for metric in ("efficiency", "consensus")]
        for Ns, mode, metric, color, line in curves:
            group = sorted([c for c in cells if c["Ns"] == Ns and c["mode"] == mode], key=lambda c: c["N"])
            if not group:
                continue
            x = [c["N"] for c in group]
            y = [c["stats"][metric]["mean"] for c in group]
            se = [c["stats"][metric]["se"] for c in group]
            lows.extend(np.subtract(y, se))
            ax.errorbar(x, y, yerr=se, color=color, ls=line, lw=.8, capsize=1)
            for xx, yy, c in zip(x, y, group):
                ax.plot(xx, yy, "o", ms=2.5, color=color, fillstyle="full" if c["confirmed"] else "none")
            text = mode if mode_panel else f"{Ns} " + ("mean" if metric == "efficiency" else "cons.")
            ends.append((x[-1], y[-1], text, color))
        ax.set(xscale="log", xlabel="population N", ylim=(0, 1.03), title="Ns=300" if mode_panel else "mean / consensus")
        ticks = sorted({c["N"] for c in cells})
        ax.set_xticks(ticks if data["quick"] else [10, 100, 1000],
                      labels=[str(N) for N in (ticks if data["quick"] else [10, 100, 1000])])
        ax.minorticks_off()
        if not data["quick"]:
            ax.set_ylim(max(0, min(lows)-.08), 1.02)
        labels(ax, ends)
    for ax, letter in zip(axes, "ABC"):
        panel_letter(ax, letter)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.title.set_size(7)
    note = "SMOKE TEST: not equilibrium" if data["quick"] else "open points: equilibrium unresolved; bars/bands: replicate SE"
    fig.text(.5, .015, note, ha="center", fontsize=5.5, family="monospace")
    fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.83, wspace=.38)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def check(args):
    begin = perf_counter()
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    row = dict(**f2.config(4, 3, K=2, T=4, every=2, u=0., seed=args.seed), mode="clonal", start="optimum")
    g, keys = f2.initial([row])
    optimal = optimum(g, env.best_z)
    assert np.allclose(f2.unit(f2.forward(optimal)[..., -1, :]), env.best_z, atol=3e-6)
    assert np.allclose(np.linalg.norm(np.asarray(optimal[..., -256:]), axis=-1), 4, atol=2e-6)
    engine = Engine(env, 8)
    info = engine.evaluate(optimal)
    assert np.allclose(info / env.I_max, 1, atol=3e-6)
    measured = engine.snapshot(optimal, info)
    assert np.allclose(measured["consensus"], measured["efficiency"], atol=3e-6)
    assert np.allclose(measured["genetic_distance"], 0, atol=1e-7)
    assert np.allclose(measured["z_variance"], 0, atol=1e-7)
    rng = np.random.default_rng(21)
    genomes = jnp.asarray(f2.original.normalize(rng.normal(size=(12, 784)), 16), dtype=jnp.float32)
    parents = jnp.asarray(rng.integers(0, len(genomes), (len(genomes), 2)))
    for mode in MODES[1:]:
        children = np.asarray(recombine(genomes, parents, random.PRNGKey(7), mode))
        a, b = np.asarray(genomes[parents[:, 0]]), np.asarray(genomes[parents[:, 1]])
        if mode == "block":
            for lo, hi in ((0, 16), (16, 272), (272, 528), (528, 784)):
                assert np.all(np.all(children[:, lo:hi] == a[:, lo:hi], -1) | np.all(children[:, lo:hi] == b[:, lo:hi], -1))
        else:
            assert np.all((children == a) | (children == b))
    # Check O(NG) distance against explicit pairs on a diverse population.
    diverse = genomes[None, None].repeat(2, axis=1)
    snapshot = engine.snapshot(diverse, engine.evaluate(diverse))
    pairs = np.asarray(genomes)[:, None] - np.asarray(genomes)[None, :]
    expected = (pairs**2).sum() / (12 * 11 * 49)
    assert np.allclose(snapshot["genetic_distance"], expected, atol=2e-6)
    for mode in MODES:
        runner = Engine(env, 8, mode)
        final, records = runner.run(optimal, keys, jnp.array([3.]), jnp.array([0.]), T=4, every=2)
        assert np.array_equal(final, optimal), mode
        assert np.allclose(records["efficiency"], 1, atol=3e-6), mode
    full = grid()
    assert len(full) == 48 and all(r["T"] % r["every"] == 0 and r["K"] == 8 for r in full)
    assert max(r["T"] for r in full) == 200000
    # Analysis regression: recover a known mixed load and its inverse threshold.
    synthetic = []
    for r in full:
        eta = 1 - (2 + .01*r["N"]*.784)/(r["N"]*r["s"])
        synthetic.append(dict(**r, tail={k: [eta]*8 for k in METRICS},
                              tail_change=[0.]*8, tail_records=6))
    cells = equilibria(synthetic)
    assert all(c["confirmed"] for c in cells)
    assert not any(c["confirmed"] for c in equilibria(synthetic, quick=True))
    fitted = theory(cells)
    assert fitted["winner"] == "drift+mutation" and fitted["adequate"]
    for r in fitted["thresholds"]:
        assert np.isclose(r["Ns"], (2 + .01*r["N"]*.784)/(1-r["target"]))
    synthetic[1]["tail"]["efficiency"] = [.5]*8
    assert not equilibria(synthetic)[0]["confirmed"]
    print(f"check ok ({perf_counter()-begin:.2f}s): optimum construction, block/site inheritance, consensus, distance, u=0 optimal stability, convergence gate, theory inversion, full schedule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "quick", "benchmark", "run", "summary", "figure"))
    parser.add_argument("--contrasts", type=Path, default=OUTPUT / "contrasts.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-chunk", type=int, default=2)
    parser.add_argument("--individual-chunk", type=int, default=128)
    parser.add_argument("--generations", type=int, help="Quick only; full schedule is fixed")
    parser.add_argument("--max-hours", type=float, default=4.5)
    parser.add_argument("--output", type=Path, help="Output stem without extension")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.config_chunk, args.individual_chunk, args.max_hours) <= 0 or not np.isfinite(args.max_hours):
        parser.error("chunks and max-hours must be positive and finite")
    if args.generations is not None and (args.command != "quick" or args.generations < 1):
        parser.error("--generations is positive and quick-only")
    if args.command == "check":
        check(args)
    elif args.command in ("summary", "figure"):
        stem = args.output or OUTPUT / "code3"
        if args.command == "figure":
            figure(stem)
        else:
            data = json.loads(stem.with_suffix(".json").read_text())
            save(data, stem)
    elif args.command == "benchmark":
        env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
        result = benchmark(env, args)
        stem = args.output or OUTPUT / "code3-benchmark"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    else:
        main(args, quick=args.command == "quick")
