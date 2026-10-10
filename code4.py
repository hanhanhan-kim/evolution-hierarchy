"""Experiment F4: variation load and quantitative selection thresholds after F3.

Run: python code4.py [check | quick | bench | run | merge | summary | figure]
     [--part {0,1}] [--resume] [--config-chunk 2] [--individual-chunk 128]
No downloads. Full run requires GPU. Part 0 has Ns=300, part 1 Ns=3000;
merge validates both grids and writes code4.*. Quick always uses *-quick names.

F2's real van Hateren channel, sigma_n=.0128, L=3, clonal WF reproduction,
Bernoulli-site u and Gaussian effect SD=.1; F3's paired random/optimum starts.
Full grid: N={10,100,1000,3000}, Ns={300,3000}, u={1e-4,3e-4,1e-3,3e-3}, K=8.
Earliest test T=ceil(max(20000,20000/s)/1000)*1000, capped at 300000.
Continue the SAME populations, information caches and PRNG streams in 10000
generation chunks until convergence or the cap. Record every 1000 generations.
Both mean AND consensus must pass F3's .01 equivalence test: paired-start
last-quarter differences and each start's late-tail change, including Student-t
95% intervals over K replicate summaries. Repeated checks are descriptive
stopping evidence, not a sequentially calibrated proof of equilibrium.
Pool paired starts only after passing; otherwise report the random-start tail
and retain both starts. Quick can never establish equilibrium.

Measures reuse F3. Segregating sites count normalized float32 genome coordinates
with unequal values among individuals (exact inequality); normalization can make
an entire touched block polymorphic. This is NOT a count of mutation events or
independently segregating loci. U=(16+256L)u; G=consensus-mean; D=1-consensus.
JSON stores replicate records, never genomes. Atomic checkpoints after each
chunk retain completed cells. Resume skips these, and deterministically replays
unfinished cells from their seeds; it does not restore populations from JSON.
Batch size/device changes may cause small floating-point differences on replay.

Custom grids: --populations 100 1000 3000 --ns 30 100 1000
--mutations 1e-4 3e-4 1e-3 3e-3 --replicates 32 --cap 300000
--exclude 3000:30 --output output/code4-step4 (no --part).
Unspecified dimensions retain full defaults; quick uses small defaults and its
--generations cap. Custom caps must be multiples of 1000. For a single B300,
start with --config-chunk 2 --individual-chunk 128: one cell, two starts,
32 replicates per start. At N=3000 the float32 genome carry is about 0.60 GB;
temporary genomes/responses add memory, while information stencils are bounded
by individual-chunk. Keep config-chunk=2 until bench at K=32 verifies headroom;
bench reports device peak memory when available. No genomes accumulate in JSON.

Descriptive fits: positive power laws y=a*product(x**p), log least squares,
ranked by leave-one-N-out efficiency-unit RMSE (also report log RMSE), using a
common positive-response subset of confirmed cells. No pseudocounts. G tests
U/s, NU, Ns, U, and all six pairs; D tests Ns and Ns+U. Rank-deficient full or
held-out designs are rejected. Ties favor fewer variables. Fits with <3 N are
unresolved. With only two Ns, functional shape is weakly constrained. Algebraic
aliases (e.g. U/s=NU/Ns) prevent causal identification from collapse alone.

Consensus thresholds invert D; individual thresholds solve D+G=1-target along
s=Ns/N at each observed N,u. Report s/U, not a gap-only threshold that spends
the entire load budget twice. Interpolation requires confirmed Ns bracketing at
the reported u (and N for individual thresholds); all other inversions are
extrapolations. Nondecreasing laws/no finite crossing yield unresolved thresholds.
The house-of-cards U/s exponent is fitted, never assumed. Fits, held-out errors
and extrapolations are descriptive, without parameter-uncertainty guarantees.
Imax is F2's best numerical representable code, not a certified global optimum.
N is census size; replacing it by Ne needs calibration.
"""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import code3 as f3
import code2 as f2
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.optimize import brentq

OUTPUT = f2.OUTPUT
POPULATIONS = (10, 100, 1000, 3000)
NS = (300, 3000)
MUTATIONS = (1e-4, 3e-4, 1e-3, 3e-3)
METRICS = (*f3.METRICS, "segregating_sites")
CAP = 300000
VERSION = 1


class Engine(f3.Engine):
    def __init__(self, env, individual_chunk=128):
        super().__init__(env, individual_chunk, "clonal")
        self.advance = jax.jit(self._advance, static_argnames=("T", "every"))

    def snapshot(self, g, info):
        old = super().snapshot(g, info)
        return {**{k: old[k] for k in f3.METRICS},
                "segregating_sites": jnp.any(g != g[:, :, :1], axis=2).sum(-1)}

    def _advance(self, carry, s, u, T, every):
        def block(carry, _):
            carry, _ = lax.scan(lambda c, _: self.step(c, (s, u)), carry, None, length=every)
            return carry, self.snapshot(carry[0], carry[1])
        return lax.scan(block, carry, None, length=T // every)


def custom_spec(args, quick=False):
    """Canonical requested spec, separate from the effective quick schedule."""
    names = ("populations", "ns", "mutations", "replicates", "cap", "exclude")
    if not any(getattr(args, name, None) is not None for name in names):
        return None
    if args.part is not None or args.output is None:
        raise ValueError("Custom grid options require --output and forbid --part")
    spec = dict(populations=sorted(args.populations if args.populations is not None else
                                   ((4, 8) if quick else POPULATIONS)),
                ns=sorted(args.ns if args.ns is not None else NS),
                mutations=sorted(args.mutations if args.mutations is not None else
                                 ((1e-3,) if quick else MUTATIONS)),
                replicates=args.replicates if args.replicates is not None else (2 if quick else 8),
                cap=args.cap if args.cap is not None else CAP)
    for name in ("populations", "ns", "mutations"):
        values = spec[name]
        if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"{name} must contain distinct positive finite values")
    if any(u > 1 for u in spec["mutations"]):
        raise ValueError("mutations are Bernoulli probabilities, at most 1")
    if spec["replicates"] < 2 or spec["cap"] < 1000 or spec["cap"] % 1000:
        raise ValueError("replicates must be >=2; cap must be a positive multiple of 1000")
    excluded = set()
    for value in args.exclude or []:
        try:
            n, ns = value.split(":")
            pair = (int(n), float(ns))
        except ValueError:
            raise ValueError(f"Invalid --exclude {value!r}; expected N:Ns") from None
        if pair[0] not in spec["populations"] or pair[1] not in spec["ns"]:
            raise ValueError(f"Excluded pair {value} is outside the requested grid")
        excluded.add(pair)
    spec["exclude"] = [list(pair) for pair in sorted(excluded)]
    if len(excluded) == len(spec["populations"])*len(spec["ns"]):
        raise ValueError("Exclusions leave an empty grid")
    return spec


def grid(seed=0, quick=False, generations=20, part=None, spec=None):
    if spec is not None and part is not None:
        raise ValueError("Custom grids cannot use --part")
    populations = spec["populations"] if spec else ((4, 8) if quick else POPULATIONS)
    strengths = spec["ns"] if spec else NS
    mutations = spec["mutations"] if spec else ((1e-3,) if quick else MUTATIONS)
    K = spec["replicates"] if spec else (2 if quick else 8)
    cap = generations if quick else spec["cap"] if spec else CAP
    excluded = {tuple(pair) for pair in spec["exclude"]} if spec else set()
    rows = []
    for N in populations:
        for Ns in strengths:
            if part is not None and Ns != NS[part]:
                continue
            if (N, Ns) in excluded:
                continue
            for u in mutations:
                first = min(cap, int(np.ceil(max(20000, 20000*N/Ns)/1000))*1000)
                row = f2.config(N, Ns/N, K=K, u=u, seed=seed,
                                T=generations if quick else first, every=1 if quick else 1000)
                rows.append(dict(**row, Ns=Ns, cap=cap, mode="clonal"))
    return rows


def identity(row):
    return tuple(row[k] for k in ("N", "Ns", "u"))


def pending(data):
    done = {identity(r) for r in data["cells"] if r["status"] in ("met", "cap reached")}
    return sorted((r for r in data["planned"] if identity(r) not in done), key=first_work)


def first_work(row):
    return 2*row["N"]*row["K"]*row["T"]


def pending_batches(data, config_chunk):
    # Shape grouping may collect nonadjacent rows; sort the resulting batches too.
    return sorted(f2.groups(pending(data), config_chunk//2), key=lambda batch: first_work(batch[0]))


def new_data(planned, quick, part, env, chunk, spec=None):
    return dict(experiment="F4", version=VERSION, quick=quick, part=part,
                seed=planned[0]["seed"], generations=planned[0]["cap"] if quick else None,
                sigma_n=env.sigma_n, environment=env.params, calibration=env.calibration,
                chunk_generations=chunk, planned=planned, cells=[], elapsed_seconds=0., definitions=__doc__,
                **({"grid_spec": spec} if spec is not None else {}))


def validate_resume(previous, requested):
    fields = ("experiment", "version", "planned", "environment", "calibration", "sigma_n",
              "quick", "part", "seed", "generations", "chunk_generations", "grid_spec")
    if any(previous.get(k) != requested.get(k) for k in fields):
        raise ValueError("Resume configuration/environment differs from checkpoint")
    planned = {identity(c): c for c in requested["planned"]}
    seen = set()
    for c in previous["cells"]:
        key = identity(c)
        if (key in seen or key not in planned or
                any(c.get(k) != v for k, v in planned[key].items()) or
                c["status"] not in ("running", "met", "cap reached")):
            raise ValueError("Resume checkpoint contains duplicate or mismatched cells")
        seen.add(key)


def summarize_cell(row, times, records, quick=False):
    arms = []
    mask = np.asarray(times) >= .75 * times[-1]
    for i, start in enumerate(("random", "optimum")):
        tail = {k: np.asarray(v)[mask, i].mean(0).tolist() for k, v in records.items()}
        changes = {}
        for k in ("efficiency", "consensus"):
            values = np.asarray(records[k])[mask, i]
            middle = len(values)//2
            changes[k] = (values[middle:].mean(0)-values[:middle].mean(0)).tolist() if middle else [0.]*row["K"]
        arms.append(dict(start=start, tail=tail, tail_change=changes, tail_records=int(mask.sum())))
    diagnostics = {}
    for k in ("efficiency", "consensus"):
        delta = np.subtract(arms[0]["tail"][k], arms[1]["tail"][k])
        stable = all(a["tail_records"] >= 4 and f3.equivalent(a["tail_change"][k]) for a in arms)
        diagnostics[k] = dict(starts_meet=bool(f3.equivalent(delta)), stable=bool(stable),
                              difference=f3.interval(delta),
                              changes=[f3.interval(a["tail_change"][k]) for a in arms])
    confirmed = bool(not quick and times[-1] >= row["T"] and
                     all(d["starts_meet"] and d["stable"] for d in diagnostics.values()))
    tail = {k: ((np.array(arms[0]["tail"][k])+arms[1]["tail"][k])/2 if confirmed else
                np.array(arms[0]["tail"][k])).tolist() for k in METRICS}
    tail.update(G=np.subtract(tail["consensus"], tail["efficiency"]).tolist(),
                D=(1-np.asarray(tail["consensus"])).tolist())
    status = "met" if confirmed else "cap reached" if times[-1] >= row["cap"] else "running"
    return dict(**row, U=row["u"]*(16+256*row["L"]), completed_T=int(times[-1]),
                confirmed=confirmed, status=status, diagnostics=diagnostics, arms=arms,
                times=list(times), records={k: np.asarray(v).tolist() for k, v in records.items()},
                tail=tail, stats={k: f3.interval(v) for k, v in tail.items()})


def execute(engine, rows, quick, chunk, checkpoint=None):
    """Paired cells; carry keys and cached information across every chunk."""
    arms = [dict(**r, start=start) for r in rows for start in ("random", "optimum")]
    g, keys = f2.initial(arms)
    for i in range(1, len(arms), 2):
        g = g.at[i].set(f3.optimum(g[i], engine.env.best_z))
    carry = (g, engine.evaluate(g), keys)
    initial = jax.device_get(engine.snapshot(carry[0], carry[1]))
    history = {k: [v] for k, v in initial.items()}
    times, active, results = [0], list(rows), {}
    elapsed, t = 0., 0
    while active:
        T = min(chunk, active[0]["cap"]-t)
        begin = perf_counter()
        carry, records = engine.advance(carry, jnp.array([r["s"] for r in active for _ in range(2)]),
            jnp.array([r["u"] for r in active for _ in range(2)]), T=T, every=active[0]["every"])
        carry[0].block_until_ready()
        seconds = perf_counter()-begin
        elapsed += seconds
        for k, values in jax.device_get(records).items():
            history[k].extend(values)
        times.extend(range(t+active[0]["every"], t+T+1, active[0]["every"]))
        t += T
        keep, current = [], []
        for i, row in enumerate(active):
            rec = {k: np.asarray(v)[:, 2*i:2*i+2] for k, v in history.items()}
            result = summarize_cell(row, times, rec, quick)
            results[identity(row)] = result
            current.append(result)
            if result["status"] == "running":
                keep.append(i)
        if checkpoint:
            checkpoint(current, seconds)
        if len(keep) != len(active) and keep:
            indices = np.array([2*i+j for i in keep for j in (0, 1)])
            carry = jax.tree.map(lambda x: x[indices], carry)
            history = {k: list(np.asarray(v)[:, indices]) for k, v in history.items()}
        active = [active[i] for i in keep]
    return [results[identity(r)] for r in rows], elapsed


def feature(c, name):
    return {"U/s": c["U"]/c["s"], "NU": c["N"]*c["U"], "Ns": c["Ns"], "U": c["U"]}[name]


def predict(fit, cell):
    logy = fit["log_a"] + sum(p*np.log(feature(cell, k)) for k, p in zip(fit["variables"], fit["exponents"]))
    return float(np.exp(np.clip(logy, -700, 700)))


def fit_models(cells, response):
    confirmed = [c for c in cells if c["confirmed"]]
    rows = [c for c in confirmed if c["stats"][response]["mean"] > 0]
    result = dict(cells=len(rows), excluded_nonpositive=len(confirmed)-len(rows), fits={}, winner=None)
    if len({r["N"] for r in rows}) < 3:
        return dict(**result, status="unresolved: need positive equilibrated cells at >=3 N")
    variables = ("U/s", "NU", "Ns", "U")
    candidates = [(x,) for x in variables] + list(combinations(variables, 2)) if response == "G" else [("Ns",), ("Ns", "U")]
    y = np.array([r["stats"][response]["mean"] for r in rows])
    logy = np.log(y)
    for names in candidates:
        x = np.array([[1., *[np.log(feature(r, k)) for k in names]] for r in rows])
        folds = [np.array([r["N"] == N for r in rows]) for N in sorted({r["N"] for r in rows})]
        if any(np.linalg.matrix_rank(z) < x.shape[1] for z in [x, *[x[~test] for test in folds]]):
            continue
        beta = np.linalg.lstsq(x, logy, rcond=None)[0]
        held = np.empty(len(y))
        for test in folds:
            held[test] = x[test] @ np.linalg.lstsq(x[~test], logy[~test], rcond=None)[0]
        result["fits"][" + ".join(names)] = dict(variables=list(names), log_a=float(beta[0]), a=float(np.exp(beta[0])),
            exponents=beta[1:].tolist(), rmse=float(np.sqrt(np.mean((y-np.exp(x@beta))**2))),
            cv_rmse=float(np.sqrt(np.mean((y-np.exp(np.clip(held, -700, 700)))**2))),
            cv_log_rmse=float(np.sqrt(np.mean((logy-held)**2))),
            held_out=[dict(N=r["N"], Ns=r["Ns"], u=r["u"], observed=float(v), predicted=float(np.exp(h)))
                      for r, v, h in zip(rows, y, held)])
    if not result["fits"]:
        return dict(**result, status="unresolved: rank-deficient designs (need both Ns for D)")
    best = min(f["cv_rmse"] for f in result["fits"].values())
    tied = [k for k, f in result["fits"].items() if f["cv_rmse"] <= best+1e-10]
    result["winner"] = min(tied, key=lambda k: len(result["fits"][k]["variables"]))
    result["adequate"] = result["fits"][result["winner"]]["cv_rmse"] <= .02
    result["status"] = "descriptive equilibrium fit" if result["adequate"] else "poor held-out fit (> .02); thresholds unsupported"
    return result


def ns_exponent(fit):
    return sum(p*(-1 if k == "U/s" else 1 if k == "Ns" else 0)
               for k, p in zip(fit["variables"], fit["exponents"]))


def crossing(fits, N, U, target):
    powers = [ns_exponent(f) for f in fits]
    if any(p > 1e-10 for p in powers) or not any(p < -1e-10 for p in powers):
        return None
    def residual(logns):
        ns = np.exp(logns)
        return sum(predict(f, dict(N=N, Ns=ns, s=ns/N, U=U)) for f in fits)-(1-target)
    if residual(-30) <= 0 or residual(40) >= 0:
        return None
    return float(np.exp(brentq(residual, -30, 40)))


def thresholds(cells, gap, deficit):
    if not deficit["winner"]:
        return []
    d = deficit["fits"][deficit["winner"]]
    g = gap["fits"].get(gap["winner"])
    result = []
    for kind in (("consensus", "individual") if g else ("consensus",)):
        for N in ([None] if kind == "consensus" else sorted({c["N"] for c in cells})):
            for u in sorted({c["u"] for c in cells}):
                U = 784*u
                for target in (.95, .99):
                    ns = crossing([d] if kind == "consensus" else [d, g], N or 1, U, target)
                    support = [c["Ns"] for c in cells if c["confirmed"] and c["u"] == u and (N is None or c["N"] == N)]
                    inside = bool(ns is not None and len(set(support)) >= 2 and min(support) <= ns <= max(support))
                    result.append(dict(kind=kind, N=N, u=u, U=U, target=target, Ns=ns,
                        s_over_U=ns/(N*U) if ns is not None and N is not None else None,
                        region="interpolated" if inside else "extrapolated" if ns is not None else "unresolved",
                        adequate=deficit["adequate"] and (kind == "consensus" or gap["adequate"])))
    return result


def analysis(data):
    cells = data["cells"]
    gap, deficit = (fit_models(cells, key) for key in ("G", "D"))
    return dict(gap=gap, deficit=deficit, thresholds=thresholds(cells, gap, deficit))


def summary(data):
    cells, a = data["cells"], data["analysis"]
    print("F4 " + ("QUICK SMOKE TEST — no equilibrium or selection-threshold inference" if data["quick"] else "variation load and selection strength"))
    print(f"Completed {sum(c['status'] != 'running' for c in cells)}/{len(data['planned'])} cells; "
          f"met={sum(c['confirmed'] for c in cells)}, cap reached={sum(c['status'] == 'cap reached' for c in cells)}; sigma_n={data['sigma_n']}")
    print("Paired starts, last-quarter tails; both mean and consensus: starts and late-tail changes within .01 including 95% replicate intervals.")
    print("N    Ns      u       U       T status       mean    cons.   best    G       D       distance seg.sites var(z)")
    for c in sorted(cells, key=identity):
        m = {k: v["mean"] for k, v in c["stats"].items()}
        print(f"{c['N']:4d} {c['Ns']:4g} {c['u']:.1e} {c['U']:.4f} {c['completed_T']:6d} {c['status']:11s} "
              + " ".join(f"{m[k]:.5f}" for k in ("efficiency", "consensus", "best", "G", "D", "genetic_distance"))
              + f" {m['segregating_sites']:.1f} {m['z_variance']:.5f}")
    print("Unresolved cells use random-start tails; confirmed cells pool paired starts. Both arms and replicate SE are in JSON.")
    print("Segregating sites = unequal normalized coordinates, not independent loci; normalization spreads coordinate differences.")
    for label, result in (("G", a["gap"]), ("D", a["deficit"])):
        print(f"{label}: {result['status']}; {result['cells']} fitted cells, {result['excluded_nonpositive']} nonpositive excluded.")
        for name, f in result["fits"].items():
            print(f"  {name:12s} a={f['a']:.6g}, exponents={np.round(f['exponents'], 4).tolist()}, "
                  f"LO-N-out RMSE={f['cv_rmse']:.6g}, log RMSE={f['cv_log_rmse']:.5g}" + (" BEST" if name == result["winner"] else ""))
    g = a["gap"]
    if g["winner"]:
        print(f"Best collapsing scaling: G = {g['fits'][g['winner']]['a']:.6g} * " +
              " * ".join(f"({k})^{p:.4g}" for k, p in zip(g['fits'][g['winner']]['variables'], g['fits'][g['winner']]['exponents'])))
        if g["winner"] == "U/s" and g["fits"][g["winner"]]["exponents"][0] > 0:
            f = g["fits"][g["winner"]]
            print(f"At fixed consensus deficit D<1-target: s/U >= ({f['a']:.6g}/(1-target-D))^(1/{f['exponents'][0]:.5g}).")
        else:
            print("No universal s/U condition is established by the winning law; use N,u-specific total-load thresholds below.")
    print("Revised numbers: invert D for consensus and D+G for average individuals; s=Ns/N. Conditional descriptive fits, not confidence bounds.")
    for r in a["thresholds"]:
        value = "no decreasing finite crossing" if r["Ns"] is None else f"Ns >= {r['Ns']:.6g}" + (f", s/U >= {r['s_over_U']:.6g}" if r["N"] else "")
        print(f"  {r['kind']:10s} N={r['N']} u={r['u']:.1e} target={r['target']:.2f}: {value}; {r['region']}" + ("; POOR FIT" if not r["adequate"] else ""))
    if not a["thresholds"]:
        print("VERDICT: 95%/99% selection thresholds unresolved; need equilibrated cells across >=3 N and both Ns.")
    else:
        print("VERDICT: conditional selection thresholds above; capped cells remain unresolved and are excluded from fits.")
    scope = (f"{len({r['Ns'] for r in data['planned']})} Ns values constrain functional shape"
             if "grid_spec" in data else "Only two Ns constrain functional shape")
    print(scope + "; collapse is descriptive. Algebraic aliases are not causal evidence; census N is not calibrated Ne.")


def save(data, stem, echo=False):
    data["cells"].sort(key=identity)
    data["analysis"] = analysis(data)
    stem.parent.mkdir(parents=True, exist_ok=True)
    target = stem.with_suffix(".json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n")
    temporary.replace(target)
    stream = StringIO()
    with redirect_stdout(stream):
        summary(data)
    stem.with_name(stem.name+"-summary.txt").write_text(stream.getvalue())
    if echo:
        print(stream.getvalue(), end="")


def merge_data(parts):
    if len(parts) != 2 or {p["part"] for p in parts} != {0, 1}:
        raise ValueError("Need exactly parts 0 and 1")
    first = parts[0]
    fields = ("experiment", "version", "quick", "seed", "generations", "sigma_n", "environment", "calibration", "chunk_generations")
    if any(any(p[k] != first[k] for k in fields) for p in parts):
        raise ValueError("Part metadata/environment mismatch")
    for p in parts:
        expected = grid(p["seed"], p["quick"], p["generations"] or 20, p["part"])
        if p["planned"] != expected or {identity(c) for c in p["cells"]} != {identity(r) for r in expected} or pending(p):
            raise ValueError("Incomplete or mismatched part; resume it before merging")
        if len(p["cells"]) != len(expected):
            raise ValueError("Duplicate cells")
    data = deepcopy(first)
    data.update(part=None, planned=grid(first["seed"], first["quick"], first["generations"] or 20),
                cells=sorted([c for p in parts for c in p["cells"]], key=identity),
                elapsed_seconds=sum(p["elapsed_seconds"] for p in parts))
    return data


def output_stem(args, quick=False):
    name = args.output or OUTPUT / ("code4" + (f"-part{args.part}" if args.part is not None else ""))
    if quick and not name.name.endswith("-quick"):
        name = name.with_name(name.name+"-quick")
    return name


def main(args, quick=False):
    stem = output_stem(args, quick)
    if not quick and jax.default_backend() != "gpu":
        raise RuntimeError("Full run requires GPU; use quick or bench --tiny on CPU")
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    spec = custom_spec(args, quick)
    planned = grid(args.seed, quick, args.generations, args.part, spec)
    chunk = min(args.chunk_generations, args.generations) if quick else args.chunk_generations
    data = new_data(planned, quick, args.part, env, chunk, spec)
    if args.resume:
        if not stem.with_suffix(".json").exists():
            raise ValueError("No JSON checkpoint at requested output stem")
        previous = json.loads(stem.with_suffix(".json").read_text())
        validate_resume(previous, data)
        data = previous
    elif stem.with_suffix(".json").exists():
        raise ValueError("Output exists; use --resume or a new --output stem")
    engine = Engine(env, args.individual_chunk)
    def checkpoint(cells, seconds):
        replacement = {identity(c): c for c in cells}
        replacement.update({identity(c): c for c in data["cells"] if identity(c) not in replacement})
        data["cells"] = list(replacement.values())
        data["elapsed_seconds"] += seconds
        save(data, stem)
        print(f"N={cells[0]['N']} T={cells[0]['completed_T']}: " +
              ", ".join(f"Ns={c['Ns']:g} u={c['u']:g} {c['status']}" for c in cells) + f" ({seconds:.2f}s)", flush=True)
    for batch in pending_batches(data, args.config_chunk):
        execute(engine, batch, quick, chunk, checkpoint)
    save(data, stem, echo=True)
    figure(stem)


def bench(args):
    gpu = jax.default_backend() == "gpu"
    if not gpu and not args.tiny:
        raise ValueError("CPU benchmark requires --tiny; representative throughput belongs on GPU")
    spec = custom_spec(args)
    if spec is not None:
        return bench_grid(args, spec)
    N, K, T, every = (4, 2, 4, 1) if args.tiny else (1000, 8, 1000, 1000)
    row = dict(**f2.config(N, 300/N, K=K, T=T, every=every, seed=args.seed), Ns=300, cap=T, mode="clonal")
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    engine = Engine(env, args.individual_chunk)
    _, cold = execute(engine, [row], True, T)
    _, warm = execute(engine, [row], True, T)
    rate = 2*N*K*T/warm
    parts = []
    for part in (0, 1):
        rows = grid(args.seed, part=part)
        initial = sum(2*r["N"]*r["K"]*min(r["cap"], int(np.ceil(r["T"]/args.chunk_generations))*args.chunk_generations)
                      for r in rows)
        cap = sum(2*r["N"]*r["K"]*r["cap"] for r in rows)
        parts.append(dict(part=part, initial_work=initial, cap_work=cap,
                          initial_hours=initial/rate/3600, cap_hours=cap/rate/3600))
    result = dict(device=str(jax.devices()[0]), tiny=args.tiny, N=N, K=K, T=T,
                  cold_seconds=cold, warm_seconds=warm, rate=rate, parts=parts,
                  note="Same-device single-cell extrapolation, excluding setup/IO/other-shape compilation; adaptive runtime lies between schedule and cap. Tiny rates are smoke tests, NOT pod throughput.")
    stem = args.output or OUTPUT / ("code4-bench-quick" if args.tiny else "code4-bench")
    if args.tiny and not stem.name.endswith("-quick"):
        stem = stem.with_name(stem.name+"-quick")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


def scheduled_T(row, chunk):
    return min(row["cap"], ((row["T"]+chunk-1)//chunk)*chunk)


def bench_grid(args, spec):
    """Requested K, cold compile then warm timing at two representative shapes."""
    rows = grid(args.seed, spec=spec)
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    engine = Engine(env, args.individual_chunk)
    probes = []
    for N in ((4,) if args.tiny else (1000, 3000)):
        T, every = (4, 1) if args.tiny else (1000, 1000)
        row = dict(**f2.config(N, spec["ns"][0]/N, K=spec["replicates"],
                              u=max(spec["mutations"]), T=T, every=every, seed=args.seed),
                   Ns=spec["ns"][0], cap=T, mode="clonal")
        _, cold = execute(engine, [row], True, T)
        _, warm = execute(engine, [row], True, T)
        probes.append(dict(N=N, K=row["K"], T=T, cold_seconds=cold, warm_seconds=warm,
                           rate=2*N*row["K"]*T/warm))
    estimates = []
    for row in sorted(rows, key=first_work):
        probe = min(probes, key=lambda p: abs(np.log(row["N"]/p["N"])))
        first = scheduled_T(row, args.chunk_generations)
        hours = 2*row["N"]*row["K"]/probe["rate"]/3600
        estimates.append(dict(N=row["N"], Ns=row["Ns"], u=row["u"], K=row["K"],
                              first_T=row["T"], scheduled_T=first, cap=row["cap"],
                              probe_N=probe["N"], scheduled_hours=hours*first, cap_hours=hours*row["cap"]))
    memory = jax.devices()[0].memory_stats() or {}
    result = dict(device=str(jax.devices()[0]), tiny=args.tiny, grid_spec=spec, probes=probes,
                  individual_chunk=args.individual_chunk, config_chunk=args.config_chunk,
                  peak_bytes_in_use=memory.get("peak_bytes_in_use"), cells=estimates,
                  scheduled_hours=sum(r["scheduled_hours"] for r in estimates),
                  cap_hours=sum(r["cap_hours"] for r in estimates),
                  note="Warm paired single-cell estimates at requested K, nearest probe N in log space; "
                       "excludes setup/IO/other-shape compilation. Uses highest requested u. "
                       "Larger config-chunk throughput is not measured. Tiny rates are smoke tests, NOT GPU throughput.")
    stem = args.output
    if args.tiny and not stem.name.endswith("-quick"):
        stem = stem.with_name(stem.name+"-quick")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(stem=OUTPUT / "code4"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullLocator
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    data = json.loads(stem.with_suffix(".json").read_text())
    cells, result = data["cells"], data["analysis"]["gap"]
    fit = result["fits"].get(result["winner"])
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    style = dict(fontsize=5.5, family="monospace")
    populations = sorted({r["N"] for r in data["planned"]})
    mutations = sorted({r["u"] for r in data["planned"]})
    strengths = sorted({r["Ns"] for r in data["planned"]})
    markers = {ns: ("o", "^", "s", "D", "v", "P", "X")[i % 7] for i, ns in enumerate(strengths)}
    if "grid_spec" not in data:
        markers = {300: "o", 3000: "^"}  # Preserve original part figures/legends.
    ax = axes[0]
    ax.set(xscale="log", yscale="log")
    skipped = 0
    for i, N in enumerate(populations):
        for c in (c for c in cells if c["N"] == N):
            x = (feature(c, fit["variables"][0]) if len(fit["variables"]) == 1 else
                 predict(fit, c)/fit["a"]) if fit else feature(c, "U/s")
            y = c["stats"]["G"]["mean"]
            if y <= 0:
                skipped += 1
                continue
            ax.plot(x, y, markers[c["Ns"]], color=f"C{i}", ms=3,
                    fillstyle="full" if c["confirmed"] else "none")
    if fit:
        limits = ax.get_xlim()
        x = np.geomspace(max(limits[0], 1e-15), limits[1], 100)
        power = fit["exponents"][0] if len(fit["variables"]) == 1 else 1
        ax.plot(x, fit["a"]*x**power, color="C0", lw=.6, alpha=.5)
        label = fit["variables"][0] if len(fit["variables"]) == 1 else " ".join(
            f"({k})^{p:.2g}" for k, p in zip(fit["variables"], fit["exponents"]))
    else:
        label = "U/s (fit unresolved)"
    ax.set(xscale="log", yscale="log", xlabel=label, ylabel="gap G", title="variation load")
    legend_style = dict(prop=dict(size=5.5, family="monospace"), frameon=False,
                        ncol=2, handlelength=1.5, handletextpad=.4,
                        columnspacing=.7, labelspacing=.3, borderpad=.3)
    handles = [Line2D([], [], color=f"C{i}", lw=1, label=f"N={N}")
               for i, N in enumerate(populations)]
    handles += [Line2D([], [], color="black", marker=marker, linestyle="none",
                       ms=3, label=f"N·s={Ns}") for Ns, marker in markers.items()]
    ax.legend(handles=handles, loc="lower right", **legend_style)
    ax = axes[1]
    colors = plt.get_cmap("Purples")(np.linspace(.4, .95, len(mutations)))
    linestyles = {ns: ("-", "--", ":", "-.")[i % 4] for i, ns in enumerate(strengths)}
    if "grid_spec" not in data:
        linestyles = {300: "-", 3000: "--"}
    for i, u in enumerate(mutations):
        for Ns, linestyle in linestyles.items():
            group = sorted((c for c in cells if c["u"] == u and c["Ns"] == Ns),
                           key=lambda c: c["N"])
            ax.plot([c["N"] for c in group], [c["stats"]["D"]["mean"] for c in group],
                    color=colors[i], linestyle=linestyle, lw=.5, zorder=1)
            for c in group:
                ax.plot(c["N"], c["stats"]["D"]["mean"], markers[Ns], ms=3,
                        color=colors[i], fillstyle="full" if c["confirmed"] else "none")
    handles = [Line2D([], [], color=colors[i], lw=1, label=f"u={u:.0e}")
               for i, u in enumerate(mutations)]
    handles += [Line2D([], [], color="black", linestyle=linestyle, lw=.5,
                       label=f"N·s={Ns}") for Ns, linestyle in linestyles.items()]
    ax.legend(handles=handles, loc="best", **legend_style)
    ax.set(xscale="log", yscale="linear", xlabel="N", ylabel="consensus deficit D",
           title="consensus deficit")
    ax.set_xticks(populations, labels=[str(n) for n in populations])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax = axes[2]
    columns = sorted({(r["Ns"], r["u"]) for r in data["planned"]})
    lookup = {identity(c): c for c in cells}
    for iy, N in enumerate(populations):
        for ix, (Ns, u) in enumerate(columns):
            c = lookup.get((N, Ns, u))
            status = c["status"] if c else "pending"
            color = f"C{0 if status == 'met' else 1}"
            ax.scatter(ix, iy, marker="s", s=220 if len(columns) < 4 else 110,
                       facecolors=color, edgecolors=color, alpha=.18 if status != "met" else .65)
            ax.text(ix, iy, {"met": "met", "cap reached": "cap", "running": "...", "pending": "—"}[status],
                    ha="center", va="center", **style)
    ax.set_xticks(range(len(columns)), labels=[f"{n}\n{u:.0e}" for n, u in columns], rotation=90)
    ax.set_yticks(range(len(populations)), labels=[str(n) for n in populations])
    ax.set(xlim=(-.6, len(columns)-.4), ylim=(len(populations)-.5, -.5),
           xlabel="N·s / u", ylabel="N", title="equilibrium: met / cap")
    ax.grid(False)
    for ax, letter in zip(axes, "ABC"):
        panel_letter(ax, letter)
        ax.tick_params(labelsize=5.5)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(7)
        ax.title.set_size(7)
    note = "SMOKE TEST: no equilibrium inference" if data["quick"] else "open points: unresolved; fits use confirmed cells only"
    if skipped:
        note += f"; {skipped} nonpositive G omitted on log axis"
    fig.text(.5, .015, note, ha="center", **style)
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.83, wspace=.48)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def check(args):
    begin = perf_counter()
    full = grid()
    split = [grid(part=p) for p in (0, 1)]
    assert len(full) == 32 and len(split[0]) == len(split[1]) == 16
    assert set(map(identity, split[0])).isdisjoint(map(identity, split[1]))
    assert set(map(identity, full)) == set(map(identity, split[0]+split[1]))
    assert all(r["K"] == 8 and r["L"] == 3 and r["cap"] == CAP for r in full)
    custom_args = argparse.Namespace(populations=[3000, 10, 1000], ns=[1000., 30., 100.],
        mutations=[.001, .0001], replicates=32, cap=250000, exclude=["3000:30", "10:100"],
        part=None, output=Path("unused"))
    spec = custom_spec(custom_args)
    custom = grid(spec=spec)
    assert len(custom) == 14 and all(r["K"] == 32 for r in custom)
    assert not any((r["N"], r["Ns"]) in ((3000, 30), (10, 100)) for r in custom)
    assert all(r["T"] == min(250000, int(np.ceil(max(20000, 20000*r["N"]/r["Ns"])/1000))*1000)
               for r in custom)
    assert {r["T"] for r in custom} >= {20000, 200000, 250000}
    for chunk_size in (2, 4, 8):
        ordered = [r for batch in pending_batches(dict(planned=custom, cells=[]), chunk_size) for r in batch]
        assert list(map(first_work, ordered)) == sorted(map(first_work, custom))
    assert scheduled_T(dict(T=21000, cap=25000), 10000) == 25000
    for changes in (dict(part=0), dict(output=None), dict(replicates=1), dict(cap=250001),
                    dict(ns=[0]), dict(ns=[float("nan")]), dict(mutations=[1.1]),
                    dict(populations=[10, 10]), dict(exclude=["bad"]), dict(exclude=["7:30"]),
                    dict(exclude=[f"{n}:{ns}" for n in custom_args.populations for ns in custom_args.ns])):
        try:
            custom_spec(argparse.Namespace(**(vars(custom_args) | changes)))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid custom spec accepted: {changes}")
    synthetic = []
    for r in full:
        U = 784*r["u"]
        G, D = .02*(U/r["s"])**.7, 2/r["Ns"]
        synthetic.append(dict(**r, U=U, confirmed=True, stats={"G": dict(mean=G), "D": dict(mean=D)}))
    gap, deficit = (fit_models(synthetic, k) for k in ("G", "D"))
    assert gap["winner"] == "U/s" and deficit["winner"] == "Ns"
    assert np.allclose(gap["fits"]["U/s"]["exponents"], [.7])
    assert gap["fits"]["U/s"]["cv_rmse"] < 1e-12
    for r in thresholds(synthetic, gap, deficit):
        c = dict(N=r["N"] or 1, Ns=r["Ns"], s=r["Ns"]/(r["N"] or 1), U=r["U"])
        load = predict(deficit["fits"]["Ns"], c)
        if r["kind"] == "individual":
            load += predict(gap["fits"]["U/s"], c)
            assert np.isclose(r["s_over_U"], c["s"]/r["U"])
        assert np.isclose(load, 1-r["target"])
    assert not fit_models([c for c in synthetic if c["Ns"] == 300], "D")["winner"]
    bad = deepcopy(synthetic)
    bad[0]["stats"]["G"]["mean"] = -1
    assert fit_models(bad, "G")["excluded_nonpositive"] == 1
    # A stable optimum at min T passes; a consensus-only disagreement fails.
    row = full[0]
    times = np.arange(0, row["T"]+1, row["every"]).tolist()
    records = {k: np.ones((len(times), 2, 8))*.99 for k in METRICS}
    assert summarize_cell(row, times, records)["confirmed"]
    records["consensus"][:, 1] -= .1
    assert not summarize_cell(row, times, records)["confirmed"]
    assert not summarize_cell(row, times, records, quick=True)["confirmed"]
    # Real tiny executions: splitting and merging must reproduce a single run.
    env = f2.channel("vanhateren", args.contrasts, args.seed, f2.DEFAULT_SIGMA_N)
    requested = new_data(custom, False, None, env, 10000, spec)
    roundtrip = json.loads(json.dumps(requested))
    validate_resume(roundtrip, requested)
    for key in ("grid_spec", "planned", "sigma_n", "environment", "calibration", "chunk_generations"):
        bad = deepcopy(roundtrip)
        bad[key] = None
        try:
            validate_resume(bad, requested)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Resume accepted changed {key}")
    engine = Engine(env, 8)
    planned = grid(args.seed, True, 2)
    single = new_data(planned, True, None, env, 1)
    for batch in f2.groups(planned, 1):
        values, _ = execute(engine, batch, True, 1)
        single["cells"].extend(values)
    parts = []
    for p in (0, 1):
        plan = grid(args.seed, True, 2, p)
        part = new_data(plan, True, p, env, 1)
        for batch in f2.groups(plan, 1):
            values, _ = execute(engine, batch, True, 1)
            part["cells"].extend(values)
        parts.append(part)
    merged = merge_data(parts)
    assert merged["cells"] == sorted(single["cells"], key=identity)
    assert analysis(merged) == analysis(single)
    assert not pending(json.loads(json.dumps(merged)))  # resume schedules zero work
    validate_resume(json.loads(json.dumps(merged)), new_data(planned, True, None, env, 1))
    partial = deepcopy(merged)
    partial["cells"][0]["status"] = "running"
    assert list(map(identity, pending(partial))) == [identity(partial["cells"][0])]
    broken = deepcopy(parts)
    broken[0]["cells"].pop()
    try:
        merge_data(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete merge accepted")
    # Chunking preserves the exact trajectory and RNG state.
    a = execute(engine, [planned[0]], True, 1)[0][0]
    b = execute(engine, [planned[0]], True, 2)[0][0]
    for k in METRICS:
        assert np.allclose(a["records"][k], b["records"][k], atol=1e-7)
    assert np.all(np.array(a["records"]["segregating_sites"])[0] == 0)
    # Real K=32 smoke, batched paired cells and a remainder individual chunk.
    small_args = argparse.Namespace(**(vars(custom_args) | dict(populations=[4], ns=[30, 100],
        mutations=[.001], exclude=None)))
    small_spec = custom_spec(small_args, quick=True)
    small = grid(args.seed, True, 2, spec=small_spec)
    batched, _ = execute(engine, small, True, 1)
    unbatched, _ = execute(Engine(env, 7), [small[0]], True, 2)
    assert all(np.asarray(c["records"]["efficiency"]).shape == (3, 2, 32) for c in batched)
    for key in METRICS:
        assert np.allclose(batched[0]["records"][key], unbatched[0]["records"][key], atol=1e-6)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="f4-check-") as directory:
        stem = Path(directory)/"custom"
        checkpoint = new_data(small, True, None, env, 1, small_spec)
        checkpoint["cells"] = batched
        save(checkpoint, stem)
        restored = json.loads(stem.with_suffix(".json").read_text())
        validate_resume(restored, new_data(small, True, None, env, 1, small_spec))
        assert not pending(restored) and not stem.with_suffix(".json.tmp").exists()
        for field, value in (("replicates", 8), ("cap", 300000), ("exclude", [[4, 30]])):
            changed = deepcopy(checkpoint)
            changed["grid_spec"][field] = value
            try:
                validate_resume(restored, changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Resume accepted changed {field}")
        restored["cells"].append(restored["cells"][0])
        try:
            validate_resume(restored, checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("Resume accepted duplicate cells")
    print(f"check ok ({perf_counter()-begin:.2f}s): custom grid, work order, resume validation, atomic checkpoint, K=32 batching, split/merge execution, chunk continuity, convergence gate, synthetic scaling/exponent, thresholds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "quick", "bench", "run", "merge", "summary", "figure"))
    parser.add_argument("--part", type=int, choices=(0, 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contrasts", type=Path, default=OUTPUT / "contrasts.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-chunk", type=int, default=2, help="Arms per batch; positive even number (paired starts)")
    parser.add_argument("--individual-chunk", type=int, default=128)
    parser.add_argument("--chunk-generations", type=int, default=10000)
    parser.add_argument("--generations", type=int, default=20, help="Quick only")
    parser.add_argument("--tiny", action="store_true", help="Tiny bench smoke test; required on CPU")
    parser.add_argument("--output", type=Path, help="Output stem without extension; quick appends -quick")
    parser.add_argument("--inputs", type=Path, nargs=2, help="Merge input JSONs; defaults to output/code4-part{0,1}.json")
    parser.add_argument("--populations", type=int, nargs="+", help="Custom N list")
    parser.add_argument("--ns", type=float, nargs="+", help="Custom N*s list")
    parser.add_argument("--mutations", type=float, nargs="+", help="Custom per-site u list")
    parser.add_argument("--replicates", type=int, help="Replicates per start (default 8; quick 2)")
    parser.add_argument("--cap", type=int, help="Custom generation cap, multiple of 1000")
    parser.add_argument("--exclude", action="append", help="Drop N:Ns pair, repeatable")
    args = parser.parse_args()
    try:
        spec = custom_spec(args, quick=args.command == "quick")
        if spec is not None and args.command not in ("run", "quick", "bench", "check"):
            parser.error("Custom grid options apply to run, quick, bench, or check")
    except ValueError as error:
        parser.error(str(error))
    if args.config_chunk < 2 or args.config_chunk % 2 or args.individual_chunk < 1:
        parser.error("config-chunk must be positive/even (>=2); individual-chunk must be positive")
    if args.generations < 1 or args.chunk_generations < 1:
        parser.error("generations and chunk-generations must be positive")
    if args.command in ("run", "bench") and args.chunk_generations % 1000:
        parser.error("Full chunk-generations must be a multiple of 1000")
    if args.command == "run" and args.generations != 20:
        parser.error("--generations is quick-only; full schedule is fixed")
    if args.command == "check":
        check(args)
    elif args.command == "bench":
        bench(args)
    elif args.command == "merge":
        paths = args.inputs or [OUTPUT / f"code4-part{p}.json" for p in (0, 1)]
        data = merge_data([json.loads(p.read_text()) for p in paths])
        stem = output_stem(args, data["quick"])
        save(data, stem, echo=True)
        figure(stem)
    elif args.command in ("summary", "figure"):
        stem = output_stem(args)
        if args.command == "figure":
            figure(stem)
        else:
            save(json.loads(stem.with_suffix(".json").read_text()), stem, echo=True)
    else:
        main(args, quick=args.command == "quick")
