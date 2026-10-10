"""F5: CPU-only re-analysis of F4; never simulates or modifies the input.

python code5.py [check|run|figure] [--input output/code4.json] [--output STEM]
run writes output/code5.{json,png,pdf} and output/code5-summary.txt by default;
--output changes the stem. figure re-renders the saved F5 JSON (and checks its source against --input).
Requires numpy/scipy; rendering also needs matplotlib. JAX is not required.

Last-quarter paired replicate means define directed random/optimum endpoints.
Direction checks split the last HALF into two equal-count windows. A negative
width at any recorded time with a pointwise Student-t 95% interval below zero
is reported only: these unadjusted diagnostics do not invalidate a bracket.
Noisy point inversions are retained as diagnostics but are not promoted: a
usable directed bracket must also have ordered point endpoints. Never swap
arms or clip a negative width into an apparently converged bracket.

B preserves confirmed F4 estimates and adds only valid narrow capped cells.
C/D move newly admitted cells to both random/optimum endpoints respectively;
their envelope is sensitivity, not a confidence interval or an extremum over
all endpoint combinations/model choices. G bounds separately use worst-case
combinations. S/S_no10 add stationary cells using last-half pooled windows:
95% drift and arm-difference intervals within +/-0.01, and level half-width
<=0.01. Level/arm uncertainty takes the larger batch and replicate-mean SE;
up to 10 contiguous batches per trajectory, with conservative Student-t df.
S bootstrap resamples whole trajectories within arms, preserving time and
cross-metric dependence. Confirmed F4 estimates remain unchanged in all sets.
Quick input can never promote cells or support inference.
Bootstrap membership is fixed; B/E sample paired replicate indices within
each cell, never individual correlated time records. Model rank/positivity,
folds, scoring, winners, and threshold inversion follow code4 unchanged.
"""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from functools import lru_cache
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

# Set before importing numpy: these tiny designs need only one BLAS thread.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.stats import t as student_t


def import_f4():
    """Import the actual modules without requiring their unused JAX backend.

    This tiny, temporary import shim supplies only names imported at module
    scope. It implements no numerical operations and is removed immediately.
    All simulation entry points still fail, rather than silently using numpy.
    """
    try:
        import code4
        return code4
    except ModuleNotFoundError as error:
        if error.name != "jax":
            raise

    def unavailable(*args, **kwargs):
        raise RuntimeError("F5 is analysis-only; JAX/simulation is unavailable")

    names = ("jax", "jax.numpy", "jax.lax", "jax.random")
    modules = {name: ModuleType(name) for name in names}
    for module in modules.values():
        module.__getattr__ = lambda name: unavailable
    modules["jax"].numpy = modules["jax.numpy"]
    modules["jax"].lax = modules["jax.lax"]
    modules["jax"].random = modules["jax.random"]
    modules["jax"].vmap = unavailable
    saved = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update(modules)
        import code4
        return code4
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


f4 = import_f4()
TOL = .01
BOOTSTRAPS = 2000
SEED = 2315
STEM = Path(__file__).resolve().parent / "output" / "code5"
METRICS = ("efficiency", "consensus")
SETS = {"A": "F4 confirmed only", "B": "confirmed + bracketed midpoints",
        "C": "B at random-start ends", "D": "B at optimum-start ends",
        "E": "B excluding N=10", "A_no10": "A excluding N=10",
        "S": "confirmed + stationary window estimates", "S_no10": "S excluding N=10"}


@lru_cache(None)
def critical(k):
    return float(student_t.ppf(.975, k-1))


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite replicate summaries")
    result = f4.f3.interval(values)
    radius = critical(len(values))*result["se"]
    return dict(**result, low=result["mean"]-radius, high=result["mean"]+radius)


def record_arrays(cell):
    times = np.asarray(cell["times"], dtype=float)
    arrays = {k: np.asarray(cell["records"][k], dtype=float) for k in METRICS}
    if (times.ndim != 1 or len(times) < 4 or not np.isfinite(times).all()
            or times[-1] <= 0 or np.any(np.diff(times) <= 0)
            or times[-1] != cell["completed_T"] or cell["K"] < 2):
        raise ValueError(f"Invalid times/replicates for {f4.identity(cell)}")
    if any(v.shape != (len(times), 2, cell["K"]) or not np.isfinite(v).all()
           for v in arrays.values()):
        raise ValueError(f"Expected finite time x 2 x K records: {f4.identity(cell)}")
    return times, arrays


def width_history(times, arrays):
    """Last-quarter widths at eight prefix endpoints; no simulation/time fitting."""
    history = []
    for index in np.unique(np.linspace(3, len(times)-1, min(8, len(times)-3), dtype=int)):
        mask = (times >= .75*times[index]) & (times <= times[index])
        if mask.sum() < 4:
            continue
        history.append(dict(T=float(times[index]), **{
            k: interval(v[mask, 1].mean(0)-v[mask, 0].mean(0)) for k, v in arrays.items()}))
    return history


def bracket_cell(cell):
    times, arrays = record_arrays(cell)
    tail_mask, half_mask = times >= .75*times[-1], times >= .5*times[-1]
    metrics = {}
    for key, values in arrays.items():
        # F4 may have reduced float32 device records before serializing them.
        # Preserve those saved replicate tails after checking extraction agreement.
        tails = np.array([cell["arms"][arm]["tail"][key] for arm in range(2)], dtype=float)
        for arm in range(2):
            matches = [np.allclose(values.astype(dtype)[tail_mask, arm].mean(0), tails[arm],
                                   rtol=0, atol=1e-12) for dtype in (np.float64, np.float32)]
            if not any(matches):
                raise ValueError(f"Stored F4 tail disagrees with records: {f4.identity(cell)}, {key}")
        late = values[half_mask]
        middle = len(late)//2
        change = late[middle:].mean(0)-late[:middle].mean(0)
        changes = [interval(change[arm]) for arm in range(2)]
        width = interval(tails[1]-tails[0])
        # Pointwise intervals across paired replicates, not across time.
        differences = values[:, 1]-values[:, 0]
        record_high = differences.mean(1) + critical(cell["K"])*differences.std(1, ddof=1)/np.sqrt(cell["K"])
        reversed_times = times[record_high < -1e-12]
        inverted = width["high"] < -1e-12
        reasons = []
        if tail_mask.sum() < 4 or len(late) < 4:
            reasons.append("insufficient tail records")
        if changes[0]["high"] < -1e-12:
            reasons.append("random arm decreasing beyond noise")
        if changes[1]["low"] > 1e-12:
            reasons.append("optimum arm increasing beyond noise")
        if inverted:
            reasons.append("inverted tail beyond 95% interval")
        valid = not reasons
        ordered = width["mean"] >= 0
        midpoint = (tails[0]+tails[1])/2
        metrics[key] = dict(ends=[float(v.mean()) for v in tails],
            endpoint_intervals=[interval(v) for v in tails], tails=tails.tolist(),
            midpoint=interval(midpoint), half_width=width["mean"]/2 if ordered else None,
            width=width, tail_change=changes, tail_records=int(tail_mask.sum()),
            last_half_records=int(half_mask.sum()), valid=valid, ordered=ordered,
            narrow=bool(valid and ordered and width["high"] <= TOL),
            inverted=bool(inverted), reversed_records=len(reversed_times),
            first_reversal_T=float(reversed_times[0]) if len(reversed_times) else None,
            noisy_reversed_records=int(np.sum((differences.mean(1) < 0) & (record_high >= -1e-12))),
            reasons=reasons)
    mean, consensus = (metrics[k] for k in METRICS)
    mid = {k: np.mean(v["tails"], axis=0) for k, v in metrics.items()}
    directed = all(v["valid"] and v["ordered"] for v in metrics.values())
    result = {k: cell[k] for k in ("N", "Ns", "u", "U", "s", "K", "completed_T")}
    result.update(f4_status=cell["status"], f4_confirmed=cell["confirmed"], metrics=metrics,
        valid=all(v["valid"] for v in metrics.values()),
        bracketed=all(v["narrow"] for v in metrics.values()),
        derived={"G": dict(midpoint=interval(mid["consensus"]-mid["efficiency"]),
                    bounds=[consensus["ends"][0]-mean["ends"][1],
                            consensus["ends"][1]-mean["ends"][0]] if directed else None),
                 "D": dict(midpoint=interval(1-mid["consensus"]),
                    bounds=[1-consensus["ends"][1], 1-consensus["ends"][0]] if directed else None)},
        width_history=width_history(times, arrays))
    return result


def window_interval(values):
    """Time x trajectory summaries; never treat correlated records as replicates.

    Contiguous, approximately equal-count batches include every observation.
    Batch variance estimates each trajectory mean's variance, then propagates
    across trajectories. Compare with between-trajectory window-mean variance.
    Use the smaller batch/trajectory df, even when the replicate SE wins.
    """
    values = np.asarray(values, dtype=float)
    means = values.mean(0)
    count = min(10, len(values)//2)
    if count < 2:
        raise ValueError("Need at least four window records for batch means")
    chunks = np.array_split(values, count)
    batches = np.array([v.mean(0) for v in chunks])
    weights = np.array([len(v)/len(values) for v in chunks])
    batch_se = float(np.sqrt(np.sum(batches.var(0, ddof=1))*np.sum(weights**2))/len(means))
    replicate_se = interval(means)["se"]
    se = max(batch_se, replicate_se)
    df = min(len(means)-1, count-1)
    half_width = critical(df+1)*se
    mean = float(means.mean())
    return dict(mean=mean, se=se, low=mean-half_width, high=mean+half_width,
                half_width=half_width, batch_se=batch_se, replicate_se=replicate_se,
                batches=count, batch_sizes=[len(v) for v in chunks], df=df)


def drift_interval(times, values, window_length):
    centered = times-times.mean()
    # Normalize each OLS slope to a change over the nominal last-half window.
    drift = np.tensordot(centered, values, axes=(0, 0))/np.dot(centered, centered)*window_length
    return interval(drift.reshape(-1))


def equivalent_interval(stats):
    return bool(stats["low"] >= -TOL and stats["high"] <= TOL)


def stationary_cell(cell):
    times, arrays = record_arrays(cell)
    mask = times >= .5*times[-1]
    metrics, windows = {}, {}
    for key, values in arrays.items():
        late = values[mask]
        windows[key] = late.mean(0)
        if len(late) < 4:
            metrics[key] = dict(estimate=float(late.mean()), half_width=None, drift=None,
                arm_difference=None, admitted=False, reasons=["insufficient window records"],
                significant_drift=False, wide_half_width=False)
            continue
        drift = drift_interval(times[mask], late, .5*times[-1])
        # Paired start differences follow F3, now using window/batch means.
        difference = window_interval(late[:, 1]-late[:, 0])
        level = window_interval(late.reshape(len(late), -1))
        trend_ok, arms_ok = equivalent_interval(drift), equivalent_interval(difference)
        wide = level["half_width"] > TOL
        significant = not trend_ok and (drift["low"] > 0 or drift["high"] < 0)
        reasons = []
        if not trend_ok:
            reasons.append("significant drift: longer runs" if significant else
                           "drift interval too wide: more replicates to resolve trend")
        if not arms_ok:
            reasons.append("arm difference excludes +/-0.01: starts have not met; no time estimate without drift"
                           if difference["low"] > TOL or difference["high"] < -TOL else
                           "arm-difference interval too wide: more replicates")
        if wide:
            reasons.append("wide level half-width: more replicates")
        metrics[key] = dict(estimate=level["mean"], half_width=level["half_width"],
            level=level, drift=drift, arm_difference=difference,
            trajectory_means=windows[key].tolist(), admitted=not reasons, reasons=reasons,
            significant_drift=bool(significant), wide_half_width=bool(wide))
    derived = {}
    if mask.sum() >= 4:
        for key, values in (("G", arrays["consensus"]-arrays["efficiency"]),
                            ("D", 1-arrays["consensus"])):
            late = values[mask]
            derived[key] = window_interval(late.reshape(len(late), -1))
    history = []
    for index in np.unique(np.linspace(3, len(times)-1, min(8, len(times)-3), dtype=int)):
        prefix = (times >= .5*times[index]) & (times <= times[index])
        if prefix.sum() >= 4:
            history.append(dict(T=float(times[index]), **{
                k: drift_interval(times[prefix], v[prefix], .5*times[index]) for k, v in arrays.items()}))
    return dict(window_start=float(.5*times[-1]), window_end=float(times[-1]),
                window_records=int(mask.sum()), trajectories=2*cell["K"], metrics=metrics,
                derived=derived, stationary=all(m["admitted"] for m in metrics.values()),
                drift_history=history)


def stationary_fit_cell(cell, bracket):
    result = dict(cell)
    if bracket["stationary"]["admitted"]:
        windows = {k: np.asarray(bracket["stationary"]["metrics"][k]["trajectory_means"])
                   for k in METRICS}
        windows.update(G=windows["consensus"]-windows["efficiency"], D=1-windows["consensus"])
        stats = {k: bracket["stationary"]["metrics"][k]["level"] for k in METRICS}
        stats.update(bracket["stationary"]["derived"])
        result.update(confirmed=True, stats=stats,
                      tail={k: v.reshape(-1).tolist() for k, v in windows.items()})
        result["bootstrap_arms"] = windows["D"].tolist()
    elif cell["confirmed"]:
        # F4's published mean is the pooled last-quarter mean. Resampling each
        # arm's trajectory tail preserves that estimator without time resampling.
        result["bootstrap_arms"] = (1-np.asarray([a["tail"]["consensus"] for a in cell["arms"]])).tolist()
    return result


def fit_cell(cell, bracket, end=None):
    """Only newly admitted cells change estimates; original F4 stats stay intact."""
    result = {k: cell[k] for k in ("N", "Ns", "u", "U", "s", "K", "confirmed", "stats", "tail")}
    if bracket["admitted"]:
        tails = {k: np.asarray(bracket["metrics"][k]["tails"]) for k in METRICS}
        tails = {k: v.mean(0) if end is None else v[end] for k, v in tails.items()}
        tails.update(G=tails["consensus"]-tails["efficiency"], D=1-tails["consensus"])
        result.update(confirmed=True, tail={k: v.tolist() for k, v in tails.items()},
                      stats={k: f4.f3.interval(v) for k, v in tails.items()})
    return result


def build_sets(cells, brackets):
    b = [fit_cell(c, r) for c, r in zip(cells, brackets)]
    stationary = [stationary_fit_cell(c, r) for c, r in zip(cells, brackets)]
    return {"A": cells, "B": b,
            "C": [fit_cell(c, r, 0) for c, r in zip(cells, brackets)],
            "D": [fit_cell(c, r, 1) for c, r in zip(cells, brackets)],
            "E": [c for c in b if c["N"] != 10],
            "A_no10": [c for c in cells if c["N"] != 10],
            "S": stationary, "S_no10": [c for c in stationary if c["N"] != 10]}


def bootstrap_distribution(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                ci95=np.quantile(values, [.025, .975]).tolist())


def bootstrap_u(cells, resamples=BOOTSTRAPS, seed=SEED):
    """Batched log least squares equals F4, including leave-one-N-out folds.

    Reuse each fixed design for all positive draws; uncommon nonpositive draws
    go through fit_models directly so F4's positive-subset/rank rules still hold.
    """
    baseline = f4.fit_models(cells, "D")
    required = ("Ns", "Ns + U")
    result = dict(seed=seed, requested=resamples, valid_resamples=0,
                  baseline=baseline, status="unresolved: both D models required")
    if not all(k in baseline["fits"] for k in required):
        return result
    rows = [c for c in cells if c["confirmed"]]
    rng = np.random.default_rng(seed)
    samples = []
    for c in rows:
        if "bootstrap_arms" in c:
            arms = np.asarray(c["bootstrap_arms"])
            samples.append(np.mean([arm[rng.integers(len(arm), size=(resamples, len(arm)))].mean(1)
                                    for arm in arms], axis=0))
        else:
            samples.append(np.asarray(c["tail"]["D"])[rng.integers(c["K"], size=(resamples, c["K"]))].mean(1))
    y = np.array(samples)
    good = np.all(y > 0, axis=0)
    exponents, delta = np.full(resamples, np.nan), np.full(resamples, np.nan)
    rmse = []
    if good.any():
        logy = np.log(y[:, good])
        for names in (("Ns",), ("Ns", "U")):
            x = np.array([[1., *[np.log(f4.feature(c, k)) for k in names]] for c in rows])
            folds = [np.array([c["N"] == n for c in rows]) for n in sorted({c["N"] for c in rows})]
            if any(np.linalg.matrix_rank(z) < x.shape[1] for z in [x, *[x[~m] for m in folds]]):
                # The baseline positive subset is full rank, so this is defensive.
                raise ValueError("Unexpected rank loss in the bootstrap design")
            beta = np.linalg.lstsq(x, logy, rcond=None)[0]
            held = np.empty_like(logy)
            for test in folds:
                held[test] = x[test] @ np.linalg.lstsq(x[~test], logy[~test], rcond=None)[0]
            rmse.append(np.sqrt(np.mean((y[:, good]-np.exp(np.clip(held, -700, 700)))**2, axis=0)))
            if len(names) == 2:
                exponents[good] = beta[-1]
        delta[good] = rmse[1]-rmse[0]
    for i in np.flatnonzero(~good):
        sample = [dict(c, stats={"D": {"mean": float(v)}}) for c, v in zip(rows, y[:, i])]
        fitted = f4.fit_models(sample, "D")["fits"]
        if all(k in fitted for k in required):
            exponents[i] = fitted["Ns + U"]["exponents"][-1]
            delta[i] = fitted["Ns + U"]["cv_rmse"]-fitted["Ns"]["cv_rmse"]
    valid = np.isfinite(exponents) & np.isfinite(delta)
    result.update(valid_resamples=int(valid.sum()), nonpositive_draws=int((~good).sum()))
    if not valid.any():
        result["status"] = "unresolved: no identifiable bootstrap draws"
        return result
    exponent = bootstrap_distribution(exponents[valid])
    difference = bootstrap_distribution(delta[valid])
    lo, hi = exponent["ci95"]
    result.update(status=("descriptive whole-trajectory bootstrap within arms" if any("bootstrap_arms" in c for c in rows)
                          else "descriptive paired-replicate bootstrap"), u_exponent=exponent,
        excludes_zero=bool(lo > 0 or hi < 0), delta_cv_rmse=difference,
        probability_u_improves=float(np.mean(delta[valid] < 0)),
        delta_definition="RMSE(Ns + U) - RMSE(Ns); negative favors U",
        # Compact joint empirical distribution; summary statistics retain precision.
        draws=np.column_stack((exponents[valid], delta[valid])).round(12).tolist())
    return result


def extension(bracket):
    """Project only unresolved significant drift; precision is not a clock.

    Fit late log upper absolute drift against time. This is descriptive, not
    a guarantee that the drift or the other admission criteria will pass.
    """
    estimates = {}
    stationary = bracket["stationary"]
    for key in METRICS:
        metric = stationary["metrics"][key]
        estimate = dict(completed_T=bracket["completed_T"], rate=None,
                        total_T=None, extra_T=None, status="no time projection: no significant unresolved drift")
        if metric["significant_drift"]:
            points = [(h["T"], max(abs(h[key]["low"]), abs(h[key]["high"])))
                      for h in stationary["drift_history"] if h["T"] >= .5*bracket["completed_T"]]
            points = [(t, d) for t, d in points if d > 0]
            estimate["status"] = "longer run needed; insufficient decaying drift history for a time estimate"
            if len(points) >= 3:
                times, drift = np.array(points).T
                slope = float(np.polyfit(times-times[-1], np.log(drift), 1)[0])
                estimate["rate"] = slope
                if slope < 0:
                    bound = max(abs(metric["drift"]["low"]), abs(metric["drift"]["high"]))
                    extra = float(np.log(bound/TOL)/-slope)
                    estimate.update(extra_T=extra, total_T=bracket["completed_T"]+extra,
                        status="descriptive drift projection only; no runtime/admission guarantee")
                else:
                    estimate["status"] = "longer run needed; drift bound not decaying, extension time unresolved"
        estimates[key] = estimate
    totals = [v["total_T"] for k, v in estimates.items() if stationary["metrics"][k]["significant_drift"]]
    return dict(**{k: bracket[k] for k in ("N", "Ns", "u", "completed_T", "f4_confirmed")},
                metrics=estimates, total_T=max(totals) if totals and all(t is not None for t in totals) else None)


def threshold_key(row):
    return row["kind"], row["N"], row["u"], row["target"]


def threshold_changes(fits, first, second):
    reference = {threshold_key(r): r for r in fits[first]["thresholds"]}
    changes = []
    for r in fits[second]["thresholds"]:
        old = reference.get(threshold_key(r))
        if old and old["Ns"] is not None and r["Ns"] is not None:
            changes.append(dict(**{k: r[k] for k in ("kind", "N", "u", "target")},
                from_Ns=old["Ns"], to_Ns=r["Ns"], ratio=r["Ns"]/old["Ns"],
                from_region=old["region"], to_region=r["region"]))
    return changes


def analysis(data, source):
    cells = data["cells"]
    if not cells or len({f4.identity(c) for c in cells}) != len(cells):
        raise ValueError("Input needs a nonempty unique cell list")
    if any(c.get("L") != 3 for c in cells):
        raise ValueError("F4 thresholds assume L=3 (U=784u)")
    brackets = [bracket_cell(c) for c in cells]
    for c, r in zip(cells, brackets):
        r["admitted"] = bool(not data["quick"] and not c["confirmed"]
                             and c["status"] == "cap reached" and r["bracketed"])
        r["usable"] = bool(c["confirmed"] or r["admitted"])
        r["stationary"] = stationary_cell(c)
        r["stationary"]["admitted"] = bool(not data["quick"] and not c["confirmed"]
            and c["status"] == "cap reached" and r["stationary"]["stationary"])
        r["stationary"]["usable"] = bool(c["confirmed"] or r["stationary"]["admitted"])
    sets = build_sets(cells, brackets)
    fits = {name: f4.analysis({"cells": rows}) for name, rows in sets.items()}
    # Both the unchanged function and the serialized published analysis are checked.
    assert fits["A"] == f4.analysis(data), "F4 reproduction failed"
    if "analysis" in data:
        assert fits["A"] == data["analysis"], "Saved F4 fits differ: input/code4 version or row order mismatch"
    bootstrap = {name: bootstrap_u(sets[name]) for name in ("B", "E", "S", "S_no10")}
    return dict(experiment="F5", version=2, source=str(Path(source).resolve()),
        quick=bool(data["quick"]), tolerance=TOL, definitions=__doc__,
        f4_reproduction="exact (including saved analysis)" if "analysis" in data else "exact (recomputed; no saved analysis)",
        cells=brackets, fit_sets=fits, set_definitions=SETS, bootstrap=bootstrap,
        threshold_changes={f"{a}_to_{b}": threshold_changes(fits, a, b)
                           for a, b in (("A", "B"), ("A", "A_no10"), ("B", "E"),
                                       ("A", "E"), ("A", "S"), ("A", "S_no10"), ("S", "S_no10"))},
        extensions=[extension(r) for r in brackets if not r["stationary"]["usable"]])


def number(value):
    return "unresolved" if value is None else f"{value:.6g}"


def summary(result):
    cells = result["cells"]
    print("F5 " + ("QUICK SMOKE TEST — no scientific inference" if result["quick"] else "F4 bracket re-analysis"))
    print(f"Source: {result['source']}; A reproduction: {result['f4_reproduction']}")
    print("Tail=last quarter; changes=split last half; paired Student-t 95% intervals.")
    print("Crossing=negative width with pointwise 95% interval below zero: unadjusted diagnostic only; never invalidates a bracket.")
    print("Point endpoint inversions within noise remain unbracketed. Width upper bound must be <=0.01 for BOTH metrics.")
    print("N Ns u F4 | mean [random,optimum] width [95%] valid | consensus [random,optimum] width [95%] valid | bracketed admitted")
    for c in sorted(cells, key=f4.identity):
        parts = []
        for k in METRICS:
            m = c["metrics"][k]
            parts.append(f"[{m['ends'][0]:.6f},{m['ends'][1]:.6f}] {m['width']['mean']:.6f} "
                         f"[{m['width']['low']:.6f},{m['width']['high']:.6f}] {m['valid']}")
        print(f"{c['N']:4d} {c['Ns']:g} {c['u']:.1e} {c['f4_status']} | " + " | ".join(parts)
              + f" | {c['bracketed']} {c['admitted']}")
        for k in METRICS:
            m = c["metrics"][k]
            print(f"    {k}: crossing records={m['reversed_records']}, first T={m['first_reversal_T']}; "
                  f"noisy point inversions={m['noisy_reversed_records']} (unadjusted)")
            if m["reasons"] or not m["ordered"]:
                print(f"    {k}: " + "; ".join(m["reasons"] or ["point endpoints inverted within noise"]))
    print("\nSTATIONARY: last half; pooled 2K trajectories; OLS drift over window; paired arm difference.")
    print("Level and arm difference use max(batch-means SE, replicate-window SE); 95% Student-t intervals.")
    for c in sorted(cells, key=f4.identity):
        stationary = c["stationary"]
        print(f"  {f4.identity(c)}: newly admitted={'yes' if stationary['admitted'] else 'no'}; "
              f"usable in S={stationary['usable']}; records={stationary['window_records']}")
        for k, m in stationary["metrics"].items():
            def bounds(stats):
                return f"[{stats['low']:.6f},{stats['high']:.6f}]" if stats else "unresolved"
            print(f"    {k}: estimate={m['estimate']:.6f}, half-width={number(m['half_width'])}; "
                  f"drift={bounds(m['drift'])}; arm difference (optimum-random)={bounds(m['arm_difference'])}; "
                  f"admitted={'yes' if m['admitted'] else 'no'}; " + ("; ".join(m['reasons']) or "all three criteria pass"))
        if not stationary["admitted"] and stationary["stationary"]:
            print("    Admission gate: " + ("quick input cannot promote cells" if result["quick"] else
                  "already F4 confirmed" if c["f4_confirmed"] else "cell has not reached cap"))
    print("Midpoints and worst-case endpoint bounds (bracket bounds, not statistical confidence intervals):")
    for c in sorted(cells, key=f4.identity):
        bits = []
        for k in METRICS:
            m = c["metrics"][k]
            bits.append(f"{k}={m['midpoint']['mean']:.6f} +/- {number(m['half_width'])}")
        for k in ("G", "D"):
            d = c["derived"][k]
            bits.append(f"{k}={d['midpoint']['mean']:.6f} bounds={d['bounds']}")
        print(f"  {f4.identity(c)}: " + "; ".join(bits))
    for name, fit in result["fit_sets"].items():
        print(f"\nFIT {name}: {SETS[name]}")
        for label, key in (("G", "gap"), ("D", "deficit")):
            model = fit[key]
            print(f"  {label}: {model['status']}; cells={model['cells']}; nonpositive={model['excluded_nonpositive']}; winner={model['winner']}")
            for law, f in model["fits"].items():
                print(f"    {law}: a={f['a']:.9g} log_a={f['log_a']:.9g} exponents={f['exponents']} "
                      f"held-out RMSE={f['cv_rmse']:.9g} log RMSE={f['cv_log_rmse']:.9g}")
        print("  kind N u target Ns s/U region supported")
        for r in fit["thresholds"]:
            print(f"  {r['kind']} {r['N']} {r['u']:.1e} {r['target']:.2f} {number(r['Ns'])} "
                  f"{number(r['s_over_U'])} {r['region']} {r['adequate']}")
        if not fit["thresholds"]:
            print("  95%/99% thresholds unresolved")
    print("\nC–D is an endpoint sensitivity envelope, not a confidence interval; winners are refitted.")
    print("\nU BOOTSTRAP: whole trajectories within cells (B/E paired; S/S_no10 within arms); fixed membership; delta = RMSE(Ns+U)-RMSE(Ns)")
    for name, b in result["bootstrap"].items():
        print(f"  {name}: {b['status']}; {b['valid_resamples']}/{b['requested']} draws; seed={b['seed']}")
        if "u_exponent" in b:
            print(f"    U exponent: {b['u_exponent']}; excludes 0={b['excludes_zero']}")
            print(f"    RMSE difference: {b['delta_cv_rmse']}; P(U improves)={b['probability_u_improves']:.4f}")
    print("\nSTEP 5 PREP: extension times only for significant unresolved drift; late log(drift bound) versus T.")
    print("N Ns u completed_T projected_total_T metric details")
    for r in result["extensions"]:
        print(f"{r['N']} {r['Ns']:g} {r['u']:.1e} {r['completed_T']} {number(r['total_T'])}")
        for key, e in r["metrics"].items():
            print(f"    {key}: slope={number(e['rate'])}, extra_T={number(e['extra_T'])}; {e['status']}")
    print("\nTHRESHOLD CHANGES")
    for comparison, changes in result["threshold_changes"].items():
        print(f"  {comparison}:")
        for r in changes:
            print(f"    {threshold_key(r)}: {r['from_Ns']:.6g} -> {r['to_Ns']:.6g}, "
                  f"ratio={r['ratio']:.6g}; {r['from_region']} -> {r['to_region']}")
        if not changes:
            print("    unresolved: no matched finite thresholds")
    print("\nVERDICT")
    if result["quick"]:
        print("SMOKE TEST ONLY: no threshold, mutation-rate, N=10, or GPU allocation conclusion.")
    print(f"F4 confirmed={sum(c['f4_confirmed'] for c in cells)}/{len(cells)}")
    print("Scope | strict | stationary")
    for strict, stationary in (("B", "S"), ("E", "S_no10")):
        columns = []
        for name in (strict, stationary):
            selected = [c for c in cells if name not in ("E", "S_no10") or c["N"] != 10]
            usable = sum(c["usable"] if name in ("B", "E") else c["stationary"]["usable"] for c in selected)
            ratios = [r["ratio"] for r in result["threshold_changes"][f"A_to_{name}"]]
            ratio = f"{min(ratios):.4g}..{max(ratios):.4g}" if ratios else "unresolved"
            b = result["bootstrap"][name]
            u = b.get("u_exponent")
            exponent = (f"{u['median']:.4g} [{u['ci95'][0]:.4g},{u['ci95'][1]:.4g}]"
                        if u else "unresolved")
            columns.append(f"{name}: usable={usable}/{len(selected)}; threshold ratios vs F4={ratio}; "
                           f"U exponent median [95%]={exponent}; "
                           f"P(held-out RMSE lower with U)={number(b.get('probability_u_improves'))}")
        print(("all N" if strict == "B" else "without N=10") + " | " + " | ".join(columns))
    for name in ("B", "E", "S", "S_no10"):
        rows = result["fit_sets"][name]["thresholds"]
        remaining = [r for r in rows if r["region"] != "interpolated" or not r["adequate"]]
        print(f"{name} extrapolated/unresolved/unsupported thresholds:")
        for r in remaining:
            print(f"  {threshold_key(r)}: Ns={number(r['Ns'])}; {r['region']}; supported fit={r['adequate']}")
        if not remaining:
            print("  none" if rows else "  unresolved: no identifiable thresholds")
    pending = [c for c in cells if not c["stationary"]["usable"]]
    print("Remaining unresolved cells under S (extension times only address drift):")
    for c in pending:
        reasons = [f"{k}: {reason}" for k, m in c["stationary"]["metrics"].items() for reason in m["reasons"]]
        print(f"  {f4.identity(c)}: " + ("; ".join(reasons) or "admission gate (quick input or cap not reached)"))
    if not pending:
        print("  none")
    if not result["quick"]:
        for name in ("S", "S_no10"):
            rows = result["fit_sets"][name]["thresholds"]
            outside = [r for r in rows if r["region"] == "extrapolated"]
            if outside:
                print(f"{name}: extrapolation remains; GPU validation at Ns=30,100,1000 is still needed "
                      "to constrain thresholds between/below the existing Ns grid.")
                beyond = [r for r in outside if r["Ns"] < 30 or r["Ns"] > 3000]
                if beyond:
                    print(f"  {len(beyond)} thresholds also lie outside Ns=30..3000; the proposed grid alone cannot bracket them.")
            elif not rows or any(r["region"] == "unresolved" or not r["adequate"] for r in rows):
                print(f"{name}: threshold support unresolved; additional Ns=30,100,1000 GPU evidence remains necessary.")
            else:
                print(f"{name}: no extrapolated thresholds; this criterion does not require Ns=30,100,1000 GPU runs.")
    print("N=10 is the strong-selection sensitivity (A_no10, E, S_no10); only two original Ns constrain shape.")


def save(result, stem=STEM):
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    stream = StringIO()
    with redirect_stdout(stream):
        summary(result)
    stem.with_name(stem.name+"-summary.txt").write_text(stream.getvalue())
    print(stream.getvalue(), end="")


def figure(result, stem=STEM):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401; loads design/ only when present, as in F4

    cells, fits = result["cells"], result["fit_sets"]
    populations = sorted({c["N"] for c in cells})
    mutations = sorted({c["u"] for c in cells})
    strengths = sorted({c["Ns"] for c in cells})
    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(3, 1, height_ratios=(1, 1.1, 1.3), hspace=.68)
    axes = []
    top = grid[0].subgridspec(1, len(strengths)+1,
                            width_ratios=[1]*len(strengths)+[.035], wspace=.45)
    vmax = max(TOL, *(max(c["metrics"][k]["width"]["high"], 0) for c in cells for k in METRICS))
    for index, ns in enumerate(strengths):
        ax = fig.add_subplot(top[0, index]); axes.append(ax)
        values = np.full((len(populations), len(mutations)), np.nan)
        lookup = {(c["N"], c["u"]): c for c in cells if c["Ns"] == ns}
        for i, n in enumerate(populations):
            for j, u in enumerate(mutations):
                c = lookup.get((n, u))
                if not c:
                    continue
                m, d = [c["metrics"][k] for k in METRICS]
                values[i, j] = max(0, m["width"]["high"], d["width"]["high"])
                label = f"{m['width']['high']:.3f}/{d['width']['high']:.3f}"
                if not c["valid"]:
                    label += " ×"
                elif c["bracketed"]:
                    label += " *"
                elif not all(c["metrics"][k]["ordered"] for k in METRICS):
                    label += " ~"
                if c["stationary"]["admitted"]:
                    ax.plot(j+.42, i, marker="D", ms=3, mfc="none", mec="C3", mew=1)
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color="black")
        im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=vmax, alpha=.65, aspect="auto")
        ax.set_xticks(range(len(mutations)), [f"{u:.0e}" for u in mutations])
        ax.set_yticks(range(len(populations)), [str(n) for n in populations])
        ax.set(xlabel="u", ylabel="N", title=f"Ns={ns:g}: mean / consensus width (upper 95%)")
        ax.grid(False)
        if index == 0:
            f4.panel_letter(ax, "A")
    colorbar = fig.colorbar(im, cax=fig.add_subplot(top[0, -1]))
    colorbar.ax.tick_params(labelsize=6)
    colorbar.set_label("max width; × invalid, * narrow; diamond stationary", fontsize=6)
    middle = grid[1].subgridspec(1, 2, wspace=.4)
    for index, (key, response) in enumerate((("gap", "G"), ("deficit", "D"))):
        ax = fig.add_subplot(middle[0, index]); axes.append(ax)
        model = fits["B"][key]
        fit = model["fits"].get(model["winner"])
        for c in cells:
            if not c["usable"] and not c["stationary"]["usable"]:
                continue
            x = f4.predict(fit, c)/fit["a"] if fit else f4.feature(c, "U/s" if response == "G" else "Ns")
            y = c["derived"][response]["midpoint"]["mean"]
            bounds = c["derived"][response]["bounds"]
            color = f"C{populations.index(c['N'])}"
            # Linear response axes retain negative G and full endpoint intervals.
            if bounds and c["usable"]:
                ax.vlines(x, bounds[0], bounds[1], color=color, lw=.65)
            if c["usable"]:
                ax.plot(x, y, "o" if c["f4_confirmed"] else "s", color=color, ms=4,
                        fillstyle="full" if c["f4_confirmed"] else "none")
            if c["stationary"]["admitted"]:
                level = c["stationary"]["derived"][response]
                ax.errorbar(x, level["mean"], yerr=level["half_width"], fmt="D", color=color,
                            ms=4, mfc="none", elinewidth=.7)
        ax.set_xscale("log")
        if fit:
            low, high = ax.get_xlim()
            x = np.geomspace(low, high, 100)
            ax.plot(x, fit["a"]*x, color="C0", lw=.8, alpha=.6)
            label = " ".join(f"({k})^{p:.3g}" for k, p in zip(fit["variables"], fit["exponents"]))
        else:
            label = "U/s" if response == "G" else "Ns"
            ax.text(.5, .5, "Fit unresolved" + (" — smoke data" if result["quick"] else ""), transform=ax.transAxes,
                    ha="center", fontsize=7)
        ax.set(xlabel=label, ylabel=response, title=f"{response} versus B scaling; diamonds = S additions")
        handles = [Line2D([], [], color=f"C{i}", lw=1, label=f"N={n}") for i, n in enumerate(populations)]
        handles += [Line2D([], [], color="black", marker="o", ls="", label="confirmed"),
                    Line2D([], [], color="black", marker="s", mfc="none", ls="", label="bracketed"),
                    Line2D([], [], color="black", marker="D", mfc="none", ls="", label="stationary")]
        ax.legend(handles=handles, fontsize=5.5, frameon=False, ncol=3)
        if index == 0:
            f4.panel_letter(ax, "B")
    bottom = grid[2].subgridspec(1, 2, wspace=.35)
    for index, target in enumerate((.95, .99)):
        ax = fig.add_subplot(bottom[0, index]); axes.append(ax)
        rows = list({threshold_key(r): r for name in ("B", "S")
                     for r in fits[name]["thresholds"] if r["target"] == target}.values())
        keys = [threshold_key(r) for r in rows]
        maps = {name: {threshold_key(r): r for r in fit["thresholds"]} for name, fit in fits.items()}
        for j, key in enumerate(keys):
            ends = [maps[name].get(key, {}).get("Ns") for name in ("C", "D")]
            if all(v is not None for v in ends):
                ax.fill_between([j-.32, j+.32], min(ends), max(ends), color="C1", alpha=.2)
            for offset, name, color in ((-.24, "A", "C0"), (-.08, "B", "C1"), (.08, "E", "C2"), (.24, "S", "C3")):
                r = maps[name].get(key)
                if r and r["Ns"] is not None:
                    ax.plot(j+offset, r["Ns"], ("D" if name == "S" else "o") if r["adequate"] else "x", color=color, ms=3,
                            fillstyle="full" if r["region"] == "interpolated" else "none")
        if keys:
            labels = [f"{'cons' if r['N'] is None else 'N='+str(r['N'])}\n{r['u']:.0e}" for r in rows]
            ax.set_xticks(range(len(keys)), labels, rotation=90)
            ax.set_yscale("log")
        else:
            ax.text(.5, .5, "95% / 99% thresholds unresolved", transform=ax.transAxes, ha="center", fontsize=7)
            ax.set_xticks([])
        ax.set(ylabel="threshold Ns", title=f"{target:.0%} efficiency", xlabel="consensus / individual N; u")
        handles = [Line2D([], [], color=f"C{i}", marker="D" if name == "S" else "o", ls="", label=name)
                   for i, name in enumerate(("A", "B", "E", "S"))]
        handles.append(Patch(color="C1", alpha=.2, label="C–D sensitivity"))
        ax.legend(handles=handles, fontsize=5.5, frameon=False, ncol=4, loc="upper left")
        if index == 0:
            f4.panel_letter(ax, "C")
    for ax in axes:
        ax.tick_params(labelsize=5.5)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(7)
        ax.title.set_size(7)
    note = "QUICK SMOKE TEST — no inference. " if result["quick"] else ""
    fig.text(.5, .025, note+"B: bars = bracket bounds; diamonds = stationary mean ±95%. C: open = extrapolated; × = poor fit; shade = C–D sensitivity.",
             ha="center", fontsize=6, family="monospace")
    fig.subplots_adjust(left=.08, right=.94, top=.95, bottom=.18)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def synthetic_cell(n=100, ns=300, u=.001, width=.004, confirmed=False):
    """Known smooth approach from both sides, with paired replicate variation."""
    times = np.arange(0, 101, 2)
    U = 784*u
    deficit, gap = .3*ns**-.5*U**.2, .015*(U/(ns/n))**.5
    noise = np.linspace(-.0001, .0001, 8)
    records = {}
    # Constant final-quarter separation; approaching arms in the last half.
    half = width/2 + .02*np.maximum(.75-times/100, 0)
    for key, center in (("efficiency", 1-deficit-gap), ("consensus", 1-deficit)):
        records[key] = np.stack((center-half[:, None]+noise, center+half[:, None]+noise), axis=1)
    tails = {k: v[times >= 75].mean(0) for k, v in records.items()}
    tail = {k: v.mean(0) if confirmed else v[0] for k, v in tails.items()}
    tail.update(G=tail["consensus"]-tail["efficiency"], D=1-tail["consensus"])
    return dict(N=n, Ns=ns, s=ns/n, u=u, U=U, K=8, L=3, confirmed=confirmed,
        status="met" if confirmed else "cap reached", completed_T=100,
        times=times.tolist(), records={k: v.tolist() for k, v in records.items()},
        arms=[dict(tail={k: v[i].tolist() for k, v in tails.items()}) for i in range(2)],
        tail={k: v.tolist() for k, v in tail.items()}, stats={k: f4.f3.interval(v) for k, v in tail.items()})


def check(input_path):
    begin = perf_counter()
    cell = synthetic_cell()
    bracket = bracket_cell(cell)
    assert bracket["bracketed"] and bracket["valid"]
    assert np.isclose(bracket["metrics"]["efficiency"]["width"]["mean"], .004)
    assert np.isclose(bracket["metrics"]["efficiency"]["half_width"], .002)
    assert not bracket_cell(synthetic_cell(width=.02))["bracketed"]
    inverted = bracket_cell(synthetic_cell(width=-.004))
    assert not inverted["valid"] and inverted["metrics"]["consensus"]["inverted"]
    crossed = deepcopy(cell)
    # Earlier crossing that returns to a good final bracket; endpoint-only tests miss it.
    for key in METRICS:
        crossed["records"][key][10][0] = [1.1]*8
    crossed = bracket_cell(crossed)
    assert crossed["valid"] and crossed["bracketed"]
    assert crossed["metrics"]["consensus"]["first_reversal_T"] == 20
    # Pairwise noise, not just the point width, must pass the 0.01 criterion.
    noisy = deepcopy(cell)
    offsets = np.linspace(-.012, .012, 8)
    for key in METRICS:
        noisy["records"][key] = (np.asarray(noisy["records"][key])
                                 + np.stack((-offsets/2, offsets/2))[None]).tolist()
        for arm in range(2):
            noisy["arms"][arm]["tail"][key] = (np.asarray(noisy["arms"][arm]["tail"][key])
                                                + (-1 if arm == 0 else 1)*offsets/2).tolist()
    noisy_bracket = bracket_cell(noisy)
    assert noisy_bracket["valid"] and not noisy_bracket["bracketed"]
    assert noisy_bracket["metrics"]["consensus"]["width"]["mean"] < TOL
    assert noisy_bracket["metrics"]["consensus"]["width"]["high"] > TOL
    assert np.allclose(bracket["derived"]["G"]["bounds"],
                       bracket["derived"]["G"]["midpoint"]["mean"]+np.array([-.004, .004]))
    # Wrong-way drift in the earlier half of the last-half window, leaving tails intact.
    wrong = deepcopy(cell)
    for key in METRICS:
        for i, time in enumerate(wrong["times"]):
            if 50 <= time < 75:
                wrong["records"][key][i][1] = (np.asarray(wrong["records"][key][i][1])-.03).tolist()
    wrong_bracket = bracket_cell(wrong)
    assert "optimum arm increasing beyond noise" in wrong_bracket["metrics"]["consensus"]["reasons"]
    # Noisy stationary trajectories: fluctuations dwarf .01 pointwise, but
    # sufficient window evidence supports equilibrium. Synthetic records only.
    rng = np.random.default_rng(481)
    stationary = synthetic_cell()
    times = np.arange(401, dtype=float)
    stationary.update(times=times.tolist(), completed_T=400)
    for key, center in (("efficiency", .65), ("consensus", .75)):
        values = center+rng.normal(0, .015, (len(times), 2, 8))
        stationary["records"][key] = values.tolist()
    def refresh_tails(c):
        times = np.asarray(c["times"])
        for key in METRICS:
            tails = np.asarray(c["records"][key])[times >= .75*times[-1]].mean(0)
            for arm in range(2):
                c["arms"][arm]["tail"][key] = tails[arm].tolist()
        return c
    refresh_tails(stationary)
    station = stationary_cell(stationary)
    assert station["stationary"]
    for m in station["metrics"].values():
        assert m["half_width"] <= TOL and equivalent_interval(m["drift"])
        assert equivalent_interval(m["arm_difference"])
        assert m["level"]["se"] == max(m["level"]["batch_se"], m["level"]["replicate_se"])
    blocks = np.repeat(np.tile([-.08, .08], 5), 20)
    batch_check = window_interval(np.tile(blocks[:, None], (1, 16)))
    assert batch_check["batch_se"] > batch_check["replicate_se"]
    assert batch_check["half_width"] > TOL
    drifting = deepcopy(stationary)
    for key in METRICS:
        drifting["records"][key] = (np.asarray(drifting["records"][key])
                                   -.2*(times/times[-1])[:, None, None]).tolist()
    drift = stationary_cell(refresh_tails(drifting))
    assert not drift["stationary"] and all(m["significant_drift"] for m in drift["metrics"].values())
    chance = deepcopy(stationary)
    for key in METRICS:
        values = np.asarray(chance["records"][key])
        # A chance excursion after meeting, followed by ordinary stationary noise.
        values[300, 0] = values[300, 1]+.006
        chance["records"][key] = values.tolist()
    chance_bracket = bracket_cell(refresh_tails(chance))
    assert all(m["reversed_records"] > 0 for m in chance_bracket["metrics"].values())
    assert stationary_cell(chance)["stationary"]
    mixed = deepcopy(stationary)
    mixed["records"]["consensus"] = drifting["records"]["consensus"]
    refresh_tails(mixed)
    for candidate, expected in ((stationary, True), (chance, True), (drifting, False), (mixed, False)):
        candidate_result = analysis(dict(cells=[candidate], quick=False), "synthetic-admission")
        row = candidate_result["cells"][0]
        assert row["stationary"]["admitted"] == expected
        fitted = build_sets([candidate], [row])["S"][0]
        assert fitted["confirmed"] == expected
        if expected:
            assert np.isclose(fitted["stats"]["efficiency"]["mean"],
                              np.asarray(candidate["records"]["efficiency"])[times >= 200].mean())
    # Precision failures must not be turned into an extension-time recommendation.
    wide = deepcopy(stationary)
    for key in METRICS:
        wide["records"][key] = (np.asarray(wide["records"][key])
                                + np.linspace(-.1, .1, 8)[None, None]).tolist()
    wide_station = stationary_cell(refresh_tails(wide))
    assert not wide_station["stationary"]
    assert all(m["wide_half_width"] and not m["significant_drift"] for m in wide_station["metrics"].values())
    wide_bracket = dict(bracket_cell(wide), stationary=wide_station)
    assert extension(wide_bracket)["total_T"] is None
    # Known exponential drift bound gives a descriptive time, only for drift.
    decaying = dict(bracket_cell(drifting), stationary=deepcopy(drift))
    decaying["stationary"]["drift_history"] = [dict(T=t, **{
        k: {"low": -.02*np.exp((400-t)/50), "high": -.01} for k in METRICS}) for t in (200, 300, 400)]
    for m in decaying["stationary"]["metrics"].values():
        m["drift"] = dict(low=-.02, high=-.01)
    assert np.isclose(extension(decaying)["total_T"], 400+50*np.log(2))
    cells = [synthetic_cell(n, ns, u, confirmed=(u == .0001))
             for n in (10, 100, 1000, 3000) for ns in (300, 3000) for u in (.0001, .001, .003)]
    data = dict(cells=cells, quick=False)
    data["analysis"] = f4.analysis(data)
    original = deepcopy(data)
    result = analysis(data, "synthetic")
    assert data == original and result["fit_sets"]["A"] == data["analysis"]
    assert sum(c["usable"] for c in result["cells"]) == len(cells)
    assert result["fit_sets"]["E"]["deficit"]["cells"] == 18
    assert result["fit_sets"]["S"]["deficit"]["cells"] == len(cells)
    assert result["fit_sets"]["S_no10"]["deficit"]["cells"] == 18
    b = result["bootstrap"]["B"]
    assert b["valid_resamples"] == BOOTSTRAPS and b["excludes_zero"]
    assert abs(b["u_exponent"]["median"]-.2) < .005
    assert b["delta_cv_rmse"]["median"] < 0
    # Compare the first vectorized bootstrap draw against the unmodified fitter.
    sets = build_sets(cells, result["cells"])
    rng = np.random.default_rng(SEED)
    sampled = []
    for c in sets["B"]:
        indices = rng.integers(c["K"], size=(BOOTSTRAPS, c["K"]))
        sampled.append(dict(c, stats={"D": {"mean": float(np.asarray(c["tail"]["D"])[indices[0]].mean())}}))
    direct = f4.fit_models(sampled, "D")["fits"]
    assert np.allclose(b["draws"][0], [direct["Ns + U"]["exponents"][-1],
                       direct["Ns + U"]["cv_rmse"]-direct["Ns"]["cv_rmse"]], atol=1e-11)
    # The S bootstrap draws independent whole trajectory windows within arms.
    rng = np.random.default_rng(SEED)
    sampled = []
    for original_cell, c in zip(cells, sets["S"]):
        if original_cell["confirmed"]:
            assert c["stats"] == original_cell["stats"] and c["tail"] == original_cell["tail"]
        arms = np.asarray(c["bootstrap_arms"])
        means = [arm[rng.integers(len(arm), size=(BOOTSTRAPS, len(arm)))][0].mean() for arm in arms]
        sampled.append(dict(c, stats={"D": {"mean": float(np.mean(means))}}))
    direct = f4.fit_models(sampled, "D")["fits"]
    sb = result["bootstrap"]["S"]
    assert sb["valid_resamples"] == BOOTSTRAPS
    assert np.allclose(sb["draws"][0], [direct["Ns + U"]["exponents"][-1],
                       direct["Ns + U"]["cv_rmse"]-direct["Ns"]["cv_rmse"]], atol=1e-11)
    # Crossing inversion and interpolation/extrapolation remain F4's own logic.
    assert {r["region"] for r in result["fit_sets"]["B"]["thresholds"]} == {"interpolated", "extrapolated"}
    for r in result["fit_sets"]["B"]["thresholds"]:
        models = result["fit_sets"]["B"]
        ds = models["deficit"]; gs = models["gap"]
        laws = [ds["fits"][ds["winner"]]] + ([gs["fits"][gs["winner"]]] if r["kind"] == "individual" else [])
        probe = dict(N=r["N"] or 1, U=r["U"], Ns=r["Ns"], s=r["Ns"]/(r["N"] or 1))
        assert np.isclose(sum(f4.predict(law, probe) for law in laws), 1-r["target"])
    quick = analysis(dict(data, quick=True), "synthetic-quick")
    assert not any(c["admitted"] or c["stationary"]["admitted"] for c in quick["cells"])
    path = input_path if input_path.exists() else f4.OUTPUT / "code4-quick.json"
    if path.exists():
        real = analysis(json.loads(path.read_text()), path)
        print(f"F4 exact reproduction passed: {path}; {len(real['cells'])} cells")
    else:
        print("No local F4 JSON; saved-data reproduction skipped")
    print(f"F5 check passed: brackets, unadjusted crossing, stationary noise/drift/precision, fit sets, "
          f"bootstrap and thresholds ({perf_counter()-begin:.2f}s; no simulation).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "figure"), nargs="?", default="check")
    parser.add_argument("--input", type=Path, default=f4.OUTPUT / "code4.json")
    parser.add_argument("--output", type=Path, default=STEM, help="output stem (also used by figure)")
    args = parser.parse_args()
    if args.command == "check":
        check(args.input)
    elif args.command == "run":
        if not args.input.exists():
            parser.error(f"F4 input missing: {args.input}; use --input output/code4-quick.json for smoke testing")
        result = analysis(json.loads(args.input.read_text()), args.input)
        save(result, args.output)
        figure(result, args.output)
    else:
        result = json.loads(args.output.with_suffix(".json").read_text())
        if Path(result["source"]).resolve() != args.input.resolve():
            parser.error("Saved F5 source differs from --input; run analysis for this input first")
        figure(result, args.output)


if __name__ == "__main__":
    main()
