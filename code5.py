"""F5: CPU-only re-analysis of F4; never simulates or modifies the input.

python code5.py [check|run|figure] [--input output/code4.json output/code4-step4.json] [--output STEM]
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
Inputs must share environment, calibration, sigma_n and quick/full mode, and
have disjoint (N,Ns,u) cells. K may differ between cells. All fit sets use the
combined cells; main is S_no10 (stationary admission, excluding N=10).
Each source's A analysis is reproduced separately, including saved F4 fits.

Model-free thresholds use main's membership and pooled last-half trajectory
window means for ALL cells (including confirmed cells); existing F4 estimators
and fits remain unchanged. Sensitivity adds non-admitted window means. Crossings
interpolate adjacent available cells in log Ns / log deficit, never extrapolate.
Bonferroni two-sided 95% difference intervals across all ordered Ns pairs,
using stationary level SEs and the smaller df, gate significant reversals.
Bootstrap intervals resample whole trajectories independently within each arm
of the fixed bracketing cells, 2000 times. Draws leaving that bracket are
censored, not discarded; unidentified draws conservatively widen both bounds.
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
        "S": "confirmed + stationary window estimates", "S_no10": "S excluding N=10",
        "main": "S_no10 over all inputs: stationary admission, excluding N=10"}


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
                trajectory_means=windows[key].tolist(),
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
            "S": stationary, "S_no10": [c for c in stationary if c["N"] != 10],
            "main": [c for c in stationary if c["N"] != 10]}


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


def log_crossing(ns, efficiency, target):
    """Invert the line through two positive deficits; caller checks brackets."""
    x = np.log(ns)
    y = np.log1p(-np.asarray(efficiency))
    return np.exp(x[0]+(np.log1p(-target)-y[0])*(x[1]-x[0])/(y[1]-y[0]))


def crossing_bootstrap(pair, metric, target, resamples=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = []
    for cell in pair:
        arms = np.asarray(cell["stationary"]["metrics"][metric]["trajectory_means"])
        draws.append(np.mean([arm[rng.integers(len(arm), size=(resamples, len(arm)))].mean(1)
                              for arm in arms], axis=0))
    left, right = draws
    physical = (left >= 0) & (left <= 1) & (right >= 0) & (right <= 1) & (right >= left)
    below = physical & (left >= target) & (right >= target)
    above = physical & (left < target) & (right < target)
    inside = physical & (left < target) & (right >= target) & (right < 1)
    unknown = ~(below | above | inside)
    # Extended-real bounds retain every draw, including lost brackets. They
    # become JSON null plus explicit censor labels, never a conditional CI.
    low, high = np.full(resamples, -np.inf), np.full(resamples, np.inf)
    low[above] = high[above] = np.inf
    low[below] = high[below] = -np.inf
    ns = [c["Ns"] for c in pair]
    low[inside] = high[inside] = log_crossing(ns, [left[inside], right[inside]], target)
    bounds = [np.quantile(low, .025, method="inverted_cdf"),
              np.quantile(high, .975, method="inverted_cdf")]
    return dict(seed=seed, requested=resamples, inside=int(inside.sum()),
                below_bracket=int(below.sum()), above_bracket=int(above.sum()),
                unidentified=int(unknown.sum()),
                ci95=[float(v) if np.isfinite(v) else None for v in bounds],
                censor=[None if np.isfinite(v) else ("below bracket" if v < 0 else "above bracket")
                        for v in bounds])


def model_free_series(cells, metric, target, version, resamples=BOOTSTRAPS, seed=SEED):
    """A single (N,u) series, with membership fixed before any resampling."""
    first = cells[0]
    selected = sorted([c for c in cells if version == "sensitivity" or c["stationary"]["usable"]],
                      key=lambda c: c["Ns"])
    row = dict(version=version, N=first["N"], u=first["u"], U=first["U"],
               kind="consensus" if metric == "consensus" else "individual", target=target,
               Ns=None, Ns_ci95=None, s_over_U=None, bracket=None, flags=[],
               measured_Ns=[c["Ns"] for c in selected], status="no admitted cells")
    if not selected:
        return row
    stats = [c["stationary"]["metrics"][metric] for c in selected]
    levels = np.array([m["estimate"] for m in stats])
    if any(not c["stationary"]["usable"] for c in selected):
        row["flags"].append("includes non-admitted cells")
    if not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)):
        row["status"] = "unresolved: efficiency outside [0,1]"
        return row
    if any("level" not in m for m in stats):
        row["status"] = "unresolved: insufficient window records for noise check"
        return row
    pairs = len(selected)*(len(selected)-1)//2
    reversals = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            a, b = stats[i]["level"], stats[j]["level"]
            radius = student_t.ppf(1-.025/max(pairs, 1), min(a["df"], b["df"])) * np.hypot(a["se"], b["se"])
            if levels[j]-levels[i]+radius < -1e-12:
                reversals.append([selected[i]["Ns"], selected[j]["Ns"]])
    row["significant_reversals"] = reversals
    if reversals:
        row["status"] = "non-monotone beyond noise"
        return row
    if np.any(np.diff(levels) < 0):
        row["flags"].append("reversals within noise")
    if levels[0] >= target or levels[-1] < target:
        below = levels[0] >= target
        row["status"] = "below range" if below else "above range"
        endpoint = selected[0 if below else -1]
        row["range_bound_Ns"] = endpoint["Ns"]
        row["range_bound_s_over_U"] = endpoint["Ns"]/(first["N"]*first["U"])
        if not endpoint["stationary"]["usable"]:
            row["flags"].append("non-admitted range endpoint")
        return row
    crossings = [i for i in range(len(selected)-1) if levels[i] < target <= levels[i+1]]
    if len(crossings) != 1:
        row["status"] = "unresolved: multiple crossings within noise"
        return row
    i = crossings[0]
    pair = selected[i:i+2]
    row["bracket"] = [c["Ns"] for c in pair]
    if any(not c["stationary"]["usable"] for c in pair):
        row["flags"].append("NON-ADMITTED BRACKET")
    if levels[i+1] == 1:
        row["status"] = "unresolved: zero deficit at bracket endpoint"
        return row
    crossing = float(log_crossing(row["bracket"], levels[i:i+2], target))
    bootstrap = crossing_bootstrap(pair, metric, target, resamples, seed)
    row.update(status="interpolated", Ns=crossing, Ns_ci95=bootstrap["ci95"],
               s_over_U=crossing/(first["N"]*first["U"]), bootstrap=bootstrap)
    if any(bootstrap["censor"]):
        row["flags"].append("95% CI censored at bracket edges")
    if bootstrap["unidentified"]:
        row["flags"].append("bootstrap includes unidentified crossings")
    return row


def model_free_thresholds(cells, resamples=BOOTSTRAPS, seed=SEED):
    groups = {}
    for c in cells:
        if c["N"] != 10:
            groups.setdefault((c["N"], c["u"]), []).append(c)
    return [model_free_series(groups[key], metric, target, version, resamples, seed)
            for version in ("main", "sensitivity") for key in sorted(groups)
            for target in (.95, .99) for metric in ("consensus", "efficiency")]


def model_free_summary(rows):
    print("MODEL-FREE THRESHOLDS")
    print("Main membership = S_no10; all estimates = pooled last-half trajectory window means.")
    print("Sensitivity includes non-admitted cells; adjacent available Ns; log Ns / log(1-efficiency) interpolation.")
    print("95% CI: 2000 whole-trajectory resamples within arms, fixed seed=2315, fixed bracket; censored draws retained.")
    print("Monotonicity: all-pair Bonferroni 95% difference intervals using stationary level SE and minimum df.")
    print("set N u target kind | crossing Ns [95% CI] | s/U | flags")
    for r in rows:
        value, ratio = r["status"], "—"
        if r["Ns"] is not None:
            bounds = [number(v) if v is not None else
                      (f"<={r['bracket'][0]:g}" if c == "below bracket" else f">={r['bracket'][1]:g}")
                      for v, c in zip(r["Ns_ci95"], r["bootstrap"]["censor"])]
            value = f"{r['Ns']:.6g} [{', '.join(bounds)}]"
            ratio = number(r["s_over_U"])
        elif "range_bound_Ns" in r:
            value += f" ({r['range_bound_Ns']:g})"
            ratio = f"{'<=' if r['status'] == 'below range' else '>'}{r['range_bound_s_over_U']:.6g}"
        print(f"{r['version']} {r['N']} {r['u']:.1e} {r['target']:.0%} {r['kind']} | "
              f"{value} | {ratio} | {'; '.join(r['flags']) or 'none'}")
    print("Descriptive crossing-Ns ratios (finite crossings only; no fitted scaling law):")
    for version in ("main", "sensitivity"):
        subset = [r for r in rows if r["version"] == version]
        for kind in ("consensus", "individual"):
            for target in (.95, .99):
                lookup = {(r["N"], r["u"]): r for r in subset if r["kind"] == kind and r["target"] == target}
                comparisons = [(f"N={n}: u=3e-3/1e-4", (n, .003), (n, .0001))
                               for n in sorted({k[0] for k in lookup})]
                comparisons += [(f"u={u:.1e}: N=3000/100", (3000, u), (100, u))
                                for u in sorted({k[1] for k in lookup})]
                bits = []
                for label, numerator, denominator in comparisons:
                    a, b = lookup.get(numerator, {}), lookup.get(denominator, {})
                    ratio = number(a["Ns"]/b["Ns"]) if a.get("Ns") and b.get("Ns") else "unresolved"
                    flagged = any("NON-ADMITTED BRACKET" in r.get("flags", []) for r in (a, b))
                    bits.append(f"{label}={ratio}" + (" [non-admitted bracket]" if flagged else ""))
                print(f"  {version} {kind} {target:.0%}: " + "; ".join(bits))


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


def merge_inputs(inputs, sources):
    """Validate before combining; never mutate or average the source records."""
    if not inputs or len(inputs) != len(sources):
        raise ValueError("Need one or more JSON inputs and matching source paths")
    fields = ("environment", "calibration", "sigma_n", "quick")
    first = inputs[0]
    cells, seen, reproduction = [], set(), []
    for data, source in zip(inputs, sources):
        if any(k not in data or k not in first or data[k] != first[k] for k in fields):
            raise ValueError(f"Input environment/calibration/sigma_n/quick mismatch: {source}")
        if not data["cells"]:
            raise ValueError(f"Empty cell list: {source}")
        for cell in data["cells"]:
            key = f4.identity(cell)
            if key in seen:
                raise ValueError(f"Duplicate (N, Ns, u) cell {key}: {source}")
            seen.add(key)
            cells.append(cell)
        # Retain the published A assert for code4.json even in a combined run.
        a = f4.analysis(data)
        if "analysis" in data:
            assert a == data["analysis"], f"Saved F4 A reproduction failed: {source}"
        reproduction.append(dict(source=str(Path(source).resolve()), cells=len(data["cells"]),
                                 saved_analysis="analysis" in data))
    merged = dict(first, cells=cells)
    if len(inputs) > 1:
        merged.pop("analysis", None)  # A now refers to the combined cell list.
    return merged, reproduction


def analyze_inputs(inputs, sources):
    data, reproduction = merge_inputs(inputs, sources)
    result = analysis(data, sources[0])
    result["sources"] = [r["source"] for r in reproduction]
    result["source_reproduction"] = reproduction
    result["f4_reproduction"] = "exact per input (saved A checked wherever present); combined A recomputed"
    # Identify newly interpolated main thresholds relative to the original F4
    # stationary/no10 analysis, with the same admission gates and estimators.
    originals = [d for d, p in zip(inputs, sources) if Path(p).name == "code4.json"]
    if originals:
        original = originals[0]
        lookup = {f4.identity(c): c for c in result["cells"]}
        brackets = [lookup[f4.identity(c)] for c in original["cells"]]
        baseline = f4.analysis({"cells": build_sets(original["cells"], brackets)["main"]})
        result["original_main"] = baseline
        result["threshold_changes"]["original_main_to_main"] = threshold_changes(
            {"original_main": baseline, "main": result["fit_sets"]["main"]}, "original_main", "main")
    return result


def load_inputs(paths):
    for path in paths:
        if not path.is_file():
            raise ValueError(f"F4 input missing: {path}")
    return analyze_inputs([json.loads(path.read_text()) for path in paths], paths)


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
    bootstrap["main"] = bootstrap["S_no10"]
    return dict(experiment="F5", version=4, source=str(Path(source).resolve()),
        sources=[str(Path(source).resolve())], main_set="main",
        quick=bool(data["quick"]), tolerance=TOL, definitions=__doc__,
        f4_reproduction="exact (including saved analysis)" if "analysis" in data else "exact (recomputed; no saved analysis)",
        cells=brackets, fit_sets=fits, set_definitions=SETS, bootstrap=bootstrap,
        model_free_thresholds=model_free_thresholds(brackets),
        threshold_changes={f"{a}_to_{b}": threshold_changes(fits, a, b)
                           for a, b in (("A", "B"), ("A", "A_no10"), ("B", "E"),
                                       ("A", "E"), ("A", "S"), ("A", "S_no10"), ("S", "S_no10"), ("A", "main"))},
        extensions=[extension(r) for r in brackets if not r["stationary"]["usable"]])


def number(value):
    return "unresolved" if value is None else f"{value:.6g}"


def summary(result):
    cells = result["cells"]
    print("F5 " + ("QUICK SMOKE TEST — no scientific inference" if result["quick"] else "F4 bracket re-analysis"))
    print(f"Sources: {', '.join(result.get('sources', [result['source']]))}; A reproduction: {result['f4_reproduction']}")
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
                      f"held-out RMSE={f['cv_rmse']:.9g} log RMSE={f['cv_log_rmse']:.9g}" +
                      ("; FAILING HELD-OUT (>0.02)" if f['cv_rmse'] > .02 else "; passes held-out (<=0.02)"))
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
    model_free_summary(result["model_free_thresholds"] if "model_free_thresholds" in result
                       else model_free_thresholds(cells))
    if result["quick"]:
        print("SMOKE TEST ONLY: no threshold, mutation-rate, N=10, or GPU allocation conclusion.")
    print("\nPOWER-LAW RESULTS (retained for comparison)")
    for key in ("gap", "deficit"):
        laws = result["fit_sets"]["main"][key]["fits"]
        failed = [name for name, law in laws.items() if law["cv_rmse"] > .02]
        print(f"  Main {key}: failing held-out (>0.02): {', '.join(failed) or 'none'}" +
              ("; ALL power-law fits fail; their thresholds are unsupported." if laws and len(failed) == len(laws) else ""))
    print("Main = S_no10 across all inputs: stationary admission, excluding N=10.")
    main_rows = result["fit_sets"]["main"]["thresholds"]
    newly_inside = {threshold_key(r) for r in result["threshold_changes"].get("original_main_to_main", [])
                    if r["from_region"] != "interpolated" and r["to_region"] == "interpolated"}
    for r in main_rows:
        print(f"  main {threshold_key(r)}: Ns={number(r['Ns'])}; s/U={number(r['s_over_U'])}; "
              f"{r['region']}; supported fit={r['adequate']}" +
              ("; NOW INTERPOLATED (original F4 main was extrapolated)" if threshold_key(r) in newly_inside else ""))
    if not main_rows:
        print("  Main 95%/99% thresholds unresolved: no identifiable thresholds.")
    inside = [threshold_key(r) for r in main_rows if r["region"] == "interpolated"]
    print(f"Main interpolated thresholds: {inside or 'none'}")
    if "original_main" in result:
        print(f"Newly interpolated versus original F4 main: {sorted(newly_inside, key=str) or 'none'}")
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
                print(f"{name}: {len(outside)} thresholds remain extrapolated; additional admitted Ns "
                      "must bracket these crossings at the relevant N,u.")
            elif not rows or any(r["region"] == "unresolved" or not r["adequate"] for r in rows):
                print(f"{name}: threshold support unresolved; more admitted cells or better held-out fits are needed.")
            else:
                print(f"{name}: no extrapolated thresholds; this criterion does not require additional Ns runs.")
    print(f"N=10 is the strong-selection sensitivity; {len({c['Ns'] for c in cells})} observed Ns values constrain shape.")


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
    heatmap_rows = (len(strengths)+1)//2
    heatmap_cols = min(2, len(strengths))
    fig = plt.figure(figsize=(10, 12+2*(heatmap_rows-1)))
    grid = fig.add_gridspec(4, 1, height_ratios=(heatmap_rows, 1.1, 1.3, 1.5), hspace=.85)
    axes = []
    top = grid[0].subgridspec(heatmap_rows, heatmap_cols+1,
                            width_ratios=[1]*heatmap_cols+[.035], wspace=.45, hspace=.9)
    vmax = max(TOL, *(max(c["metrics"][k]["width"]["high"], 0) for c in cells for k in METRICS))
    for index, ns in enumerate(strengths):
        ax = fig.add_subplot(top[index//heatmap_cols, index % heatmap_cols]); axes.append(ax)
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
    colorbar = fig.colorbar(im, cax=fig.add_subplot(top[:, -1]))
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
    threshold_rows = (result["model_free_thresholds"] if "model_free_thresholds" in result
                      else model_free_thresholds(cells))
    threshold_ns = [n for n in populations if n != 10]
    panel = grid[3].subgridspec(1, max(1, len(threshold_ns)), wspace=.25)
    if not threshold_ns:
        ax = fig.add_subplot(panel[0, 0]); axes.append(ax)
        ax.text(.5, .5, "Model-free thresholds: no N != 10 cells", ha="center", transform=ax.transAxes)
    for index, n in enumerate(threshold_ns):
        ax = fig.add_subplot(panel[0, index]); axes.append(ax)
        for target in (.95, .99):
            ax.axhline(target, color=".5", lw=.6, ls=":")
            ax.text(.99, target, f"{target:.0%}", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=6)
        for j, u in enumerate(mutations):
            series = sorted([c for c in cells if c["N"] == n and c["u"] == u], key=lambda c: c["Ns"])
            admitted = [c for c in series if c["stationary"]["usable"]]
            rejected = [c for c in series if not c["stationary"]["usable"]]
            color = f"C{j}"
            for metric, style, marker in (("consensus", "-", "o"), ("efficiency", "--", "s")):
                for selected, alpha, width in ((series, .3, .7), (admitted, 1, 1.1)):
                    ax.plot([c["Ns"] for c in selected],
                            [c["stationary"]["metrics"][metric]["estimate"] for c in selected],
                            color=color, ls=style, lw=width, alpha=alpha)
                for selected, fill in ((admitted, color), (rejected, "none")):
                    ax.plot([c["Ns"] for c in selected],
                            [c["stationary"]["metrics"][metric]["estimate"] for c in selected],
                            color=color, marker=marker, mfc=fill, ls="", ms=3)
            for r in threshold_rows:
                if r["N"] != n or r["u"] != u or r["Ns"] is None:
                    continue
                main = r["version"] == "main"
                ax.plot(r["Ns"], r["target"], marker="^" if main else "v", ls="", color=color,
                        mfc=color if main else "none", ms=5, zorder=5)
                if main and all(v is not None for v in r["Ns_ci95"]):
                    ax.hlines(r["target"], *r["Ns_ci95"], color=color, lw=1, alpha=.7)
        ax.set(xscale="log", xlabel="Ns", ylabel="efficiency" if index == 0 else "",
               title=f"N = {n}", ylim=(0, 1.045))
        if index == 0:
            f4.panel_letter(ax, "D")
    handles = [Line2D([], [], color=f"C{j}", label=f"u={u:.0e}") for j, u in enumerate(mutations)]
    handles += [Line2D([], [], color="black", ls=style, marker=marker, ms=3, label=label)
                for label, style, marker in (("consensus", "-", "o"), ("mean individual", "--", "s"))]
    handles += [Line2D([], [], color="black", ls="", marker=marker, mfc=fill, ms=4, label=label)
                for label, marker, fill in (("non-admitted cell", "o", "none"),
                                            ("main crossing / finite 95% CI", "^", "black"),
                                            ("sensitivity crossing", "v", "none"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .042),
               fontsize=6.5, frameon=False, ncol=4)
    for ax in axes:
        ax.tick_params(labelsize=5.5)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(7)
        ax.title.set_size(7)
    note = "QUICK SMOKE TEST — no inference. " if result["quick"] else ""
    fig.text(.5, .025, note+"B: bars = bracket bounds; diamonds = stationary mean ±95%. C: open = extrapolated; × = poor fit; shade = C–D sensitivity.",
             ha="center", fontsize=6, family="monospace")
    fig.subplots_adjust(left=.08, right=.94, top=.95, bottom=.13)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    threshold_figure(result, stem, threshold_rows)


def displayed_threshold(rows, n, u, kind, target):
    """Choose the saved crossing for display, flagging an admission-only fallback."""
    matching = {r["version"]: r for r in rows
                if (r["N"], r["u"], r["kind"], r["target"]) == (n, u, kind, target)}
    main, sensitivity = matching.get("main"), matching.get("sensitivity")
    if (main and main["status"] == "below range" and sensitivity
            and sensitivity["status"] == "interpolated"
            and "NON-ADMITTED BRACKET" in sensitivity["flags"]
            and sensitivity["bracket"][0] < min(main["measured_Ns"])):
        return sensitivity, True
    return main, False


def threshold_figure(result, stem, rows):
    """Focused companion figure; consumes saved intervals and crossings only."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    populations = (100, 1000, 3000)
    cells = [c for c in result["cells"] if c["N"] in populations]
    mutations = sorted({c["u"] for c in cells})
    if not cells:
        return
    strengths = sorted({c["Ns"] for c in cells})
    fig = plt.figure(figsize=(10, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.35, 1), hspace=.55)
    top = grid[0].subgridspec(1, 3, wspace=.15)
    axes = []
    positive = [.01, .05]
    for c in cells:
        for metric in ("consensus", "efficiency"):
            stats = c["stationary"]["metrics"][metric]
            positive.append(1-stats["estimate"])
            if "level" in stats:
                positive.extend((1-stats["level"]["high"], 1-stats["level"]["low"]))
    positive = [v for v in positive if np.isfinite(v) and v > 0]
    limits = (min(positive)/1.4, max(positive)*1.3)
    for index, n in enumerate(populations):
        ax = fig.add_subplot(top[index], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set(xscale="log", yscale="log", xlabel="Ns", title=f"N = {n}",
               ylim=limits, xlim=(min(strengths)/1.2, max(strengths)*1.25))
        ax.xaxis.set_major_locator(FixedLocator(strengths))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.tick_params(labelleft=index == 0)
        if index == 0:
            ax.set_ylabel("shortfall  1 − efficiency")
        for target in (.95, .99):
            deficit = 1-target
            ax.axhline(deficit, color=".5", lw=.7, ls=":")
            ax.text(.99, deficit, f"{target:.0%}", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.3, alpha=.85))
        for j, u in enumerate(mutations):
            series = sorted([c for c in cells if (c["N"], c["u"]) == (n, u)],
                            key=lambda c: c["Ns"])
            color = f"C{j}"
            for metric, style, marker in (("consensus", "-", "o"), ("efficiency", "--", "s")):
                ax.plot([c["Ns"] for c in series],
                        [1-c["stationary"]["metrics"][metric]["estimate"] for c in series],
                        color=color, ls=style, lw=1, zorder=2)
                for c in series:
                    stats = c["stationary"]["metrics"][metric]
                    deficit = 1-stats["estimate"]
                    if deficit <= 0:
                        continue  # Exact zero has no location on a logarithmic axis.
                    bounds = stats.get("level")
                    # Reverse the efficiency interval under 1 − efficiency.
                    error = ([[max(0, bounds["high"]-stats["estimate"])],
                              [max(0, stats["estimate"]-bounds["low"])]] if bounds else None)
                    ax.errorbar(c["Ns"], deficit, yerr=error, fmt=marker, color=color,
                                mfc=color if c["stationary"]["usable"] else "white",
                                ms=3.5, mew=.8, elinewidth=.65, capsize=1.5, zorder=3)
            for kind in ("consensus", "individual"):
                for target in (.95, .99):
                    row, fallback = displayed_threshold(rows, n, u, kind, target)
                    if row and row["Ns"] is not None:
                        ax.plot(row["Ns"], 1-target, marker="^", ls="", color=color,
                                mfc="white" if fallback else color, ms=5, mew=.9, zorder=5)
        f4.panel_letter(ax, chr(ord("A")+index))

    # One shared log normalization permits direct comparison between both metrics.
    chosen = {(n, u, kind): displayed_threshold(rows, n, u, kind, .95)
              for n in populations for u in mutations for kind in ("consensus", "individual")}
    finite = [r["Ns"] for r, _ in chosen.values() if r and r["Ns"] is not None]
    vmin, vmax = min([100]+finite), max([3000]+finite)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")
    symbol_fonts = [*plt.rcParams["font.family"], "DejaVu Sans"]
    bottom = grid[1].subgridspec(1, 3, width_ratios=(1, 1, .035), wspace=.3)
    for index, (kind, title) in enumerate((("consensus", "Consensus"), ("individual", "Mean individual"))):
        ax = fig.add_subplot(bottom[index]); axes.append(ax)
        values = np.full((len(populations), len(mutations)), np.nan)
        labels = {}
        for i, n in enumerate(populations):
            for j, u in enumerate(mutations):
                row, fallback = chosen[n, u, kind]
                label = "unresolved"
                if row and row["Ns"] is not None:
                    values[i, j] = row["Ns"]
                    bounds = []
                    for k, value in enumerate(row["Ns_ci95"]):
                        if value is not None:
                            bounds.append(f"{value:.0f}")
                        else:
                            censor = row["bootstrap"]["censor"][k]
                            edge = row["bracket"][0 if censor == "below bracket" else 1]
                            bounds.append(f"{'<' if censor == 'below bracket' else '>'}{edge:g}")
                    label = f"{row['Ns']:.0f}{'†' if fallback else ''}\n[{bounds[0]}, {bounds[1]}]"
                elif row and row["status"] in ("above range", "below range"):
                    above = row["status"] == "above range"
                    label = f"{'>' if above else '<'}{row['range_bound_Ns']:g}"
                    values[i, j] = vmax if above else vmin
                labels[i, j] = label
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for (i, j), label in labels.items():
            rgba = cmap(norm(values[i, j])) if np.isfinite(values[i, j]) else (1, 1, 1, 1)
            luminance = np.dot(rgba[:3], [.299, .587, .114])
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color="white" if luminance < .45 else "black", linespacing=1.6,
                    fontfamily=symbol_fonts)
        ax.set_xticks(range(len(mutations)), [f"{u:.0e}" for u in mutations])
        ax.set_yticks(range(len(populations)), [str(n) for n in populations])
        ax.set(xlabel="u", ylabel="N", title=f"{title}: 95% threshold Ns [95% CI]")
        ax.grid(False)
        f4.panel_letter(ax, chr(ord("D")+index))
    colorbar = fig.colorbar(im, cax=fig.add_subplot(bottom[2]), extend="both")
    colorbar.set_ticks([100, 300, 1000, 3000], labels=["100", "300", "1000", "3000"])
    colorbar.set_label("threshold Ns (log)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    handles = [Line2D([], [], color=f"C{j}", label=f"u={u:.0e}") for j, u in enumerate(mutations)]
    handles += [Line2D([], [], color="black", ls=style, marker=marker, ms=3.5, label=label)
                for label, style, marker in (("consensus", "-", "o"), ("mean individual", "--", "s"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               fontsize=7.5, frameon=False, ncol=len(handles))
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(8)
    note = "SMOKE TEST. " if result["quick"] else ""
    fig.text(.5, .025, note+"○/□ open: non-admitted; bars: 95% CI; ▲: main crossing; △ / †: sensitivity crossing, lower-Ns bracket not admitted.",
             ha="center", fontsize=7, fontfamily=symbol_fonts)
    fig.subplots_adjust(left=.075, right=.94, top=.89, bottom=.12)
    target = stem.with_name(stem.name+"-thresholds")
    fig.savefig(target.with_suffix(".png"), dpi=200)
    fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)


def synthetic_cell(n=100, ns=300, u=.001, width=.004, confirmed=False, K=8):
    """Known smooth approach from both sides, with paired replicate variation."""
    times = np.arange(0, 101, 2)
    U = 784*u
    deficit, gap = .3*ns**-.5*U**.2, .015*(U/(ns/n))**.5
    noise = np.linspace(-.0001, .0001, K)
    records = {}
    # Constant final-quarter separation; approaching arms in the last half.
    half = width/2 + .02*np.maximum(.75-times/100, 0)
    for key, center in (("efficiency", 1-deficit-gap), ("consensus", 1-deficit)):
        records[key] = np.stack((center-half[:, None]+noise, center+half[:, None]+noise), axis=1)
    tails = {k: v[times >= 75].mean(0) for k, v in records.items()}
    tail = {k: v.mean(0) if confirmed else v[0] for k, v in tails.items()}
    tail.update(G=tail["consensus"]-tail["efficiency"], D=1-tail["consensus"])
    return dict(N=n, Ns=ns, s=ns/n, u=u, U=U, K=K, L=3, confirmed=confirmed,
        status="met" if confirmed else "cap reached", completed_T=100,
        times=times.tolist(), records={k: v.tolist() for k, v in records.items()},
        arms=[dict(tail={k: v[i].tolist() for k, v in tails.items()}) for i in range(2)],
        tail={k: v.tolist() for k, v in tail.items()}, stats={k: f4.f3.interval(v) for k, v in tail.items()})


def check(input_paths):
    begin = perf_counter()
    # Exact log-deficit lines, using only tiny synthetic trajectory records.
    def threshold_cells(levels=None, noise=.0001, n=100):
        rows = []
        for index, ns in enumerate((30, 100, 300, 1000, 3000)):
            c = synthetic_cell(n=n, ns=ns)
            for metric, deficit in (("consensus", 9), ("efficiency", 18)):
                center = 1-deficit/ns if levels is None else levels[index]
                c["records"][metric] = np.broadcast_to(
                    center+np.linspace(-noise, noise, c["K"]), (51, 2, c["K"])).tolist()
            station = stationary_cell(c)
            station.update(usable=True, admitted=True)
            rows.append(dict(c, stationary=station))
        return rows

    threshold_grid = threshold_cells()
    free = model_free_thresholds(threshold_grid)
    assert len(free) == 8
    assert free == model_free_thresholds(threshold_grid)
    # Independently reproduce percentile bounds from whole-arm trajectory draws.
    rng = np.random.default_rng(SEED)
    sampled = []
    for c in threshold_grid[1:3]:
        arms = np.asarray(c["stationary"]["metrics"]["consensus"]["trajectory_means"])
        sampled.append(np.mean([arm[rng.integers(len(arm), size=(2000, len(arm)))].mean(1)
                                for arm in arms], axis=0))
    fraction = (np.log(.05)-np.log1p(-sampled[0]))/(np.log1p(-sampled[1])-np.log1p(-sampled[0]))
    manual = np.exp(np.log(100)+fraction*np.log(3))
    assert np.allclose(free[0]["Ns_ci95"], np.quantile(manual, [.025, .975], method="inverted_cdf"))
    for r in free:
        expected = (9 if r["kind"] == "consensus" else 18)/(1-r["target"])
        assert r["status"] == "interpolated" and np.isclose(r["Ns"], expected)
        assert np.isclose(r["s_over_U"], expected/(100*784*.001))
        assert r["Ns_ci95"][0] < expected < r["Ns_ci95"][1]
        assert r["bootstrap"]["requested"] == 2000 and r["bootstrap"]["inside"] == 2000
        assert not r["flags"]
    assert model_free_thresholds(threshold_cells(n=10)) == []
    below = model_free_thresholds(threshold_cells([.999]*5))
    above = model_free_thresholds(threshold_cells([.5, .6, .7, .8, .9]))
    assert all(r["status"] == "below range" and r["range_bound_Ns"] == 30 for r in below)
    assert all(r["status"] == "above range" and r["range_bound_Ns"] == 3000 for r in above)
    nonmonotone = model_free_thresholds(threshold_cells([.8, .96, .9, .98, .995]))
    assert all(r["status"] == "non-monotone beyond noise" and r["Ns"] is None for r in nonmonotone)
    noisy_turn = model_free_series(threshold_cells([.94, .9501, .9499, .98, .995], noise=.004),
                                  "consensus", .95, "main")
    assert noisy_turn["status"] == "unresolved: multiple crossings within noise"
    assert "reversals within noise" in noisy_turn["flags"]
    sensitivity_cells = deepcopy(threshold_grid)
    sensitivity_cells[2]["stationary"].update(usable=False, admitted=False)
    main = model_free_series(sensitivity_cells, "consensus", .95, "main")
    sensitivity = model_free_series(sensitivity_cells, "consensus", .95, "sensitivity")
    assert main["bracket"] == [100, 1000] and not main["flags"]
    assert sensitivity["bracket"] == [100, 300] and "NON-ADMITTED BRACKET" in sensitivity["flags"]
    assert np.isclose(main["Ns"], sensitivity["Ns"])
    for c in sensitivity_cells:
        c["stationary"]["usable"] = False
    assert model_free_series(sensitivity_cells, "consensus", .95, "main")["status"] == "no admitted cells"
    censored = model_free_series(threshold_cells([.94999, .98, .99, .995, .999], noise=.002),
                                 "consensus", .95, "main")
    assert censored["bootstrap"]["below_bracket"] > 50
    assert censored["Ns_ci95"][0] is None and censored["bootstrap"]["censor"][0] == "below bracket"
    assert sum(censored["bootstrap"][k] for k in ("inside", "above_bracket", "below_bracket", "unidentified")) == 2000
    exact = model_free_series(threshold_cells([.8, .9, .95, .98, .99], noise=0), "consensus", .95, "main")
    assert np.isclose(exact["Ns"], 300)
    zero = model_free_series(threshold_cells([.8, .9, 1., 1., 1.], noise=0), "consensus", .95, "main")
    assert zero["status"] == "unresolved: zero deficit at bracket endpoint"
    # No NaN/Infinity may leak into saved JSON, including censored intervals.
    json.dumps([*free, *below, *above, *nonmonotone, noisy_turn, censored, zero], allow_nan=False)
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
    assert result["fit_sets"]["main"] == result["fit_sets"]["S_no10"]
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
    # Disjoint extra strengths at K=32; metadata and A reproduction are checked
    # per source, while every fit set and bootstrap sees the combined grid.
    metadata = dict(environment={"channel": "synthetic"}, calibration={"Imax": 1}, sigma_n=.0128)
    original_input = dict(data, **metadata)
    extra = dict(cells=[synthetic_cell(n, ns, u, K=32, confirmed=(u == .0001))
                        for n in (10, 100, 1000, 3000) for ns in (30, 100, 1000)
                        for u in (.0001, .001, .003)], quick=False, **metadata)
    extra["analysis"] = f4.analysis(extra)
    sources = [Path("code4.json"), Path("step4.json")]
    merged = analyze_inputs([original_input, extra], sources)
    assert len(merged["cells"]) == 60 and {c["K"] for c in merged["cells"]} == {8, 32}
    assert merged["fit_sets"]["main"] == merged["fit_sets"]["S_no10"]
    assert merged["fit_sets"]["main"]["deficit"]["cells"] == 45
    assert merged["original_main"] == result["fit_sets"]["main"]
    assert all(r["saved_analysis"] for r in merged["source_reproduction"])
    assert merged["bootstrap"]["main"]["valid_resamples"] == BOOTSTRAPS
    assert merged["bootstrap"]["B"]["valid_resamples"] == BOOTSTRAPS
    for K in (8, 32):
        v = np.linspace(-.1, .1, K)
        assert np.isclose(interval(v)["high"], v.mean()+student_t.ppf(.975, K-1)*v.std(ddof=1)/np.sqrt(K))
    for field in (*metadata, "quick", "duplicate"):
        bad = deepcopy(extra)
        if field == "duplicate":
            bad["cells"].append(cells[0])
        else:
            bad[field] = None
        try:
            merge_inputs([original_input, bad], sources)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted mismatched/duplicate input: {field}")
    bad = deepcopy(original_input)
    bad["analysis"]["gap"]["winner"] = "tampered"
    try:
        merge_inputs([bad, extra], sources)
    except AssertionError:
        pass
    else:
        raise AssertionError("Lost saved code4.json A reproduction assert")
    stream = StringIO()
    with redirect_stdout(stream):
        summary(merged)
    verdict = stream.getvalue().split("\nVERDICT\n")[1]
    assert verdict.startswith("MODEL-FREE THRESHOLDS") and "Main interpolated thresholds:" in verdict
    assert "NON-ADMITTED BRACKET" in verdict or not any(
        "NON-ADMITTED BRACKET" in r["flags"] for r in merged["model_free_thresholds"])
    assert "Descriptive crossing-Ns ratios" in verdict and "POWER-LAW RESULTS" in verdict
    assert "NOW INTERPOLATED" in verdict
    # Exercise multi-row heatmaps only in a temporary directory.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="f5-check-") as directory:
        stem = Path(directory)/"five-strengths"
        figure(merged, stem)
        for suffix in (".png", ".pdf", "-thresholds.png", "-thresholds.pdf"):
            assert (stem.parent/(stem.name+suffix)).stat().st_size > 0
    # The display may substitute sensitivity only for an admission-lost lower bracket.
    display_main = dict(version="main", N=100, u=.001, kind="consensus", target=.95,
                        status="below range", measured_Ns=[1000, 3000], Ns=None)
    display_sensitivity = dict(display_main, version="sensitivity", status="interpolated",
                               bracket=[300, 1000], Ns=700, flags=["NON-ADMITTED BRACKET"])
    display_rows = [display_main, display_sensitivity]
    assert displayed_threshold(display_rows, 100, .001, "consensus", .95) == (display_sensitivity, True)
    for status in ("interpolated", "above range", "non-monotone beyond noise", "no admitted cells"):
        display_main["status"] = status
        assert displayed_threshold(display_rows, 100, .001, "consensus", .95) == (display_main, False)
    display_main.update(status="below range", measured_Ns=[100, 1000, 3000])
    assert displayed_threshold(display_rows, 100, .001, "consensus", .95) == (display_main, False)
    paths = input_paths
    if paths == [f4.OUTPUT / "code4.json"] and not paths[0].exists():
        paths = [f4.OUTPUT / "code4-quick.json"]
    if all(p.exists() for p in paths) or input_paths != [f4.OUTPUT / "code4.json"]:
        real = load_inputs(paths)
        print(f"F4 exact reproduction passed: {paths}; {len(real['cells'])} cells")
    else:
        print("No local F4 JSON; saved-data reproduction skipped")
    print(f"F5 check passed: brackets, unadjusted crossing, stationary noise/drift/precision, fit sets, "
          f"mixed-K merging, duplicate/metadata refusal, main verdict, five-Ns figure, "
          f"bootstrap, model-free known crossings/range bounds/non-monotonicity/censoring and thresholds "
          f"({perf_counter()-begin:.2f}s; no simulation).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "figure"), nargs="?", default="check")
    parser.add_argument("--input", type=Path, nargs="+", default=[f4.OUTPUT / "code4.json"])
    parser.add_argument("--output", type=Path, default=STEM, help="output stem (also used by figure)")
    args = parser.parse_args()
    if args.command == "check":
        check(args.input)
    elif args.command == "run":
        try:
            result = load_inputs(args.input)
        except ValueError as error:
            parser.error(str(error))
        save(result, args.output)
        figure(result, args.output)
    else:
        result = json.loads(args.output.with_suffix(".json").read_text())
        if result.get("sources", [result["source"]]) != [str(p.resolve()) for p in args.input]:
            parser.error("Saved F5 sources differ from --input; run analysis for these inputs first")
        figure(result, args.output)


if __name__ == "__main__":
    main()
