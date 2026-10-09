"""Experiment E: pyloric rhythm conservation during conductance drift.

Independent JAX implementation of the equations in Prinz, Billimoria & Marder
(2003), J Neurophysiol 90:3998–4015, doi:10.1152/jn.00641.2003, and Prinz,
Bucher & Marder (2004), Nat Neurosci 7:1345–1352, doi:10.1038/nn1352.
Local xolotl reference files were used to cross-check equations, units and the
canonical parameters; no simulator code is incorporated.

Units: ms, mV, intracellular/extracellular Ca in uM; intrinsic gbar in uS/mm²,
area .0628 mm², specific capacitance 10 nF/mm². Density times mV is nA/mm²;
calcium influx uses total nA. Synaptic gmax is nS (divide by 1000*area).
Temperature is 11 C. pyloric.m assigns gbars ALPHABETICALLY: A, CaS, CaT, H,
KCa, Kd, Leak, NaV, not in its channels declaration order. Thus AB leak, LP
CaT/KCa and PY KCa are zero. Evolution starts these at 1e-6 uS/mm²; the exact
zero-valued reference is separately validated. Genome order below is NaV,
CaT, CaS, A, KCa, Kd, H, leak for AB/PD, LP, PY, then seven synapses.

Spikes: upward -20 mV crossings, 2 ms refractory. Bursts: >=3 spikes;
split at gaps >max(100 ms, 3*median of the lower half of finite ISIs in the
analysis window). The lower half estimates intraburst ISIs without long
interburst gaps. Discard smaller groups; isolated interburst spikes are allowed.
Simulate 12000 ms, discard 4000 ms (both configurable). Exclude left-censored
bursts and bursts without a full gap before the right edge (a censored AB onset
can still close the preceding cycle). Require >=3 complete bursts per cell;
match exactly one LP and PY burst in every complete AB onset-to-onset cycle.
Require AB period CV <.1, all per-cycle duties in (0,.6), AB offset < LP onset,
LP onset < PY onset, and LP offset < PY offset. This follows the pyloric-like
ordering in Prinz et al. (2004), with the requested CV/duty bounds, rather than
their separate 15-feature experimental-range classifier. LP/PY overlap is allowed.
Features are mean AB period (ms), three mean per-cycle duties, LP on/off and
PY on/off phases relative to each cycle's AB onset and period; offsets unwrapped.
Exponential Euler updates gates, calcium and voltage at .05 ms, with old
voltage/calcium and the preceding calcium current, as in the reference.

Wright–Fisher selection then independent Bernoulli/log-normal mutation;
selected burn-in, identical population fork, matched random streams in the
selected and strictly neutral arms (neutral reproductive weights are all 1,
including nonpyloric networks). Extinction of a selected lineage is an error,
not a silent neutral rescue. Omega uses the prespecified final generation.
Correlations are descriptive: shared ancestry and finite samples can produce
neutral correlations. The quick experiment cannot establish compensation or
replicate Schulz et al. (2006), Nat Neurosci 9:356–362; model maximal currents
are not the same observables as measured channel expression.

Run: python stg.py [check|bench|quick|run|figure] [--input output/stg-quick.json]
Quick always writes stg-quick.*; run writes stg.*. No full grid is launched.
Bench measures a warmed 2048-network, 12-second assay (both arms together).
Run/quick/bench tune scan unroll 8/16/32 on the current device before use.
Evolution retains only window spikes (1024/cell, explicit overflow rejection),
keeps genomes and precomputed matched random draws on device, and uses float32.
Figures retain 0.5-ms traces; check also records every integration step.
A failing canonical assay produces a clearly marked quick diagnostic artifact,
not invented evolutionary trajectories. Check and quick then exit nonzero.

E2: replicates defaults to R=8 independent runs (seed+r), each with its own
selected burn-in and matched selected/neutral fork. The legacy run/quick
commands accept --replicates (default 1) without changing their default path.
Use --chunk-size to bound networks per simulation batch; random plans are
always bounded to 32 generations. Correlations use all final individuals or
one mean-log-conductance point per lineage. CIs and paired arm tests use
independent replicates. figure-replicates reads stg-replicates.json;
bench-replicates times a precompiled generation and projects runtime, not biology.
Example pipeline test: replicates --replicates 2 --N 8 --K 2 --B 5 --T 10
  --every 5 --output output/stg-replicates-quick.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

jax.config.update("jax_enable_x64", False)  # All simulation/evolution device floats are float32.

OUTPUT = Path(__file__).resolve().parent / "output"
CELLS = ("AB", "LP", "PY")
CHANNELS = ("NaV", "CaT", "CaS", "A", "KCa", "Kd", "H", "leak")
SYNAPSES = ("AB-LP-chol", "AB-PY-chol", "AB-LP-glut", "AB-PY-glut",
            "LP-PY-glut", "PY-LP-glut", "LP-AB-glut")
SITES = [f"{c}.{g}" for c in CELLS for g in CHANNELS] + list(SYNAPSES)
GBAR = np.array([[1000, 25, 60, 500, 50, 1000, .1, 0],
                 [1000, 0, 40, 200, 0, 250, .5, .3],
                 [1000, 24, 20, 500, 0, 1250, .5, .1]], dtype=np.float32)
GMAX = np.array([30, 3, 30, 10, 1, 30, 30], dtype=np.float32)
CANONICAL = np.r_[GBAR.ravel(), GMAX]
FLOOR, AREA, CM = 1e-6, .0628, 10.
ANCESTOR = np.log10(np.maximum(CANONICAL, FLOOR))
PRE = jnp.array([0, 0, 0, 0, 1, 2, 1])
POST = jnp.array([1, 2, 1, 2, 2, 1, 0])
SYN_E = jnp.array([-80, -80, -70, -70, -70, -70, -70])
SYN_K = jnp.array([.01, .01, .025, .025, .025, .025, .025])
POWERS = jnp.array([3, 3, 3, 3, 4, 4, 1, 0])
DURATION, WINDOW, GAP, REFRACTORY = 12000., 4000., 100., 2.
CV_MAX, MIN_BURSTS, MIN_SPIKES = .1, 3, 3
MAX_SPIKES, MAX_BURSTS = 1024, 64
FEATURES = ("period_ms", "AB_duty", "LP_duty", "PY_duty", "LP_on", "LP_off", "PY_on", "PY_off")


def kinetics(v, ca):
    """Steady states and time constants, vectorized over cells/channels."""
    v = v[..., None]
    sig = jax.nn.sigmoid
    m = sig((v + jnp.array([25.5, 27.1, 33, 27.2, 28.3, 12.3, 75, 0])) /
            jnp.array([5.29, 7.2, 8.1, 8.7, 12.6, 11.8, -5.5, 1]))
    m = m.at[..., 4].multiply(ca / (ca + 3))
    m = m.at[..., 7].set(1.)
    h = sig(-(v + jnp.array([48.9, 32.1, 60, 56.9])) / jnp.array([5.18, 5.5, 6.2, 4.9]))
    v = v[..., 0]
    tm = jnp.stack([2.64 - 2.52 * sig((v + 120) / 25),
                    43.4 - 42.6 * sig((v + 68.1) / 20.5),
                    2.8 + 14 / (jnp.exp((v + 27) / 10) + jnp.exp(-(v + 70) / 13)),
                    23.2 - 20.8 * sig((v + 32.9) / 15.2),
                    180.6 - 150.2 * sig((v + 46) / 22.7),
                    14.4 - 12.8 * sig((v + 28.3) / 19.2),
                    2 / (jnp.exp(-(v + 169.7) / 11.6) + jnp.exp((v - 26.7) / 14.3)),
                    jnp.ones_like(v)], -1)
    th = jnp.stack([1.34 * sig((v + 62.9) / 10) * (1.5 + sig(-(v + 34.9) / 3.6)),
                    210 - 179.6 * sig((v + 55) / 16.9),
                    120 + 300 / (jnp.exp((v + 55) / 9) + jnp.exp(-(v + 65) / 16)),
                    77.2 - 58.4 * sig((v + 38.9) / 26.5)], -1)
    return m, h, tm, th


def simulate_one(conductances, dt=.05, trace=False, duration_ms=DURATION,
                 window_start_ms=0., unroll=16, online=False, full_trace=False):
    """Bounded spike buffers keep population simulations resident on the GPU."""
    conductances = jnp.asarray(conductances, dtype=jnp.float32)
    gbar, gmax = conductances[:24].reshape(3, 8), conductances[24:]
    v, ca = jnp.full(3, -60.), jnp.full(3, .05)
    m, h, _, _ = kinetics(v, ca)
    state = (v, ca, m, h, jnp.zeros(7), jnp.zeros(3),
             jnp.full((3, MAX_SPIKES), jnp.inf), jnp.zeros(3, jnp.int32),
             jnp.full(3, -1e6), jnp.full(3, -jnp.inf), jnp.zeros(3, dtype=bool))

    def step(state, i):
        v, ca, m, h, syn, ica, times, counts, last, previous, overflow = state
        mi, hi, tm, th = kinetics(v, ca)
        m = mi + (m - mi) * jnp.exp(-dt / tm)
        h = hi + (h - hi) * jnp.exp(-dt / th)
        ec = (1000 * 8.314 * (273.15 + 11) / (2 * 96485)) * jnp.log(3000 / ca)
        e = jnp.broadcast_to(jnp.array([50., 0, 0, -80, -80, -80, -20, -50]), (3, 8))
        e = e.at[:, 1:3].set(ec[:, None])
        g = gbar * m**POWERS
        g = g.at[:, :4].multiply(h)
        ci = .05 - 14.96 * AREA * ica
        ca = ci + (ca - ci) * jnp.exp(-dt / 200)
        ica = jnp.sum(g[:, 1:3] * (v[:, None] - ec[:, None]), axis=1)
        si = jax.nn.sigmoid((v[PRE] + 35) / 5)
        # sigmoid(-x) avoids cancellation in 1 - sigmoid(x).
        tau = jax.nn.sigmoid(-(v[PRE] + 35) / 5) / SYN_K
        syn = si + (syn - si) * jnp.exp(-dt / tau)
        gs = gmax * syn / (1000 * AREA)
        total = g.sum(-1) + jnp.zeros(3).at[POST].add(gs)
        drive = (g * e).sum(-1) + jnp.zeros(3).at[POST].add(gs * SYN_E)
        vi = drive / total
        vn = vi + (v - vi) * jnp.exp(-dt * total / CM)
        t = (i + 1) * dt
        spike = (v < -20) & (vn >= -20) & (t - last >= REFRACTORY)
        # Linear crossing interpolation; indices beyond capacity are dropped.
        crossing = i * dt + dt * (-20 - v) / jnp.where(vn != v, vn - v, 1)
        record = spike & (crossing >= window_start_ms)
        previous = jnp.where(spike & ~record, crossing, previous)
        # Refractory state includes the transient, but the event buffer does not.
        times = times.at[jnp.arange(3), jnp.where(record, counts, MAX_SPIKES)].set(crossing, mode="drop")
        overflow |= record & (counts >= MAX_SPIKES)
        counts += record.astype(jnp.int32)
        last = jnp.where(spike, t, last)
        return (vn, ca, m, h, syn, ica, times, counts, last, previous, overflow), vn if full_trace else None

    if full_trace:
        final, voltage = jax.lax.scan(step, state, jnp.arange(round(duration_ms / dt)), unroll=unroll)
    elif trace:
        stride = round(.5 / dt)
        def block(state, b):
            state, _ = jax.lax.scan(step, state, b * stride + jnp.arange(stride), unroll=unroll)
            return state, state[0]
        final, voltage = jax.lax.scan(block, state, jnp.arange(round(duration_ms / dt) // stride))
    else:
        # No scan outputs: memory is O(networks * spike capacity), not O(steps).
        final, voltage = jax.lax.scan(step, state, jnp.arange(round(duration_ms / dt)), unroll=unroll)
    if online:
        return final[6], final[7], final[9], final[10]
    return final[6], final[7], voltage


simulate = jax.jit(jax.vmap(simulate_one, in_axes=(0, None, None, None)), static_argnums=(1, 2, 3))
single = jax.jit(simulate_one, static_argnums=tuple(range(1, 8)))


def burst_gap(spikes, window_start_ms=WINDOW):
    """Robust intraburst estimate: median of the lower half of window ISIs."""
    isi = jnp.diff(spikes)
    usable = jnp.isfinite(isi) & (isi > 0) & (spikes[:-1] >= window_start_ms)
    median = jnp.nanmedian(jnp.where(usable, isi, jnp.nan))
    fast = jnp.nanmedian(jnp.where(usable & (isi <= median), isi, jnp.nan))
    return jnp.maximum(GAP, 3 * jnp.nan_to_num(fast, nan=0.))


def bursts(spikes, duration_ms=DURATION, window_start_ms=WINDOW, previous_ms=-jnp.inf):
    gap = burst_gap(spikes, window_start_ms)
    present = jnp.isfinite(spikes)
    previous = jnp.r_[-jnp.inf, spikes[:-1]]
    new = present & ((spikes - previous) > gap)
    ids = jnp.cumsum(new.astype(jnp.int32)) - 1
    ids = jnp.where(present, ids, MAX_BURSTS)
    starts = jax.ops.segment_min(spikes, ids, MAX_BURSTS)
    ends = jax.ops.segment_max(spikes, ids, MAX_BURSTS)
    sizes = jax.ops.segment_sum(present.astype(jnp.int32), ids, MAX_BURSTS)
    # Remove singleton/doublet groups from both diagnostics and cycle matching.
    starts = jnp.where(sizes >= MIN_SPIKES, starts, jnp.inf)
    ends = jnp.where(sizes >= MIN_SPIKES, ends, -jnp.inf)
    # The last transient spike suffices to identify a left-censored group,
    # even though its preceding spikes are deliberately not buffered.
    starts = starts.at[0].set(jnp.where(spikes[0] - previous_ms <= gap, -jnp.inf, starts[0]))
    valid = (sizes >= MIN_SPIKES) & (starts >= window_start_ms) & (ends <= duration_ms - gap)
    return starts, ends, valid, jnp.sum(new) <= MAX_BURSTS


def phenotype(spikes, counts, duration_ms=DURATION, window_start_ms=WINDOW,
              previous=None, overflow=None):
    previous = jnp.full(3, -jnp.inf) if previous is None else previous
    overflow = jnp.zeros(3, dtype=bool) if overflow is None else overflow
    starts, ends, valid, capacity = jax.vmap(bursts, in_axes=(0, None, None, 0))(
        spikes, duration_ms, window_start_ms, previous)
    complete_counts = valid.sum(1)
    # A right-censored AB burst still supplies the endpoint of the preceding
    # complete cycle. Its own duration is never measured without a next onset.
    valid = valid.at[0].set(jnp.isfinite(starts[0]) & (starts[0] >= window_start_ms))
    # Pack complete bursts chronologically without data-dependent array shapes.
    order = jnp.argsort(jnp.where(valid, starts, jnp.inf), axis=1)
    starts = jnp.take_along_axis(starts, order, 1)
    ends = jnp.take_along_axis(ends, order, 1)
    n = valid.sum(1)
    mask = jnp.arange(MAX_BURSTS)[None, :] < n[:, None]
    pmask = jnp.arange(MAX_BURSTS - 1)[None, :] < (n - 1)[:, None]
    intervals = jnp.where(pmask, jnp.diff(starts, axis=1), 0.)
    period = intervals.sum(1) / jnp.maximum(n - 1, 1)
    variance = jnp.where(pmask, (intervals - period[:, None])**2, 0).sum(1) / jnp.maximum(n - 1, 1)
    cv = jnp.sqrt(variance) / jnp.maximum(period, 1)
    # Require both followers in EVERY measured cycle, not just any matching pair.
    cycles = pmask[0]
    refs, next_refs = starts[0, :-1], starts[0, 1:]
    lengths = jnp.where(cycles, next_refs - refs, 1.)
    pair = (cycles[None, :, None] & mask[1:, None, :] &
            (starts[1:, None, :] >= refs[None, :, None]) &
            (starts[1:, None, :] < next_refs[None, :, None]))
    matched = jnp.all(pair.sum(-1) == 1, axis=0)
    follower_start = jnp.where(pair, starts[1:, None, :], 0).sum(-1)
    follower_end = jnp.where(pair, ends[1:, None, :], 0).sum(-1)
    on = (follower_start - refs) / lengths
    off = (follower_end - refs) / lengths
    duties = jnp.concatenate([((ends[0, :-1] - refs) / lengths)[None, :], off - on])
    cycle_count = jnp.maximum(cycles.sum(), 1)
    duty = jnp.where(cycles, duties, 0).sum(1) / cycle_count
    phases = jnp.stack([jnp.where(cycles, on, 0).sum(1),
                        jnp.where(cycles, off, 0).sum(1)], -1) / cycle_count
    ordered = ((follower_start[0] > ends[0, :-1]) &
               (follower_start[1] > follower_start[0]) &
               (follower_end[1] > follower_end[0]))
    bounded = jnp.all((duties > 0) & (duties < .6), axis=0)
    measurable = (jnp.all(complete_counts >= MIN_BURSTS) &
                  jnp.all(~cycles | matched))
    ok = (measurable & (cv[0] < CV_MAX) &
          jnp.all(~cycles | (ordered & bounded)) &
          jnp.all(counts <= MAX_SPIKES) & ~jnp.any(overflow) & jnp.all(capacity))
    f = jnp.r_[period[0], duty, phases.ravel()]
    f = jnp.where(measurable, f, jnp.full(8, jnp.nan))
    return f, ok & jnp.all(jnp.isfinite(f)), jnp.where(n >= 2, cv, jnp.nan), complete_counts


# Incremented by Python tracing, never by execution of a compiled generation.
EVALUATE_TRACES = 0


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def evaluate(genomes, dt=.05, duration_ms=DURATION, window_start_ms=WINDOW, unroll=16):
    global EVALUATE_TRACES
    EVALUATE_TRACES += 1
    def assay_one(genome):
        spikes, counts, previous, overflow = simulate_one(
            10**genome, dt, False, duration_ms, window_start_ms, unroll, True)
        return phenotype(spikes, counts, duration_ms, window_start_ms, previous, overflow)
    return jax.vmap(assay_one)(jnp.asarray(genomes, dtype=jnp.float32))


def tune_unroll(batch_size, dt):
    """Time warmed, synchronized 100 ms probes on the actual device/batch size."""
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (batch_size, 31))
    timings = {}
    for unroll in (8, 16, 32):
        fn = evaluate.lower(g, dt, 100., 0., unroll).compile()
        jax.block_until_ready(fn(g))
        start = perf_counter()
        for _ in range(3):
            jax.block_until_ready(fn(g))
        timings[unroll] = (perf_counter() - start) / 3
    chosen = min(timings, key=timings.get)
    print(f"Unroll probe ({batch_size} networks, 100 ms): {timings}; using {chosen}", flush=True)
    return chosen


def features(conductances=CANONICAL, dt=.05, duration_ms=DURATION, window_start_ms=WINDOW):
    spikes, counts, _ = single(jnp.asarray(conductances), dt, False, duration_ms)
    f, ok, cv, n = phenotype(spikes, counts, duration_ms, window_start_ms)
    return np.asarray(f), bool(ok), np.asarray(cv), np.asarray(n)


def validate_dt(dt):
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(.5 / dt, round(.5 / dt)):
        raise ValueError("dt must be positive and divide 0.5 ms exactly")


def validate_assay(dt, duration_ms, window_start_ms):
    validate_dt(dt)
    if (not np.isfinite(duration_ms) or not np.isfinite(window_start_ms) or
            not 0 <= window_start_ms < duration_ms or
            not np.isclose(duration_ms / .5, round(duration_ms / .5))):
        raise ValueError("Require 0 <= window-start-ms < duration-ms, duration a multiple of 0.5 ms")


def tolerance(optimum):
    return np.r_[.1 * optimum[0], np.full(7, .05)]


def displacements(f, optimum):
    delta = f - optimum
    delta[..., 4:] = (delta[..., 4:] + .5) % 1 - .5
    return delta


def fitness(f, valid, optimum, s):
    """Assay fitness; neutral reproduction separately uses exactly uniform weights."""
    d = np.mean((displacements(f, optimum) / tolerance(optimum))**2, axis=-1)
    return np.where(valid, np.exp(-s * d), 0.), d


class Assay:
    def __init__(self, dt, duration_ms=DURATION, window_start_ms=WINDOW, batch_size=None):
        self.duration_ms, self.window_start_ms = duration_ms, window_start_ms
        self.dt, self.seconds, self.network_seconds = dt, 0., 0.
        self.warm_seconds, self.warm_network_seconds, self.shapes = 0., 0., set()
        self.unroll = tune_unroll(batch_size, dt) if batch_size else 16
        self.compiled = {}
        self.compile_count = 0

    def __call__(self, g):
        shape = g.shape[:-1]
        g = jnp.asarray(g, dtype=jnp.float32).reshape(-1, 31)
        start = perf_counter()
        if g.shape not in self.compiled:
            self.compiled[g.shape] = evaluate.lower(
                g, self.dt, self.duration_ms, self.window_start_ms, self.unroll).compile()
            self.compile_count += 1
        # Calling an explicit executable cannot retrace in a generation loop.
        f, valid, cv, n = self.compiled[g.shape](g)
        f, valid = np.asarray(f), np.asarray(valid)
        elapsed, work = perf_counter() - start, np.prod(shape) * self.duration_ms / 1000
        self.seconds += elapsed
        self.network_seconds += work
        if shape in self.shapes:
            self.warm_seconds += elapsed
            self.warm_network_seconds += work
        self.shapes.add(shape)
        return f.reshape(*shape, 8), valid.reshape(shape)

    def report(self):
        return dict(device=str(jax.devices()[0]), evaluation_wall_seconds=self.seconds,
                    network_simulated_seconds=float(self.network_seconds),
                    including_compile=self.network_seconds / max(self.seconds, 1e-12),
                    warmed=self.warm_network_seconds / max(self.warm_seconds, 1e-12),
                    unroll=self.unroll, compile_count=self.compile_count, dtype="float32",
                    unit="networks * simulated seconds / wall second")


def bench():
    """One warmed paired generation at the scientific defaults; no evolution."""
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (2, 16, 64, 31))
    assay = Assay(.05, batch_size=2 * 16 * 64)
    assay(g)
    traces = EVALUATE_TRACES
    assay(g)
    assert EVALUATE_TRACES == traces and assay.compile_count == 1
    print(json.dumps(dict(assay.report(), networks=2048, simulated_seconds=12.,
                         wall_seconds_per_generation=assay.warm_seconds,
                         retraces_after_warmup=EVALUATE_TRACES - traces)), flush=True)


def step(g, f, valid, optimum, s, u, sigma, rng):
    K, N = g.shape[:2]
    if s == 0:
        weights = np.ones((K, N))
    else:
        if np.any(~valid.any(1)):
            raise RuntimeError(f"No pyloric parents in lineage(s) {np.flatnonzero(~valid.any(1)).tolist()}")
        _, d = fitness(f, valid, optimum, s)
        # Subtract each lineage's best distance to avoid reproductive underflow.
        best = np.min(np.where(valid, d, np.inf), axis=1, keepdims=True)
        weights = np.where(valid, np.exp(-s * np.maximum(d - best, 0)), 0.)
    cdf = np.cumsum(weights, axis=1)
    draws = rng.random((K, N, 1)) * cdf[:, -1:, None]
    parents = np.minimum((draws >= cdf[:, None, :]).sum(-1), N - 1)
    children = g[np.arange(K)[:, None], parents].copy()
    count = rng.binomial(children.size, u)
    sites = rng.choice(children.size, count, replace=False)
    children.reshape(-1)[sites] += rng.normal(0, sigma, count)
    return children


def random_plan(shape, generations, u, sigma, rng):
    """Preserve NumPy's seeded draws; upload once, before the generation loop.

    Only O(T*K*N) parent uniforms and sparse mutations are retained. Both
    arms use the same plan, exactly as their formerly separate matched RNGs.
    """
    draws, sites, changes = [], [], []
    for _ in range(generations):
        draws.append(rng.random((*shape[:2], 1)).astype(np.float32))
        count = rng.binomial(np.prod(shape), u)
        sites.append(rng.choice(np.prod(shape), count, replace=False).astype(np.int32))
        changes.append(rng.normal(0, sigma, count).astype(np.float32))
    width = max(map(len, sites), default=0)
    ids = np.full((generations, width), np.prod(shape), dtype=np.int32)
    delta = np.zeros((generations, width), dtype=np.float32)
    for i, (idx, d) in enumerate(zip(sites, changes)):
        ids[i, :len(idx)], delta[i, :len(d)] = idx, d
    return (jnp.asarray(np.asarray(draws).reshape(generations, *shape[:2], 1)),
            jnp.asarray(ids), jnp.asarray(delta))


@jax.jit
def device_step(g, f, valid, optimum, strength, draws, sites, changes):
    K, N = g.shape[:2]
    delta = f - optimum
    delta = delta.at[..., 4:].set((delta[..., 4:] + .5) % 1 - .5)
    scale = jnp.concatenate((.1 * optimum[:1], jnp.full(7, .05)))
    d = jnp.mean((delta / scale)**2, axis=-1)
    best = jnp.min(jnp.where(valid, d, jnp.inf), axis=1, keepdims=True)
    weights = jnp.where(strength == 0, jnp.ones((K, N)),
                        jnp.where(valid, jnp.exp(-strength * jnp.maximum(d - best, 0)), 0.))
    cdf = jnp.cumsum(weights, axis=1)
    parents = jnp.minimum((draws * cdf[:, -1:, None] >= cdf[:, None, :]).sum(-1), N - 1)
    children = g[jnp.arange(K)[:, None], parents]
    return children.reshape(-1).at[sites].add(changes, mode="drop").reshape(g.shape)


STEP_TRACES = 0


@jax.jit
def planned_step(g, f, valid, optimum, strength, plan, generation):
    global STEP_TRACES
    STEP_TRACES += 1
    # Dynamic indexing inside jit avoids compiling an eager slice for every t.
    return device_step(g, f, valid, optimum, strength, *(x[generation] for x in plan))


@jax.jit
def population_moments(g):
    return jnp.var(g.mean(1), axis=0), jnp.var(g, axis=1).mean(0)


def measure(g, f, valid, optimum, s):
    """Population variance (ddof=0); phenotype statistics condition on pyloricity.

    An arm with any lineage lacking valid phenotypes has undefined between-
    lineage feature divergence, represented by null rather than zero.
    """
    between, within = map(np.asarray, population_moments(jnp.asarray(g, dtype=jnp.float32)))
    delta = displacements(f, optimum)
    means, feature_within = [], []
    for x, mask in zip(delta, valid):
        means.append(x[mask].mean(0) if mask.any() else np.full(8, np.nan))
        feature_within.append(x[mask].var(0) if mask.any() else np.full(8, np.nan))
    fd = np.var(means, axis=0)
    fw = np.mean(feature_within, axis=0)
    w, _ = fitness(f, valid, optimum, s)
    return dict(mean_fitness=float(w.mean()), fraction_pyloric=float(valid.mean()),
                pyloric_by_lineage=valid.mean(1).tolist(),
                conductance_divergence=between.tolist(), conductance_divergence_sum=float(between.sum()),
                within_conductance=within.tolist(), within_conductance_sum=float(within.sum()),
                feature_divergence=fd.tolist(),
                feature_divergence_normalized=float(np.mean(fd / tolerance(optimum)**2)),
                within_feature=fw.tolist())


def correlations(g):
    result = {}
    for cell, name in enumerate(CELLS):
        x = g[..., cell * 8:(cell + 1) * 8].reshape(-1, 8)
        x = x - x.mean(0)
        norm = np.sqrt((x*x).sum(0))
        matrix = np.divide(x.T @ x, norm[:, None] * norm[None, :],
                           out=np.full((8, 8), np.nan), where=(norm[:, None] * norm[None, :]) > 1e-20)
        pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.isfinite(matrix[i, j])]
        strongest = sorted(pairs, key=lambda ij: -abs(matrix[ij]))[:5]
        result[name] = dict(matrix=matrix.tolist(), strongest=[dict(pair=[CHANNELS[i], CHANNELS[j]],
                            r=float(matrix[i, j])) for i, j in strongest],
                            candidate_K_Na={CHANNELS[j]: float(matrix[0, j]) for j in (3, 4, 5)})
    return result


def hybrids(g, assay, optimum, s, rng, samples=64):
    K, N = g.shape[:2]
    a = rng.integers(K, size=samples)
    b = (a + rng.integers(1, K, size=samples)) % K
    p = g[a, rng.integers(N, size=samples)]
    q = g[b, rng.integers(N, size=samples)]
    r = g[a, rng.integers(N, size=samples)]
    cross = np.where(rng.random(p.shape) < .5, p, q)
    within = np.where(rng.random(p.shape) < .5, p, r)
    group = np.concatenate([p, q, r, cross, within])
    f, valid = assay(group)
    w, _ = fitness(f, valid, optimum, s)
    results = {}
    for name, ids in (("cross_parents", np.r_[0:2*samples]),
                      ("within_parents", np.r_[0:samples, 2*samples:3*samples]),
                      ("cross_lineage", np.r_[3*samples:4*samples]),
                      ("within_lineage", np.r_[4*samples:5*samples])):
        results[name] = dict(n=len(ids), fraction_pyloric=float(valid[ids].mean()),
                             mean_fitness=float(w[ids].mean()))
    return results


def add_omega(selected, neutral):
    """Fixed final record, same generation and fork in both arms."""
    assert selected[-1]["t"] == neutral[-1]["t"]
    a, b = [np.asarray(x[-1]["conductance_divergence"]) for x in (selected, neutral)]
    return dict(t_star=selected[-1]["t"], per_site=np.divide(a, b, out=np.full(31, np.nan), where=b > 0).tolist(),
                summed=float(a.sum() / b.sum()) if b.sum() > 0 else None)


def clean(value):
    """Strict JSON: undefined measurements become null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run(N=64, K=16, B=200, T=2000, every=20, u=.02, sigma=.05, s=1., seed=0,
        dt=.05, hybrid_samples=64, duration_ms=DURATION, window_start_ms=WINDOW,
        replicates=1, chunk_size=None):
    if replicates != 1 or chunk_size is not None:
        return run_replicates(N=N, K=K, B=B, T=T, every=every, u=u, sigma=sigma, s=s,
                              seed=seed, dt=dt, hybrid_samples=hybrid_samples,
                              duration_ms=duration_ms, window_start_ms=window_start_ms,
                              replicates=replicates, chunk_size=chunk_size)
    validate_assay(dt, duration_ms, window_start_ms)
    if N < 1 or K < 2 or min(B, T) < 0 or every < 1 or hybrid_samples < 1:
        raise ValueError("Require N>=1, K>=2, B,T>=0, every,hybrid_samples>=1")
    if not 0 <= u <= 1 or min(sigma, s) < 0:
        raise ValueError("Require u in [0,1], sigma,s>=0")
    params = dict(N=N, K=K, B=B, T=T, every=every, u=u, sigma=sigma, s=s, seed=seed,
                  dt=dt, hybrid_samples=hybrid_samples, duration_ms=duration_ms,
                  window_start_ms=window_start_ms, zero_floor=FLOOR)
    start = perf_counter()
    optimum, ok, canonical_cv, n = features(dt=dt, duration_ms=duration_ms, window_start_ms=window_start_ms)
    if not ok:
        raise RuntimeError(f"Canonical validation failed: features={optimum.tolist()}, bursts={n.tolist()}, CV={canonical_cv.tolist()}")
    assay = Assay(dt, duration_ms, window_start_ms, 2 * K * N)
    burn_seed, fork_seed, hybrid_seed = np.random.SeedSequence(seed).spawn(3)
    burn_plan = random_plan((1, N, 31), B, u, sigma, np.random.default_rng(burn_seed))
    fork_plan = random_plan((K, N, 31), T, u, sigma, np.random.default_rng(fork_seed))
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (1, N, 31))
    f, valid = assay(g)
    # Burn-in is one shared selected population, before either arm exists.
    for t in trange(B, desc="burn-in"):
        if s != 0 and np.any(~valid.any(1)):
            raise RuntimeError(f"No pyloric parents in lineage(s) {np.flatnonzero(~valid.any(1)).tolist()}")
        g = planned_step(g, f, valid, optimum, s, burn_plan, jnp.int32(t))
        f, valid = assay(g)
    fork = g[0]
    arms = jnp.broadcast_to(g, (2, K, N, 31))
    records = [[], []]
    compile_count, step_traces = None, None
    for t in trange(T + 1, desc="forked lineages"):
        all_f, all_valid = assay(arms)
        if compile_count is None:
            compile_count = assay.compile_count
        assert assay.compile_count == compile_count, "Generation assay recompiled"
        children = []
        for i, strength in enumerate((s, 0.)):
            g, f, valid = arms[i], all_f[i], all_valid[i]
            if t % every == 0 or t == T:
                record = dict(t=t, **measure(g, f, valid, optimum, s))
                record["mean_reproductive_weight"] = 1. if strength == 0 else record["mean_fitness"]
                records[i].append(record)
            if t < T:
                if strength != 0 and np.any(~valid.any(1)):
                    raise RuntimeError(f"No pyloric parents in lineage(s) {np.flatnonzero(~valid.any(1)).tolist()}")
                children.append(planned_step(g, f, valid, optimum, strength, fork_plan, jnp.int32(t)))
        if t < T:
            arms = jnp.stack(children)
            if step_traces is None:
                step_traces = STEP_TRACES
            assert STEP_TRACES == step_traces, "Evolution step retraced"
    # Full genomes leave the device only after evolution, for final reports.
    arms = np.asarray(arms)
    corr = {name: correlations(g) for name, g in zip(("selected", "neutral"), arms)}
    hyb = {name: hybrids(g, assay, optimum, s, np.random.default_rng(hybrid_seed), hybrid_samples)
           for name, g in zip(("selected", "neutral"), arms)}
    # Two most separated selected lineage consensus genomes, without selecting
    # for their rhythm or silently replacing failed consensus networks.
    means = arms[0].mean(1)
    dist = ((means[:, None] - means[None, :])**2).sum(-1)
    pair = np.unravel_index(np.argmax(dist), dist.shape)
    consensus = means[list(pair)]
    cf, consensus_valid = assay(consensus)
    return clean(dict(params=params, canonical=dict(features=optimum.tolist(), cv=canonical_cv.tolist()),
                      feature_names=FEATURES, sites=SITES, trajectories=dict(zip(("selected", "neutral"), records)),
                      fork_mean=fork.mean(0).tolist(), correlations=corr, hybrids=hyb,
                      omega=add_omega(*records), throughput=assay.report(),
                      examples=dict(lineages=list(map(int, pair)), genomes=consensus.tolist(),
                                    features=cf.tolist(), pyloric=consensus_valid.tolist()),
                      elapsed_seconds=perf_counter() - start))


def summary(data):
    p, lines = data["params"], []
    lines.append(f"Experiment E: N={p['N']} K={p['K']} B={p['B']} T={p['T']} s={p['s']} seed={p['seed']}")
    if data.get("status") == "canonical_validation_failed":
        lines.extend(["BLOCKED: " + data["reason"],
                      "No generations evolved; no conductance correlations, hybrids or omega were estimated.",
                      "Canonical diagnostic: " + json.dumps(data["canonical"]),
                      "Throughput: " + str(data["throughput"])])
        return "\n".join(lines) + "\n"
    lines.append("Canonical " + str(dict(zip(FEATURES, data["canonical"]["features"]))))
    for name in ("selected", "neutral"):
        r = data["trajectories"][name][-1]
        lines.append(f"{name}: fitness={r['mean_fitness']:.4g}, pyloric={r['fraction_pyloric']:.3f}, "
                     f"log-g divergence={r['conductance_divergence_sum']:.5g}, rhythm divergence={r['feature_divergence_normalized']}")
        for cell in CELLS:
            c = data["correlations"][name][cell]
            lines.append(f"{name} {cell} strongest correlations: {c['strongest']}")
            lines.append(f"{name} {cell} NaV vs A/KCa/Kd (exploratory Schulz comparison): {c['candidate_K_Na']}")
            lines.append(f"{name} {cell} correlation matrix ({','.join(CHANNELS)}): {c['matrix']}")
        lines.append(f"{name} hybrids: {data['hybrids'][name]}")
    lines.append(f"Omega at t={data['omega']['t_star']}: {data['omega']['per_site']}")
    lines.append(f"Throughput: {data['throughput']}")
    lines.append("Correlations alone do not establish compensation: neutral finite samples and shared ancestry can correlate. "
                 "Compare replicated long runs; this short run, if quick, is only a pipeline test. "
                 "Schulz expression correlations are not direct measurements of these model gbar values.")
    return "\n".join(lines) + "\n"


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(path=OUTPUT / "stg.json"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("experiment") == "E2":
        return figure_replicates(path)
    if data.get("status") == "canonical_validation_failed":
        return diagnostic_figure(data, path, plt)
    gs = np.vstack([CANONICAL, 10**np.asarray(data["examples"]["genomes"])])
    duration = data["params"].get("duration_ms", DURATION)
    window = data["params"].get("window_start_ms", WINDOW)
    _, _, traces = simulate(jnp.asarray(gs), data["params"]["dt"], True, duration)
    traces = np.asarray(traces)[:, round(window / .5):]
    span = (duration - window) / 1000
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": .55}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.85, wspace=.7)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        time = np.arange(traces.shape[1]) * .0005
        for row, label in enumerate(["canonical"] + [f"lineage {i}" for i in data["examples"]["lineages"]]):
            for cell in range(3):
                ax.plot(time, traces[row, :, cell] / 110 + (2-row)*3.7 + (2-cell), color=f"C{cell}", lw=.35)
            suffix = "" if row == 0 or data["examples"]["pyloric"][row-1] else " (fails)"
            ax.text(0, (2-row)*3.7+2.65, label+suffix, fontsize=5, family="monospace")
        for cell, label in enumerate(CELLS):
            ax.text(span * 1.02, 9.4-cell, label, color=f"C{cell}", fontsize=5, family="monospace")
        ax.set(xlabel="Time (s)", yticks=[], xlim=(0, span), ylim=(-1.1, 10.4))
        ax.plot([1.75, 1.75], [-.98, -.98+50/110], color="C0")
        ax.text(1.7, -.98, "50 mV", ha="right", fontsize=5, family="monospace")
        ax = axes[1]
        inset = ax.inset_axes([.12, .52, .56, .42])
        for name, color, style in (("selected", "C0", "-"), ("neutral", "C1", "--")):
            records = data["trajectories"][name]
            t = [r["t"] for r in records]
            y = [r["conductance_divergence_sum"] for r in records]
            ax.plot(t, y, color=color, linestyle=style)
            ax.text(.05, .40 if name == "selected" else .30, name, color=color,
                    transform=ax.transAxes, family="monospace", fontsize=5)
            inset.plot(t, [np.nan if r["feature_divergence_normalized"] is None else r["feature_divergence_normalized"]
                           for r in records], color=color, linestyle=style)
        ax.set(xlabel="Generation", ylabel="Log-conductance divergence", ylim=(0, None))
        inset.set_title("rhythm divergence", fontsize=5, family="monospace")
        inset.tick_params(labelsize=4, length=2)
        inset.spines[["top", "right"]].set_visible(False)
        ax = axes[2]
        matrices = [np.array(data["correlations"][name]["LP"]["matrix"], dtype=float) for name in ("selected", "neutral")]
        joined = np.hstack([matrices[0], np.full((8, 1), np.nan), matrices[1]])
        im = ax.imshow(joined, vmin=-1, vmax=1, aspect="auto", cmap=plt.get_cmap())
        ax.grid(False)
        ax.set(yticks=np.arange(8), yticklabels=CHANNELS, xticks=np.r_[0:8, 9:17],
               xticklabels=CHANNELS*2)
        ax.tick_params(axis="x", labelrotation=90, labelsize=4, length=0)
        ax.tick_params(axis="y", labelsize=5, length=0)
        ax.text(.23, 1.02, "LP selected", transform=ax.transAxes, ha="center", fontsize=5, family="monospace")
        ax.text(.78, 1.02, "neutral", transform=ax.transAxes, ha="center", fontsize=5, family="monospace")
        bar = fig.colorbar(im, ax=ax, fraction=.04, pad=.03, ticks=[-1, 0, 1])
        bar.ax.tick_params(labelsize=5, length=2)
        if data.get("quick"):
            fig.suptitle("Quick pipeline test: N=8, K=2, B=5, T=10", fontsize=6, family="monospace", y=.99)
        for suffix in (".pdf", ".png"):
            fig.savefig(path.with_suffix(suffix), dpi=200)
        plt.close(fig)
    print(f"Wrote {path.with_suffix('.pdf')} and {path.with_suffix('.png')}")


def diagnostic_figure(data, path, plt):
    """Keep invalid runs visibly distinct from scientific experiment results."""
    duration = data["params"].get("duration_ms", DURATION)
    window = data["params"].get("window_start_ms", WINDOW)
    _, _, v = single(jnp.asarray(CANONICAL), data["params"]["dt"], True, duration)
    v = np.asarray(v)[round(window / .5):]
    span = (duration - window) / 1000
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.80, wspace=.65)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        for c, label in enumerate(CELLS):
            ax.plot(np.arange(len(v))*.0005, v[:, c]/110+2-c, color=f"C{c}", lw=.5)
            ax.text(span * 1.015, 2-c, label, color=f"C{c}", fontsize=6, family="monospace")
        ax.set(xlabel="Time after transient (s)", ylabel="Canonical voltage", yticks=[], xlim=(0, span), ylim=(-.75, 3.))
        ax.plot([.08, .08], [2.45, 2.45+50/110], color="C0")
        ax.text(.14, 2.57, "50 mV", fontsize=5, family="monospace")
        counts = "   ".join(f"{c}: {n}" for c, n in zip(CELLS, data["canonical"]["complete_bursts"]))
        for ax, heading, explanation in zip(axes[1:], ("Evolution not started", "No correlation estimate"),
                (f"Complete bursts in {window/1000:g}–{duration/1000:g} s:\n{counts}\n\nRequires at least {MIN_BURSTS}/cell.\nCanonical fitness is zero.",
                 "No selected lineages,\nneutral comparison,\nhybrids or omega.\n\nSee stg-check.json.")):
            ax.set_axis_off()
            ax.text(0, .85, heading, transform=ax.transAxes, fontsize=7, family="monospace")
            ax.text(0, .67, explanation, transform=ax.transAxes, va="top", fontsize=6, family="monospace", linespacing=1.5)
        fig.suptitle("Canonical validation failed — diagnostic only", fontsize=8, family="monospace", y=.98)
        for suffix in (".pdf", ".png"):
            fig.savefig(path.with_suffix(suffix), dpi=200)
        plt.close(fig)
    print(f"Wrote diagnostic {path.with_suffix('.pdf')} and {path.with_suffix('.png')}")


def diagnostic(dt=.05, duration_ms=DURATION, window_start_ms=WINDOW):
    """Raw events expose failures without inventing a measurable phenotype."""
    spikes, counts, _ = single(jnp.asarray(CANONICAL), dt, False, duration_ms)
    f, ok, cv, n = phenotype(spikes, counts, duration_ms, window_start_ms)
    cells = {}
    for name, events in zip(CELLS, spikes):
        starts, ends, valid, _ = bursts(events, duration_ms, window_start_ms)
        starts, ends, valid = map(np.asarray, (starts, ends, valid))
        finite = np.isfinite(starts)
        cells[name] = dict(gap_threshold_ms=float(burst_gap(events, window_start_ms)),
                           burst_starts_ms=starts[finite].tolist(), burst_ends_ms=ends[finite].tolist(),
                           complete_in_window=valid[finite].tolist())
    return clean(dict(pyloric=bool(ok), features=np.asarray(f).tolist(),
                      period_cv=np.asarray(cv).tolist(), complete_bursts=np.asarray(n).tolist(),
                      spike_counts=np.asarray(counts).tolist(), cells=cells))


def trace_events(voltage, dt):
    """Independent host reference: detect events from every integration step."""
    voltage = np.concatenate((np.full((1, 3), -60., dtype=np.float32), np.asarray(voltage)))
    events = np.full((3, MAX_SPIKES), np.inf, dtype=np.float32)
    counts = np.zeros(3, dtype=np.int32)
    for cell in range(3):
        v, vn = voltage[:-1, cell], voltage[1:, cell]
        last = -1e6
        for i in np.flatnonzero((v < -20) & (vn >= -20)):
            t = np.float32(i + 1) * np.float32(dt)
            if t - last >= REFRACTORY:
                crossing = np.float32(i) * np.float32(dt) + np.float32(dt) * (-20 - v[i]) / (vn[i] - v[i])
                if counts[cell] < MAX_SPIKES:
                    events[cell, counts[cell]] = crossing
                counts[cell] += 1
                last = t
    return events, counts


def check(duration_ms=DURATION, window_start_ms=WINDOW):
    validate_assay(.05, duration_ms, window_start_ms)
    start, failures = perf_counter(), []
    def test(name, condition):
        print(f"{'PASS' if condition else 'FAIL'} {name}", flush=True)
        if not condition:
            failures.append(name)

    report = diagnostic(.05, duration_ms, window_start_ms)
    print("Canonical: " + json.dumps(report), flush=True)
    test(f"canonical regular triphasic rhythm in {window_start_ms/1000:g}–{duration_ms/1000:g} s", report["pyloric"])
    isolated = CANONICAL.copy()
    isolated[24:] = 0
    spikes, _, _ = single(jnp.asarray(isolated), .05, False, duration_ms)
    bs, be, complete, _ = bursts(spikes[0], duration_ms, window_start_ms)
    complete = np.asarray(complete)
    intervals = np.diff(np.asarray(bs)[complete])
    test("isolated AB bursts regularly", complete.sum() >= 2 and intervals.std()/intervals.mean() < CV_MAX)
    isolated[1:3] = 0
    off_spikes, _, _ = single(jnp.asarray(isolated), .05, False, duration_ms)
    off_start, off_end, _, _ = bursts(off_spikes[0], duration_ms, window_start_ms)
    test("removing CaT and CaS abolishes AB bursts", not np.any(np.isfinite(np.asarray(off_start)) &
                                                               (np.asarray(off_end) > np.asarray(off_start))))
    # Two unequal genomes ensure vmap cannot pass by broadcasting one answer.
    inputs = np.vstack([CANONICAL, CANONICAL * np.linspace(.99, 1.01, 31)])
    # Full-resolution traces allow an explicit one-step timing envelope;
    # float32 reduction order can shift a sharp upstroke by a fraction of dt.
    trace_batch = jax.jit(jax.vmap(partial(simulate_one, dt=.05, duration_ms=duration_ms,
                                          full_trace=True)))
    bt, bn, bv = trace_batch(jnp.asarray(inputs, dtype=jnp.float32))
    max_error = 0.
    for i in range(2):
        st, sn, sv = single(jnp.asarray(inputs[i]), .05, False, duration_ms, 0., 16, False, True)
        finite = np.isfinite(st) & np.isfinite(bt[i])
        max_error = max(max_error, float(np.max(np.abs(np.asarray(st)[finite] - np.asarray(bt[i])[finite]))))
        test(f"vmap network {i} spike counts", np.array_equal(bn[i], sn))
        test(f"vmap network {i} crossing times", np.allclose(bt[i], st, atol=.05, rtol=0))
        reference = np.asarray(sv)
        neighbors = np.stack((np.vstack((reference[:1], reference[:-1])), reference,
                              np.vstack((reference[1:], reference[-1:]))))
        # Allow 0.2 mV at sampled extrema, where a sub-step peak shift also
        # changes sampled amplitude; event timing is separately bounded by dt.
        test(f"vmap network {i} voltages within one step and 0.2 mV",
             np.all((np.asarray(bv[i]) >= neighbors.min(0) - .2) &
                    (np.asarray(bv[i]) <= neighbors.max(0) + .2)))

    rng = np.random.default_rng(1729)
    mutants = CANONICAL[None, :] * np.exp(rng.normal(0, .015, (3, 31))).astype(np.float32)
    feature_errors = []
    networks = np.vstack((CANONICAL, mutants))
    online_batch = jax.jit(jax.vmap(partial(simulate_one, dt=.05, duration_ms=duration_ms,
                                           window_start_ms=window_start_ms, online=True)))
    batch_events = online_batch(jnp.asarray(networks))
    # Match batch shape/precision so this tests event extraction, independently
    # of trajectory divergence from scalar versus SIMD arithmetic ordering.
    _, _, reference_voltages = trace_batch(jnp.asarray(networks))
    for i, conductances in enumerate(networks):
        events, counts, previous, overflow = (x[i] for x in batch_events)
        voltage = reference_voltages[i]
        reference, ref_counts = trace_events(voltage, .05)
        online_f = phenotype(events, counts, duration_ms, window_start_ms, previous, overflow)
        trace_f = phenotype(jnp.asarray(reference), jnp.asarray(ref_counts), duration_ms, window_start_ms)
        max_timing = 0.
        for c in range(3):
            expected_events = reference[c, np.isfinite(reference[c]) & (reference[c] >= window_start_ms)]
            actual = np.asarray(events[c])[:int(counts[c])]
            test(f"online/trace network {i} cell {c} spike count", len(actual) == len(expected_events))
            error = np.max(np.abs(actual - expected_events), initial=0.) if len(actual) == len(expected_events) else np.inf
            max_timing = max(max_timing, float(error))
        test(f"online/trace network {i} spike times within one step", max_timing <= .05)
        # Period in ms; phase/duty tolerances correspond to one step per edge.
        period = max(float(trace_f[0][0]), 1.)
        atol = np.r_[.05, np.full(7, 2 * .05 / period)]
        error = np.abs(np.asarray(online_f[0]) - np.asarray(trace_f[0]))
        feature_errors.append(error.tolist())
        test(f"online/trace network {i} features", np.all(
            np.isclose(online_f[0], trace_f[0], atol=atol, rtol=1e-5, equal_nan=True)))
        test(f"online/trace network {i} classification and complete bursts",
             bool(online_f[1]) == bool(trace_f[1]) and np.array_equal(online_f[3], trace_f[3]))
        test(f"online/trace network {i} float32 and no overflow",
             events.dtype == voltage.dtype == jnp.float32 and not np.any(overflow))

    def synthetic(phases=(0., .35, .65), durations=(.2, .2, .2), irregular=False):
        result = np.full((3, MAX_SPIKES), np.inf, dtype=np.float32)
        for c in range(3):
            times = []
            for cycle in range(int(DURATION / 500)):
                base = 500 * (cycle + phases[c]) + (200 if irregular and cycle == 12 else 0)
                times.extend(base + np.arange(round(durations[c] * 500 / 10) + 1) * 10)
            a = np.array(times)
            a = np.sort(a[a < DURATION])
            result[c, :len(a)] = a
        return result
    expected = np.array([500, .2, .2, .2, .35, .55, .65, .85])
    fixture = synthetic()
    ff, good, _, _ = phenotype(jnp.asarray(fixture), jnp.isfinite(fixture).sum(1))
    test("synthetic burst period, duties and phases", bool(good) and np.allclose(ff, expected, atol=1e-6))
    for name, fixture in (("reversed order", synthetic((0, .65, .35))),
                          ("AB/LP overlap", synthetic(durations=(.4, .2, .2))),
                          ("LP offset after PY", synthetic(durations=(.2, .55, .2))),
                          ("excess duty", synthetic(durations=(.2, .2, .65))),
                          ("irregular", synthetic(irregular=True))):
        _, good, _, _ = phenotype(jnp.asarray(fixture), jnp.isfinite(fixture).sum(1))
        test("synthetic rejects " + name, not bool(good))
    test("synthetic silence", not bool(phenotype(jnp.full((3, MAX_SPIKES), jnp.inf), jnp.zeros(3))[1]))
    fixture = synthetic(durations=(.2, .4, .2))
    test("synthetic permits LP/PY overlap", bool(phenotype(
        jnp.asarray(fixture), np.isfinite(fixture).sum(1))[1]))
    fixture = synthetic()
    # Remove one follower burst: remaining matched cycles must not hide it.
    fixture[2, (fixture[2] >= 6325) & (fixture[2] <= 6425)] = np.inf
    fixture.sort(axis=1)
    test("synthetic rejects missing PY cycle", not bool(phenotype(
        jnp.asarray(fixture), np.isfinite(fixture).sum(1))[1]))
    fixture = synthetic()
    extra = np.arange(24) * 500 + 100  # >100 ms from both neighboring PY bursts.
    times = np.sort(np.r_[fixture[2, np.isfinite(fixture[2])], extra])
    fixture[2] = np.inf
    fixture[2, :len(times)] = times
    ff, good, _, _ = phenotype(jnp.asarray(fixture), np.isfinite(fixture).sum(1))
    test("synthetic ignores isolated PY spikes", bool(good) and np.allclose(ff, expected))
    events = np.full(MAX_SPIKES, np.inf)
    # Slow within-burst ISIs exceed the old fixed threshold; singleton/doublet
    # groups remain separated by >360 ms and must disappear from diagnostics.
    events[:18] = [1000, 1120, 1240, 3000, 4000, 4050, 5000, 5120, 5240,
                   7000, 7120, 7240, 9000, 9120, 9240, 11000, 11120, 11240]
    bs, be, complete, _ = bursts(jnp.asarray(events))
    test("adaptive detector joins slow bursts and drops groups below three spikes",
         np.isclose(burst_gap(jnp.asarray(events)), 360) and
         np.array_equal(np.asarray(bs)[np.isfinite(bs)], [1000, 5000, 7000, 9000, 11000]) and
         int(complete.sum()) == 4 and np.all(np.asarray(be)[np.isfinite(bs)] > np.asarray(bs)[np.isfinite(bs)]))
    test("spike buffer overflow rejects phenotype", not bool(phenotype(
        jnp.asarray(synthetic()), jnp.full(3, MAX_SPIKES + 1))[1]))
    fixture = synthetic()
    # Put the left boundary inside a burst, then compare full/window-only input.
    boundary = 4050.
    trimmed = np.full_like(fixture, np.inf)
    previous = np.full(3, -np.inf, dtype=np.float32)
    for c in range(3):
        before = fixture[c, fixture[c] < boundary]
        after = fixture[c, np.isfinite(fixture[c]) & (fixture[c] >= boundary)]
        previous[c] = before[-1] if len(before) else -np.inf
        trimmed[c, :len(after)] = after
    full = phenotype(jnp.asarray(fixture), np.isfinite(fixture).sum(1), DURATION, boundary)
    windowed = phenotype(jnp.asarray(trimmed), np.isfinite(trimmed).sum(1), DURATION,
                         boundary, jnp.asarray(previous))
    test("window buffer preserves left-censored burst exclusion",
         np.allclose(full[0], windowed[0], equal_nan=True) and np.array_equal(full[3], windowed[3]))
    test("explicit overflow flag rejects phenotype", not bool(phenotype(
        jnp.asarray(synthetic()), jnp.full(3, 10), overflow=jnp.array([True, False, False]))[1]))
    small = diagnostic(.025, duration_ms, window_start_ms)
    f0, f1 = np.array(report["features"], float), np.array(small["features"], float)
    convergence = None
    if np.isfinite(f0).all() and np.isfinite(f1).all():
        # Relative period error <3%; absolute duty/phase errors <0.05.
        errors = np.abs(f1-f0) / np.r_[f0[0], np.ones(7)]
        convergence = errors.tolist()
    test("dt feature convergence: period <3%, duties/phases <0.05",
         small["pyloric"] and convergence is not None and convergence[0] < .03 and max(convergence[1:]) < .05)
    # Report crossing convergence even when the canonical feature assay fails.
    a = np.array(report["cells"]["AB"]["burst_starts_ms"])
    b = np.array(small["cells"]["AB"]["burst_starts_ms"])
    raw_periods = [float(np.diff(x).mean()) for x in (a, b)]
    print(f"Raw AB onset periods (whole simulation, including transient): {raw_periods} ms", flush=True)
    floor_f, floor_ok, _, _ = features(10**ANCESTOR, .05, duration_ms, window_start_ms)
    test("conductance floor preserves classification", floor_ok == report["pyloric"])
    rng = np.random.default_rng(5)
    g = rng.normal(size=(2, 8, 31))
    f = np.broadcast_to(expected, (2, 8, 8)).copy()
    valid = np.ones((2, 8), bool)
    a = step(g, f, valid, expected, 0, .02, .05, np.random.default_rng(4))
    b = step(g, f, np.zeros_like(valid), expected, 0, .02, .05, np.random.default_rng(4))
    test("neutral reproduction ignores pyloricity", np.array_equal(a, b))
    a = step(g, f, valid, expected, 1, .02, .05, np.random.default_rng(4))
    test("equal fitness matches neutral ancestry and mutation", np.array_equal(a, b))
    plan = random_plan(g.shape, 3, .02, .05, np.random.default_rng(4))
    host, device = g.astype(np.float32), jnp.asarray(g, dtype=jnp.float32)
    reference_rng = np.random.default_rng(4)
    for t in range(3):
        host = step(host, f, valid, expected, 1, .02, .05, reference_rng)
        device = planned_step(device, f, valid, expected, 1., plan, jnp.int32(t))
        if t == 0:
            step_traces = STEP_TRACES
    test("evolution step does not retrace per generation", STEP_TRACES == step_traces)
    test("device reproduction preserves seeded ancestry and mutations",
         np.allclose(host, device, atol=3e-7, rtol=1e-6) and device.dtype == jnp.float32)
    neutral_device = device_step(jnp.asarray(g, dtype=jnp.float32), f, np.zeros_like(valid),
                                 expected, 0., *(x[0] for x in plan))
    selected_device = device_step(jnp.asarray(g, dtype=jnp.float32), f, valid,
                                  expected, 1., *(x[0] for x in plan))
    test("device neutral ignores validity and matches equal-fitness selection",
         np.array_equal(neutral_device, selected_device))
    weighted_f = f.astype(np.float32)
    weighted_f[..., 0] += np.arange(8, dtype=np.float32) * 30
    weighted_f[..., 4] += np.linspace(-.6, .6, 8, dtype=np.float32)
    weighted_valid = valid.copy()
    weighted_valid[:, 0] = False
    weighted_host = step(g.astype(np.float32), weighted_f, weighted_valid, expected,
                         1., .02, .05, np.random.default_rng(4))
    weighted_device = device_step(jnp.asarray(g, dtype=jnp.float32), weighted_f, weighted_valid,
                                  expected, 1., *(x[0] for x in plan))
    test("device weighted selection and wrapped phases match reference",
         np.allclose(weighted_host, weighted_device, atol=3e-7, rtol=1e-6))
    try:
        step(g, f, np.zeros_like(valid), expected, 1, .02, .05, rng)
        raised = False
    except RuntimeError:
        raised = True
    test("selected extinction is explicit", raised)
    same = np.repeat(g[:1], 2, axis=0)
    record = measure(same, f, valid, expected, 1)
    test("identical fork has zero divergence", record["conductance_divergence_sum"] == 0)
    selected = [dict(t=10, conductance_divergence=[.5]*31)]
    neutral = [dict(t=10, conductance_divergence=[2.]*30+[0.])]
    omega = add_omega(selected, neutral)
    test("fixed-time omega and zero denominator", np.allclose(omega["per_site"][:30], .25) and
         np.isnan(omega["per_site"][-1]))
    check_replicates(test, duration_ms, window_start_ms)
    assay = Assay(.05, duration_ms, window_start_ms)
    genomes = np.broadcast_to(ANCESTOR, (2, 8, 31)).copy()
    assay(genomes)
    assay(genomes)
    test("warmed assay compiles once", assay.compile_count == 1)
    throughput = assay.report()
    print(f"Throughput: {throughput}", flush=True)
    result = clean(dict(status="passed" if not failures else "failed", failures=failures,
                        params=dict(duration_ms=duration_ms, window_start_ms=window_start_ms),
                        feature_names=FEATURES, canonical=report, half_dt=small, dt_feature_errors=convergence,
                        raw_AB_period_ms=raw_periods, batched_max_crossing_error_ms=max_error, online_trace_feature_errors=feature_errors,
                        isolated_AB_period_ms=float(intervals.mean()), throughput=throughput,
                        seconds=perf_counter()-start))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "stg-check.json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    print(f"check: {len(failures)} failures ({perf_counter()-start:.1f}s)", flush=True)
    return not failures


# Experiment E2 is additive: the original E runner and output schema stay intact.
ARMS = ("selected", "neutral")


def check_replicates(test, duration_ms, window_start_ms):
    seeds = replicate_streams(0, 3)
    streams = [np.random.default_rng(x[1]).random(128) for x in seeds]
    test("replicate streams differ", all(not np.array_equal(streams[i], streams[j])
                                         for i in range(3) for j in range(i+1, 3)))
    test("replicate r has standalone seed r stream", all(np.array_equal(
        streams[r], np.random.default_rng(np.random.SeedSequence(r).spawn(3)[1]).random(128))
        for r in range(3)))
    test("seed offset and replicate count preserve streams", all(np.array_equal(
        streams[r], np.random.default_rng(replicate_streams(r, 1)[0][1]).random(128)) for r in range(3)))
    phase_draws = [[np.random.default_rng(seed).random(32) for seed in phases] for phases in seeds]
    test("burn, fork and hybrid streams are independent across replicates", all(
        not np.array_equal(phase_draws[i][phase], phase_draws[j][phase])
        for phase in range(3) for i in range(3) for j in range(i+1, 3)))
    full = random_plan((2, 4, 31), 5, .2, .05, np.random.default_rng(seeds[0][1]))
    blocked = ReplicatePlans((2, 4, 31), 5, .2, .05, [seeds[0][1]], block_size=2)
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (2, 4, 31))
    a, b = g, g
    f = np.broadcast_to([500., .2, .2, .2, .35, .55, .65, .85], (2, 4, 8))
    valid = np.ones((2, 4), bool)
    for t in range(5):
        plans, index = blocked.at(t)
        a = planned_step(a, f, valid, f[0, 0], 0., full, jnp.int32(t))
        b = planned_step(b, f, valid, f[0, 0], 0., plans[0], index)
        if t == 0:
            traces = STEP_TRACES
    test("bounded plans preserve exact draws across block boundaries", np.array_equal(a, b))
    test("bounded plans do not retrace at block boundaries", STEP_TRACES == traces)
    ci = replicate_ci([1., 2., None])
    test("replicate CI counts missing values and uses replicate sample size",
         ci["n"] == 2 and ci["mean"] == 1.5 and ci["ci95"][0] < 1 and ci["ci95"][1] > 2)
    test("single replicate CI is undefined", np.isnan(replicate_ci([1.])["ci95"]).all())
    def correlation_fixture(selected, neutral):
        fixtures = []
        for rs, rn in zip(selected, neutral):
            result = {arm: {} for arm in ARMS}
            for arm, value in zip(ARMS, (rs, rn)):
                for level in ("individual", "lineage"):
                    matrix = np.full((8, 8), np.nan)
                    matrix[0, 3] = matrix[3, 0] = value
                    result[arm][level] = {cell: dict(matrix=matrix.tolist()) for cell in CELLS}
            fixtures.append(dict(correlations=result))
        return replicate_correlations(fixtures, 16)["lineage"][2]  # AB NaV-A
    rs, rn = np.linspace(.6, .7, 8), np.linspace(.1, .12, 8)
    supported = correlation_fixture(rs, rn)
    test("consistent selected correlation differs from paired neutrals", supported["selection_associated"] and
         supported["selected"]["same_sign"] == 8 and supported["paired_holm_p"] < .05)
    test("matching selected/neutral correlations are not selection-associated",
         not correlation_fixture(rs, rs)["selection_associated"])
    rs[:2] *= -1
    test("six of eight selected signs do not pass consistency",
         not correlation_fixture(rs, rn)["selection_associated"])
    # Exercise actual burn-in, fork, selection, neutral control and hybrid paths.
    kwargs = dict(N=2, K=2, B=1, T=2, every=1, seed=9, hybrid_samples=2,
                  duration_ms=duration_ms, window_start_ms=window_start_ms)
    legacy = run(**kwargs)
    batched = run_replicates(**kwargs, replicates=1)
    one = batched["replicates"][0]
    for key in ("fork_mean", "trajectories", "omega"):
        test(f"R=1 exactly preserves legacy short-run {key}", one[key] == legacy[key])
    test("R=1 exactly preserves individual correlations", all(
        one["correlations"][arm]["individual"] == legacy["correlations"][arm] for arm in ARMS))
    test("R=1 exactly preserves hybrid sampling and results", one["hybrids"] == legacy["hybrids"])
    test("K=2 never produces selection-associated calls", not any(
        row["selection_associated"] for row in batched["aggregate"]["correlations"]["lineage"]))
    # Test the chunk boundary and output order independently of phenotype dynamics.
    probe = ReplicateAssay(.05, 100., 0., 4, chunk_size=3)
    genomes = np.broadcast_to(ANCESTOR, (2, 2, 31)).copy()
    genomes[1, 0, 0] += .1
    cf, cv = probe(genomes)
    reference = Assay(.05, 100., 0.)
    reference.unroll = probe.unroll
    ff, fv = reference(genomes)
    test("assay chunks preserve shape, order and classification", cf.shape == ff.shape and
         np.array_equal(cv, fv) and np.array_equal(cf, ff, equal_nan=True))


class ReplicateAssay(Assay):
    """Flatten R x arms x K x N into the existing vmap; optionally bound memory."""
    def __init__(self, dt, duration_ms, window_start_ms, batch_size, chunk_size=None):
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk-size must be positive")
        self.chunk_size = chunk_size
        super().__init__(dt, duration_ms, window_start_ms,
                         min(batch_size, chunk_size) if chunk_size else batch_size)

    def __call__(self, g):
        if not self.chunk_size or np.prod(g.shape[:-1]) <= self.chunk_size:
            return super().__call__(g)
        shape = g.shape[:-1]
        flat = g.reshape(-1, 31)
        results = [super(ReplicateAssay, self).__call__(flat[i:i+self.chunk_size])
                   for i in range(0, len(flat), self.chunk_size)]
        return (np.concatenate([x[0] for x in results]).reshape(*shape, 8),
                np.concatenate([x[1] for x in results]).reshape(shape))


def replicate_streams(seed, replicates):
    """Index r is the legacy standalone seed seed+r, including r=0 exactly."""
    return [np.random.SeedSequence(seed + r).spawn(3) for r in range(replicates)]


class ReplicatePlans:
    """Bound storage to 32 generations with fixed shapes and unchanged RNG order.

    Padding to the maximum mutation count avoids recompiling reproduction at
    every block. At R=8, K=16, N=64 the fork plans occupy about 66 MB.
    """
    def __init__(self, shape, generations, u, sigma, seeds, block_size=32):
        self.shape, self.generations = shape, generations
        self.u, self.sigma, self.block_size = u, sigma, block_size
        self.rngs = [np.random.default_rng(seed) for seed in seeds]
        self.plans = None

    def at(self, t):
        if t % self.block_size == 0:
            self.plans = []
            length = min(self.block_size, self.generations-t)
            size = int(np.prod(self.shape))
            for rng in self.rngs:
                raw = random_plan(self.shape, length, self.u, self.sigma, rng)
                draws = np.zeros((self.block_size, *self.shape[:2], 1), dtype=np.float32)
                sites = np.full((self.block_size, size), size, dtype=np.int32)
                changes = np.zeros((self.block_size, size), dtype=np.float32)
                draws[:length] = np.asarray(raw[0])
                sites[:length, :raw[1].shape[1]] = np.asarray(raw[1])
                changes[:length, :raw[2].shape[1]] = np.asarray(raw[2])
                self.plans.append(tuple(jnp.asarray(x) for x in (draws, sites, changes)))
        return self.plans, jnp.int32(t % self.block_size)


def replicate_hybrid_group(g, rng, samples):
    """Same parent sampling and recombination draws as hybrids(), before assay."""
    K, N = g.shape[:2]
    a = rng.integers(K, size=samples)
    b = (a + rng.integers(1, K, size=samples)) % K
    p = g[a, rng.integers(N, size=samples)]
    q = g[b, rng.integers(N, size=samples)]
    r = g[a, rng.integers(N, size=samples)]
    cross = np.where(rng.random(p.shape) < .5, p, q)
    within = np.where(rng.random(p.shape) < .5, p, r)
    return np.concatenate([p, q, r, cross, within])


def replicate_ci(values):
    """Pointwise Student-t CI across independent replicates, never individuals.

    Missing measurements are excluded explicitly; n is reported at each point.
    One observation has a mean but no estimable confidence interval.
    """
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    n = finite.sum(axis=0)
    mean = np.sum(np.where(finite, x, 0), axis=0) / np.maximum(n, 1)
    ss = np.sum(np.where(finite, (x - mean)**2, 0), axis=0)
    se = np.sqrt(ss / np.maximum(n - 1, 1) / np.maximum(n, 1))
    half = t.ppf(.975, np.maximum(n - 1, 1)) * se
    return dict(mean=np.where(n > 0, mean, np.nan).tolist(),
                ci95=np.stack([np.where(n > 1, mean-half, np.nan),
                               np.where(n > 1, mean+half, np.nan)], axis=-1).tolist(),
                n=n.tolist())


def replicate_signs(values):
    from scipy.stats import binomtest
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    return dict(**replicate_ci(x), positive=pos, negative=neg, zero=int((x == 0).sum()),
                same_sign=max(pos, neg), majority_sign=1 if pos > neg else -1 if neg > pos else 0,
                sign_test_p=float(binomtest(pos, pos+neg, .5).pvalue) if pos+neg else None)


def replicate_correlations(results, K):
    """Raw-r paired t tests, with Holm correction over 84 pairs per level."""
    from scipy.stats import t
    R = len(results)
    levels = {}
    for level in ("individual", "lineage"):
        rows = []
        for cell in CELLS:
            for i in range(8):
                for j in range(i+1, 8):
                    values = {arm: np.array([r["correlations"][arm][level][cell]["matrix"][i][j]
                                             for r in results], dtype=float) for arm in ARMS}
                    delta = values["selected"] - values["neutral"]
                    delta = delta[np.isfinite(delta)]
                    p = None
                    if len(delta) >= 2:
                        se = delta.std(ddof=1) / np.sqrt(len(delta))
                        p = float(2*t.sf(abs(delta.mean()/se), len(delta)-1)) if se > 0 else (1. if delta.mean() == 0 else 0.)
                    rows.append(dict(cell=cell, pair=[CHANNELS[i], CHANNELS[j]],
                                     **{arm: replicate_signs(x) for arm, x in values.items()},
                                     paired_difference=replicate_ci(delta), paired_t_p=p,
                                     paired_holm_p=None, selection_associated=False))
        ordered = sorted((i for i, row in enumerate(rows) if row["paired_t_p"] is not None),
                         key=lambda i: rows[i]["paired_t_p"])
        adjusted = 0.
        for rank, i in enumerate(ordered):
            # Keep the full prespecified family even if some pairs are undefined.
            adjusted = max(adjusted, min(1., (len(rows)-rank)*rows[i]["paired_t_p"]))
            rows[i]["paired_holm_p"] = adjusted
            rows[i]["selection_associated"] = bool(
                level == "lineage" and K >= 3 and R >= 3 and
                rows[i]["selected"]["same_sign"] >= int(np.ceil(.875*R)) and
                rows[i]["paired_difference"]["n"] == R and adjusted < .05)
        levels[level] = rows
    return levels


def aggregate_replicates(results, K):
    aggregate = dict(trajectories={}, hybrids={}, omega={}, correlations=replicate_correlations(results, K))
    metrics = ("fraction_pyloric", "feature_divergence_normalized", "conductance_divergence_sum",
               "conductance_divergence", "mean_fitness")
    for arm in ARMS:
        aggregate["trajectories"][arm] = [dict(t=record["t"], **{
            metric: replicate_ci([r["trajectories"][arm][i][metric] for r in results])
            for metric in metrics}) for i, record in enumerate(results[0]["trajectories"][arm])]
        aggregate["hybrids"][arm] = {group: {
            metric: replicate_ci([r["hybrids"][arm][group][metric] for r in results])
            for metric in ("fraction_pyloric", "mean_fitness")}
            for group in results[0]["hybrids"][arm]}
    aggregate["omega"] = {key: replicate_ci([r["omega"][key] for r in results])
                          for key in ("per_site", "summed")}
    return aggregate


def run_replicates(N=64, K=16, B=200, T=2000, every=20, u=.02, sigma=.05, s=1., seed=0,
                   dt=.05, hybrid_samples=64, duration_ms=DURATION, window_start_ms=WINDOW,
                   replicates=8, chunk_size=None):
    validate_assay(dt, duration_ms, window_start_ms)
    if N < 1 or K < 2 or min(B, T) < 0 or every < 1 or hybrid_samples < 1 or replicates < 1 or seed < 0:
        raise ValueError("Require N,R>=1, K>=2, B,T,seed>=0, every,hybrid-samples>=1")
    if not 0 <= u <= 1 or not np.isfinite([sigma, s]).all() or min(sigma, s) < 0:
        raise ValueError("Require u in [0,1], finite sigma,s>=0")
    start = perf_counter()
    optimum, ok, cv, n = features(dt=dt, duration_ms=duration_ms, window_start_ms=window_start_ms)
    if not ok:
        raise RuntimeError(f"Canonical validation failed: features={optimum.tolist()}, bursts={n.tolist()}, CV={cv.tolist()}")
    R = replicates
    params = dict(N=N, K=K, B=B, T=T, every=every, u=u, sigma=sigma, s=s, seed=seed, dt=dt,
                  hybrid_samples=hybrid_samples, duration_ms=duration_ms, window_start_ms=window_start_ms,
                  zero_floor=FLOOR, replicates=R, chunk_size=chunk_size)
    assay = ReplicateAssay(dt, duration_ms, window_start_ms, R*2*K*N, chunk_size)
    seeds = replicate_streams(seed, R)
    burn = ReplicatePlans((1, N, 31), B, u, sigma, [x[0] for x in seeds])
    fork = ReplicatePlans((K, N, 31), T, u, sigma, [x[1] for x in seeds])
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (R, 1, N, 31))
    f, valid = assay(g)

    def require_parents(valid, phase, generation):
        if s != 0 and np.any(~valid.any(-1)):
            raise RuntimeError(f"No pyloric parents: {phase} generation {generation}, "
                               f"[replicate, lineage] {np.argwhere(~valid.any(-1)).tolist()}")

    for t in trange(B, desc="replicate burn-in"):
        require_parents(valid, "burn-in", t)
        plans, index = burn.at(t)
        g = jnp.stack([planned_step(g[r], f[r], valid[r], optimum, s, plans[r], index) for r in range(R)])
        f, valid = assay(g)
    fork_means = np.asarray(g[:, 0].mean(1))
    arms = jnp.broadcast_to(g[:, None], (R, 2, K, N, 31))
    results = [dict(index=r, seed=seed+r, fork_mean=fork_means[r].tolist(),
                    trajectories={arm: [] for arm in ARMS}) for r in range(R)]
    compile_count = None
    for t in trange(T+1, desc="replicate forked lineages"):
        all_f, all_valid = assay(arms)
        if compile_count is None:
            compile_count = assay.compile_count
        assert assay.compile_count == compile_count, "Generation assay recompiled"
        if t % every == 0 or t == T:
            for r in range(R):
                for a, arm in enumerate(ARMS):
                    record = dict(t=t, **measure(arms[r, a], all_f[r, a], all_valid[r, a], optimum, s))
                    record["mean_reproductive_weight"] = record["mean_fitness"] if a == 0 and s != 0 else 1.
                    results[r]["trajectories"][arm].append(record)
        if t < T:
            require_parents(all_valid[:, 0], "selected fork", t)
            plans, index = fork.at(t)
            arms = jnp.stack([jnp.stack([
                planned_step(arms[r, a], all_f[r, a], all_valid[r, a], optimum,
                             s if a == 0 else 0., plans[r], index) for a in range(2)]) for r in range(R)])
    arms = np.asarray(arms)
    # Batch every replicate's matched parents and hybrids into the same assay.
    groups = np.stack([np.stack([replicate_hybrid_group(
        arms[r, a], np.random.default_rng(seeds[r][2]), hybrid_samples) for a in range(2)]) for r in range(R)])
    hf, hv = assay(groups)
    m = hybrid_samples
    for r in range(R):
        results[r]["correlations"], results[r]["hybrids"] = {}, {}
        for a, arm in enumerate(ARMS):
            results[r]["correlations"][arm] = dict(individual=correlations(arms[r, a]),
                lineage=correlations(arms[r, a].astype(float).mean(1)))
            weights, _ = fitness(hf[r, a], hv[r, a], optimum, s)
            results[r]["hybrids"][arm] = {}
            for group, ids in (("cross_parents", np.r_[0:2*m]), ("within_parents", np.r_[0:m, 2*m:3*m]),
                               ("cross_lineage", np.r_[3*m:4*m]), ("within_lineage", np.r_[4*m:5*m])):
                results[r]["hybrids"][arm][group] = dict(n=len(ids),
                    fraction_pyloric=float(hv[r, a, ids].mean()), mean_fitness=float(weights[ids].mean()))
        results[r]["omega"] = add_omega(*(results[r]["trajectories"][arm] for arm in ARMS))
    return clean(dict(experiment="E2", params=params, sites=SITES, feature_names=FEATURES,
                      canonical=dict(features=optimum.tolist(), cv=cv.tolist()), replicates=results,
                      aggregate=aggregate_replicates(results, K), throughput=assay.report(),
                      elapsed_seconds=perf_counter()-start,
                      inference=dict(seed_rule="standalone seed = seed + zero-based replicate index",
                        ci="pointwise 95% Student-t across finite replicate values; n reported; undefined if n<2",
                        test="two-sided paired t-test on selected minus neutral raw r; Holm across 84 pairs per level",
                        selection_associated="lineage level only: >=ceil(0.875*R) same selected sign, all R paired, Holm p<0.05, R>=3, K>=3",
                        rhythm="conditioned on pyloricity; undefined if any lineage has no pyloric networks",
                        measurement="Model gbar correlations are not the same measurement as mRNA correlations.")))


def summary_replicates(data):
    p, a = data["params"], data["aggregate"]
    def fmt(x):
        if x["mean"] is None:
            return f"undefined (n={x['n']})"
        lo, hi = x["ci95"]
        interval = f"[{lo:.4g}, {hi:.4g}]" if lo is not None else "undefined"
        return f"{x['mean']:.4g} CI95 {interval} (n={x['n']})"
    lines = [f"Experiment E2: R={p['replicates']} N={p['N']} K={p['K']} B={p['B']} T={p['T']} seed={p['seed']}"]
    lines.extend(data["inference"].values())
    if p["K"] < 3 or p["replicates"] < 3:
        lines.append("PIPELINE TEST ONLY: K=2 gives degenerate +/-1 lineage correlations; R<3 cannot establish consistency. No selection-associated calls.")
    for arm in ARMS:
        last = a["trajectories"][arm][-1]
        for metric in ("fraction_pyloric", "feature_divergence_normalized", "conductance_divergence_sum"):
            lines.append(f"{arm} {metric}: {fmt(last[metric])}")
        for group, values in a["hybrids"][arm].items():
            lines.append(f"{arm} hybrid {group}: {fmt(values['fraction_pyloric'])}")
    lines.append("Omega summed: " + fmt(a["omega"]["summed"]))
    for i, site in enumerate(SITES):
        value = {k: v[i] for k, v in a["omega"]["per_site"].items()}
        lines.append(f"Omega {site}: {fmt(value)}")
    lines.append("Per-replicate final metrics (rhythm divergence conditions on pyloricity):")
    for r in data["replicates"]:
        for arm in ARMS:
            last = r["trajectories"][arm][-1]
            lines.append(f"replicate {r['index']} seed={r['seed']} {arm}: pyloric={last['fraction_pyloric']:.4g}, "
                         f"rhythm={last['feature_divergence_normalized']}, log-g={last['conductance_divergence_sum']:.4g}, "
                         f"between/within hybrids={r['hybrids'][arm]['cross_lineage']['fraction_pyloric']:.4g}/"
                         f"{r['hybrids'][arm]['within_lineage']['fraction_pyloric']:.4g}")
    lines.extend(["Schulz et al. (2006) comparison: requested candidate pairs, motivated by the supplied LP/PD context.",
                  "The supplied context describes positive Na-A and IA-IKd associations, especially LP and PD; NaV-Kd and CaS-A are additional model comparisons.",
                  "Model AB represents the coupled AB/PD pacemaker; it is not a separate measured PD neuron.",
                  "Model gbar correlations are not the same measurement as mRNA correlations; these are exploratory comparisons, not a replication.",
                  "Cell | pair | selected lineage r (95% CI) | neutral lineage r (95% CI) | sign count | paired Holm p | selection-associated"])
    candidates = {("NaV", "A"), ("A", "Kd"), ("NaV", "Kd"), ("CaS", "A")}
    def row_text(row):
        return (f"{row['cell']} | {'-'.join(row['pair'])} | {fmt(row['selected'])} | {fmt(row['neutral'])} | "
                f"{row['selected']['same_sign']}/{p['replicates']} | {row['paired_holm_p']} | {row['selection_associated']}")
    lines.extend(row_text(row) for row in a["correlations"]["lineage"] if tuple(row["pair"]) in candidates)
    for level, rows in a["correlations"].items():
        lines.append(f"All {level} pairs: mean r, CI, same-sign counts and two-sided sign tests; paired arm tests")
        for row in rows:
            lines.append(row_text(row) + f" | neutral sign={row['neutral']['same_sign']}/{p['replicates']} "
                         f"| sign p S/N={row['selected']['sign_test_p']}/{row['neutral']['sign_test_p']} "
                         f"| paired p={row['paired_t_p']}")
    lines.append("Throughput: " + str(data["throughput"]))
    return "\n".join(lines) + "\n"


def figure_replicates(path=OUTPUT / "stg-replicates.json"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401
    path = Path(path)
    data = json.loads(path.read_text())
    aggregate = data["aggregate"]
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": .55}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.85, wspace=.7)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        for c, cell in enumerate(CELLS):
            for row in aggregate["correlations"]["lineage"]:
                if row["cell"] != cell:
                    continue
                x, y = row["selected"], row["neutral"]
                if x["mean"] is None or y["mean"] is None:
                    continue
                def error(v):
                    ci = np.asarray(v["ci95"], float)
                    # Correlation means lie in [-1,1]; truncate displayed t intervals.
                    return np.abs(np.clip(ci, -1, 1) - v["mean"]).reshape(2, 1) if np.isfinite(ci).all() else None
                ax.errorbar(x["mean"], y["mean"], xerr=error(x), yerr=error(y),
                            fmt="*" if row["selection_associated"] else "o", color=f"C{c}",
                            ms=4 if row["selection_associated"] else 2, alpha=.6, elinewidth=.3)
            ax.plot([], [], "o", color=f"C{c}", ms=3, label=cell)
        ax.axhline(0, color=".7", lw=.4)
        ax.axvline(0, color=".7", lw=.4)
        ax.set(xlabel="Selected mean lineage r", ylabel="Neutral mean lineage r", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
        ax.legend(fontsize=4, frameon=False, loc="upper left", ncol=3, columnspacing=.5, handletextpad=.3)
        ax.text(.02, -.31, "95% CI clipped to ±1; * consistency + Holm p < .05",
                transform=ax.transAxes, fontsize=4)
        ax = axes[1]
        inset = ax.inset_axes([.16, .57, .60, .36])
        for arm, color, style in (("selected", "C0", "-"), ("neutral", "C1", "--")):
            records = aggregate["trajectories"][arm]
            times = [r["t"] for r in records]
            for target, metric in ((ax, "conductance_divergence_sum"), (inset, "feature_divergence_normalized")):
                means = np.asarray([r[metric]["mean"] for r in records], float)
                ci = np.asarray([r[metric]["ci95"] for r in records], float)
                target.plot(times, means, color=color, ls=style, label=arm)
                target.fill_between(times, np.maximum(0, ci[:, 0]), ci[:, 1], color=color, alpha=.15, lw=0)
            ax.text(.04, .44 if arm == "selected" else .34, arm, color=color, transform=ax.transAxes,
                    fontsize=5, family="monospace")
        ax.set(xlabel="Generation", ylabel="Log-conductance divergence", ylim=(0, None))
        inset.set_title("Rhythm divergence", fontsize=5, family="monospace")
        inset.tick_params(labelsize=4, length=2)
        inset.spines[["top", "right"]].set_visible(False)
        inset.set_ylim(bottom=0)
        ax = axes[2]
        for a, arm in enumerate(ARMS):
            for g, group in enumerate(("cross_lineage", "within_lineage")):
                x = g + (a-.5)*.25
                y = [r["hybrids"][arm][group]["fraction_pyloric"] for r in data["replicates"]]
                ax.scatter(x+np.linspace(-.055, .055, len(y)), y, s=9, color=f"C{a}", alpha=.65,
                           marker="o" if a == 0 else "s", label=arm if g == 0 else None)
                ax.plot([x-.09, x+.09], [np.mean(y)]*2, color=f"C{a}", lw=1.3)
        ax.set(xticks=[0, 1], xticklabels=["Between", "Within"], xlabel="Hybrid parent lineages",
               ylabel="Fraction pyloric", ylim=(-.05, 1.05), xlim=(-.4, 1.4))
        ax.legend(fontsize=4, frameon=False, loc="center")
        p = data["params"]
        title = f"E2: R={p['replicates']}, N={p['N']}, K={p['K']}, B={p['B']}, T={p['T']}"
        if p["K"] < 3 or p["replicates"] < 3:
            title += " — pipeline test only"
        fig.suptitle(title, fontsize=6, family="monospace", y=.99)
        for suffix in (".pdf", ".png"):
            fig.savefig(path.with_suffix(suffix), dpi=200)
        plt.close(fig)
    print(f"Wrote {path.with_suffix('.pdf')} and {path.with_suffix('.png')}")


def replicate_work(R, N, K, B, T, hybrid_samples, duration_ms):
    # Initial/burn assays, t=0..T fork assays, five hybrid groups per arm.
    return R*((B+1)*N + (T+1)*2*K*N + 10*hybrid_samples)*duration_ms/1000


def bench_replicates(replicates=8, N=64, K=16, B=200, T=2000, hybrid_samples=64,
                     dt=.05, duration_ms=DURATION, window_start_ms=WINDOW, chunk_size=None):
    validate_assay(dt, duration_ms, window_start_ms)
    if min(replicates, N, hybrid_samples) < 1 or K < 2 or min(B, T) < 0:
        raise ValueError("Require R,N,hybrid-samples>=1, K>=2, B,T>=0")
    assay = ReplicateAssay(dt, duration_ms, window_start_ms, replicates*2*K*N, chunk_size)
    g = jnp.broadcast_to(jnp.asarray(ANCESTOR), (replicates, 2, K, N, 31))
    print(f"Precompiling one generation: {replicates*2*K*N} networks on {jax.devices()[0]}", flush=True)
    # The unroll probes warm execution for 100 ms. Compile the full-duration
    # executable separately so CPU benchmarking needs only one full generation.
    flat = g.reshape(-1, 31)
    stride = chunk_size or len(flat)
    for offset in range(0, len(flat), stride):
        sample = flat[offset:offset+stride]
        if sample.shape not in assay.compiled:
            assay.compiled[sample.shape] = evaluate.lower(
                sample, dt, duration_ms, window_start_ms, assay.unroll).compile()
            assay.compile_count += 1
    plan_blocks = ReplicatePlans((K, N, 31), 1, .02, .05,
                                [x[1] for x in replicate_streams(0, replicates)])
    plans, index = plan_blocks.at(0)
    optimum = np.asarray(features(dt=dt, duration_ms=duration_ms, window_start_ms=window_start_ms)[0])
    f = np.broadcast_to(optimum, (*g.shape[:-1], 8))
    valid = np.ones(g.shape[:-1], dtype=bool)
    def reproduction(f, valid):
        return jax.block_until_ready(jnp.stack([jnp.stack([
            planned_step(g[r, a], f[r, a], valid[r, a], optimum, 1. if a == 0 else 0.,
                         plans[r], index) for a in range(2)]) for r in range(replicates)]))
    reproduction(f, valid)  # Warm reproduction too; timed call has no compilation.
    count = assay.compile_count
    print("Timing the full assay plus reproduction (compilation excluded).", flush=True)
    start = perf_counter()
    f, valid = assay(g)
    reproduction(f, valid)
    elapsed = perf_counter()-start
    assert assay.compile_count == count
    work = replicate_work(replicates, N, K, B, T, hybrid_samples, duration_ms)
    rate = replicates*2*K*N*duration_ms/1000/elapsed
    report = dict(assay.report(), replicates=replicates, networks=replicates*2*K*N,
                  N=N, K=K, B=B, T=T, hybrid_samples=hybrid_samples, chunk_size=chunk_size,
                  duration_ms=duration_ms, window_start_ms=window_start_ms, dt=dt,
                  wall_seconds_per_generation=elapsed, full_run_network_seconds=work,
                  generation_network_seconds_per_wall_second=rate,
                  timing_protocol="one precompiled full generation after warmed 100-ms probes and warmed reproduction",
                  projected_full_run_hours=work/rate/3600,
                  b300_baseline_network_seconds_per_second=3100,
                  b300_constant_throughput_hours=work/3100/3600,
                  b300_ideal_batch_scaling_hours=work/(3100*replicates)/3600,
                  projection_note="CPU projection scales measured generation by network work; burn/hybrid shapes and host overhead may differ. B300 scenarios are extrapolations, not measured R=8 timings; ideal assumes R-fold throughput.")
    print(json.dumps(clean(report)), flush=True)
    return report



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bench")
    check_cli = sub.add_parser("check")
    for flag, default in (("duration-ms", DURATION), ("window-start-ms", WINDOW)):
        check_cli.add_argument("--" + flag, type=float, default=default)
    fig = sub.add_parser("figure")
    fig.add_argument("--input", type=Path, default=OUTPUT / "stg.json")
    fig_r = sub.add_parser("figure-replicates")
    fig_r.add_argument("--input", type=Path, default=OUTPUT / "stg-replicates.json")
    bench_r = sub.add_parser("bench-replicates")
    for flag, default in dict(replicates=8, N=64, K=16, B=200, T=2000, hybrid_samples=64).items():
        bench_r.add_argument("--"+flag.replace("_", "-"), type=int, default=default)
    for flag, default in dict(dt=.05, duration_ms=DURATION, window_start_ms=WINDOW).items():
        bench_r.add_argument("--"+flag.replace("_", "-"), type=float, default=default)
    bench_r.add_argument("--chunk-size", type=int, default=None)
    for name in ("quick", "run", "replicates"):
        cli = sub.add_parser(name)
        cli.add_argument("--replicates", type=int, default=8 if name == "replicates" else 1)
        cli.add_argument("--chunk-size", type=int, default=None,
                         help="Maximum networks per assay chunk (default: entire batch)")
        if name == "replicates":
            cli.add_argument("--output", type=Path, default=OUTPUT / "stg-replicates.json")
        for flag, default in dict(N=64, K=16, B=200, T=2000, every=20, seed=0, hybrid_samples=64).items():
            cli.add_argument("--"+flag.replace("_", "-"), type=int, default=default)
        for flag, default in dict(dt=.05, u=.02, sigma=.05, s=1., duration_ms=DURATION, window_start_ms=WINDOW).items():
            cli.add_argument("--"+flag.replace("_", "-"), type=float, default=default)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "bench":
        bench()
        return
    if command == "bench-replicates":
        bench_replicates(**args)
        return
    if command == "check":
        raise SystemExit(0 if check(**args) else 1)
    if command == "figure":
        figure(args["input"])
        return
    if command == "figure-replicates":
        figure_replicates(args["input"])
        return
    if command == "replicates":
        path = args.pop("output")
        data = run_replicates(**args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n")
        text = summary_replicates(data)
        path.with_name(path.stem+"-summary.txt").write_text(text)
        print(text)
        return
    quick = command == "quick"
    if quick:
        args.update(N=8, K=2, B=5, T=10, every=5)
    if args["replicates"] == 1 and args["chunk_size"] is None:
        # Keep even the legacy quick diagnostic's params/output unchanged.
        args.pop("replicates")
        args.pop("chunk_size")
    try:
        data = run(**args)
    except RuntimeError as error:
        if not quick or not str(error).startswith("Canonical validation failed"):
            raise
        assay = Assay(args["dt"], args["duration_ms"], args["window_start_ms"])
        g = np.broadcast_to(ANCESTOR, (args["K"], args["N"], 31)).copy()
        assay(g)
        assay(g)
        data = dict(status="canonical_validation_failed", reason=str(error), params=args,
                    canonical=diagnostic(args["dt"], args["duration_ms"], args["window_start_ms"]), trajectories={}, correlations=None,
                    hybrids=None, omega=None, throughput=assay.report(), generations_evolved=0)
    data["quick"] = quick
    OUTPUT.mkdir(exist_ok=True)
    stem = "stg-quick" if quick else "stg"
    (OUTPUT / f"{stem}.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n")
    text = summary_replicates(data) if data.get("experiment") == "E2" else summary(data)
    (OUTPUT / f"{stem}-summary.txt").write_text(text)
    print(text)
    if data.get("status") == "canonical_validation_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
