"""Experiment E3: development from scratch on inherited inverse-tau directions.

Genome: log10 Ca targets (3), tau_m (3 x 7), leaks (3), synapses (7).
Starting from m=g=0 removes integral-controller offsets: tau_m,i*m_i is
common within each cell. At equilibrium g=m=physical_alpha_c/tau_m,i.
We solve that reduced equilibrium problem, not the developmental trajectory.
Alpha below is dimensionless, in units of the canonical integrated Ca scale;
alpha=1 gives canonical conductances and anchors the ancestor and burn-in start.
Offspring warm-start the numerical solve from their parent's log alpha; hybrids
use the mean parental log alpha. This initial guess does not inherit biological
m or g: development still starts at m=g=0 and solves the same fixed-point
equations. Finite iteration outputs are used even when unconverged.

Calcium is accumulated online using stg's full-step voltage and calcium equations.
Targets and regulation use the same short window (1 s transient + 2 s mean),
so the canonical genome is a fixed point by construction. Rhythm is measured
separately with E's longer spike-buffer assay. Meeting mean-calcium targets
does not guarantee a pyloric state; selection acts on the developed rhythm.
Within-cell g correlations reflect shared alpha and variation in tau ratios.
No reference simulator code is incorporated; the GPL reference informed only
the method. Fast states restart each observation, as in E.

Commands: check, validate, replicates, bench, figure. Bench is opt-in.
Tiny smoke: replicates --R 1 --N 2 --K 2 --B 1 --T 1 --every 1
 --hybrid-samples 1 --output output/stg-homeo-quick.json
"""

import argparse
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
from time import perf_counter

import jax
# The CLI watchdog must not reserve GPU memory before spawning its worker.
# Spawn imports this file as __mp_main__, so only the worker uses the GPU.
if __name__ == '__main__' and sys.argv[1:2] == ['bench']:
    with jax.default_device(jax.devices('cpu')[0]):
        import stg as E
else:
    import stg as E
import jax.numpy as jnp
import numpy as np
from tqdm import trange

OUTPUT = E.OUTPUT
REG = np.array([8*c+i for c in range(3) for i in range(7)])
DIRECT = np.r_[7, 15, 23, np.arange(24, 31)]
G0 = np.maximum(E.CANONICAL[REG], E.FLOOR).reshape(3, 7)
SITES = ([f'{c}.Ca_target' for c in E.CELLS] +
         [f'{c}.tau_{g}' for c in E.CELLS for g in E.CHANNELS[:7]] +
         [E.SITES[i] for i in DIRECT])
CLASSES = dict(Ca_targets=list(range(3)), tau_m=list(range(3, 24)),
               leaks=list(range(24, 27)), synapses=list(range(27, 34)))
SCHULZ = [('PD', 'IH', 'shal', .76), ('PD', 'IH', 'BKKCa', .87),
          ('PD', 'IH', 'para', .91), ('PD', 'BKKCa', 'shal', .89),
          ('PD', 'BKKCa', 'para', .90), ('LP', 'IH', 'shal', .92),
          ('LP', 'IH', 'shab', .87), ('LP', 'IH', 'shaw', .90),
          ('LP', 'shal', 'shab', .88), ('LP', 'shal', 'shaw', .97)]
CHANNEL_MAP = dict(para='NaV', shal='A', shab='Kd', shaw='Kd', BKKCa='KCa', IH='H')


@jax.jit
def conductances(genome, g):
    out = jnp.zeros((*genome.shape[:-1], 31), jnp.float32)
    out = out.at[..., REG].set(g.reshape(*g.shape[:-2], 21))
    return out.at[..., DIRECT].set(10.**genome[..., 24:])


@partial(jax.jit, static_argnums=(1, 2, 3))
def reference_calcium(conductance, dt, duration, transient):
    """Observe exactly the calcium trajectory underlying E's full-step voltage."""
    _, _, voltage = E.simulate_one(conductance, dt, False, duration, 0., 8, False, True)
    old_v = jnp.concatenate((jnp.full((1, 3), -60.), voltage[:-1]))
    ca = jnp.full(3, .05)
    m, h, _, _ = E.kinetics(old_v[0], ca)
    def step(state, xs):
        ca, m, h, ica, total = state
        i, v = xs
        mi, hi, tm, th = E.kinetics(v, ca)
        m, h = mi + (m-mi)*jnp.exp(-dt/tm), hi + (h-hi)*jnp.exp(-dt/th)
        ec = (1000*8.314*(273.15+11)/(2*96485))*jnp.log(3000/ca)
        gc = conductance[:24].reshape(3, 8)[:, 1:3]*m[:, 1:3]**3*h[:, 1:3]
        ci = .05 - 14.96*E.AREA*ica
        ca = ci + (ca-ci)*jnp.exp(-dt/200)
        ica = jnp.sum(gc*(v[:, None]-ec[:, None]), -1)
        total += jnp.where(i*dt >= transient, ca, 0.)
        return (ca, m, h, ica, total), None
    state, _ = jax.lax.scan(step, (ca, m, h, jnp.zeros(3), jnp.zeros(3)),
                            (jnp.arange(len(old_v)), old_v), unroll=8)
    return state[-1] / (round(duration/dt)-round(transient/dt))


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def calcium(conductance, dt, duration, transient, unroll=16):
    """E.simulate_one's dynamics and the calcium observer in one online scan.

    Preserve both update orders, including the one-step lag in I_Ca. The
    observer uses the original scalar m**3 calcium-channel expression, whose
    float32 rounding can differ from the voltage solver's vector powers.
    Keeping its tiny state preserves the old measurement without voltage traces.
    """
    conductance = jnp.asarray(conductance, dtype=jnp.float32)
    gbar, gmax = conductance[:24].reshape(3, 8), conductance[24:]
    v, ca = jnp.full(3, -60., jnp.float32), jnp.full(3, .05, jnp.float32)
    m, h, _, _ = E.kinetics(v, ca)
    def step(state, i):
        v, ca, m, h, syn, ica, observed_ca, observed_m, observed_h, observed_ica, accumulated = state
        omi, ohi, otm, oth = E.kinetics(v, observed_ca)
        observed_m = omi + (observed_m-omi)*jnp.exp(-dt/otm)
        observed_h = ohi + (observed_h-ohi)*jnp.exp(-dt/oth)
        oec = (1000*8.314*(273.15+11)/(2*96485))*jnp.log(3000/observed_ca)
        gc = gbar[:, 1:3]*observed_m[:, 1:3]**3*observed_h[:, 1:3]
        oci = .05 - 14.96*E.AREA*observed_ica
        observed_ca = oci + (observed_ca-oci)*jnp.exp(-dt/200)
        observed_ica = jnp.sum(gc*(v[:, None]-oec[:, None]), -1)
        mi, hi, tm, th = E.kinetics(v, ca)
        m = mi + (m-mi)*jnp.exp(-dt/tm)
        h = hi + (h-hi)*jnp.exp(-dt/th)
        ec = (1000*8.314*(273.15+11)/(2*96485))*jnp.log(3000/ca)
        e = jnp.broadcast_to(jnp.array([50., 0, 0, -80, -80, -80, -20, -50],
                                       dtype=jnp.float32), (3, 8))
        e = e.at[:, 1:3].set(ec[:, None])
        g = gbar * m**E.POWERS
        g = g.at[:, :4].multiply(h)
        ci = .05 - 14.96*E.AREA*ica
        ca = ci + (ca-ci)*jnp.exp(-dt/200)
        ica = jnp.sum(g[:, 1:3]*(v[:, None]-ec[:, None]), axis=1)
        si = jax.nn.sigmoid((v[E.PRE]+35)/5)
        tau = jax.nn.sigmoid(-(v[E.PRE]+35)/5)/E.SYN_K
        syn = si + (syn-si)*jnp.exp(-dt/tau)
        gs = gmax*syn/(1000*E.AREA)
        total = g.sum(-1) + jnp.zeros(3, jnp.float32).at[E.POST].add(gs)
        drive = (g*e).sum(-1) + jnp.zeros(3, jnp.float32).at[E.POST].add(gs*E.SYN_E)
        vi = drive/total
        v = vi + (v-vi)*jnp.exp(-dt*total/E.CM)
        accumulated += jnp.where(i*dt >= transient, observed_ca, 0.)
        return (v, ca, m, h, syn, ica, observed_ca, observed_m, observed_h, observed_ica, accumulated), None
    state = (v, ca, m, h, jnp.zeros(7, jnp.float32),
             jnp.zeros(3, jnp.float32), ca, m, h, jnp.zeros(3, jnp.float32), jnp.zeros(3, jnp.float32))
    final, _ = jax.lax.scan(step, state, jnp.arange(round(duration/dt)), unroll=unroll)
    return final[-1] / (round(duration/dt)-round(transient/dt))


CALCIUM_TRACES = 0


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def batch_calcium(gs, dt, duration, transient, unroll):
    global CALCIUM_TRACES
    CALCIUM_TRACES += 1  # Python tracing, never compiled execution.
    return jax.vmap(calcium, in_axes=(0, None, None, None, None))(
        gs, dt, duration, transient, unroll)


@jax.jit
def regulation_state(genome, direction, alpha):
    g = direction * alpha[..., None]
    return g, conductances(genome, g)


class Homeostasis:
    def __init__(self, dt=.05, reg_window_ms=2000., reg_transient_ms=1000.,
                 reg_gain=.3, chunk_size=None, ca_tol=.02, controller_seconds=300.):
        self.dt, self.window, self.transient = dt, reg_window_ms, reg_transient_ms
        self.gain, self.chunk, self.ca_tol = reg_gain, chunk_size, ca_tol
        if chunk_size is not None and chunk_size < 1:
            raise ValueError('chunk_size must be positive or None')
        # A fixed unit (300 uM*s by default), independent of mutated Ca targets.
        self.tau0 = jnp.log10(controller_seconds / jnp.asarray(G0))
        self.compiled, self.wall, self.network_seconds = {}, 0., 0.
        self.compile_count, self.compile_seconds = 0, 0.
        self.probe_compile_count, self.probe_compile_seconds = 0, 0.
        self.probe_seconds, self.probe_timings = 0., {}
        self.unroll = 16
        self.progress = None

    def report(self):
        return dict(compile_count=self.compile_count, compile_seconds=self.compile_seconds,
                    probe_compile_count=self.probe_compile_count,
                    probe_compile_seconds=self.probe_compile_seconds,
                    probe_seconds=self.probe_seconds, probe_timings=self.probe_timings,
                    unroll=self.unroll, wall_seconds=self.wall,
                    network_simulated_seconds=self.network_seconds, dtype='float32')

    def notify(self, phase, **extra):
        if self.progress is not None:
            self.progress(phase, regulation=self.report(), **extra)

    def tune(self, sample):
        # Single-network calibration should not choose the GPU batch unroll.
        if self.probe_timings or len(sample) == 1:
            return
        start = perf_counter()
        duration = min(100., self.window+self.transient)
        for unroll in (8, 16, 32):
            self.notify('regulation unroll probe', candidate_unroll=unroll)
            tick = perf_counter()
            fn = batch_calcium.lower(sample, self.dt, duration, 0., unroll).compile()
            self.probe_compile_seconds += perf_counter()-tick
            self.probe_compile_count += 1
            self.notify('timing regulation unroll probe', candidate_unroll=unroll)
            jax.block_until_ready(fn(sample))
            tick = perf_counter()
            for _ in range(3):
                jax.block_until_ready(fn(sample))
            self.probe_timings[unroll] = (perf_counter()-tick)/3
        self.unroll = min(self.probe_timings, key=self.probe_timings.get)
        self.probe_seconds = perf_counter()-start
        self.notify('regulation unroll chosen')

    def executable(self, sample, duration, transient):
        key = (sample.shape, duration, transient, self.unroll)
        if key not in self.compiled:
            self.notify('compiling regulation', batch_shape=sample.shape)
            start = perf_counter()
            self.compiled[key] = batch_calcium.lower(
                sample, self.dt, duration, transient, self.unroll).compile()
            self.compile_seconds += perf_counter()-start
            self.compile_count += 1
            self.notify('compiled regulation')
        return self.compiled[key]

    def prepare(self, gs, duration=None, transient=None):
        flat = jnp.asarray(gs, dtype=jnp.float32).reshape(-1, 31)
        duration = self.window+self.transient if duration is None else duration
        transient = self.transient if transient is None else transient
        size = min(self.chunk or len(flat), len(flat))
        self.tune(flat[:size])
        # Only the main chunk and optional tail can have distinct shapes.
        for n in sorted({size, len(flat) % size} - {0}, reverse=True):
            self.executable(flat[:n], duration, transient)

    def mean_ca(self, gs, duration=None, transient=None):
        shape = gs.shape[:-1]
        flat = jnp.asarray(gs, dtype=jnp.float32).reshape(-1, 31)
        duration = self.window+self.transient if duration is None else duration
        transient = self.transient if transient is None else transient
        start = perf_counter()
        self.prepare(flat, duration, transient)
        if self.chunk is None or len(flat) <= self.chunk:
            out = self.executable(flat, duration, transient)(flat)
        else:
            # Memory-limited opt-in only; the default has one dispatch.
            result = []
            for i in range(0, len(flat), self.chunk):
                sample = flat[i:i+self.chunk]
                result.append(self.executable(sample, duration, transient)(sample))
            out = jnp.concatenate(result)
        out = out.reshape(*shape, 3)
        out.block_until_ready()
        self.wall += perf_counter()-start
        self.network_seconds += len(flat)*duration/1000
        return out

    def direction(self, genome):
        """Canonical-unit inverse tau, evaluated as a ratio for an exact anchor."""
        tau = genome[..., 3:24].reshape(*genome.shape[:-1], 3, 7)
        return jnp.asarray(G0) * 10.**(self.tau0 - tau)

    def iterate(self, genome, iterations, initial_alpha=None, tolerance=None):
        """Damped log-scale fixed point with a bracketed step-length fallback.

        Gain .3 limits overshoot on the short, phase-sensitive Ca window.
        If a trial increases the residual norm more than tenfold, bisect its step-length
        bracket (accepted endpoint, rejected trial). Cross-cell responses need
        not be monotone; this is a line safeguard, not a scalar root bracket.
        There is no claim of global convergence or a unique pyloric root.

        initial_alpha is a positive, dimensionless numerical initial guess
        (not log alpha); None means the canonical alpha=1 anchor. Warm starts
        change neither the biological m=g=0 origin enforced by direction()
        nor the fixed-point equations. Finite-budget endpoints can differ.
        Stop once 98% of individuals meet tolerance, after at least two updates
        (or the supplied budget if smaller). iterations counts updates after
        observation zero; each returned endpoint has measured calcium.
        """
        genome = jnp.asarray(genome, dtype=jnp.float32)
        target = np.asarray(10.**genome[..., :3])
        direction = self.direction(genome)
        # Preserve the host solver's original log-scale arithmetic. Only this
        # small controller state is float64; every device array/scan is float32.
        x = np.zeros(target.shape) if initial_alpha is None else np.log(
            np.broadcast_to(initial_alpha, target.shape)).copy()
        tol = self.ca_tol if tolerance is None else tolerance
        previous_x = previous_f = previous_ca = None
        fallback_count = np.zeros(target.shape[:-1], int)
        history, first = [], np.full(target.shape[:-1], -1, int)
        first_solver = first.copy()
        for i in range(iterations+1):
            tick, compiled_before = perf_counter(), self.compile_seconds
            g, gs = regulation_state(genome, direction, jnp.asarray(np.exp(x), jnp.float32))
            # Only three Ca means per network return to the host each iteration.
            ca = np.asarray(self.mean_ca(gs))
            if not np.isfinite(ca).all() or np.any(ca <= 0):
                raise RuntimeError('Nonfinite or nonpositive calcium during regulation')
            f = np.log(ca/target)
            # A rejected trial is not an endpoint. Before either stopping
            # condition retain its last accepted alpha and measured Ca.
            ready = i >= min(2, iterations)
            stop = ready and np.mean(np.max(np.abs(ca-target)/target, axis=-1) < tol) >= .98
            if (i == iterations or stop) and previous_x is not None:
                reject = np.linalg.norm(f, axis=-1) > 10*np.linalg.norm(previous_f, axis=-1)
                fallback_count += reject
                x = np.where(reject[..., None], previous_x, x)
                ca = np.where(reject[..., None], previous_ca, ca)
                f = np.log(ca/target)
                g = direction * jnp.asarray(np.exp(x))[..., None]
            error = np.abs(ca-target)/target
            done = np.max(error, axis=-1) < tol
            within = np.max(error, axis=-1) < self.ca_tol
            first = np.where((first < 0) & within, i, first)
            first_solver = np.where((first_solver < 0) & done, i, first_solver)
            history.append(dict(iteration=i, max_relative_ca_error=float(error.max()),
                                fraction_converged=float(within.mean()), fraction_solver_converged=float(done.mean()),
                                wall_seconds=perf_counter()-tick,
                                compile_seconds=self.compile_seconds-compiled_before,
                                compile_count=self.compile_count))
            if (ready and done.mean() >= .98) or i == iterations:
                history[-1]['wall_seconds'] = perf_counter()-tick
                self.notify('regulation iteration', iteration=history[-1])
                break
            step = -self.gain*f
            step = np.clip(step, -.35, .35)
            proposal = np.clip(x+step, -8., 8.)
            reject = np.zeros(done.shape, bool)
            if previous_x is not None:
                reject = np.linalg.norm(f, axis=-1) > 10*np.linalg.norm(previous_f, axis=-1)
                proposal = np.where(reject[..., None], (previous_x+x)/2, proposal)
                fallback_count += reject
            previous_x = x.copy() if previous_x is None else np.where(reject[..., None], previous_x, x)
            previous_f = f.copy() if previous_f is None else np.where(reject[..., None], previous_f, f)
            previous_ca = ca.copy() if previous_ca is None else np.where(reject[..., None], previous_ca, ca)
            x = np.where(done[..., None], x, proposal)
            history[-1]['wall_seconds'] = perf_counter()-tick
            self.notify('regulation iteration', iteration=history[-1])
        converged = np.max(error, axis=-1) < self.ca_tol
        def report(err, conv, al, needed):
            return dict(relative_ca_error=err.tolist(), max_relative_ca_error=float(err.max()),
                        fraction_converged=float(conv.mean()),
                        fraction_within_2pct=float((np.max(err, axis=-1) < .02).mean()), alpha=al.tolist(),
                        iterations=i,
                        iterations_needed=np.where(needed >= 0, needed, None).tolist())
        alpha = np.exp(x)
        diag = report(error, converged, alpha, first)
        # Leading axes are replicate/arm; the last two are lineage/individual.
        group_shape = (-1, *error.shape[-3:]) if error.ndim >= 4 else (-1, 1, 3)
        diag.update(iterations=i, history=history, ca_tolerance=self.ca_tol,
            solver_iterations_needed=np.where(first_solver >= 0, first_solver, None).tolist(),
            bracket_fallbacks=fallback_count.tolist(),
            solver_tolerance=tol, groups=[report(e, np.max(e, -1) < self.ca_tol, a, n)
                for e, a, n in zip(error.reshape(group_shape), alpha.reshape(group_shape),
                                  first.reshape(error.reshape(group_shape).shape[:-1]))])
        return jnp.asarray(alpha), g, diag


class HomeoAssay(E.ReplicateAssay):
    """Expose explicit assay compilation separately from synchronized execution."""
    compile_seconds = 0.

    def prepare(self, g):
        flat = jnp.asarray(g, jnp.float32).reshape(-1, 31)
        size = min(self.chunk_size or len(flat), len(flat))
        for n in sorted({size, len(flat) % size} - {0}, reverse=True):
            sample = flat[:n]
            if sample.shape not in self.compiled:
                tick = perf_counter()
                self.compiled[sample.shape] = E.evaluate.lower(
                    sample, self.dt, self.duration_ms, self.window_start_ms, self.unroll).compile()
                self.compile_seconds += perf_counter()-tick
                self.compile_count += 1

    def __call__(self, g):
        self.prepare(g)
        return super().__call__(g)

    def report(self):
        return dict(super().report(), compile_seconds=self.compile_seconds)


def make_assay(p):
    # Regulation probes its own kernel; the assay reuses E's online kernel.
    assay = HomeoAssay(p['dt'], p['duration_ms'], p['window_start_ms'], None, None)
    assay.chunk_size = p['chunk_size']
    return assay


def homeostasis(p):
    return Homeostasis(**{k: p[k] for k in ('dt', 'reg_window_ms', 'reg_transient_ms',
                                           'reg_gain', 'chunk_size', 'ca_tol', 'controller_seconds')})


def canonical(p, homeo, assay, recovery=True):
    """Short-window calibration and a handful of full-network recovery tests."""
    seed = jnp.asarray(np.r_[np.zeros(3), np.asarray(homeo.tau0).ravel(), E.ANCESTOR[DIRECT]])
    base = conductances(seed, jnp.asarray(G0))
    target = np.asarray(homeo.mean_ca(base))
    genome = seed.at[:3].set(jnp.log10(target))
    f, valid = assay(jnp.log10(base)[None])
    alpha, fixed_g, fixed = homeo.iterate(genome, 1)
    report = dict(target=target.tolist(), genome=np.asarray(genome).tolist(), features=f[0].tolist(),
        canonical_pyloric=bool(valid[0]), fixed_point=fixed,
        fixed_point_relative_g_error=float(np.max(abs(np.asarray(fixed_g)-G0)/G0)),
        fixed_point_passed=bool(fixed['fraction_converged'] == 1 and np.allclose(alpha, 1)),
        target_window_ms=[homeo.transient, homeo.transient+homeo.window])
    report['passed'] = bool(valid[0] and report['fixed_point_passed'])
    if not recovery:
        return genome, f[0], report
    starts = np.array([[.5, .5, .5], [2., 2., 2.], [.5, 2., .5]])
    genomes = jnp.broadcast_to(genome, (len(starts), 34))
    alpha, g, diag = homeo.iterate(genomes, p['validation_iters'], starts, tolerance=.001)
    rf, rv = assay(jnp.log10(jnp.maximum(conductances(genomes, g), E.FLOOR)))
    errors = E.displacements(rf, f[0]) / E.tolerance(f[0])
    report.update(recovery=diag, recovery_features=rf.tolist(), recovery_pyloric=rv.tolist(),
        recovery_feature_errors_in_E_tolerances=errors.tolist(),
        recovery_relative_g_error=np.max(abs(np.asarray(g)-G0)/G0, axis=(-2, -1)).tolist(),
        perturbation=f'initial alpha {starts.tolist()}', iterations_needed=diag['iterations_needed'],
        acceptance='Ca error <2%; alpha within 3%; period within 2%, duties/phases within .02')
    report['recovery_passed'] = bool(np.all(rv) and np.isfinite(errors).all() and
        np.max(abs(errors[:, 0])) < .2 and np.max(abs(errors[:, 1:])) < .4 and
        np.max(abs(np.asarray(alpha)-1)) < .03 and
        diag['fraction_converged'] == 1)
    # One nonzero calcium channel: only AB.CaS changes its inverse-tau direction.
    mutant = genome.at[3+2].add(np.log10(1.05))
    ma, mg, md = homeo.iterate(mutant, p['validation_iters'], tolerance=.001)
    ratio = np.asarray(homeo.direction(mutant)/homeo.direction(genome))
    expected = np.ones((3, 7)); expected[0, 2] = 1/1.05
    report['tau_perturbation'] = dict(site=SITES[5], factor=1.05, direction_ratio=ratio.tolist(),
        alpha=np.asarray(ma).tolist(), convergence=md,
        only_changed_channel_direction=bool(np.allclose(ratio, expected, rtol=1e-6)),
        compensated=bool(md['fraction_converged'] == 1 and abs(float(ma[0])-1) > 1e-4))
    report['passed'] &= bool(report['recovery_passed'] and
        report['tau_perturbation']['only_changed_channel_direction'] and report['tau_perturbation']['compensated'])
    return genome, f[0], report


@jax.jit
def inherit(genome, f, valid, optimum, strength, plan, index, log_alpha=None):
    """Carry the selected parent's solver guess; biological development is fresh."""
    draws, sites, changes = (x[index] for x in plan)
    children = E.device_step(genome, f, valid, optimum, strength, draws, sites, changes)
    if log_alpha is None:
        return children
    # Reuse exactly E's selection with the same draws, but no mutations, for
    # the numerical state. This consumes no RNG and preserves mutation sites.
    guesses = E.device_step(log_alpha, f, valid, optimum, strength, draws,
                            jnp.empty(0, jnp.int32), jnp.empty(0, log_alpha.dtype))
    return children, guesses


def hybrid_group(genome, g, rng, samples, log_alpha=None):
    """Acute tau-site donors supply parent g; rescue discards this mixed g."""
    K, N = genome.shape[:2]
    a = rng.integers(K, size=samples)
    b = (a+rng.integers(1, K, size=samples)) % K
    ids = [(a, rng.integers(N, size=samples)), (b, rng.integers(N, size=samples)),
           (a, rng.integers(N, size=samples))]
    parents = [[np.asarray(x)[ix] for ix in ids] for x in (genome, g)]
    masks = [rng.random((samples, 34)) < .5 for _ in range(2)]
    groups = []
    for k, (p, q, r) in enumerate(parents):
        mask = masks if k == 0 else [x[:, 3:24].reshape(samples, 3, 7) for x in masks]
        groups.append(jnp.asarray(np.concatenate([p, q, r, np.where(mask[0], p, q), np.where(mask[1], p, r)])))
    if log_alpha is not None:
        p, q, r = [np.asarray(log_alpha)[ix] for ix in ids]
        groups.append(jnp.asarray(np.concatenate([p, q, r, (p+q)/2, (p+r)/2])))
    return tuple(groups)


def within_correlations(logg):
    matrices = [E.correlations(x) for x in logg]
    result = {}
    for cell in E.CELLS:
        values = np.array([x[cell]['matrix'] for x in matrices])
        count = np.isfinite(values).sum(0)
        mean = np.divide(np.nansum(values, 0), count, out=np.full((8, 8), np.nan), where=count > 0)
        result[cell] = dict(matrix=mean.tolist(), finite_lineages=count.tolist())
    return result


def omega(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return dict(per_site=np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0).tolist(),
                summed=float(a.sum()/b.sum()) if b.sum() > 0 else None)


def comparison(baseline, correlations):
    rows = []
    for cell, x, y, r in SCHULZ:
        model_cell, pair = 'AB' if cell == 'PD' else cell, {CHANNEL_MAP[x], CHANNEL_MAP[y]}
        def lookup(data, level, arm):
            row = next(v for v in data[level] if v['cell'] == model_cell and set(v['pair']) == pair)
            return row[arm]['mean']
        rows.append(dict(cell=cell, pair=[x, y], model_pair=[CHANNEL_MAP[x], CHANNEL_MAP[y]], schulz=r,
            E2_pooled_individual_r=lookup(baseline['aggregate']['correlations'], 'individual', 'selected'),
            E2_lineage_r=lookup(baseline['aggregate']['correlations'], 'lineage', 'selected'),
            E3_within_lineage_r=lookup(correlations, 'individual', 'selected'),
            E3_neutral_within_lineage_r=lookup(correlations, 'individual', 'neutral')))
    return rows


def run(p):
    start = perf_counter()
    homeo, assay = homeostasis(p), make_assay(p)
    ancestor, optimum, validation = canonical(p, homeo, assay, recovery=False)
    data = dict(experiment='E3', params=p, sites=SITES, conductance_sites=E.SITES,
                feature_names=E.FEATURES, canonical=validation, status='validation_failed',
                inference=dict(measurement='Model g correlations are not the same measurement as mRNA correlations. '
                    'Development from zero fixes inverse-tau directions; within-cell correlations among regulated '
                    'conductances come from shared alpha variation plus variation in tau_m ratios.',
                    baseline='E2 individual r pools animals across lineages; within-lineage E2 r is unavailable '
                    'without saved populations. shab/shaw both map to Kd; these are not independent model pairs.',
                    convergence='Final alpha is used even without convergence. Mean-calcium convergence does not imply a pyloric '
                    'fixed point; selection acts on the developed rhythm. Acute hybrids mix parental endpoint g; '
                    'rescued hybrids develop from scratch. Unconverged parental endpoints are not steady states.',
                    tests='E2 raw-r paired t, Holm over 84 pairs per level; independent-replicate CIs/sign tests. '
                    'Individual level here means mean of within-lineage r. Consistency calls require R>=3 and K>=3.'))
    if not validation['passed']:
        data['reason'] = 'Canonical short-window fixed point or pyloric assay failed; evolution stopped. Run validate for recovery diagnostics.'
        data['elapsed_seconds'] = perf_counter()-start
        print(data['reason'], flush=True)
        return E.clean(data)
    baseline = json.loads(Path(p['baseline']).read_text())
    R, K, N = p['R'], p['K'], p['N']
    seeds = E.replicate_streams(p['seed'], R)
    genome = jnp.broadcast_to(ancestor, (R, 1, N, 34))
    alpha, g, initial_diag = homeo.iterate(genome, p['n_reg'])
    f, valid = assay(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
    log_alpha = jnp.log(alpha)
    def generation(genome, log_alpha, f, valid, plans, index, fork=False):
        if p['s'] and np.any(~(valid[:, 0] if fork else valid).any(-1)):
            raise RuntimeError('Selected lineage has no pyloric parents; no neutral rescue')
        states, guesses = [], []
        for r in range(R):
            if fork:
                offspring = [inherit(genome[r, a], f[r, a], valid[r, a], optimum,
                    p['s'] if a == 0 else 0., plans[r], index, log_alpha[r, a]) for a in range(2)]
                child, guess = (jnp.stack([x[k] for x in offspring]) for k in range(2))
            else:
                child, guess = inherit(genome[r], f[r], valid[r], optimum, p['s'], plans[r], index, log_alpha[r])
            states.append(child)
            guesses.append(guess)
        genome = jnp.stack(states)
        alpha, g, diag = homeo.iterate(genome, p['n_reg'], initial_alpha=np.exp(np.asarray(jnp.stack(guesses))))
        f, valid = assay(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
        diag['fraction_pyloric'] = float(valid.mean())
        for group, v in zip(diag['groups'], np.asarray(valid).reshape(-1, *valid.shape[-2:])):
            group['fraction_pyloric'] = float(v.mean())
        return genome, jnp.log(alpha), g, f, valid, diag
    results = [dict(index=r, seed=p['seed']+r, trajectories={a: [] for a in E.ARMS},
                    convergence=[]) for r in range(R)]
    initial_diag['fraction_pyloric'] = float(valid.mean())
    diagnostics = [dict(phase='initial', t=0, **initial_diag)]
    burn = E.ReplicatePlans((1, N, 34), p['B'], p['u'], p['sigma'], [x[0] for x in seeds])
    for t in trange(p['B'], desc='E3 burn-in'):
        plans, index = burn.at(t)
        genome, log_alpha, g, f, valid, diag = generation(genome, log_alpha, f, valid, plans, index)
        diagnostics.append(dict(phase='burn', t=t+1, **diag))
        for r in range(R):
            results[r]['convergence'].append(dict(phase='burn', t=t+1, **diag['groups'][r]))
    genome = jnp.broadcast_to(genome[:, None], (R, 2, K, N, 34))
    log_alpha = jnp.broadcast_to(log_alpha[:, None], (R, 2, K, N, 3))
    g = jnp.broadcast_to(g[:, None], (R, 2, K, N, 3, 7))
    f, valid = assay(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
    fork = E.ReplicatePlans((K, N, 34), p['T'], p['u'], p['sigma'], [x[1] for x in seeds])
    for t in trange(p['T']+1, desc='E3 forked lineages'):
        if t % p['every'] == 0 or t == p['T']:
            logs = np.asarray(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
            for r in range(R):
                for a, arm in enumerate(E.ARMS):
                    record = dict(t=t, **E.measure(logs[r, a], f[r, a], valid[r, a], optimum, p['s']))
                    between, within = map(np.asarray, E.population_moments(genome[r, a]))
                    record.update(genome_divergence=between.tolist(), within_genome=within.tolist(),
                                  mean_reproductive_weight=record['mean_fitness'] if a == 0 and p['s'] else 1.)
                    results[r]['trajectories'][arm].append(record)
        if t < p['T']:
            plans, index = fork.at(t)
            genome, log_alpha, g, f, valid, diag = generation(genome, log_alpha, f, valid, plans, index, True)
            diagnostics.append(dict(phase='fork', t=t+1, **diag))
            for r in range(R):
                results[r]['convergence'].append(dict(phase='fork', t=t+1,
                    **{arm: diag['groups'][2*r+a] for a, arm in enumerate(E.ARMS)}))
    logs = np.asarray(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
    groups = [[hybrid_group(genome[r, a], g[r, a], np.random.default_rng(seeds[r][2]),
                           p['hybrid_samples'], log_alpha[r, a]) for a in range(2)] for r in range(R)]
    hg, hc, hx = [jnp.stack([jnp.stack([groups[r][a][i] for a in range(2)]) for r in range(R)]) for i in range(3)]
    af, av = assay(jnp.log10(jnp.maximum(conductances(hg, hc), E.FLOOR)))
    ha, hc, hybrid_diag = homeo.iterate(hg, p['hybrid_iters'], initial_alpha=np.exp(np.asarray(hx)))
    hf, hv = assay(jnp.log10(jnp.maximum(conductances(hg, hc), E.FLOOR)))
    hybrid_diag['fraction_pyloric'] = float(hv.mean())
    count = p['hybrid_samples']
    for r in range(R):
        result = results[r]
        result['correlations'], result['hybrids'] = {}, {}
        for a, arm in enumerate(E.ARMS):
            result['correlations'][arm] = dict(individual=within_correlations(logs[r, a]),
                pooled_individual=E.correlations(logs[r, a]), lineage=E.correlations(logs[r, a].astype(float).mean(1)))
            result['hybrids'][arm] = {}
            for name, ids in (('cross_parents', np.r_[0:2*count]), ('within_parents', np.r_[0:count, 2*count:3*count]),
                              ('cross_lineage', np.r_[3*count:4*count]), ('within_lineage', np.r_[4*count:5*count])):
                result['hybrids'][arm][name] = dict(n=len(ids), acute=float(av[r, a, ids].mean()),
                    after=float(hv[r, a, ids].mean()), rescued=float(hv[r, a, ids].mean()),
                    fraction_converged=float((np.max(np.asarray(hybrid_diag['relative_ca_error'])[r, a, ids], -1) < p['ca_tol']).mean()),
                    after_label='developed from scratch')
        last = [result['trajectories'][a][-1] for a in E.ARMS]
        result['omega'] = omega(*(x['conductance_divergence'] for x in last))
        result['genome_omega'] = omega(*(x['genome_divergence'] for x in last))
        result['omega_classes'] = {c: omega(*(np.asarray(x['genome_divergence'])[ids] for x in last))['summed']
                                   for c, ids in CLASSES.items()}
        result['omega_classes']['emergent_g'] = omega(*(np.asarray(x['conductance_divergence'])[REG] for x in last))['summed']
    correlations = E.replicate_correlations(results, K)
    # E2's consistency rule applied at both requested E3 levels (never with K=2).
    for row in correlations['individual']:
        row['selection_associated'] = bool(R >= 3 and K >= 3 and row['selected']['same_sign'] >= int(np.ceil(.875*R))
            and row['paired_difference']['n'] == R and row['paired_holm_p'] is not None and row['paired_holm_p'] < .05)
    aggregate = dict(correlations=correlations,
        omega_classes={c: E.replicate_ci([r['omega_classes'][c] for r in results]) for c in (*CLASSES, 'emergent_g')},
        omega=E.replicate_ci([r['omega']['per_site'] for r in results]),
        genome_omega=E.replicate_ci([r['genome_omega']['per_site'] for r in results]))
    baseline_omega = {name: E.replicate_ci([omega(*(np.asarray(r['trajectories'][a][-1]['conductance_divergence'])[ids]
        for a in E.ARMS))['summed'] for r in baseline['replicates']]) for name, ids in
        (('emergent_g', REG), ('synapses', np.arange(24, 31)))}
    data.update(status='complete', replicates=results, aggregate=aggregate, convergence=diagnostics,
        hybrid_convergence=hybrid_diag, schulz=comparison(baseline, correlations), baseline_omega=baseline_omega,
        baseline_params=baseline['params'], elapsed_seconds=perf_counter()-start,
        throughput=dict(assay=assay.report(), regulation=homeo.report(), regulation_wall_seconds=homeo.wall,
                        regulation_network_seconds=homeo.network_seconds))
    return E.clean(data)


def benchmark_generation(p, publish):
    """Compile first, then execute exactly one paired population generation."""
    start = perf_counter()
    homeo, assay = homeostasis(p), make_assay(p)
    homeo.progress = publish
    R, K, N = p['R'], p['K'], p['N']
    networks = R*2*K*N
    sample = jnp.broadcast_to(jnp.asarray(E.CANONICAL), (networks, 31))
    publish('preparing regulation', device=str(jax.devices()[0]), networks=networks)
    # Probe the actual population shape before the single-network calibration.
    homeo.prepare(sample)
    publish('canonical calibration and assay')
    ancestor, optimum, validation = canonical(p, homeo, assay, recovery=False)
    publish('canonical complete', canonical=validation, assay=assay.report())
    if not validation['passed']:
        publish('validation failed', status='validation_failed',
                reason='Canonical calibration failed; use the normal assay window for bench.')
        return
    genome = jnp.broadcast_to(ancestor, (R, 2, K, N, 34))
    f = jnp.broadcast_to(jnp.asarray(optimum), (R, 2, K, N, 8))
    valid = jnp.ones((R, 2, K, N), bool)
    plans, index = E.ReplicatePlans((K, N, 34), 1, p['u'], p['sigma'],
        [x[1] for x in E.replicate_streams(p['seed'], R)]).at(0)

    @jax.jit
    def reproduction(genome, f, valid):
        return jnp.stack([jnp.stack([inherit(genome[r, a], f[r, a], valid[r, a], optimum,
            p['s'] if a == 0 else 0., plans[r], index) for a in range(2)]) for r in range(R)])

    publish('compiling reproduction')
    tick = perf_counter()
    reproduce = reproduction.lower(genome, f, valid).compile()
    reproduction_compile_seconds = perf_counter()-tick
    publish('reproduction compiled', reproduction_compile_seconds=reproduction_compile_seconds)
    # Warm only the cheap state/shape operations, never a population simulation.
    jax.block_until_ready(regulation_state(genome, homeo.direction(genome),
                                          jnp.ones((*genome.shape[:-1], 3), jnp.float32)))
    publish('compiling assay')
    assay.prepare(jnp.log10(jnp.maximum(sample, E.FLOOR)))
    def compile_counts():
        return dict(regulation=homeo.compile_count, assay=assay.compile_count,
                    regulation_traces=CALCIUM_TRACES, assay_traces=E.EVALUATE_TRACES)
    counts = compile_counts()
    compile_seconds = (homeo.compile_seconds + homeo.probe_compile_seconds +
                       assay.compile_seconds + reproduction_compile_seconds)
    setup_seconds = perf_counter()-start
    publish('generation start', compile_seconds=compile_seconds, setup_seconds=setup_seconds,
            assay=assay.report(), compile_counts_before_generation=counts,
            reproduction_compile_seconds=reproduction_compile_seconds)
    tick = perf_counter()
    genome = jax.block_until_ready(reproduce(genome, f, valid))
    reproduction_seconds = perf_counter()-tick
    reg_start = perf_counter()
    reg_work_before = homeo.network_seconds
    _, g, diag = homeo.iterate(genome, p['n_reg'])
    reg_seconds = perf_counter()-reg_start
    reg_work = homeo.network_seconds-reg_work_before
    publish('generation assay', regulation_seconds=reg_seconds,
            regulation_network_seconds=reg_work, assay=assay.report())
    assay_start = perf_counter()
    _, valid = assay(jnp.log10(jnp.maximum(conductances(genome, g), E.FLOOR)))
    assay_seconds = perf_counter()-assay_start
    seconds = perf_counter()-tick
    after = compile_counts()
    assert counts == after, 'Generation unexpectedly recompiled a simulation'
    assay_work = networks*p['duration_ms']/1000
    reg_rate, assay_rate = reg_work/reg_seconds, assay_work/assay_seconds

    def projection(q):
        # Include initial/burn development, forked generations and both acute
        # and regulated hybrid assays. Budget every regulation observation,
        # including the endpoint (n_reg+1), even if this generation ended early.
        r, k, n, b, t = (q[x] for x in ('R', 'K', 'N', 'B', 'T'))
        reg_networks = r*n*(b+1) + 2*r*k*n*t
        reg_work = (reg_networks*(q['n_reg']+1) +
                    10*r*q['hybrid_samples']*(q['hybrid_iters']+1)) * (
                        q['reg_window_ms']+q['reg_transient_ms'])/1000
        assay_work = r*((b+1)*n + (t+1)*2*k*n + 20*q['hybrid_samples'])*q['duration_ms']/1000
        # Scale step work when the smoke test used a different timestep.
        evolution = (reg_work/reg_rate + assay_work/assay_rate)*p['dt']/q['dt']
        evolution += reproduction_seconds*(reg_networks/networks)
        return dict(sizes={x: q[x] for x in ('R', 'K', 'N', 'B', 'T', 'n_reg')},
                    regulation_network_seconds=reg_work, assay_network_seconds=assay_work,
                    projected_seconds=evolution+setup_seconds,
                    projected_hours=(evolution+setup_seconds)/3600)

    defaults = dict(p, R=4, K=16, N=64, B=200, T=2000, n_reg=12,
                    hybrid_samples=64, hybrid_iters=12, dt=.05, duration_ms=12000.,
                    reg_window_ms=2000., reg_transient_ms=1000.)
    publish('complete', status='benchmark', generations_timed=1,
        generation_seconds=seconds, reproduction_seconds=reproduction_seconds,
        regulation_seconds=reg_seconds, assay_seconds=assay_seconds,
        iteration_wall_seconds=[h['wall_seconds'] for h in diag['history']],
        regulation_observations=len(diag['history']),
        regulation_network_seconds_per_wall_second=reg_rate,
        assay_network_seconds_per_wall_second=assay_rate,
        network_seconds_per_wall_second=(reg_work+assay_work)/seconds,
        throughput_unit='networks * simulated seconds / wall second',
        compile_counts_after_generation=after, no_generation_recompiles=counts == after,
        regulation=homeo.report(), assay=assay.report(), fraction_pyloric=float(valid.mean()),
        projections=dict(requested=projection(p), default=projection(defaults),
                         reduced=projection(dict(defaults, R=4, T=1000))),
        projection_note='Work-scaled estimates at measured throughput, with full iteration budgets. '
            'Includes measured setup, burn, fork and hybrids; excludes recovery validation, '
            'additional shape compilation and output analysis. Batch scaling, long-window behavior '
            'and evolved populations can differ; CPU smoke timings do not predict GPU performance.',
        timing_protocol='One generation from canonical parents; simulation compilation and '
            'short unroll probes excluded from generation timing. Observation 0 measures the initial state; '
            'up to n_reg subsequent observations measure the damped updates.',
        compile_timing_scope='Explicit regulation, assay, reproduction and probe compilation; '
            'setup_seconds also includes calibration, probe execution and small JAX primitive setup.')


def benchmark_worker(p, queue):
    def publish(phase, **values):
        queue.put(E.clean(dict(phase=phase, **values)))
    try:
        benchmark_generation(p, publish)
    except Exception as exc:
        publish('error', status='failed', reason=f'{type(exc).__name__}: {exc}')
    finally:
        queue.put(None)


def bench(p):
    """A separate process makes the deadline interrupt compilation/device waits."""
    context = mp.get_context('spawn')  # Never fork an initialized JAX runtime.
    queue = context.Queue()
    worker = context.Process(target=benchmark_worker, args=(p, queue))
    start = perf_counter()
    report = dict(status='running', phase='starting worker', params=p, iterations=[], compile_seconds=0.)
    worker.start()
    try:
        while True:
            remaining = p['max_seconds']-(perf_counter()-start)
            if remaining <= 0:
                report.update(status='timeout', reason='Hard wall-time limit reached; partial report.')
                break
            try:
                update = queue.get(timeout=min(remaining, .1))
            except Empty:
                if not worker.is_alive():
                    report.update(status='failed', reason=f'Benchmark worker exited ({worker.exitcode})')
                    break
                continue
            if update is None:
                break
            if update['phase'] == 'generation start':
                report['iterations'] = []  # Exclude canonical calibration observations.
            if 'iteration' in update:
                report['iterations'].append(update.pop('iteration'))
            report.update(update)
            # Also useful if the deadline interrupts setup before generation 0.
            regulation = report.get('regulation', {})
            report['compile_seconds'] = (regulation.get('compile_seconds', 0.) +
                regulation.get('probe_compile_seconds', 0.) +
                report.get('assay', {}).get('compile_seconds', 0.) +
                report.get('reproduction_compile_seconds', 0.))
    finally:
        # Kill directly on timeout: SIGTERM need not interrupt a stuck runtime.
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=1.)
        queue.close()
    report['elapsed_seconds'] = perf_counter()-start
    return report


def validate(p):
    start = perf_counter()
    _, _, report = canonical(p, homeostasis(p), make_assay(p))
    return E.clean(dict(experiment='E3 validation', status='passed' if report['passed'] else 'failed',
                        canonical=report, params=p, elapsed_seconds=perf_counter()-start))


def check(p):
    """Tiny structural checks plus the canonical alpha=1 network check."""
    checks = {}
    def test(name, ok):
        checks[name] = bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)
    homeo = homeostasis(p)
    base = jnp.asarray(np.r_[np.zeros(3), np.asarray(homeo.tau0).ravel(), E.ANCESTOR[DIRECT]])
    genome = np.arange(2*3*34, dtype=np.float32).reshape(2, 3, 34)/100
    g = genome[..., 3:24].reshape(2, 3, 3, 7)
    guesses = genome[..., :3]
    hg, hc, hx = map(np.asarray, hybrid_group(genome, g, np.random.default_rng(3), 4, guesses))
    test('acute hybrids use matching tau-site donor conductances',
         all(np.all((hg[12+i*4:16+i*4] == hg[:4]) | (hg[12+i*4:16+i*4] == hg[(i+1)*4:(i+2)*4]))
             for i in range(2)) and np.array_equal(hg[:, 3:24].reshape(hc.shape), hc))
    test('hybrid guesses average the two parental log scales',
         np.array_equal(hx[:12], hg[:12, :3]) and
         np.array_equal(hx[12:16], (hx[:4]+hx[4:8])/2) and
         np.array_equal(hx[16:20], (hx[:4]+hx[8:12])/2))
    direction = np.asarray(homeo.direction(base))
    test('canonical direction equals G0', np.array_equal(direction, G0))
    tau = 10.**np.asarray(base[3:24]).reshape(3, 7)
    test('zero-origin steady states have common tau*g within each cell',
         np.allclose(direction*tau, (direction*tau)[:, :1], rtol=2e-6))
    mutant = base.at[5].add(np.log10(2.))
    expected = np.ones((3, 7)); expected[0, 2] = .5
    test('single tau mutation changes only that channel direction',
         np.allclose(homeo.direction(mutant)/direction, expected, rtol=1e-6))
    # An analytic calcium sensor tests convergence and safeguards without
    # additional network runs. For this genome mean(g/G0) equals alpha.
    sensor = homeostasis(p)
    sensor.mean_ca = lambda gs: jnp.mean(gs[..., REG].reshape(*gs.shape[:-1], 3, 7)/G0, axis=-1)
    sa, sg, sd = sensor.iterate(base, 24, initial_alpha=[.5, 2., .5])
    test('linear sensor recovers all scales and reports endpoint Ca',
         sd['fraction_converged'] == 1 and np.allclose(sa, 1, atol=.02) and
         np.allclose(sd['relative_ca_error'], np.abs(np.asarray(sensor.mean_ca(conductances(base, sg)))-1)))
    sensor.gain = 16.
    _, sg, sd = sensor.iterate(base, 2, initial_alpha=[1.01]*3, tolerance=1e-6)
    test('oversized trial uses bracket fallback and keeps measured endpoint',
         sd['bracket_fallbacks'] > 0 and
         np.allclose(sd['relative_ca_error'], np.abs(np.asarray(sensor.mean_ca(conductances(base, sg)))-1)))
    sensor.gain = p['reg_gain']
    batch = jnp.broadcast_to(base, (50, 34))
    starts = np.ones((50, 3)); starts[-1] = 2.
    _, _, enough = sensor.iterate(batch, 12, starts)
    starts[-2] = 2.
    _, _, limited = sensor.iterate(batch, 3, starts)
    test('98 percent stopping respects minimum and maximum update budgets',
         enough['iterations'] == 2 and enough['fraction_converged'] == .98 and
         limited['iterations'] == 3 and limited['fraction_converged'] == .96)
    genomes = jnp.broadcast_to(base, (2, 2, 34))
    fakef = jnp.broadcast_to(jnp.array([500., .2, .2, .2, .35, .55, .65, .85]), (2, 2, 8))
    plan = E.random_plan((2, 2, 34), 2, .1, .01, np.random.default_rng(5))
    a = b = genomes
    for t in range(2):
        a = inherit(a, fakef, jnp.ones((2, 2), bool), fakef[0, 0], 0., plan, t)
        b = jax.vmap(inherit, in_axes=(0, 0, 0, None, None, None, None))(
            b[None], fakef[None], jnp.ones((1, 2, 2), bool), fakef[0, 0], 0., plan, t)[0]
    test('R=1 genome reproduction equals unbatched for two generations', np.array_equal(a, b))
    guesses = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)/10
    child, inherited = inherit(genomes, fakef, jnp.ones((2, 2), bool), fakef[0, 0], 0., plan, 0, guesses)
    batched_child, batched_guess = jax.vmap(inherit, in_axes=(0, 0, 0, None, None, None, None, 0))(
        genomes[None], fakef[None], jnp.ones((1, 2, 2), bool), fakef[0, 0], 0., plan, 0, guesses[None])
    parents = np.minimum((np.asarray(plan[0][0, ..., 0])*2).astype(int), 1)
    test('solver guesses follow selected parents without altering genomes or batching',
         np.array_equal(child, inherit(genomes, fakef, jnp.ones((2, 2), bool), fakef[0, 0], 0., plan, 0)) and
         np.array_equal(inherited, guesses[jnp.arange(2)[:, None], parents]) and
         np.array_equal(child, batched_child[0]) and np.array_equal(inherited, batched_guess[0]))
    _, selected = inherit(genomes, fakef, jnp.array([[True, False], [False, True]]),
                          fakef[0, 0], 1., plan, 0, guesses)
    test('selected offspring inherit the valid parent solver guess',
         np.array_equal(selected, np.stack([np.repeat(np.asarray(guesses[k, k])[None], 2, axis=0)
                                           for k in range(2)])))
    test('replicate r retains seed plus r streams',
         all(np.array_equal(x.generate_state(4), y.generate_state(4))
             for r, streams in enumerate(E.replicate_streams(p['seed'], 2))
             for x, y in zip(streams, E.replicate_streams(p['seed']+r, 1)[0])))
    tiny = dict(p, duration_ms=100., window_start_ms=0., reg_transient_ms=0., reg_window_ms=20., chunk_size=None)
    short, assay = homeostasis(tiny), make_assay(tiny)
    aa, ag, ad = short.iterate(a, 1)
    compile_counts = (short.compile_count, CALCIUM_TRACES)
    ba, bg, bd = short.iterate(b[None], 1)
    af, av = assay(jnp.log10(conductances(a, ag)))
    bf, bv = assay(jnp.log10(conductances(b[None], bg)))
    test('R=1 regulation and assay equal unbatched', np.array_equal(ag, bg[0]) and
         np.array_equal(aa, ba[0]) and np.array_equal(af, bf[0], equal_nan=True) and np.array_equal(av, bv[0]))
    test('endpoint diagnostics retain every cell and individual',
         np.shape(ad['relative_ca_error']) == (2, 2, 3) and np.shape(bd['relative_ca_error']) == (1, 2, 2, 3))
    _, fresh, _ = short.iterate(a, 1)
    test('development repeats from scratch without parental state', np.array_equal(fresh, ag))
    test('no regulation compilation or tracing after first generation',
         compile_counts == (short.compile_count, CALCIUM_TRACES) and short.compile_count == 1)
    test('each regulation observation has synchronized timing and compile count',
         all(h['wall_seconds'] > 0 and h['compile_count'] == 1 for h in ad['history']) and
         all(h['compile_seconds'] == 0 for h in bd['history']))
    gs = conductances(a, ag).reshape(-1, 31)
    observed = short.mean_ca(gs, duration=20., transient=5.)
    reference = jax.jit(jax.vmap(lambda x: reference_calcium(x, p['dt'], 20., 5.)))(gs)
    test('online calcium agrees with trace reconstruction including transient',
         observed.dtype == jnp.float32 and np.allclose(observed, reference, rtol=2e-5, atol=1e-5))
    chunked = homeostasis(dict(tiny, chunk_size=3))
    # Reuse the measured unroll to isolate batching from tuning differences.
    chunked.unroll, chunked.probe_timings = short.unroll, dict(short.probe_timings)
    limited = chunked.mean_ca(gs, duration=20., transient=5.)
    test('optional chunking including a short tail agrees with whole batch',
         chunked.compile_count == 2 and np.allclose(limited, observed, rtol=2e-5, atol=1e-5))
    count = short.compile_count
    short.mean_ca(gs, duration=20., transient=5.)
    test('duration and transient executables are cached', short.compile_count == count == 2)
    # Two nearby genomes on the actual tiny network, with a nonunit parental
    # endpoint. Both child solves use identical targets and inverse-tau ratios.
    target = short.mean_ca(conductances(base, jnp.asarray(G0)))
    parent = base.at[:3].set(jnp.log10(target)).at[3:24].add(np.log10(1.1))
    children = jnp.stack([parent.at[3:24].add(np.log10(scale)) for scale in (1.01, 1.02)])
    # The 20 ms Ca response is weak: resolve residuals more tightly than the
    # production tolerance before comparing alpha at that tolerance.
    solver_tol = min(.001, short.ca_tol/20)
    pa, _, pd = short.iterate(parent, 256, tolerance=solver_tol)
    cold, _, cd = short.iterate(children, 256, tolerance=solver_tol)
    warm, _, wd = short.iterate(children, 256, initial_alpha=pa, tolerance=solver_tol)
    alpha_error = float(np.max(np.abs(np.asarray(warm)/np.asarray(cold)-1)))
    test('warm and cold starts converge to equivalent alpha on the same tiny genomes',
         all(d['history'][-1]['fraction_solver_converged'] == 1 for d in (pd, cd, wd)) and
         alpha_error <= short.ca_tol)
    equivalence = dict(max_relative_alpha_difference=alpha_error, solver_tolerance=short.ca_tol,
                       validation_ca_tolerance=solver_tol,
                       parent_alpha=np.asarray(pa).tolist(), warm=wd, cold=cd)
    _, _, validation = canonical(p, homeo, make_assay(p), recovery=False)
    test('canonical alpha=1 is on target and pyloric', validation['passed'])
    return E.clean(dict(experiment='E3 check', status='passed' if all(checks.values()) else 'failed',
                        checks=checks, canonical=validation, params=p,
                        regulation=short.report(), iteration_timings=ad['history'],
                        warm_cold_equivalence=equivalence))


def summary(data):
    lines = [f"Experiment E3: {data['status']}", str(data.get('params', {}))]
    lines += list(data.get('inference', {}).values())
    if 'reason' in data:
        lines.append(data['reason'])
    v = data.get('canonical')
    if v:
        lines += [f"Canonical mean Ca targets: {v['target']}",
            f"Canonical features ({', '.join(E.FEATURES)}): {v['features']}",
            f"Short-window fixed-point max relative g error: {v['fixed_point_relative_g_error']:.6g}",
            f"Fixed-point convergence: {v['fixed_point']}", f"Validation passed: {v['passed']}"]
        if 'recovery' in v:
            lines += [f"Recovery: {v['perturbation']}", f"Recovery features: {v['recovery_features']}",
                f"Recovery normalized feature errors: {v['recovery_feature_errors_in_E_tolerances']}",
                f"Recovery iterations attempted: {v['recovery']['iterations']}; iterations needed: {v['iterations_needed']}",
                f"Recovery convergence: {v['recovery']}", f"Single-tau perturbation: {v['tau_perturbation']}"]
    if data['status'] == 'complete':
        lines += [f"Last-generation convergence: {data['convergence'][-1] if data['convergence'] else 'no generations'}",
                  f"Hybrid convergence: {data['hybrid_convergence']}",
                  'Schulz 2007 Table 1 | E2 pooled individual r (within-lineage unavailable) | E3 mean within-lineage r']
        for row in data['schulz']:
            lines.append(f"{row['cell']} {'-'.join(row['pair'])}: {row['schulz']} | "
                         f"{row['E2_pooled_individual_r']} | {row['E3_within_lineage_r']}")
        for r in data['replicates']:
            for arm in E.ARMS:
                last = r['trajectories'][arm][-1]
                lines.append(f"replicate {r['index']} {arm}: fraction pyloric={last['fraction_pyloric']}, "
                             f"rhythm divergence={last['feature_divergence_normalized']}; hybrids={r['hybrids'][arm]}")
            lines.append(f"replicate {r['index']} omega classes: {r['omega_classes']}")
        lines.append('Omega by site (genome): ' + str(data['aggregate']['genome_omega']))
        lines.append('Omega by site (emergent conductances): ' + str(data['aggregate']['omega']))
        for level, rows in data['aggregate']['correlations'].items():
            lines.append(f'{level} correlations: selected/neutral r, consistency, paired Holm')
            for row in rows:
                lines.append(f"{row['cell']} {'-'.join(row['pair'])}: {row['selected']['mean']} / {row['neutral']['mean']}; "
                    f"signs {row['selected']['same_sign']}/{data['params']['R']}; Holm={row['paired_holm_p']}; "
                    f"associated={row['selection_associated']}")
    if 'checks' in data:
        lines += [f"{k}: {v}" for k, v in data['checks'].items()]
    return '\n'.join(lines)+'\n'


def figure(path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    if not hasattr(cm, 'get_cmap'):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401
    data = json.loads(path.read_text())
    with plt.rc_context({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'lines.linewidth': .55}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.85, wspace=.65)
        for ax, label in zip(axes, 'ABC'):
            E.panel_letter(ax, label)
            ax.spines[['top', 'right']].set_visible(False)
        if data['status'] != 'complete':
            for ax in axes:
                ax.set_axis_off()
            axes[1].text(.5, .5, 'Canonical regulation validation failed\nEvolution was not run\nSee summary for recovery diagnostics',
                         ha='center', va='center', transform=axes[1].transAxes)
        else:
            for key, label, color, marker in (('E2_pooled_individual_r', 'E2 pooled', 'C1', 'o'),
                                             ('E3_within_lineage_r', 'E3 within', 'C0', 's')):
                axes[0].scatter([r['schulz'] for r in data['schulz']],
                                [r[key] for r in data['schulz']], s=12, label=label, color=color, marker=marker)
            axes[0].plot([.7, 1], [.7, 1], ':', color='.5')
            axes[0].set(xlabel='Schulz mRNA r', ylabel='Model log-g r', ylim=(-1.05, 1.05))
            axes[0].legend(frameon=False, fontsize=5)
            for arm, color in (('selected', 'C0'), ('neutral', '0.55')):
                for group, style in (('cross_lineage', '-'), ('within_lineage', '--'), ('cross_parents', ':')):
                    values = np.array([[r['hybrids'][arm][group][stage] for stage in ('acute', 'after')]
                                       for r in data['replicates']])
                    axes[1].plot([0, 1], values.mean(0), style, color=color, marker='o', ms=2)
            axes[1].set(xticks=[0, 1], xticklabels=['Acute', 'Regulated'], ylabel='Fraction pyloric', ylim=(-.05, 1.05))
            handles = [Line2D([], [], color='0.2', ls=style, label=label)
                       for label, style in (('between', '-'), ('within', '--'), ('parents', ':'))]
            handles += [Line2D([], [], color=color, marker='o', ls='None', ms=3, label=arm)
                        for arm, color in (('selected', 'C0'), ('neutral', '0.55'))]
            axes[1].legend(handles=handles, frameon=False, fontsize=5, ncol=3,
                           loc='upper center', bbox_to_anchor=(.5, -.25),
                           borderaxespad=0, columnspacing=.8)
            if data['hybrid_convergence']['fraction_converged'] < 1:
                axes[1].set_title('Iteration limit; not all converged', fontsize=5)
            classes = ('Ca_targets', 'tau_m', 'emergent_g', 'synapses', 'leaks')
            axes[2].scatter(range(len(classes)), [data['aggregate']['omega_classes'][c]['mean'] for c in classes], s=12, label='E3', color='C0')
            baseline_classes = [(i, c) for i, c in enumerate(classes) if c in data['baseline_omega']]
            axes[2].scatter([i+.12 for i, c in baseline_classes],
                            [data['baseline_omega'][c]['mean'] for i, c in baseline_classes], s=12, label='E2', color='C1')
            axes[2].axhline(1, color='.6', ls=':')
            axes[2].set(xticks=range(len(classes)), xticklabels=['Ca target', r'$\tau_m$', 'g', 'synapse', 'leaks'], ylabel=r'$\omega$ (selected / neutral)')
            axes[2].tick_params(axis='x', labelsize=5)
            axes[2].legend(frameon=False, fontsize=5)
            axes[2].text(0, -.36, 'E2 has no Ca-target or tau-m genome sites', fontsize=4, transform=axes[2].transAxes)
        fig.suptitle('E3: activity-dependent regulation' + (' — smoke test' if data['params']['R'] < 3 else ''), fontsize=7)
        for suffix in ('.pdf', '.png'):
            fig.savefig(path.with_suffix(suffix), dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fig = sub.add_parser('figure')
    fig.add_argument('--input', type=Path, default=OUTPUT/'stg-homeo.json')
    for command in ('check', 'validate', 'replicates', 'bench'):
        cli = sub.add_parser(command)
        cli.add_argument('--R', '--replicates', type=int, default=4)
        for name, default in dict(N=64, K=16, B=200, T=2000, every=20, seed=0, n_reg=12,
                                  hybrid_samples=64, hybrid_iters=12, validation_iters=24).items():
            cli.add_argument('--'+name.replace('_', '-'), type=int, default=default)
        for name, default in dict(u=.02, sigma=.05, s=1., dt=.05, duration_ms=12000., window_start_ms=4000.,
            reg_window_ms=2000., reg_transient_ms=1000., reg_gain=.3, controller_seconds=300., ca_tol=.02).items():
            cli.add_argument('--'+name.replace('_', '-'), type=float, default=default)
        cli.add_argument('--chunk-size', type=int, nargs='?', const=16384, default=None,
                         help='memory-limited batch cap; bare flag uses 16384; default is all networks')
        if command == 'bench':
            cli.add_argument('--max-seconds', type=float, default=900.,
                             help='hard wall limit including setup/compile; print partial report on timeout')
        cli.add_argument('--baseline', type=str, default=str(OUTPUT/'stg-replicates.json'))
        cli.add_argument('--output', type=Path, default=OUTPUT/(f'stg-homeo-{command}.json' if command in ('check', 'validate') else 'stg-homeo.json'))
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'figure':
        figure(args['input'])
        return
    path = args.pop('output')
    E.validate_assay(args['dt'], args['duration_ms'], args['window_start_ms'])
    E.validate_assay(args['dt'], args['reg_window_ms']+args['reg_transient_ms'], args['reg_transient_ms'])
    if (min(args[k] for k in ('R', 'N', 'every', 'n_reg', 'hybrid_samples', 'hybrid_iters', 'validation_iters')) < 1
        or args['n_reg'] < 2 or (args['chunk_size'] is not None and args['chunk_size'] < 1)
        or args.get('max_seconds', 900.) <= 0 or args['K'] < 2 or min(args[k] for k in ('B', 'T', 'seed', 's', 'sigma')) < 0 or not 0 <= args['u'] <= 1
        or min(args[k] for k in ('reg_gain', 'controller_seconds', 'ca_tol')) <= 0
        or not all(np.isfinite(x) for x in args.values() if isinstance(x, (int, float)))):
        parser.error('Require positive sizes/times/tolerances, n_reg>=2, K>=2, B,T,seed,s,sigma>=0, 0<=u<=1')
    data = check(args) if command == 'check' else validate(args) if command == 'validate' else bench(args) if command == 'bench' else run(args)
    if command == 'bench':
        print(json.dumps(data, separators=(',', ':'), allow_nan=False))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
        path.with_name(path.stem+'-summary.txt').write_text(summary(data))
        print(summary(data) if data['status'] != 'complete' else f'Wrote {path}; final convergence: {data["convergence"][-1]}')
    if data['status'] in ('failed', 'validation_failed'):
        raise SystemExit(1)
    if data['status'] == 'timeout':
        raise SystemExit(124)


if __name__ == '__main__':
    main()
