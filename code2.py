"""Experiment F2: F's efficient code in JAX, natural contrasts, depth and load.

Run: python code2.py [check | validate | quick | benchmark | run | figure]
     [--contrasts output/contrasts.json] [--config-chunk 2]
     [--individual-chunk 128] [--max-hours 4] [--sigma-n 0.0128]
Both quick and run write output/code2.* (override with --output); quick is
explicitly labelled a smoke test. Use --contrasts output/contrasts.json.
validate defaults to a short CPU comparison; --full uses F's 20,000 generations,
K=8 adaptation and K=32 forks. Only that full comparison tests reproduction of
the saved equilibrium table. --cells 10:3,10:30 limits validation to small cells.
Short runs cannot establish equilibrium or omega.

The model, gain and omega definition are F's (code.py/drift.py). Output noise is
fixed at F's 0.0128 for BOTH distributions, with no CDF-based noise calibration.
I_max is the best numerical representable-code information, optimized over z;
unrestricted monotone optimization is a separate diagnostic, not the fitness
denominator. Neither optimization certifies a global optimum. A paired van
Hateren adaptation cell at sigma_n=0.0011 is saved under low_noise_sensitivity,
outside the main grid and drift comparisons. Float32 evolution uses discrete-grid
mutual information with Gaussian support truncated at >=8 SD (omitted mass < 2e-15); check compares
against F's float64 calculation. The 64-point CDF is the small-noise reference,
not a certified finite-noise optimum. Empty measured bins receive a 1e-12 floor
for the log-slope projection, explicitly recorded in the environment.

Independent Bernoulli mutation at every site, normalization ONLY of touched
blocks, and Wright-Fisher resampling are vmapped over configs/replicates/sites'
individual genomes. Generations use nested lax.scan; information is evaluated
in bounded individual chunks. Configs with equal shapes are batched in small
chunks. Neutral forks share the selected fork's starting population and random
keys, without neutral readaptation. Every fork uses the first adapted replicate.
XLA preallocation defaults off; an existing environment setting is respected.

Full: real N={10,30,100,300}, s={1,3,10,30,100,300,1000}; real N={1000,3000}
at Ns={300,1000,3000,10000,30000}; stand-in N=100 with the same seven s values;
depth L={1,3,6}, N=100, s={100,1000}; load Ns={300,3000}, N={10,30,100,300,1000},
u={.0001,.0003,.001,.003}. K=8, 20,000 adaptation generations throughout;
depth forks have 32 lineages and 20,000 generations, paired with s=0.
Identical cells are reused. Last-quarter summaries are finite-run estimates.
Collapse ranks fixed candidate scalings by leave-one-N-out, log-coordinate
local regression, with a common interpolation-only support mask. It is
descriptive, not evidence that one dimensionless group is universal.

benchmark prints measured throughput and full-grid work estimates before run.
On CPU the GPU projection is an explicitly hypothetical 20--100x speedup, not
a measured B300 forecast. On GPU a same-device pilot gates the full run at
--max-hours; no generations are silently shortened. Contention, compilation,
and convergence can increase runtime. JSON contains no individual genomes.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
from time import perf_counter

import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as np
from scipy.optimize import minimize

import code as original
from drift import add_omega

OUTPUT = original.OUTPUT
DEFAULT_SIGMA_N = 0.0128
SENSITIVITY_SIGMA_N = 0.0011


def channel(distribution="standin", contrasts=OUTPUT / "contrasts.json", seed=0,
            sigma_n=DEFAULT_SIGMA_N):
    if not np.isfinite(sigma_n) or sigma_n <= 0:
        raise ValueError("sigma_n must be finite and positive")
    env = original.Channel(seed=seed, sigma_n=sigma_n)
    if distribution == "vanhateren":
        data = json.loads(Path(contrasts).read_text())
        p = np.asarray(data["probabilities"], dtype=float)
        if (p.shape != (64,) or not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.isclose(p.sum(), 1) or not np.allclose(data["contrasts"], env.c)):
            raise ValueError("Contrasts must be normalized nonnegative probabilities on F's grid")
        env.p = np.maximum(p, 1e-12)
        env.p /= env.p.sum()
        cdf = env.p.cumsum()
        env.star = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        env.z_star = env.Phi.T @ np.log(env.p[1:])
        env.gain = float(np.linalg.norm(env.z_star) / env.ancestor_top_norm)
        env.params.update(distribution=distribution, gain=env.gain,
                          probability_floor=1e-12, raw_probabilities=p.tolist(),
                          contrast_source=str(contrasts), contrast_metadata={k: data[k] for k in
                          ("images", "sigma_pixels", "pixel_count", "moments", "fraction_clipped", "sample")})
        env.projected_response = env.coefficient_response(env.z_star)
        env.projected_z = env.phenotype_for_coefficients(env.z_star)
        env.set_noise(env.requested_sigma_n)
    elif distribution != "standin":
        raise ValueError(distribution)
    env.params["distribution"] = distribution
    fixed_noise_reference(env)
    print(f"{distribution}: sigma_n={env.sigma_n:g}, I_max={env.I_max:.7f}, "
          f"CDF/I_max={env.cdf_information / env.I_max:.5f}, gain={env.gain:.4f}", flush=True)
    return env


def fixed_noise_reference(env, starts=6):
    """Optimize at the requested noise; never tune noise to favor the CDF."""
    rng = np.random.default_rng(env.params["seed"] + 7123)
    uniform = np.zeros(env.Phi.shape[1])
    uniform[0] = -1
    guesses = [uniform, env.projected_z]
    guesses += list(original.unit(rng.normal(size=(max(0, starts - 2), len(uniform)))))
    constraint = dict(type="eq", fun=lambda z: z @ z - 1, jac=lambda z: 2 * z)
    results = [minimize(env.objective_z, z, jac=True, method="SLSQP",
                        constraints=[constraint], options=dict(ftol=1e-11, maxiter=600))
               for z in guesses]
    best = min(results, key=lambda r: r.fun)
    if not best.success:
        raise RuntimeError(f"Representable reference did not converge: {best.message}")
    env.best_z = original.unit(best.x)
    best_response = env.response(env.best_z)
    env.I_max = float(env.information(best_response))

    # The unrestricted diagnostic has 62 interior responses and fixed endpoints.
    # Include the best representable response as a feasible start, and the linear
    # response (already near the entropy ceiling at low noise).
    n = len(env.c) - 2
    def objective(interior):
        value, grad = env.information(np.r_[0, interior, 1], gradient=True)
        return -float(value), -grad[1:-1]

    direct_results = [minimize(objective, f[1:-1], jac=True, method="SLSQP",
                               bounds=[(0, 1)] * n,
                               constraints=[dict(type="ineq", fun=lambda x: np.diff(x),
                                                 jac=lambda x: np.diff(np.eye(n), axis=0))],
                               options=dict(ftol=1e-11, maxiter=500))
                      for f in (env.star, best_response, np.linspace(0, 1, len(env.c)))]
    direct_best = min(direct_results, key=lambda r: r.fun)
    if not direct_best.success or np.min(np.diff(direct_best.x)) < -1e-8:
        raise RuntimeError(f"Unrestricted reference did not converge: {direct_best.message}")
    direct = dict(information=-float(direct_best.fun), response=np.r_[0, direct_best.x, 1].tolist(),
                  success=bool(direct_best.success), starts=[dict(information=-float(r.fun),
                  success=bool(r.success), message=str(r.message)) for r in direct_results])

    # Identical draws across noise levels: these diagnose broad plateaus rather
    # than estimating a probability under an unspecified 'random monotone' prior.
    rng = np.random.default_rng(env.params["seed"] + 7124)
    increments = rng.dirichlet(np.ones(len(env.c) - 1), size=5)
    random_codes = np.column_stack((np.zeros(5), increments.cumsum(axis=1)))
    random_z = original.unit(rng.normal(size=(5, len(uniform))))
    flatness = dict(linear_efficiency=float(env.information(np.linspace(0, 1, len(env.c))) / env.I_max),
                    random_monotone_efficiencies=(env.information(random_codes) / env.I_max).tolist(),
                    random_representable_efficiencies=(env.information(env.response(random_z)) / env.I_max).tolist(),
                    seed=env.params["seed"] + 7124,
                    note="Five Dirichlet(1) increment codes and five isotropic z directions; same draws at each noise. Fractions of representable I_max; unrestricted codes may exceed 1.")
    projected_information = float(env.information(env.projected_response))
    env.calibration = dict(I_max=env.I_max, cdf_information=env.cdf_information,
                           best_z_information=env.I_max, best_z=env.best_z.tolist(),
                           best_response=best_response.tolist(),
                           best_cdf_ks=float(np.max(np.abs(best_response - env.star))),
                           cdf_fit_ks=float(np.max(np.abs(env.projected_response - env.star))),
                           z_star=env.z_star.tolist(), projected_z=env.projected_z.tolist(),
                           projected_information=projected_information,
                           ancestor_top_norm=env.ancestor_top_norm,
                           requested_sigma_n=env.requested_sigma_n, noise_policy="fixed; no recalibration",
                           I_max_definition="best numerical representable code (direct optimization over z)",
                           direct_optimization=direct, flatness=flatness,
                           starts=[dict(information=-float(r.fun), success=bool(r.success),
                                        message=str(r.message)) for r in results],
                           optimization_note="Multistart local numerical optima, not certified global maxima; KS at low noise depends on which near-degenerate optimum is found.")


def unit(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def normalize(g, d=16):
    maps = g[..., d:].reshape(*g.shape[:-1], -1, d * d)
    return jnp.concatenate((unit(g[..., :d]), (unit(maps) * np.sqrt(d)).reshape(*g.shape[:-1], -1)), -1)


def forward(g, d=16):
    maps = g[..., d:].reshape(*g.shape[:-1], -1, d, d)
    x = g[..., :d]
    states = [x]
    for layer in range(maps.shape[-3]):
        x = jnp.einsum("...ij,...j->...i", maps[..., layer, :, :], x)
        states.append(x)
    return jnp.stack(states, axis=-2)


def mutate_one(g, mask, noise, sigma, d=16):
    changed = g + mask * noise * sigma
    x = jnp.where(jnp.any(mask[:d]), unit(changed[:d]), g[:d])
    maps = changed[d:].reshape(-1, d * d)
    touched = mask[d:].reshape(-1, d * d).any(-1, keepdims=True)
    maps = jnp.where(touched, unit(maps) * np.sqrt(d), g[d:].reshape(maps.shape))
    return jnp.concatenate((x, maps.ravel()))


class Engine:
    def __init__(self, env, individual_chunk=128):
        self.env, self.chunk = env, individual_chunk
        self.p, self.Phi = jnp.array(env.p), jnp.array(env.Phi)
        self.r = jnp.array(env.r)
        self.dr = float(env.r[1] - env.r[0])
        radius = int(np.ceil(8 * env.sigma_n / self.dr)) + 1
        self.offsets = jnp.arange(-radius, radius + 1)
        self.run = jax.jit(self._run, static_argnames=("T", "every", "neutral"))

    def response(self, z):
        a = jax.nn.softmax((self.env.gain * unit(z)) @ self.Phi.T, axis=-1)
        f = jnp.concatenate((jnp.zeros((*a.shape[:-1], 1)), jnp.cumsum(a, axis=-1)), axis=-1)
        return f / f[..., -1:]

    def information_one(self, f):
        # Scatter a short Gaussian stencil into the SAME response grid as F.
        index = jnp.rint((f - self.r[0]) / self.dr).astype(jnp.int32)[:, None] + self.offsets
        valid = (index >= 0) & (index < len(self.r))
        index = jnp.clip(index, 0, len(self.r) - 1)
        logq = -0.5 * ((self.r[index] - f[:, None]) / self.env.sigma_n)**2
        q = jnp.where(valid, jnp.exp(logq), 0)
        norm = q.sum(-1, keepdims=True)
        q /= norm
        mix = jnp.zeros(len(self.r)).at[index.ravel()].add((self.p[:, None] * q).ravel())
        conditional = self.p @ (jnp.log(norm[:, 0]) - (q * logq).sum(-1))
        return -(mix * jnp.log(jnp.maximum(mix, 1e-30))).sum() - conditional

    def information(self, f):
        flat = f.reshape(-1, f.shape[-1])
        return lax.map(self.information_one, flat, batch_size=min(self.chunk, len(flat))).reshape(f.shape[:-1])

    def evaluate(self, g):
        # Three explicit vmap axes: config, replicate, individual.
        f = vmap(vmap(vmap(lambda x: self.response(forward(x)[-1]))))(g)
        return self.information(f)

    def step(self, carry, params, neutral=False):
        g, info, keys = carry
        s, u = params

        def population(genomes, values, key, selection, mutation):
            next_key, parent_key, mask_key, noise_key = random.split(key, 4)
            logw = selection * (values / self.env.I_max - 1)
            cdf = jnp.cumsum(jnp.exp(logw - jnp.max(logw)))
            draws = random.uniform(parent_key, (len(genomes),)) * cdf[-1]
            parents = jnp.minimum(jnp.searchsorted(cdf, draws, side="left"), len(genomes) - 1)
            inherited = genomes[parents]
            mask = random.bernoulli(mask_key, mutation, inherited.shape)
            noise = random.normal(noise_key, inherited.shape)
            offspring = vmap(mutate_one, in_axes=(0, 0, 0, None))(inherited, mask, noise, 0.1)
            return offspring, values[parents], next_key, mask.any(-1)

        batch = vmap(vmap(population, in_axes=(0, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0))
        offspring, inherited_info, next_keys, changed = batch(g, info, keys, s, u)
        # At u=0 preserve cached values exactly. Neutral arms only evaluate at records.
        values = inherited_info if neutral else jnp.where(changed, self.evaluate(offspring), inherited_info)
        return (offspring, values, next_keys), None

    def snapshot(self, g, info):
        states = forward(g)
        f = self.response(states[..., -1, :]).mean(axis=2)
        directions = unit(states)
        means = unit(directions.mean(axis=2))
        maps = g[..., 16:].reshape(*g.shape[:3], -1, 256)
        map_means = unit(maps.mean(axis=2))
        K = g.shape[1]

        def divergence(x):
            return jnp.clip((K * K - (x.sum(axis=1)**2).sum(-1)) / (K * (K - 1)), 0, 2)

        return dict(efficiency=(info / self.env.I_max).mean(axis=2), responses=f,
                    ks=jnp.max(jnp.abs(f - jnp.array(self.env.star)), axis=-1),
                    ks_best=jnp.max(jnp.abs(f - jnp.array(self.env.calibration["best_response"])), axis=-1),
                    state_divergence=divergence(means), map_divergence=divergence(map_means),
                    within_variation=jnp.clip(1 - (directions * means[:, :, None]).sum(-1), 0, 2).mean(axis=(1, 2)),
                    response_divergence=2 * K / (K - 1) * ((f - f.mean(axis=1, keepdims=True))**2).mean(axis=(1, 2)))

    def _run(self, g, keys, s, u, T, every, neutral=False):
        info = self.evaluate(g)
        first = self.snapshot(g, info)

        def block(carry, _):
            carry, _ = lax.scan(lambda c, _: self.step(c, (s, u), neutral), carry, None, length=every)
            genomes, values, keys = carry
            measured = self.evaluate(genomes) if neutral else values
            return carry, self.snapshot(genomes, measured)

        carry, records = lax.scan(block, (g, info, keys), None, length=T // every)
        records = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), first, records)
        if T % every:
            carry, _ = lax.scan(lambda c, _: self.step(c, (s, u), neutral), carry, None, length=T % every)
            measured = self.evaluate(carry[0]) if neutral else carry[1]
            last = self.snapshot(carry[0], measured)
            records = jax.tree.map(lambda a, b: jnp.concatenate((a, b[None])), records, last)
        return carry[0], records


def config(N, s, L=3, u=1e-3, distribution="vanhateren", K=8, T=20000, every=200, seed=0, tags=()):
    return dict(N=N, s=float(s), L=L, u=u, distribution=distribution, K=K,
                T=T, every=every, seed=seed, d=16, sigma=0.1, tags=list(tags))


def grid(seed=0, quick=False, generations=None):
    rows = {}

    def add(N, s, tag, **kw):
        row = config(N, s, seed=seed, **kw)
        key = (N, float(s), row["L"], row["u"], row["distribution"])
        if key not in rows:
            rows[key] = row
        rows[key]["tags"].append(tag)

    for N in (10, 30, 100, 300):
        for s in (1, 3, 10, 30, 100, 300, 1000):
            add(N, s, "adaptation")
    for N in (1000, 3000):
        for ns in (300, 1000, 3000, 10000, 30000):
            add(N, ns / N, "adaptation")
    for s in (1, 3, 10, 30, 100, 300, 1000):
        add(100, s, "comparison", distribution="standin")
    for L in (1, 3, 6):
        for s in (100, 1000):
            add(100, s, "depth", L=L)
    for ns in (300, 3000):
        for N in (10, 30, 100, 300, 1000):
            for u in (1e-4, 3e-4, 1e-3, 3e-3):
                add(N, ns / N, "load", u=u)
    result = list(rows.values())
    if quick:
        # Same scientific categories, tiny population/replicate/time sizes.
        result = [config(N, s, K=2, T=8, every=2, seed=seed, tags=("adaptation",))
                  for N in (4, 8) for s in (3, 1000)]
        result += [config(8, 1000, L=L, K=2, T=8, every=2, seed=seed, tags=("depth",)) for L in (1, 3, 6)]
        result += [config(8, s, distribution="standin", K=2, T=8, every=2, seed=seed,
                          tags=("comparison",)) for s in (3, 1000)]
        result += [config(N, ns / N, u=u, K=2, T=8, every=2, seed=seed, tags=("load",))
                   for N in (4, 8, 12) for ns in (300, 3000) for u in (1e-4, 3e-3)]
    if generations is not None:
        for row in result:
            row.update(T=generations, every=min(row["every"], max(1, generations // 4)))
    return result


def seed_for(row, phase):
    # Stable under chunk ordering and fractional s (unlike int(s) seeds).
    s_bits = np.float64(row.get("source_s", row["s"])).view(np.uint64).item()
    u_bits = np.float64(row["u"]).view(np.uint64).item()
    entropy = [row["seed"], row["N"], row["L"], s_bits & 0xffffffff, s_bits >> 32,
               u_bits & 0xffffffff, u_bits >> 32, phase]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def initial(rows):
    genomes, keys = [], []
    for row in rows:
        rng = np.random.default_rng(seed_for(row, 1))
        ancestors = original.normalize(rng.normal(size=(row["K"], 16 + row["L"] * 256)), 16)
        genomes.append(np.repeat(ancestors[:, None], row["N"], axis=1))
        keys.append(random.split(random.PRNGKey(seed_for(row, 2)), row["K"]))
    return jnp.array(np.array(genomes), dtype=jnp.float32), jnp.stack(keys)


def execute(engine, rows, populations=None, neutral=False):
    if populations is None:
        g, keys = initial(rows)
    else:
        g = jnp.array(np.array([np.repeat(p[None], r["K"], axis=0) for r, p in zip(rows, populations)]))
        keys = jnp.stack([random.split(random.PRNGKey(seed_for(r, 91)), r["K"]) for r in rows])
    start = perf_counter()
    final, records = engine.run(g, keys, jnp.array([r["s"] for r in rows]),
                                jnp.array([r["u"] for r in rows]),
                                T=rows[0]["T"], every=rows[0]["every"], neutral=neutral)
    final.block_until_ready()
    seconds = perf_counter() - start
    records = jax.device_get(records)
    results = []
    for index, row in enumerate(rows):
        times = list(range(0, row["T"] + 1, row["every"]))
        if times[-1] != row["T"]:
            times.append(row["T"])
        rec = {k: np.asarray(v[:, index]) for k, v in records.items()}
        tail = np.asarray(times) >= 0.75 * row["T"]
        efficiencies = rec["efficiency"][tail]
        replicate_means = efficiencies.mean(axis=0)
        responses = rec["responses"][tail].mean(axis=0)
        middle = len(efficiencies) // 2
        eq = dict(N=row["N"], s=row["s"], Ns=row["N"] * row["s"], u=row["u"], L=row["L"],
                  distribution=row["distribution"], sigma_n=engine.env.sigma_n,
                  tags=row["tags"], mean=float(replicate_means.mean()),
                  replicate_sd=float(replicate_means.std(ddof=1)), replicate_means=replicate_means.tolist(),
                  temporal_sd=float(efficiencies.std(axis=0).mean()),
                  tail_change=float(efficiencies[middle:].mean() - efficiencies[:middle].mean()) if middle else 0.,
                  cdf_efficiency=float(replicate_means.mean() * engine.env.I_max / engine.env.cdf_information),
                  mean_ks=float(rec["ks"][tail].mean()),
                  mean_ks_best=float(rec["ks_best"][tail].mean()),
                  mean_response=responses.mean(axis=0).tolist(),
                  response_sd=responses.std(axis=0, ddof=1).tolist())
        compact = [dict(t=t, **{k: v[j].tolist() for k, v in rec.items() if k != "responses"})
                   for j, t in enumerate(times)]
        results.append(dict(**row, sigma_n=engine.env.sigma_n, equilibrium=eq, records=compact, batch_seconds=seconds,
                            batch_configs=len(rows)))
    return results, np.asarray(final[:, 0]), seconds


def groups(rows, chunk):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("distribution", "N", "L", "K", "T", "every"))].append(row)
    for batch in grouped.values():
        for start in range(0, len(batch), chunk):
            yield batch[start:start + chunk]


def collapse(table):
    """Compare predictions for held-out population sizes, on common support."""
    transforms = {
        "Ns": lambda r: [r["Ns"]], "s/u": lambda r: [r["s"] / r["u"]],
        "Ns/u": lambda r: [r["Ns"] / r["u"]],
        "Ns,Nu": lambda r: [r["Ns"], r["N"] * r["u"]],
    }
    if len({r["N"] for r in table}) < 3:
        return dict(best="Ns", scores={}, support=0, note="At least three N values required")
    y = np.array([r["mean"] for r in table])
    populations = np.array([r["N"] for r in table])
    predictions, supports = {}, []
    for name, transform in transforms.items():
        x = np.log10([transform(r) for r in table])
        x /= np.maximum(np.std(x, axis=0), 1e-12)
        pred, support = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
        for i in range(len(y)):
            train = populations != populations[i]
            dx = x[train] - x[i]
            support[i] = np.all((x[i] >= x[train].min(0) - 1e-9) & (x[i] <= x[train].max(0) + 1e-9))
            # Fixed bandwidth/ridge, same method for each prespecified scaling.
            weights = np.exp(-0.5 * (dx**2).sum(1) / 0.5**2)
            design = np.column_stack((np.ones(train.sum()), dx))
            ridge = np.diag([1e-10] + [1e-5] * x.shape[1])
            beta = np.linalg.solve(design.T @ (weights[:, None] * design) + ridge,
                                   design.T @ (weights * y[train]))
            pred[i] = beta[0]
        predictions[name] = pred
        supports.append(support)
    common = np.all(supports, axis=0)
    if not common.any():
        return dict(best="Ns", scores={}, support=0, note="No common interpolation support")
    scores = {name: float(np.sqrt(np.mean((pred[common] - y[common])**2))) for name, pred in predictions.items()}
    best = min(scores, key=scores.get)
    return dict(best=best, scores=scores, support=int(common.sum()), total=len(table),
                tolerance=0.02, approximate_collapse=scores[best] <= 0.02,
                improvement_over_Ns=scores["Ns"] - scores[best],
                max_prediction_error=float(np.max(np.abs(predictions[best][common] - y[common]))),
                note="Leave-one-N-out RMSE; common interpolation support; 2D Ns,Nu tests residual load dependence. Descriptive only.")


def work(rows, fork_T=20000, fork_K=32):
    adaptation = sum(r["N"] * r["K"] * r["T"] for r in rows)
    forks = sum(2 * r["N"] * fork_K * fork_T for r in rows if "depth" in r["tags"])
    return dict(adaptation_individual_generations=adaptation, fork_individual_generations=forks,
                total_individual_generations=adaptation + forks)


def benchmark(engine, seed=0, config_chunk=2):
    gpu = jax.default_backend() == "gpu"
    # Largest population and replicate count on device; CPU uses a tiny pilot.
    rows = [config(3000 if gpu else 100, 3 + i, K=8 if gpu else 4, T=100,
                   every=100, seed=seed) for i in range(config_chunk)]
    _, _, cold = execute(engine, rows)
    _, _, warm = execute(engine, rows)
    count = work(rows, 0)["total_individual_generations"]
    rate = count / warm
    full_work = work(grid(seed) + [config(100, 1000, seed=seed, tags=("low_noise_sensitivity",))])
    hours = full_work["total_individual_generations"] / rate / 3600
    result = dict(backend=jax.default_backend(), device=str(jax.devices()[0]),
                  individual_generations_per_second=rate, compile_and_first_seconds=cold,
                  warm_seconds=warm, pilot=rows[0], pilot_configs=len(rows), **full_work,
                  same_device_hours=hours, hypothetical_gpu_hours=[hours / 100, hours / 20] if not gpu else None,
                  budget_hours=2 * hours, budget_margin=2,
                  pilot_genome_megabytes=sum(r["N"] * r["K"] * (16 + r["L"] * 256) * 4 for r in rows) / 1e6,
                  caveat="Linear individual-generation extrapolation; pilot includes record overhead; deeper maps, neutral shortcuts, compilation and contention differ. GPU speedup range on CPU is assumed, not measured.")
    print(f"Pilot {result['device']}: {rate:,.0f} individual-generations/s; compile+first={cold:.2f}s, warm={warm:.2f}s.", flush=True)
    print(f"Full grid: {full_work['total_individual_generations']:,} individual-generations; "
          f"same-device projection {hours:.2f} h.", flush=True)
    print(f"Scheduling budget with 2x margin: {result['budget_hours']:.2f} h; "
          f"pilot genomes {result['pilot_genome_megabytes']:.1f} MB (scratch/compiled programs additional).", flush=True)
    if not gpu:
        print(f"Hypothetical B300 at 20--100x CPU: {hours / 100:.2f}--{hours / 20:.2f} h; "
              "unverified. Run benchmark on the shared GPU before the full grid.", flush=True)
    return result


def forks(engine, row, population, quick=False):
    common = {**row, "K": 3 if quick else 32, "T": row["T"] if quick else 20000,
              "source_s": row["s"], "init": "gaussian", "tanh": False}
    arms = []
    for s in (row["s"], 0):
        result, _, _ = execute(engine, [{**common, "s": s}], [population], neutral=s == 0)
        arms.extend(result)
    add_omega(arms)
    return arms


def summary(data):
    print("F2 " + ("QUICK smoke test; no equilibrium conclusions" if data["quick"] else "finite-run results"))
    print("Main comparisons use the same fixed output noise for both distributions; no recalibration.")
    print("Information in nats; I_max = best numerical representable code, not a certified global maximum.")
    environments = dict(data["environments"])
    environments["vanhateren low-noise sensitivity"] = data["low_noise_sensitivity"]["environment"]
    for name, env in environments.items():
        print(f"{name}: sigma={env['sigma_n']:g}, I(CDF)={env['cdf_information']:.7f}, "
              f"I(best representable)=Imax={env['I_max']:.7f}, "
              f"I(best unrestricted monotone)={env['direct_optimization']['information']:.7f}, "
              f"CDF/Imax={env['cdf_information'] / env['I_max']:.5f}, "
              f"KS(best,CDF)={env['best_cdf_ks']:.5f}, projected CDF KS={env['cdf_fit_ks']:.5f}")
        flat = env["flatness"]
        print(f"  Flatness I/Imax: linear={flat['linear_efficiency']:.6f}; "
              f"random monotone={np.round(flat['random_monotone_efficiencies'], 6).tolist()}; "
              f"random representable={np.round(flat['random_representable_efficiencies'], 6).tolist()}")
        print("  " + flat["note"])
        print("  " + env["optimization_note"])
        if "contrast_metadata" in env:
            meta = env["contrast_metadata"]
            print(f"  {len(meta['images'])} images, sigma={meta['sigma_pixels']}px, "
                  f"{meta['pixel_count']:,} pixels; sample={meta['sample']}")
    print("KS values average distances of replicate population-mean codes over the last quarter; ks/mean_ks refer to CDF.")
    print("distribution  L     N       s       u        Ns    efficiency +/- SD    KS(CDF) KS(best) tail change")
    for r in data["equilibrium"]:
        print(f"{r['distribution']:12s} {r['L']:2d} {r['N']:5d} {r['s']:7.2g} {r['u']:7.1g} "
              f"{r['Ns']:9.2g}  {r['mean']:.4f} +/- {r['replicate_sd']:.4f}  "
              f"{r['mean_ks']:.4f}  {r['mean_ks_best']:.4f} {r['tail_change']:+.4f}")
    sensitivity = data["low_noise_sensitivity"]
    print("Low-noise sensitivity (excluded from thresholds, collapse and main drift comparison):")
    print(sensitivity["note"])
    for r in sensitivity["equilibrium"]:
        print(f"  N={r['N']}, s={r['s']:g}, sigma={r['sigma_n']:g}: efficiency={r['mean']:.6f}, "
              f"KS(CDF)={r['mean_ks']:.6f}, KS(best)={r['mean_ks_best']:.6f}")
    if sensitivity["first_run_reference"] is not None:
        old = sensitivity["first_run_reference"]
        print(f"  Historical first full run ({old['source']}), sigma={old['sigma_n']:.8f}: "
              f"CDF/Imax={old['cdf_efficiency']:.6f}, N=100 s=1000 evolved efficiency="
              f"{old['equilibrium']['mean']:.6f}, KS(CDF)={old['equilibrium']['mean_ks']:.6f}.")
    print("At low noise, well-spaced monotone responses approach the input-entropy ceiling: high efficiency need not imply CDF agreement.")
    print("The first run's noise change confounds both its Laughlin comparison and its F-vs-F2 drift comparison; extra near-neutral code directions can permit drift underneath the code.")
    print("Thresholds (first sampled Ns exceeding target, not interpolated):")
    print(json.dumps(data["Ns_crit"]))
    print("Load collapse:", json.dumps(data["collapse"]))
    if "approximate_collapse" in data["collapse"]:
        c = data["collapse"]
        print(f"Best combination: {c['best']}; " + ("held-out RMSE <= 0.02." if c["approximate_collapse"]
              else "does not collapse within 0.02 held-out RMSE.") +
              f" Improvement over Ns alone: {c['improvement_over_Ns']:.5f} RMSE.")
    print("Depth forks: L, source s, arm s, t*, omega (genotype to phenotype)")
    for r in data["drift"]:
        print(r["L"], r["source_s"], r["s"], r["t_star"], r["omega"])
    depths = [r for r in data["drift"] if r["s"] == 1000 and r["omega"] is not None]
    if depths:
        depths.sort(key=lambda r: r["L"])
        print("Mean omega below top layer: " + "; ".join(f"L={r['L']}: {np.mean(r['omega'][:-1]):.5f}" for r in depths))
        if len(depths) == 3:
            increasing = np.all(np.diff([np.mean(r["omega"][:-1]) for r in depths]) > 0)
            print("Drift below the top code " + ("increases monotonically with depth." if increasing else "does not increase monotonically with depth."))
        print("Depth comparison is descriptive: one adapted population per cell, no independent fork-source error bars.")
    else:
        print("Depth dependence unresolved: neutral threshold not reached.")
    real = {r["s"]: r for r in data["equilibrium"] if "adaptation" in r["tags"] and r["N"] == 100}
    comparison = [r for r in data["equilibrium"] if "comparison" in r["tags"] and r["N"] == 100]
    for r in comparison:
        if r["s"] in real:
            print(f"N=100, s={r['s']:g}: real-minus-standin efficiency {real[r['s']]['mean'] - r['mean']:+.5f}")
    runtime = data["runtime_estimate"]
    print(f"CPU/GPU pilot: {runtime['individual_generations_per_second']:,.0f} individual-generations/s; "
          f"same-device full projection {runtime['same_device_hours']:.2f} h; "
          f"hypothetical GPU hours {runtime['hypothetical_gpu_hours']}. {runtime['caveat']}")
    print(f"Elapsed {data['elapsed_seconds']:.1f}s. Tail changes and replicate SD are diagnostics, not stationarity guarantees.")


def save(data, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    stream = StringIO()
    with redirect_stdout(stream):
        summary(data)
    text = stream.getvalue()
    stem.with_name(stem.name + "-summary.txt").write_text(text)
    print(text, end="")


def main(args, quick=False):
    start = perf_counter()
    envs = {name: channel(name, args.contrasts, args.seed, args.sigma_n) for name in ("standin", "vanhateren")}
    engines = {name: Engine(env, args.individual_chunk) for name, env in envs.items()}
    estimate = benchmark(engines["vanhateren"], args.seed, args.config_chunk)
    if not quick and (jax.default_backend() != "gpu" or estimate["budget_hours"] > args.max_hours):
        raise SystemExit("Full grid not started: requires GPU and pilot estimate <= --max-hours. Use quick locally; benchmark/tune chunks on GPU.")
    if not quick and envs["vanhateren"].params["contrast_metadata"]["sample"]:
        raise SystemExit("Full grid requires full-dataset contrasts, not the two-image sample.")
    rows = grid(args.seed, quick, args.generations if quick else None)
    adapted, drift = [], []
    from tqdm import tqdm
    for batch in tqdm(list(groups(rows, args.config_chunk)), desc="adaptation batches"):
        engine = engines[batch[0]["distribution"]]
        results, populations, seconds = execute(engine, batch)
        adapted.extend(results)
        print(f"N={batch[0]['N']}, L={batch[0]['L']}, {len(batch)} configs: "
              f"{sum(r['N'] * r['K'] * r['T'] for r in batch) / seconds:,.0f} individual-generations/s (includes compile)", flush=True)
        for row, population in zip(batch, populations):
            if "depth" in row["tags"]:
                drift.extend(forks(engine, row, population, quick))
    # Pair the sensitivity cell with the main N=100, s=1000, L=3 adaptation
    # (N=8 for quick), including ancestry and mutation/resampling random streams.
    sensitivity_row = next(r for r in rows if r["N"] == (8 if quick else 100)
                           and r["s"] == 1000 and "adaptation" in r["tags"])
    low_env = channel("vanhateren", args.contrasts, args.seed, SENSITIVITY_SIGMA_N)
    sensitivity_rows, _, _ = execute(Engine(low_env, args.individual_chunk),
                                     [{**sensitivity_row, "tags": ["low_noise_sensitivity"]}])
    historical = None
    historical_path = OUTPUT / "first-code2.json"
    if historical_path.exists():
        first = json.loads(historical_path.read_text())
        old_env = first["environments"]["vanhateren"]
        old_cell = next((r for r in first["equilibrium"] if r["distribution"] == "vanhateren"
                         and r["N"] == 100 and r["s"] == 1000 and "adaptation" in r["tags"]), None)
        if old_cell is not None:
            historical = dict(source=str(historical_path), sigma_n=old_env["sigma_n"],
                              cdf_efficiency=old_env["cdf_information"] / old_env["I_max"],
                              equilibrium=old_cell)
    sensitivity = dict(label="van Hateren low-noise sensitivity", environment=low_env.export(),
                       adaptation=sensitivity_rows, equilibrium=[r["equilibrium"] for r in sensitivity_rows],
                       first_run_reference=historical,
                       note="One paired L=3, s=1000 adaptation cell; sigma_n=0.0011 approximates the first run's 0.00109951. Same seed, N, K, T, u and gain as the main cell; no low-noise depth forks.")
    table = [r["equilibrium"] for r in adapted]
    data = dict(schema=2, quick=quick, environments={k: e.export() for k, e in envs.items()},
                low_noise_sensitivity=sensitivity,
                adaptation=adapted, equilibrium=table, drift=drift,
                Ns_crit=original.thresholds([r for r in table if "adaptation" in r["tags"]]),
                collapse=collapse([r for r in table if "load" in r["tags"]]),
                runtime_estimate=estimate, elapsed_seconds=perf_counter() - start,
                execution=dict(config_chunk=args.config_chunk, individual_chunk=args.individual_chunk,
                               preallocate=os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], precision="float32"))
    stem = args.output or OUTPUT / "code2"
    save(data, stem)
    figure(stem)


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(stem=OUTPUT / "code2"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    stem = Path(stem)
    data = json.loads(stem.with_suffix(".json").read_text())
    table = data["equilibrium"]
    env = data["environments"]["vanhateren"]
    c = np.array(env["contrasts"])

    def labels(ax, items):
        low, high = ax.get_ylim()
        gap = min(0.13, 0.85 / max(len(items), 1))
        placed = -gap
        for i, (x, y, label, color) in enumerate(sorted(items, key=lambda a: a[1])):
            pos = np.clip((y - low) / (high - low), 0.03, 0.97 - gap * (len(items) - i - 1))
            pos = max(pos, placed + gap)
            placed = pos
            ax.annotate(label, (x, y), xytext=(1.02, pos), textcoords="axes fraction", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.4), annotation_clip=False,
                        fontsize=5, family="monospace", va="center")

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "axes.titleweight": "normal", "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "lines.linewidth": 1}):
        fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
        fig.subplots_adjust(left=0.075, right=0.89, bottom=0.10, top=0.88, wspace=1.02, hspace=0.62)
        for ax, letter in zip(axes.flat, "ABCDEF"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0, 0]
        items = []
        for i, (p, name) in enumerate(((env.get("raw_probabilities", env["probabilities"]), "measured"),
                                       (data["environments"]["standin"]["probabilities"], "stand-in"))):
            density = np.array(p) / (c[1] - c[0])
            ax.plot(c, density, color=f"C{i}")
            anchor = int(np.argmax(density))
            items.append((c[anchor], density[anchor], name, f"C{i}"))
        ax.set(xlabel="Contrast", ylabel="Probability / grid spacing")
        labels(ax, items)
        ax = axes[0, 1]
        N = 8 if data["quick"] else 100
        rows = sorted((r for r in table if r["N"] == N and "adaptation" in r["tags"]), key=lambda r: r["s"])
        rows = [rows[i] for i in sorted({0, len(rows) // 2, len(rows) - 1})]
        ax.plot(c, env["f_star"], "--", color="C0")
        ax.plot(c, env["best_response"], ":", color="black")
        anchor = 39
        items = [(c[anchor], env["f_star"][anchor], "CDF", "C0"),
                 (c[anchor], env["best_response"][anchor], "best repr.", "black")]
        for i, r in enumerate(rows, 1):
            f, sd = np.array(r["mean_response"]), np.array(r["response_sd"])
            ax.plot(c, f, color=f"C{i}")
            ax.fill_between(c, np.maximum(0, f - sd), np.minimum(1, f + sd), color=f"C{i}", alpha=.12, lw=0)
            items.append((c[anchor], f[anchor], f"s={r['s']:g}", f"C{i}"))
        ax.set(xlabel="Contrast", ylabel="Response (mean ± SD)", ylim=(0, 1.03),
               title=f"N={N}; " + rf"$\sigma_n={env['sigma_n']:g}$")
        labels(ax, items)
        ax = axes[0, 2]
        items = []
        rows = [r for r in table if "adaptation" in r["tags"]]
        for i, n in enumerate(sorted({r["N"] for r in rows})):
            group = sorted((r for r in rows if r["N"] == n), key=lambda r: r["Ns"])
            x, y = [r["Ns"] for r in group], [r["mean"] for r in group]
            ax.plot(x, y, ".-", color=f"C{i}")
            items.append((x[-1], y[-1], f"N={n}", f"C{i}"))
        for i, cutoff in enumerate((.95, .99), 6):
            ax.axhline(cutoff, color=f"C{i}", ls=":", lw=.6)
            items.append((max(r["Ns"] for r in rows), cutoff, f"{cutoff:.0%}", f"C{i}"))
        ax.set(xscale="log", xlabel=r"$N\,s$", ylabel="Efficiency")
        labels(ax, items)
        ax = axes[1, 0]
        items = []
        for i, r in enumerate(r for r in data["drift"] if r["s"] == 1000):
            if r["omega"] is not None:
                ax.plot(range(r["L"] + 1), r["omega"], ".-", color=f"C{i}")
                items.append((r["L"], r["omega"][-1], f"L={r['L']}", f"C{i}"))
        if not items:
            ax.text(.05, .5, "Neutral threshold\nnot reached", transform=ax.transAxes, fontsize=6, family="monospace")
        ax.set(xlabel="Layer (genotype to phenotype)", ylabel=r"$\omega$", title="s=1000")
        labels(ax, items)
        ax = axes[1, 1]
        best = data["collapse"]["best"]
        rows = [r for r in table if "load" in r["tags"]]
        items = []
        # For the two-variable winner show fixed-Nu slices (exact matched slices).
        key = (lambda r: r["N"] * r["u"]) if best == "Ns,Nu" else (lambda r: r["N"])
        values = sorted({round(key(r), 9) for r in rows})
        if best == "Ns,Nu":
            values = sorted(values, key=lambda v: -sum(np.isclose(key(r), v) for r in rows))[:4]
        for i, value in enumerate(values):
            group = [r for r in rows if np.isclose(key(r), value)]
            transform = lambda r: r["s"] / r["u"] if best == "s/u" else r["Ns"] / r["u"] if best == "Ns/u" else r["Ns"]
            group = sorted(group, key=transform)
            x, y = [transform(r) for r in group], [r["mean"] for r in group]
            ax.plot(x, y, "." if best != "Ns,Nu" else ".-", color=f"C{i}", ms=3)
            items.append((x[-1], y[-1], f"{'Nu' if best == 'Ns,Nu' else 'N'}={value:g}", f"C{i}"))
        ax.set(xscale="log", xlabel="Ns (fixed Nu)" if best == "Ns,Nu" else best, ylabel="Efficiency", title="Load: held-out N prediction")
        labels(ax, items)
        ax = axes[1, 2]
        items = []
        all_y = []
        for i, distribution in enumerate(("standin", "vanhateren")):
            rows = sorted((r for r in table if r["N"] == N and r["distribution"] == distribution
                           and ("comparison" in r["tags"] or "adaptation" in r["tags"])), key=lambda r: r["Ns"])
            x, y = [r["Ns"] for r in rows], [r["mean"] for r in rows]
            all_y.extend(y)
            ax.plot(x, y, ".-", color=f"C{i}")
            items.append((x[-1], y[-1], "stand-in" if i == 0 else "measured", f"C{i}"))
        ax.set(xscale="log", xlabel=r"$N\,s$", ylabel="Efficiency",
               title=f"N={N}; " + rf"$\sigma_n={env['sigma_n']:g}$")
        # Reserve space below the main curves for the separately labelled arm.
        span = max(np.ptp(all_y), .04)
        ax.set_ylim(min(all_y) - .9 * span, max(all_y) + .15 * span)
        low = data["low_noise_sensitivity"]
        low_env, low_row = low["environment"], low["equilibrium"][0]
        ax.text(.02, .03, rf"Sensitivity: $\sigma_n={low_env['sigma_n']:g}$" + "\n" +
                f"CDF/Imax={low_env['cdf_information'] / low_env['I_max']:.3f}; "
                f"linear={low_env['flatness']['linear_efficiency']:.3f}\n"
                f"s=1000: eff.={low_row['mean']:.3f}\n"
                f"KS CDF/best={low_row['mean_ks']:.3f}/{low_row['mean_ks_best']:.3f}",
                transform=ax.transAxes, fontsize=4.8, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1))
        labels(ax, items)
        from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
        for ax in (axes[0, 2], axes[1, 1], axes[1, 2]):
            ax.xaxis.set_major_locator(LogLocator(numticks=3))
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle("F2 · " + ("CPU smoke test: tiny sizes; no equilibrium inference" if data["quick"] else "natural contrasts, depth and mutational load"),
                     fontsize=7, family="monospace", y=.985)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=200)
        plt.close(fig)


def check(seed=0, contrasts=OUTPUT / "contrasts.json", sigma_n=DEFAULT_SIGMA_N):
    env = channel(seed=seed, sigma_n=sigma_n)
    engine = Engine(env, 16)
    rng = np.random.default_rng(123)
    genomes = original.normalize(rng.normal(size=(2, 3, 5, 784)), 16).astype(np.float32)
    assert np.allclose(forward(jnp.array(genomes)), np.stack(original.forward(genomes, 16), axis=-2), atol=2e-6)
    f = np.asarray(engine.response(forward(jnp.array(genomes))[..., -1, :]))
    expected = env.response(original.forward(genomes, 16)[-1])
    assert np.allclose(f, expected, atol=3e-6)
    infos = np.asarray(jax.jit(engine.information)(jnp.array(f)))
    error = float(np.max(np.abs(infos - env.information(f))))
    assert error < 4e-6, error
    assert abs(float(engine.information_one(jnp.full(64, .5)))) < 3e-6
    measured = jax.device_get(engine.snapshot(jnp.array(genomes), jnp.array(infos)))
    reference = original.drift_measure(genomes[0], env, 16, infos[0])
    for name in ("state_divergence", "map_divergence", "within_variation", "response_divergence"):
        assert np.allclose(measured[name][0], reference[name], atol=3e-6), name
    # Wright-Fisher inverse-CDF selection with the identical uniform draws.
    clone_info = jnp.array(infos)
    wf_keys = random.split(random.PRNGKey(19), 6).reshape(2, 3, 2)
    (selected, _, _), _ = jax.jit(engine.step)((jnp.array(genomes), clone_info, wf_keys),
                                              (jnp.array([3., 100.]), jnp.zeros(2)))
    for ci, selection in enumerate((3, 100)):
        for ri in range(3):
            key = random.split(wf_keys[ci, ri], 4)[1]
            logw = selection * (infos[ci, ri] / env.I_max - 1)
            cdf = np.cumsum(np.exp(logw - logw.max()))
            draws = np.asarray(random.uniform(key, (5,))) * cdf[-1]
            parents = np.searchsorted(cdf, draws)
            assert np.array_equal(selected[ci, ri], genomes[ci, ri, parents])
    row = config(5, 100, distribution="standin", K=3, T=5, every=2, seed=seed)
    g, keys = initial([row])
    info = engine.evaluate(g)
    (unchanged, values, _), _ = jax.jit(engine.step)((g, info, keys), (jnp.array([100.]), jnp.array([0.])))
    assert np.array_equal(unchanged, g) and np.array_equal(values, info)  # clonal, no mutation
    # Per-site frequency, and exactly untouched genome blocks.
    mask = random.bernoulli(random.PRNGKey(1), .003, (3000, 784))
    assert abs(float(mask.mean()) - .003) < 5 * np.sqrt(.003 * .997 / mask.size)
    flat = jnp.array(genomes.reshape(-1, 784))
    masks = random.bernoulli(random.PRNGKey(2), .001, flat.shape)
    mutated = vmap(mutate_one, in_axes=(0, 0, 0, None))(flat, masks, jnp.ones(flat.shape), .1)
    for offset, width in ((0, 16), (16, 256), (272, 256), (528, 256)):
        untouched = ~np.asarray(masks[:, offset:offset + width]).any(-1)
        assert np.array_equal(np.asarray(mutated)[untouched, offset:offset + width], np.asarray(flat)[untouched, offset:offset + width])
    # One-step cache, chunk invariance, scan final remainder and matched fork starts.
    (next_g, values, _), _ = jax.jit(engine.step)((g, info, keys), (jnp.array([100.]), jnp.array([.001])))
    assert np.allclose(values, engine.evaluate(next_g), atol=4e-6)
    rows, populations, _ = execute(engine, [row])
    assert [r["t"] for r in rows[0]["records"]] == [0, 2, 4, 5]
    arms = forks(engine, row, populations[0], quick=True)
    assert arms[0]["records"][0] == arms[1]["records"][0]
    assert max(arms[0]["records"][0]["state_divergence"]) < 1e-6
    assert arms[0]["records"][0]["response_divergence"] < 1e-12
    other = {**row, "s": 3.}
    batch, _, _ = execute(engine, [row, other])
    assert np.allclose(batch[0]["equilibrium"]["replicate_means"], rows[0]["equilibrium"]["replicate_means"], atol=5e-6)
    full = grid()
    assert len([r for r in full if "depth" in r["tags"]]) == 6
    assert len([r for r in full if "adaptation" in r["tags"]]) == 38
    assert len([r for r in full if "load" in r["tags"]]) == 40
    # Guard the confound directly, using the requested measured distribution at
    # both noises, plus F's stand-in. Compare the sparse JAX MI to dense float64.
    environments = [env] + [channel("vanhateren", contrasts, seed, sigma)
                            for sigma in dict.fromkeys((sigma_n, SENSITIVITY_SIGMA_N))]
    for reference_env in environments:
        assert reference_env.sigma_n == reference_env.requested_sigma_n
        assert reference_env.calibration["noise_policy"] == "fixed; no recalibration"
        best = np.array(reference_env.calibration["best_response"])
        assert np.isclose(reference_env.I_max, reference_env.information(best), atol=1e-10)
        assert reference_env.calibration["direct_optimization"]["information"] >= reference_env.I_max - 1e-7
        reference_engine = Engine(reference_env, 16)
        probes = np.vstack((reference_env.star, best, np.linspace(0, 1, 64),
                            reference_env.response(rng.normal(size=(5, 16)))))
        mi_error = np.max(np.abs(np.asarray(jax.jit(reference_engine.information)(jnp.array(probes)))
                                 - reference_env.information(probes)))
        assert mi_error < 5e-6, mi_error
        snaps = jax.device_get(reference_engine.snapshot(jnp.array(genomes), jnp.array(infos)))
        responses = reference_env.response(original.forward(genomes, 16)[-1]).mean(axis=2)
        assert np.allclose(snaps["ks"], np.max(np.abs(responses - reference_env.star), axis=-1), atol=3e-6)
        assert np.allclose(snaps["ks_best"], np.max(np.abs(responses - best), axis=-1), atol=3e-6)
        print(f"  fixed-noise check {reference_env.params['distribution']} sigma={reference_env.sigma_n:g}: "
              f"MI error={mi_error:.3g}, both KS references verified", flush=True)
    assert rows[0]["sigma_n"] == sigma_n
    tail = [r for r in rows[0]["records"] if r["t"] >= .75 * row["T"]]
    assert np.isclose(rows[0]["equilibrium"]["mean_ks_best"], np.mean([r["ks_best"] for r in tail]))
    print(f"check ok: MI max absolute error={error:.3g} nats; development, sparse-site mutation, cache, scans, fork pairing, batching, grid")


def reference_table():
    text = (OUTPUT / "code-summary.txt").read_text()
    pattern = r"^\s*(\d+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+\+/-\s+([\d.]+)"
    return {(int(n), float(s)): (float(mean), float(sd)) for n, s, mean, sd in re.findall(pattern, text, flags=re.M)}


def validate(args):
    env = channel(seed=args.seed, sigma_n=args.sigma_n)
    engine = Engine(env, args.individual_chunk)
    T = 20000 if args.full else (args.generations or 12)
    K = 8 if args.full else 3
    cells = [(int(cell.split(":")[0]), float(cell.split(":")[1])) for cell in args.cells.split(",")]
    refs = reference_table()
    if not cells or any(cell not in refs for cell in cells):
        raise ValueError("Validation cells must occur in output/code-summary.txt")
    results, drift = [], []
    for N, s in cells:
        row = config(N, s, distribution="standin", K=K, T=T, every=200 if args.full else max(1, T // 4), seed=args.seed)
        result, populations, _ = execute(engine, [row])
        eq = result[0]["equilibrium"]
        mean, sd = refs[N, s]
        tolerance = float(3 * np.sqrt(sd**2 / 8 + eq["replicate_sd"]**2 / K))
        result[0]["saved_comparison"] = dict(mean=mean, replicate_sd=sd, difference=eq["mean"] - mean,
                                               three_standard_errors=tolerance,
                                               within_replicate_noise=bool(abs(eq["mean"] - mean) <= tolerance) if args.full else None)
        # A short NumPy run checks transients, not the saved equilibrium.
        if not args.full:
            numpy_row, _ = original.adapt(dict(row, T_adapt=T), env)
            result[0]["short_numpy_mean"] = numpy_row["equilibrium"]["mean"]
            print(f"  NumPy at the same short duration (independent ancestry): {result[0]['short_numpy_mean']:.5f}", flush=True)
        results.extend(result)
        print(f"N={N}, s={s}: JAX {eq['mean']:.5f} +/- {eq['replicate_sd']:.5f}; "
              f"saved F {mean:.5f} +/- {sd:.5f}; "
              f"{'within 3 SE: ' + str(abs(eq['mean'] - mean) <= tolerance) if args.full else 'SHORT: equilibrium comparison deferred'}", flush=True)
        if (N, s) == (100, 1000):
            drift.extend(forks(engine, row, populations[0], quick=not args.full))
    omega = next((r["omega"] for r in drift if r["s"] == 1000), None)
    terminal_ratios = None
    if drift:
        selected, neutral = drift[:2]
        a = np.array(selected["records"][-1]["state_divergence"])
        b = np.array(neutral["records"][-1]["state_divergence"])
        terminal_ratios = [float(x / y) if y > 0 else None for x, y in zip(a, b)]
    payload = dict(full=args.full, generations=T, K=K, environment=env.export(), results=results, drift=drift,
                   saved_fork_omega=[.001, .006, .004, .020],
                   terminal_divergence_ratios=terminal_ratios,
                   qualitative_fork_suppression=None if omega is None else all(v is not None and v < .1 for v in omega),
                   note="Full validation compares independent replicate streams; short mode exercises code only. F saved selected omega is strongly suppressed (<0.1 at each layer).")
    stem = args.output or OUTPUT / ("code2-validation" if args.full else "code2-validation-quick")
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    Path(stem).with_suffix(".json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    report = "\n".join(f"N={r['N']}, s={r['s']}: JAX={r['equilibrium']['mean']:.6f}; "
                       f"short NumPy={r.get('short_numpy_mean')}; {r['saved_comparison']}" for r in results)
    report += f"\nJAX fork omega: {omega}; saved F: {payload['saved_fork_omega']}\n"
    report += f"Terminal selected/neutral divergence (not omega): {terminal_ratios}\n{payload['note']}\n"
    Path(stem).with_name(Path(stem).name + "-summary.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "validate", "quick", "benchmark", "run", "figure"), nargs="?", default="quick")
    parser.add_argument("--contrasts", type=Path, default=OUTPUT / "contrasts.json")
    parser.add_argument("--distribution", choices=("standin", "vanhateren"), default="vanhateren", help="Environment for benchmark; run covers both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-n", type=float, default=DEFAULT_SIGMA_N,
                        help="Fixed output noise shared by both main distributions (default: F's 0.0128); sensitivity stays at 0.0011")
    parser.add_argument("--config-chunk", type=int, default=2)
    parser.add_argument("--individual-chunk", type=int, default=128)
    parser.add_argument("--generations", type=int, help="Override quick/short-validation time only")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--full", action="store_true", help="Full validation at F's original sizes")
    parser.add_argument("--cells", default="10:3,10:30,100:1000", help="Validation cells as comma-separated N:s pairs")
    parser.add_argument("--output", type=Path, help="Output stem, without extension")
    args = parser.parse_args()
    if not np.isfinite(args.sigma_n) or args.sigma_n <= 0:
        parser.error("sigma-n must be finite and positive")
    if min(args.config_chunk, args.individual_chunk, args.max_hours) <= 0 or (args.generations is not None and args.generations < 1):
        parser.error("chunks, max-hours and generations must be positive")
    if args.command == "check":
        check(args.seed, args.contrasts, args.sigma_n)
    elif args.command == "validate":
        validate(args)
    elif args.command == "figure":
        figure(args.output or OUTPUT / "code2")
    elif args.command == "benchmark":
        benchmark(Engine(channel(args.distribution, args.contrasts, args.seed, args.sigma_n), args.individual_chunk), args.seed, args.config_chunk)
    else:
        main(args, quick=args.command == "quick")
