"""Experiment G: neuroevolution of homologous neurons in Brax swimmer.

Run (offline): uv run --offline --with 'jax==0.7.*' --with brax --with optax
  --with numpy --with scipy --with matplotlib --with tqdm python neuroevo.py
  [check | quick | run | combine | figure] [--input output/neuroevo-quick.json]
Quick writes only neuroevo-quick.*; run writes neuroevo.*. No gradient learning.
Combine accepts --input FILE [FILE ...], defaults to the three seed 0/1 runs,
and writes neuroevo-combined.json and neuroevo-combined-summary.txt.

Generalized is the fastest supported swimmer backend: the cached Brax rejects
positional. Use the standard four physics substeps, float32, and one fixed-key
reset/rollout per agent. JIT/vmap keep evolution and physics on the accelerator.
The linear development, block norms, haploid Wright--Fisher reproduction,
independent Bernoulli site mutations, consensus and population fork mirror drift.
x0 has norm 1; maps have Frobenius norm sqrt(d). R ~ N(0,1) is fixed per seed;
theta = gain * R @ unit(z), ordered W1,b1,W2,b2 (8*16+16+16*2+2 = 178).
Gain starts at .5 and is halved until the random ancestor's entire evaluation
rollout has max |action| <= .8. It is then frozen, including during adaptation.

Selection uses log w = s*(return - within-population mean)/fixed_return_scale.
Sampling uses softmax (no clipped selection). Adaptation's scale is measured
once in a one-generation mutant founder panel; phase 2's is the final adapted
population SD (floor 1e-6). Each s burns in from the adapted consensus, then
selected and neutral arms copy exactly the same polymorphic population into K
lineages, with matched random streams. Default burn-in B=2000 follows drift.py.
Adaptation runs a fixed budget and reports, rather than assumes, a plateau.

State divergences follow drift's mean individual directions; maps and theta
use directions of lineage means. Representation is 1-linear-CKA on one fixed
ancestor probe set (2000 observations, deterministic distinct reset keys).
Return divergence is mean |a-b|/(|a|+|b|+fixed_return_scale), avoiding the
degeneracy of scalar cosine distance. Gait is cosine distance of concatenated
joint-angle power at the first eight non-DC Fourier bins (power / steps**2).
Omega for every level uses the first neutral x0 divergence >= .25, otherwise
null, exactly as drift.add_omega. Fitness overflow is null, with log mean saved.

Homology uses consensus policies, unpermuted unit indices, signed strongest
Pearson tuning to q1,q2,v1,v2, and raw return drops after single-unit ablation.
Ancestor comparisons use the adapted consensus; fork comparisons are also
saved to separate burn-in from post-fork change. A Si1-like candidate must lose
>=10% of a fixed return reference on ablation in one lineage, <=1% in another,
with both intact returns within 10% of the fork return. Otherwise report no
qualifying example and show the largest contrast without calling it essential.
Hybrids assay every distinct consensus-parent pair and all 14 non-parental
choices of the four whole genome blocks, including reciprocal crosses.
These are analogies of functional reassignment, not claims of biological identity.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import faulthandler
import itertools
import json
import signal
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from tqdm import tqdm

OUTPUT = Path(__file__).resolve().parent / "output"
D, L, P, H = 32, 3, 178, 16
LEVELS = ["x0", "x1", "x2", "z", "M1", "M2", "M3", "theta", "representation", "return", "gait"]
DISPLAY = ["x0", "M1", "M2", "M3", "z", "theta", "representation", "return", "gait"]


def unit(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def normalize(g):
    maps = g[..., D:].reshape(*g.shape[:-1], L, D * D)
    return jnp.concatenate((unit(g[..., :D]), (unit(maps) * np.sqrt(D)).reshape(*g.shape[:-1], -1)), -1)


def forward(g):
    x, states = g[..., :D], [g[..., :D]]
    maps = g[..., D:].reshape(*g.shape[:-1], L, D, D)
    for l in range(L):
        x = jnp.einsum("...ij,...j->...i", maps[..., l, :, :], x)
        states.append(x)
    return jnp.stack(states, axis=-2)


def parameters(g, readout, gain):
    return gain * (unit(forward(g)[..., -1, :]) @ readout.T)


def consensus(g):
    return normalize(g.mean(axis=-2))


def policy(theta, obs, mask):
    hidden = jnp.tanh(obs @ theta[:128].reshape(8, H) + theta[128:144]) * mask
    action = jnp.tanh(hidden @ theta[144:176].reshape(H, 2) + theta[176:])
    return action, hidden


def mutate(g, key, u, sigma):
    sites, noise = jax.random.split(key)
    mask = jax.random.bernoulli(sites, u, g.shape)
    changed = g + mask * (sigma * jax.random.normal(noise, g.shape))
    normed = normalize(changed)
    # Untouched blocks are bitwise unchanged, as in drift.mutate.
    touched = jnp.concatenate((jnp.repeat(mask[..., :D].any(-1, keepdims=True), D, -1),
                              jnp.repeat(mask[..., D:].reshape(*g.shape[:-1], L, D * D).any(-1), D * D, -1)), -1)
    return jnp.where(touched, normed, g)


def probabilities(returns, s, scale):
    return jax.nn.softmax(s * (returns - returns.mean(-1, keepdims=True)) / scale, axis=-1)


def sample_parents(returns, key, s, scale):
    cdf = jnp.cumsum(probabilities(returns, s, scale), axis=-1).at[..., -1].set(1.)
    draws = jax.random.uniform(key, returns.shape)
    return jax.vmap(lambda c, r: jnp.searchsorted(c, r, side="right"))(cdf, draws)


class Swimmer:
    """Separate lean fitness and history kernels; padded assays reuse compilation."""

    def __init__(self, steps=300, assay_batch=32):
        from brax import envs
        self.env = envs.get_environment("swimmer", backend="generalized")
        self.steps, self.assay_batch = steps, assay_batch
        assert self.env.observation_size == 8 and self.env.action_size == 2
        self.reset_key = jax.random.PRNGKey(0)

        def rollout(theta, mask, key, trace=False):
            initial = self.env.reset(key)

            def body(carry, _):
                state, total = carry
                action, hidden = policy(theta, state.obs, mask)
                # Brax updates metrics in place: keep the reset object unmodified.
                state = state.replace(metrics=dict(state.metrics))
                next_state = self.env.step(state, action)
                history = (state.obs, hidden, action) if trace else None
                return (next_state, total + next_state.reward), history

            (_, total), history = jax.lax.scan(body, (initial, jnp.float32(0)), None, length=steps)
            return (total, history) if trace else total

        self.single = jax.jit(lambda theta, mask: rollout(theta, mask, self.reset_key))
        self.batch = jax.jit(jax.vmap(lambda theta, mask: rollout(theta, mask, self.reset_key)))
        self.traces = jax.jit(jax.vmap(lambda theta, mask, key: rollout(theta, mask, key, True)))

    def evaluate(self, theta, masks=None):
        masks = jnp.ones((len(theta), H)) if masks is None else masks
        return self.batch(theta, masks)

    def assay(self, theta, masks=None, trace=False, keys=None):
        theta = jnp.asarray(theta)
        masks = jnp.ones((len(theta), H)) if masks is None else jnp.asarray(masks)
        keys = jnp.broadcast_to(self.reset_key, (len(theta), 2)) if keys is None else keys
        chunks = []
        for start in range(0, len(theta), self.assay_batch):
            stop = min(start + self.assay_batch, len(theta))
            ids = np.minimum(np.arange(start, start + self.assay_batch), stop - 1)
            result = (self.traces(theta[ids], masks[ids], keys[ids]) if trace
                      else self.evaluate(theta[ids], masks[ids]))
            chunks.append(jax.tree.map(lambda x: np.asarray(x)[:stop - start], result))
        result = jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(result)):
            raise FloatingPointError("Non-finite swimmer assay; no invalid returns are silently selected.")
        return result


def make_step(swimmer, readout, gain, u, sigma):
    @jax.jit
    def step(g, returns, key, s, scale):
        key, reproduction, mutation = jax.random.split(key, 3)
        parents = sample_parents(returns, reproduction, s, scale)
        offspring = mutate(g[jnp.arange(len(g))[:, None], parents], mutation, u, sigma)
        theta = parameters(offspring, readout, gain).reshape(-1, P)
        returns = swimmer.evaluate(theta).reshape(g.shape[:2])
        return offspring, returns, key
    return step


def divergence(x):
    """Mean cosine distance of distinct pairs; zero vectors have no direction."""
    x = np.asarray(x, dtype=float)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)
    return float(np.clip(np.mean([1 - np.sum(x[i] * x[j])
                                 for i in range(len(x)) for j in range(i)]), 0, 2))


def linear_cka(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    # Identical constant representations share the same zero centered Gram matrix.
    if denominator < 1e-30:
        return float(np.array_equal(a, b))
    return float(np.clip(np.sum((a.T @ b)**2) / denominator, 0, 1))


def fitness_stats(returns, s, scale):
    logw = s * (returns - returns.mean(axis=-1, keepdims=True)) / scale
    logmean = float(logsumexp(logw) - np.log(logw.size))
    return dict(mean_fitness=float(np.exp(logmean)) if logmean < 709 else None,
                log_mean_fitness=logmean, mean_return=float(returns.mean()),
                return_by_lineage=returns.mean(axis=-1).tolist())


def measure(g, returns, swimmer, readout, gain, probes, s, scale, t):
    states = np.asarray(unit(forward(g))).mean(axis=1)
    maps = np.asarray(g[..., D:]).reshape(*g.shape[:2], L, D * D).mean(axis=1)
    theta = parameters(g, readout, gain)
    cons = parameters(consensus(g), readout, gain)
    ret, (obs, _, _) = swimmer.assay(cons, trace=True)
    activations = np.asarray(jax.vmap(lambda p: policy(p, probes, jnp.ones(H))[1])(cons))
    K = len(g)
    cka = np.array([[linear_cka(a, b) for b in activations] for a in activations])
    angles = obs[:, :, 1:3]
    power = abs(np.fft.rfft(angles - angles.mean(axis=1, keepdims=True), axis=1))**2 / swimmer.steps**2
    gait = power[:, 1:9].reshape(K, -1)
    means = np.asarray(returns, float).mean(axis=1)
    div = dict(zip(("x0", "x1", "x2", "z"), [divergence(states[:, l]) for l in range(L + 1)]))
    div.update({f"M{l + 1}": divergence(maps[:, l]) for l in range(L)})
    div.update(theta=divergence(np.asarray(theta).mean(axis=1)),
               representation=float(np.mean(1 - cka[np.triu_indices(K, 1)])),
               gait=divergence(gait),
               **{"return": float(np.mean([abs(means[i] - means[j]) /
                    (abs(means[i]) + abs(means[j]) + scale) for i in range(K) for j in range(i)]))})
    return dict(t=t, divergence=div, cka=cka.tolist(), consensus_return=ret.tolist(),
                gait_signature=gait.tolist(), **fitness_stats(np.asarray(returns, float), s, scale))


def rank_correlation(a, b):
    a, b = rankdata(a), rankdata(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def roles(theta, swimmer):
    ret, (obs, hidden, _) = swimmer.assay(theta, trace=True)
    masks = np.tile(1 - np.eye(H, dtype=np.float32), (len(theta), 1))
    ablated = swimmer.assay(jnp.repeat(theta, H, axis=0), masks).reshape(-1, H)
    hidden = hidden.astype(float) - hidden.mean(axis=1, keepdims=True)
    variables = obs[:, :, [1, 2, 6, 7]].astype(float)
    variables -= variables.mean(axis=1, keepdims=True)
    numerator = np.einsum("kti,ktj->kij", hidden, variables)
    denominator = np.sqrt(np.sum(hidden**2, axis=1)[:, :, None] * np.sum(variables**2, axis=1)[:, None, :])
    corr = numerator / np.maximum(denominator, 1e-30)
    best = np.argmax(abs(corr), axis=-1)
    signed = np.take_along_axis(corr, best[..., None], axis=-1)[..., 0]
    role = np.where(abs(signed) > 1e-8, 2 * best + (signed < 0), -1)
    return dict(returns=ret.tolist(), importance=(ret[:, None] - ablated).tolist(),
                tuning_role=role.tolist(), tuning_correlation=corr.tolist())


def homology(g, swimmer, readout, gain, ancestor_roles, fork_roles, scale):
    result = roles(parameters(consensus(g), readout, gain), swimmer)
    role, importance = np.array(result["tuning_role"]), np.array(result["importance"])
    reference = np.array(ancestor_roles["importance"])[0]
    result["role_change_fraction"] = (role != np.array(ancestor_roles["tuning_role"])[0]).mean(axis=1).tolist()
    result["role_change_since_fork"] = (role != np.array(fork_roles["tuning_role"])[0]).mean(axis=1).tolist()
    result["importance_rank_vs_ancestor"] = [rank_correlation(a, reference) for a in importance]
    result["importance_rank_between_lineages"] = [[rank_correlation(a, b) for b in importance] for a in importance]
    fork_return = fork_roles["returns"][0]
    reference_scale = max(abs(fork_return), scale, 1e-6)
    maintained = abs(np.array(result["returns"]) - fork_return) <= .1 * reference_scale
    result["example"] = example_unit(importance, maintained, reference_scale)
    result["return_maintained"] = maintained.tolist()
    return result


def example_unit(importance, maintained, reference_scale):
    # Search every pair: the minimum raw importance may be strongly negative,
    # while another lineage has the genuinely dispensable (near-zero) unit.
    contrast = importance[:, None, :] - importance[None, :, :]
    distinct = ~np.eye(len(importance), dtype=bool)[:, :, None]
    qualifies = ((importance[:, None, :] >= .1 * reference_scale) &
                 (abs(importance[None, :, :]) <= .01 * reference_scale) &
                 maintained[:, None, None] & maintained[None, :, None] & distinct)
    eligible = qualifies if qualifies.any() else distinct
    high, low, i = np.unravel_index(np.argmax(np.where(eligible, contrast, -np.inf)), contrast.shape)
    return dict(unit=int(i), essential_lineage=int(high), dispensable_lineage=int(low),
                return_drop_high=float(importance[high, i]), return_drop_low=float(importance[low, i]),
                reference_scale=reference_scale, qualifies=bool(qualifies[high, low, i]))


def hybrids(g, swimmer, readout, gain):
    genomes = np.asarray(consensus(g))
    pairs = np.array(list(itertools.combinations(range(len(g)), 2)))
    masks = np.array(list(itertools.product((0, 1), repeat=L + 1)))[1:-1]
    sites = np.repeat(np.arange(L + 1), [D] + [D * D] * L)
    a, b = genomes[pairs[:, 0]], genomes[pairs[:, 1]]
    crosses = np.where(masks[None, :, sites], a[:, None], b[:, None])
    returns = swimmer.assay(parameters(jnp.asarray(crosses.reshape(-1, genomes.shape[-1])), readout, gain)).reshape(len(pairs), -1)
    parent = swimmer.assay(parameters(jnp.asarray(genomes), readout, gain))
    midparent = parent[pairs].mean(axis=1)
    return dict(pairs=pairs.tolist(), block_masks=masks.tolist(), parent_returns=parent.tolist(),
                hybrid_returns=returns.tolist(), mean_parent_return=float(midparent.mean()),
                mean_hybrid_return=float(returns.mean()),
                hybrid_minus_midparent=(returns.mean(axis=1) - midparent).tolist())


def add_omega(rows):
    for row in rows:
        neutral = next(r for r in rows if r["pair_s"] == row["pair_s"] and r["s"] == 0)
        hit = next((i for i, r in enumerate(neutral["records"]) if r["divergence"]["x0"] >= .25), None)
        row["t_star"], row["omega"] = None, None
        if hit is not None:
            row["t_star"] = neutral["records"][hit]["t"]
            a, b = row["records"][hit]["divergence"], neutral["records"][hit]["divergence"]
            row["omega"] = {k: a[k] / b[k] if b[k] > 0 else None for k in LEVELS}


def evolve(g, returns, key, s, scale, generations, step, desc, callback=None, every=1):
    start = perf_counter()
    if callback is not None:
        callback(0, g, returns)
    for t in tqdm(range(1, generations + 1), desc=desc):
        g, returns, key = step(g, returns, key, jnp.float32(s), jnp.float32(scale))
        if t % every == 0 or t == generations:
            jax.block_until_ready(returns)
            if not np.isfinite(np.asarray(returns)).all():
                raise FloatingPointError(f"Invalid population return at {desc}, generation {t}")
            if callback is not None:
                callback(t, g, returns)
    jax.block_until_ready(returns)
    return g, returns, key, perf_counter() - start


def run(N=64, K=16, steps=300, T_adapt=500, N_adapt=256, B=2000, T=2000,
        every=20, strengths=(3., 30.), s_adapt=10., u=1e-3, sigma=.1, seed=0,
        probe_count=2000, quick=False, output=None):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if min(N, N_adapt, every, probe_count) < 1 or K < 2 or steps < 4 or min(B, T, T_adapt) < 0:
        raise ValueError("Require N,N_adapt,every,probes >=1, K>=2, steps>=4, B,T,T_adapt>=0")
    if not 0 <= u <= 1 or sigma < 0 or s_adapt < 0 or not strengths or min(strengths) <= 0:
        raise ValueError("Require 0<=u<=1, sigma,s_adapt>=0 and positive phase-2 strengths")
    if not np.isfinite([*strengths, s_adapt, u, sigma]).all() or len(set(strengths)) != len(strengths):
        raise ValueError("Require finite parameters and distinct selection strengths")
    path = Path(output) if output else OUTPUT / ("neuroevo-quick.json" if quick else "neuroevo.json")
    if quick and path.stem in ("neuroevo", "neuroevo-summary"):
        raise ValueError("Quick output must not use the final experiment names")
    partial_path = path.with_name(path.stem + ".partial.json")
    start = perf_counter()
    params = dict(N=N, K=K, steps=steps, T_adapt=T_adapt, N_adapt=N_adapt, B=B, T=T,
                  every=every, strengths=list(strengths), s_adapt=s_adapt, u=u, sigma=sigma,
                  seed=seed, probe_count=probe_count, d=D, L=L, P=P, backend="generalized",
                  physics_substeps=4, reset_key=0, device=str(jax.devices()[0]), jax_version=jax.__version__,
                  brax_version=version("brax"), dtype="float32")
    swimmer = Swimmer(steps)
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    ancestor = normalize(jax.random.normal(keys[0], (D + L * D * D,)))
    readout = jax.random.normal(keys[1], (P, D))
    gain = .5
    for _ in range(20):
        _, (_, _, actions) = swimmer.assay(parameters(ancestor[None], readout, gain), trace=True)
        if abs(actions).max() <= .8:
            break
        gain /= 2
    else:
        raise RuntimeError("Unable to calibrate nonsaturated ancestor actions")
    params.update(gain=gain, initial_action_rms=float(np.sqrt(np.mean(actions**2))),
                  initial_action_max=float(abs(actions).max()))
    step = make_step(swimmer, readout, gain, u, sigma)
    g = jnp.broadcast_to(ancestor, (1, N_adapt, ancestor.size))
    panel = mutate(g, keys[2], u, sigma)
    panel_returns = np.asarray(swimmer.evaluate(parameters(panel, readout, gain).reshape(-1, P)))
    adapt_scale = max(float(panel_returns.std()), 1e-6)
    ret = swimmer.evaluate(parameters(g, readout, gain).reshape(-1, P)).reshape(1, N_adapt)
    compile_start = perf_counter()
    jax.block_until_ready(step(g, ret, keys[3], jnp.float32(s_adapt), jnp.float32(adapt_scale)))
    compile_seconds = perf_counter() - compile_start
    curve = []

    def learning(t, genomes, returns):
        values = np.asarray(returns)
        curve.append(dict(t=t, mean_return=float(values.mean()), sd_return=float(values.std()),
                          best_return=float(values.max()), **{k: v for k, v in fitness_stats(values.astype(float), s_adapt, adapt_scale).items()
                                                           if k in ("mean_fitness", "log_mean_fitness")}))

    g, ret, _, adaptation_seconds = evolve(g, ret, keys[3], s_adapt, adapt_scale, T_adapt, step, "adapt", learning)
    adapted = consensus(g)[0]
    scale = max(float(np.asarray(ret).std()), 1e-6)
    params.update(adaptation_ret_scale=adapt_scale, ret_scale=scale)
    # A plateau is descriptive, never an automatic scientific conclusion.
    window = min(50, len(curve) // 2)
    improvement = (float(np.mean([r["mean_return"] for r in curve[-window:]]) -
                         np.mean([r["mean_return"] for r in curve[-2 * window:-window]])) if window else None)
    plateau = dict(window=window, recent_mean_change=improvement,
                   detected=bool(window >= 20 and abs(improvement) < .01 * max(abs(curve[-1]["mean_return"]), scale)))
    n_probe_episodes = (probe_count + steps - 1) // steps
    probe_keys = jax.random.split(keys[4], n_probe_episodes)
    ancestor_theta = parameters(adapted[None], readout, gain)
    _, (probe_obs, _, _) = swimmer.assay(jnp.repeat(ancestor_theta, n_probe_episodes, axis=0), trace=True, keys=probe_keys)
    probes = jnp.asarray(probe_obs.reshape(-1, 8)[:probe_count])
    ancestor_roles = roles(ancestor_theta, swimmer)
    rows, burn_seconds, drift_seconds = [], 0., 0.
    # Warm both population shapes, then measure a synchronized, already compiled batch.
    benchmark_theta = jnp.repeat(ancestor_theta, K * N, axis=0)
    jax.block_until_ready(swimmer.evaluate(benchmark_theta))
    bench_start = perf_counter()
    for _ in range(3):
        jax.block_until_ready(swimmer.evaluate(benchmark_theta))
    bench_seconds = perf_counter() - bench_start
    throughput = dict(agent_steps_per_second=3 * K * N * steps / bench_seconds,
                      benchmark_agents=K * N, benchmark_repeats=3, benchmark_seconds=bench_seconds,
                      device=params["device"])
    for s in strengths:
        base = jnp.broadcast_to(adapted, (1, N, ancestor.size))
        base_ret = swimmer.evaluate(parameters(base, readout, gain).reshape(-1, P)).reshape(1, N)
        burn_key = keys[5]  # same burn random stream across s, as in drift
        compile_start = perf_counter()
        jax.block_until_ready(step(base, base_ret, burn_key, jnp.float32(s), jnp.float32(scale)))
        compile_seconds += perf_counter() - compile_start
        base, base_ret, _, seconds = evolve(base, base_ret, burn_key, s, scale, B, step, f"burn s={s:g}", every=every)
        burn_seconds += seconds
        fork = consensus(base)[0]
        fork_roles = roles(parameters(fork[None], readout, gain), swimmer)
        fork_g, fork_ret = jnp.repeat(base, K, axis=0), jnp.repeat(base_ret, K, axis=0)
        compile_start = perf_counter()
        jax.block_until_ready(step(fork_g, fork_ret, keys[6], jnp.float32(s), jnp.float32(scale)))
        compile_seconds += perf_counter() - compile_start
        for arm in (float(s), 0.):
            records = []

            def record(t, genomes, returns):
                records.append(measure(genomes, returns, swimmer, readout, gain, probes, arm, scale, t))

            final_g, _, _, seconds = evolve(fork_g, fork_ret, keys[6], arm, scale, T, step,
                                            f"drift s={arm:g} (fork {s:g})", record, every)
            drift_seconds += seconds
            # Genomes first: a hang in the post-drift assays then costs only the assays.
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path.with_name(f"{path.stem}.final_g-fork{s:g}-s{arm:g}.npy"), np.asarray(final_g))
            row = dict(pair_s=float(s), s=arm, records=records, seconds=seconds,
                       fork_genome=np.asarray(fork).tolist(), fork_homology=fork_roles)
            print(f"[post] fork s={s:g} arm s={arm:g}: homology", flush=True)
            row["homology"] = homology(final_g, swimmer, readout, gain, ancestor_roles, fork_roles, scale)
            print(f"[post] fork s={s:g} arm s={arm:g}: hybrids", flush=True)
            row["hybrids"] = hybrids(final_g, swimmer, readout, gain)
            rows.append(row)
            partial = dict(quick=quick, params=params, learning_curve=curve, plateau=plateau,
                           ancestor_genome=np.asarray(adapted).tolist(), ancestor_homology=ancestor_roles,
                           rows=rows, throughput=dict(throughput, adaptation_seconds=adaptation_seconds,
                               burn_seconds=burn_seconds, drift_with_records_seconds=drift_seconds,
                               step_warmup_seconds=compile_seconds), elapsed_seconds=perf_counter() - start)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(partial, separators=(",", ":"), allow_nan=False) + "\n")
            temporary.replace(partial_path)
    add_omega(rows)
    # Illustrative rates, not a B300 measurement. Physics kernels dominate runtime.
    full_steps = (501 * 256 + 2 * 2000 * 64 + 4 * 2000 * 16 * 64) * 300
    throughput.update(adaptation_seconds=adaptation_seconds, burn_seconds=burn_seconds,
                      drift_with_records_seconds=drift_seconds, step_warmup_seconds=compile_seconds,
                      default_full_evolution_agent_steps=full_steps,
                      default_full_cpu_hours=full_steps / throughput["agent_steps_per_second"] / 3600,
                      gpu_scenarios=[dict(agent_steps_per_second=rate, evolution_hours=full_steps / rate / 3600)
                                     for rate in (1e5, 1e6)],
                      timing_note="Warmed fitness throughput; GPU rates are assumptions. Assays, JIT and mutation add overhead.")
    data = dict(quick=quick, params=params, learning_curve=curve, plateau=plateau,
                ancestor_genome=np.asarray(adapted).tolist(), ancestor_homology=ancestor_roles,
                rows=rows, throughput=throughput, elapsed_seconds=perf_counter() - start)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[post] final write: {path}", flush=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    partial_path.unlink()
    report = summary(data)
    path.with_name(path.stem + "-summary.txt").write_text(report + "\n")
    print(report)
    return data


def summary(data):
    if data.get("combined"):
        return combined_summary(data)
    p, curve, speed = data["params"], data["learning_curve"], data["throughput"]
    lines = [f"Experiment G ({'quick validation' if data['quick'] else 'full run'}): generalized swimmer, {p['device']}",
             f"N={p['N']}, K={p['K']}, steps={p['steps']}, adaptation={p['T_adapt']}, burn={p['B']}, drift={p['T']}",
             f"Gain={p['gain']:.5g}; initial action RMS/max={p['initial_action_rms']:.3g}/{p['initial_action_max']:.3g}",
             f"Adaptation return {curve[0]['mean_return']:.5g} -> {curve[-1]['mean_return']:.5g}; plateau={data['plateau']['detected']}",
             "Learning curve (generation:return): " + ", ".join(f"{curve[i]['t']}:{curve[i]['mean_return']:.4g}"
                  for i in sorted(set(np.linspace(0, len(curve) - 1, min(6, len(curve)), dtype=int)))),
             f"Fixed return scales: adaptation={p['adaptation_ret_scale']:.5g}, phase 2={p['ret_scale']:.5g}"]
    for r in data["rows"]:
        h, first, last = r["homology"], r["records"][0], r["records"][-1]
        rank = [v for v in h["importance_rank_vs_ancestor"] if v is not None]
        pairs = [h["importance_rank_between_lineages"][i][j] for i in range(p["K"]) for j in range(i)
                 if h["importance_rank_between_lineages"][i][j] is not None]
        lines.append(f"fork s={r['pair_s']:g}, arm s={r['s']:g}: return {first['mean_return']:.5g} -> {last['mean_return']:.5g}; "
                     f"CKA divergence={last['divergence']['representation']:.4g}; tuning changes={np.mean(h['role_change_fraction']):.1%}; "
                     f"importance rank vs ancestor={np.mean(rank) if rank else 'NA'}, between lineages={np.mean(pairs) if pairs else 'NA'}")
        lines.append(f"  t*={r['t_star']}; omega={r['omega']}; intact return within 10% of fork: {sum(h['return_maintained'])}/{p['K']}")
        e = h["example"]
        lines.append(f"  {'Si1-like example' if e['qualifies'] else 'Largest contrast (no qualifying Si1-like example)'}: unit {e['unit']}, "
                     f"lineage {e['essential_lineage']} drop={e['return_drop_high']:.5g}, lineage {e['dispensable_lineage']} drop={e['return_drop_low']:.5g}")
        lines.append(f"  Hybrid return={r['hybrids']['mean_hybrid_return']:.5g}; parent return={r['hybrids']['mean_parent_return']:.5g}")
    lines += [f"Warmed CPU/GPU throughput: {speed['agent_steps_per_second']:,.0f} agent-steps/s; elapsed={data['elapsed_seconds']:.1f}s.",
              f"Default full evolution: {speed['default_full_evolution_agent_steps']:,} agent-steps; CPU extrapolation={speed['default_full_cpu_hours']:.1f}h.",
              "B300 not measured: assuming 0.1--1 million agent-steps/s gives " +
              f"{speed['gpu_scenarios'][1]['evolution_hours']:.2f}--{speed['gpu_scenarios'][0]['evolution_hours']:.2f}h, plus compilation and assays.",
              "Fixed-reset performance selection does not constrain gait; inspect returns AND gait before claiming conserved behaviour.",
              "Short runs validate machinery, not drift, plateau, or functional reassignment; missing omega means neutral x0 never reached .25."]
    return "\n".join(lines)


def row_measures(row):
    """Scalar summaries within a seed; unit/lineage identities stay in raw rows."""
    h, first, last = row["homology"], row["records"][0], row["records"][-1]

    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    measures = {f"divergence.{level}": last["divergence"][level] for level in LEVELS}
    measures.update({f"omega.{level}": (row["omega"] or {}).get(level) for level in LEVELS})
    measures.update(t_star=row["t_star"], return_initial=first["mean_return"],
                    return_final=last["mean_return"],
                    tuning_change=mean(h["role_change_fraction"]),
                    tuning_change_since_fork=mean(h["role_change_since_fork"]),
                    importance_rank_vs_ancestor=mean(h["importance_rank_vs_ancestor"]),
                    importance_rank_between_lineages=mean([
                        v for i, values in enumerate(h["importance_rank_between_lineages"]) for v in values[:i]]),
                    return_maintained_count=sum(h["return_maintained"]),
                    return_maintained_fraction=mean(h["return_maintained"]),
                    si1_qualifies=int(h["example"]["qualifies"]),
                    example_drop_high=h["example"]["return_drop_high"],
                    example_drop_low=h["example"]["return_drop_low"],
                    example_reference_scale=h["example"]["reference_scale"],
                    hybrid_return=row["hybrids"]["mean_hybrid_return"],
                    parent_return=row["hybrids"]["mean_parent_return"],
                    hybrid_minus_parent=row["hybrids"]["mean_hybrid_return"] - row["hybrids"]["mean_parent_return"])
    return measures


def seed_statistics(values):
    valid = [v for v in values.values() if v is not None]
    return dict(per_seed=values, n=len(valid), mean=float(np.mean(valid)) if valid else None,
                range=[min(valid), max(valid)] if valid else None,
                spread=max(valid) - min(valid) if valid else None)


def combine_data(inputs):
    """Keep one independent observation per (seed, strength), with its own ancestor."""
    # These are measured outputs of calibration/adaptation, not shared settings.
    derived = {"gain", "initial_action_rms", "initial_action_max", "adaptation_ret_scale", "ret_scale"}
    by_seed, common, quick = {}, None, None
    for source, data in inputs:
        if data.get("combined"):
            raise ValueError(f"{source}: expected an individual run, not combined JSON")
        p = data["params"]
        settings = {k: v for k, v in p.items() if k not in derived | {"seed", "strengths"}}
        if common is None:
            common, quick = settings, data["quick"]
        mismatches = [k for k in sorted(common.keys() | settings.keys())
                      if k not in common or k not in settings or common[k] != settings[k]]
        if data["quick"] != quick:
            mismatches.append("quick")
        if mismatches:
            detail = "; ".join(f"{k}: {common.get(k, quick if k == 'quick' else '<missing>')!r} != "
                               f"{settings.get(k, data['quick'] if k == 'quick' else '<missing>')!r}" for k in mismatches)
            raise ValueError(f"{source}: mismatched parameters: {detail}")
        seed = str(p["seed"])
        if set(p["strengths"]) != {r["pair_s"] for r in data["rows"]}:
            raise ValueError(f"{source}: strengths do not match row pair_s values")
        for s in sorted(p["strengths"]):
            rows = [r for r in data["rows"] if r["pair_s"] == s]
            if len(rows) != 2 or sorted(r["s"] for r in rows) != [0, s]:
                raise ValueError(f"{source}: s={s:g} needs exactly one selected and one neutral arm")
            entry = {k: v for k, v in data.items() if k != "rows"}
            entry.update(params={**p, "strengths": [s]}, rows=rows,
                         measures={"neutral" if r["s"] == 0 else "selected": row_measures(r) for r in rows})
            key = f"{s:g}"
            previous = by_seed.setdefault(seed, {}).get(key)
            if previous is not None and previous != entry:
                raise ValueError(f"{source}: conflicting duplicate (seed={seed}, s={key})")
            by_seed[seed][key] = entry
    if common is None:
        raise ValueError("combine needs at least one input JSON")
    by_strength, disagreements = {}, []

    def flag(label, values):
        if len(set(values.values())) > 1:
            disagreements.append(label + ": " + ", ".join(f"seed {seed}={v}" for seed, v in values.items()))

    def ordering(a, b):
        if a is None or b is None:
            return "NA"
        return "<" if a < b else ">" if a > b else "="

    for s in sorted({s for entries in by_seed.values() for s in entries}, key=float):
        entries = {seed: entries[s] for seed, entries in sorted(by_seed.items(), key=lambda kv: int(kv[0])) if s in entries}
        arms = {}
        for arm in ("selected", "neutral"):
            measures = {seed: e["measures"][arm] for seed, e in entries.items()}
            arms[arm] = {name: seed_statistics({seed: m[name] for seed, m in measures.items()})
                         for name in next(iter(measures.values()))}
            for name, reference in (("si1_qualifies", 0), ("return_maintained_fraction", 1),
                                    ("tuning_change", 0), ("tuning_change_since_fork", 0),
                                    ("importance_rank_vs_ancestor", 0), ("importance_rank_between_lineages", 0),
                                    ("hybrid_minus_parent", 0)):
                flag(f"s={s} {arm}: {name} vs {reference}",
                     {seed: ordering(m[name], reference) for seed, m in measures.items()})
            if arm == "selected":
                for level in LEVELS:
                    flag(f"s={s}: omega.{level} vs 1", {seed: ordering(m[f"omega.{level}"], 1) for seed, m in measures.items()})
                for a, b in itertools.combinations(LEVELS, 2):
                    flag(f"s={s}: omega ordering {a} vs {b}",
                         {seed: ordering(m[f"omega.{a}"], m[f"omega.{b}"]) for seed, m in measures.items()})
        by_strength[s] = arms
    for a, b in itertools.combinations(by_strength, 2):
        for level in LEVELS:
            flag(f"omega.{level}: s={a} vs s={b}", {
                seed: ordering(entries[a]["measures"]["selected"][f"omega.{level}"],
                               entries[b]["measures"]["selected"][f"omega.{level}"])
                for seed, entries in by_seed.items() if a in entries and b in entries})
    return dict(combined=True, quick=quick, params=common, by_seed=by_seed,
                by_strength=by_strength, disagreements=disagreements)


def combined_summary(data):
    def fmt(value):
        return "NA" if value is None else f"{value:.6g}"

    lines = ["Experiment G: combined independent seeds" + (" (quick validation)" if data["quick"] else ""),
             "Equal weight per distinct seed; duplicate (seed, s) inputs count once.",
             "Divergences are final-generation values; omega uses each seed's own neutral t*.",
             "Means use available values (n shown); NA is not zero. Range is [min, max]; no CIs.",
             "Ancestors, calibrated gains and return scales remain specific to each (seed, s).",
             "Example units/lineages are local identities; their indices are not averaged."]
    seeds = set(data["by_seed"])
    for s, arms in data["by_strength"].items():
        present = {seed for seed, entries in data["by_seed"].items() if s in entries}
        lines.append(f"\ns={s}; seeds={', '.join(sorted(present, key=int))}")
        if present != seeds:
            lines.append("  Missing seeds: " + ", ".join(sorted(seeds - present, key=int)))
        for seed in sorted(present, key=int):
            entry = data["by_seed"][seed][s]
            p = entry["params"]
            lines.append(f"  seed {seed}: gain={p['gain']:.6g}, adaptation scale={p['adaptation_ret_scale']:.6g}, phase-2 scale={p['ret_scale']:.6g}")
            for row in entry["rows"]:
                e = row["homology"]["example"]
                lines.append(f"    arm s={row['s']:g}: {'Si1-like example' if e['qualifies'] else 'no qualifying Si1-like example; largest contrast'}: "
                             f"unit {e['unit']}, lineages {e['essential_lineage']}/{e['dispensable_lineage']}")
        for arm, measures in arms.items():
            lines.append(f"  {arm}:")
            for name, stat in measures.items():
                values = ", ".join(f"seed {seed}={fmt(v)}" for seed, v in stat["per_seed"].items())
                bounds = "NA" if stat["range"] is None else "[" + ", ".join(map(fmt, stat["range"])) + "]"
                lines.append(f"    {name}: {values}; mean={fmt(stat['mean'])}; range={bounds}; spread={fmt(stat['spread'])}; n={stat['n']}")
    lines.append("\nQualitative disagreements (NA differences indicate availability, not opposite conclusions):")
    lines.extend("  " + text for text in data["disagreements"])
    if not data["disagreements"]:
        lines.append("  None detected." if len(seeds) > 1 else "  Not assessable: only one distinct seed.")
    return "\n".join(lines)


def combine(paths=None, output=None):
    paths = paths if paths is not None else [OUTPUT / name for name in (
        "neuroevo.json", "neuroevo-seed1-s3.json", "neuroevo-seed1-s30.json")]
    data = combine_data([(str(path), json.loads(Path(path).read_text())) for path in paths])
    path = Path(output) if output else OUTPUT / "neuroevo-combined.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = summary(data)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    path.with_name(path.stem + "-summary.txt").write_text(report + "\n")
    print(report)
    return data


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(path=OUTPUT / "neuroevo.json"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("combined"):
        rows = [{**row, "seed": int(seed)} for seed, entries in data["by_seed"].items()
                for entry in entries.values() for row in entry["rows"]]
    else:
        rows = [{**row, "seed": data["params"]["seed"]} for row in data["rows"]]
    strengths = sorted({r["s"] for r in rows if r["s"] > 0})
    seeds = sorted({r["seed"] for r in rows})
    selected_rows = sorted((r for r in rows if r["s"] == max(strengths)), key=lambda r: r["seed"])
    styles = {seed: ("-" if seed == 0 else "--" if seed == 1 else (0, (1, seed + 1))) for seed in seeds}

    def end_labels(ax, labels):
        low, high = ax.get_ylim()
        previous, gap = -.06, .095
        for i, (x, y, name, color) in enumerate(sorted(labels, key=lambda x: x[1])):
            pos = max(previous + gap, np.clip((y - low) / (high - low), .025, .975 - gap * (len(labels) - i - 1)))
            ax.annotate(name, xy=(x, y), xytext=(1.025, pos), textcoords="axes fraction", color=color,
                        fontsize=5.2, family="monospace", va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=color, lw=.35))
            previous = pos

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": .9}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=.075, right=.925, bottom=.28, top=.84, wspace=.9)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax, labels = axes[0], []
        names = dict(theta="weights", representation="1-CKA", **{"return": "return"})
        peak = 0.
        for i, level in enumerate(DISPLAY):
            ends = []
            for selected in selected_rows:
                ts = [r["t"] for r in selected["records"]]
                ys = [r["divergence"][level] for r in selected["records"]]
                ax.plot(ts, ys, color=f"C{i}", linestyle=styles[selected["seed"]])
                ends.append(ys[-1])
                peak = max(peak, max(ys))
            labels.append((ts[-1], np.mean(ends), names.get(level, level), f"C{i}"))
        ends = []
        for selected in selected_rows:
            neutral = next(r for r in rows if r["s"] == 0 and r["pair_s"] == selected["s"] and r["seed"] == selected["seed"])
            ts = [r["t"] for r in neutral["records"]]
            ys = [r["divergence"]["z"] for r in neutral["records"]]
            ax.plot(ts, ys, color="C9", linestyle=styles[selected["seed"]])
            ends.append(ys[-1])
            peak = max(peak, max(ys))
        labels.append((ts[-1], np.mean(ends), "neutral z", "C9"))
        ax.set(xlabel="Generation after fork", ylabel="Divergence", ylim=(0, max(.01, peak) * 1.12))
        ax.set_title(f"s={max(strengths):g}", fontsize=6, family="monospace")
        for seed in seeds:
            ax.plot([], [], color="0.3", linestyle=styles[seed], label=f"seed {seed}")
        ax.legend(fontsize=5, loc="upper left", frameon=False)
        end_labels(ax, labels)
        ax, labels = axes[1], []
        missing = 0
        for row in sorted((r for r in rows if r["s"] > 0), key=lambda r: (r["s"], r["seed"])):
            i = strengths.index(row["s"])
            if row["omega"] is None:
                ax.text(.02, .94 - missing * .14, f"s={row['s']:g}, seed {row['seed']}: no t*", transform=ax.transAxes,
                        fontsize=5.5, family="monospace", va="top", color=f"C{i}")
                missing += 1
                continue
            y = [np.nan if row["omega"][l] is None else row["omega"][l] for l in DISPLAY]
            ax.plot(range(len(DISPLAY)), y, marker=".", color=f"C{i}", linestyle=styles[row["seed"]])
            valid = np.flatnonzero(np.isfinite(y))
            if len(valid):
                j = valid[-1]
                labels.append((j, y[j], f"s={row['s']:g}, seed {row['seed']}", f"C{i}"))
        ax.set(xlabel="Genome to behaviour", ylabel=r"$\omega$ (selected / neutral)", ylim=(0, None),
               xticks=range(len(DISPLAY)), xticklabels=["x0", "M1", "M2", "M3", "z", r"$\theta$", "CKA", "ret", "gait"])
        ax.tick_params(axis="x", rotation=65, labelsize=5)
        end_labels(ax, labels)
        ax = axes[2]
        ax.grid(False)
        # Separate seed blocks: neither ancestors nor neuron identities are pooled.
        importance = np.concatenate([np.array(r["homology"]["importance"]).T for r in selected_rows], axis=1)
        limit = max(float(abs(importance).max()), 1e-6)
        im = ax.imshow(importance, aspect="auto", origin="lower", cmap=plt.get_cmap(), vmin=-limit, vmax=limit)
        ticks, titles, offset = [], [], 0
        for selected in selected_rows:
            e = selected["homology"]["example"]
            count = len(selected["homology"]["importance"])
            for lineage in (e["essential_lineage"], e["dispensable_lineage"]):
                ax.scatter(offset + lineage, e["unit"], marker="s", s=38, facecolors="none", edgecolors="C3", linewidths=.8)
            ticks.append((offset + (count - 1) / 2, f"seed {selected['seed']}\n0–{count - 1}"))
            titles.append(f"seed {selected['seed']}: {'Si1-like' if e['qualifies'] else 'max contrast'} u{e['unit']}")
            if offset:
                ax.axvline(offset - .5, color="white", lw=1)
            offset += count
        ax.set(xlabel="Lineage", ylabel="Homologous unit", yticks=[0, 5, 10, 15],
               xticks=[t for t, _ in ticks], xticklabels=[label for _, label in ticks])
        ax.set_title("\n".join(titles), fontsize=5.5, family="monospace")
        bar = fig.colorbar(im, ax=ax, fraction=.065, pad=.06)
        bar.set_label("Ablation return drop", fontsize=6)
        bar.ax.tick_params(labelsize=5)
        if data["quick"]:
            fig.suptitle("Quick validation • short trajectories", fontsize=7, family="monospace", y=.99)
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
    print(f"Figure: {path.with_suffix('.pdf')} and .png")


def check_combine():
    """Tiny synthetic JSONs exercise aggregation without evolving populations."""
    from copy import deepcopy

    rows = []
    for s in (3., 30.):
        for arm in (s, 0.):
            rows.append(dict(pair_s=s, s=arm, t_star=1, omega=dict.fromkeys(LEVELS, .5 if arm else 1.),
                             records=[dict(t=t, mean_return=2. + t,
                                           divergence=dict.fromkeys(LEVELS, .1 * t)) for t in (0, 1)],
                             homology=dict(role_change_fraction=[0., .5], role_change_since_fork=[0., .25],
                                           importance_rank_vs_ancestor=[None, .5],
                                           importance_rank_between_lineages=[[1., .2], [.2, 1.]],
                                           return_maintained=[True, True], importance=[[0.] * H, [0.] * H],
                                           example=dict(qualifies=False, unit=0, essential_lineage=0, dispensable_lineage=1,
                                                        return_drop_high=.1, return_drop_low=0., reference_scale=2.)),
                             hybrids=dict(mean_hybrid_return=1., mean_parent_return=2.)))
    original = dict(quick=True, params=dict(seed=0, strengths=[3., 30.], N=2, K=2, T=1,
                                           gain=.2, initial_action_rms=.1, initial_action_max=.2,
                                           adaptation_ret_scale=1., ret_scale=1.),
                    ancestor_genome=[1., 0.], rows=rows)
    # Round-trip the on-disk format and pass exactly the same JSON twice.
    original = json.loads(json.dumps(original))
    same = combine_data([("a.json", original), ("a.json", original)])
    assert same == combine_data([("a.json", original)])
    for s, arms in same["by_strength"].items():
        for arm, measures in arms.items():
            row = next(r for r in rows if f"{r['pair_s']:g}" == s and (r["s"] == 0) == (arm == "neutral"))
            assert same["by_seed"]["0"][s]["measures"][arm] == row_measures(row)
            assert all(v["spread"] == 0 and v["n"] == 1 and v["mean"] == v["per_seed"]["0"] for v in measures.values())
    other = deepcopy(original)
    other["params"].update(seed=1, gain=.3, ret_scale=2.)
    other["ancestor_genome"] = [0., 1.]
    other["rows"][0]["omega"]["x0"] = 2.
    other["rows"][0]["homology"]["example"]["qualifies"] = True
    other["rows"][0]["homology"]["importance_rank_vs_ancestor"] = [None, None]
    split = []
    for s in (3., 30.):
        part = deepcopy(other)
        part["params"]["strengths"] = [s]
        part["rows"] = [r for r in part["rows"] if r["pair_s"] == s]
        part["params"]["ret_scale"] += s  # Separately adapted arms may differ.
        split.append((f"seed1-s{s:g}.json", part))
    combined = combine_data([("seed0.json", original), *split])
    stat = combined["by_strength"]["3"]["selected"]["omega.x0"]
    assert stat == dict(per_seed={"0": .5, "1": 2.}, n=2, mean=1.25, range=[.5, 2.], spread=1.5)
    assert combined["by_strength"]["3"]["selected"]["importance_rank_vs_ancestor"]["n"] == 1
    assert combined["by_seed"]["1"]["3"]["params"]["ret_scale"] == 5.
    assert combined["by_seed"]["1"]["30"]["params"]["ret_scale"] == 32.
    assert any("omega ordering x0 vs x1" in line for line in combined["disagreements"])
    assert any("si1_qualifies" in line for line in combined["disagreements"])
    assert "range=[0.5, 2]" in summary(combined)
    missing = deepcopy(original)
    missing["rows"][0].update(omega=None, t_star=None)
    assert combine_data([("missing", missing)])["by_strength"]["3"]["selected"]["omega.x0"]["mean"] is None
    for field, value in (("N", 3), ("extra_setting", True), ("quick", False)):
        bad = deepcopy(other)
        (bad if field == "quick" else bad["params"])[field] = value
        try:
            combine_data([("a.json", original), ("bad.json", bad)])
        except ValueError as error:
            assert "mismatched parameters" in str(error) and field in str(error)
        else:
            raise AssertionError(f"Accepted mismatched {field}")
    conflict = deepcopy(original)
    conflict["rows"][0]["records"][-1]["mean_return"] += 1
    try:
        combine_data([("a.json", original), ("conflict.json", conflict)])
    except ValueError as error:
        assert "conflicting duplicate (seed=0, s=3)" in str(error)
    else:
        raise AssertionError("Accepted conflicting duplicate")


def check():
    start = perf_counter()
    check_combine()
    swimmer = Swimmer(8)
    rng = np.random.default_rng(7)
    g = normalize(jnp.asarray(rng.normal(size=(3, D + L * D * D)), dtype=jnp.float32))
    readout = jnp.asarray(rng.normal(size=(P, D)), dtype=jnp.float32)
    theta = parameters(g, readout, .2)
    assert theta.shape == (3, P) and np.array_equal(theta, parameters(g, readout, .2))
    reset = jax.jit(swimmer.env.reset)
    env_step = jax.jit(lambda state, action: swimmer.env.step(state.replace(metrics=dict(state.metrics)), action))
    a, b = reset(swimmer.reset_key), reset(swimmer.reset_key)
    for _ in range(2):
        a, b = env_step(a, jnp.zeros(2)), env_step(b, jnp.zeros(2))
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    batch = np.asarray(swimmer.evaluate(theta))
    single = np.array([swimmer.single(t, jnp.ones(H)) for t in theta])
    assert np.allclose(batch, single, rtol=2e-5, atol=2e-5), (batch, single)
    assert np.array_equal(batch, swimmer.evaluate(theta))
    assert np.allclose(batch, swimmer.assay(theta, trace=True)[0], atol=2e-5)
    x, y = rng.normal(size=(2000, H)), rng.normal(size=(2000, H))
    assert abs(linear_cka(x, x) - 1) < 1e-12 and linear_cka(x, y) < .03
    zero = theta[0].at[144:146].set(0.)
    assert np.allclose(swimmer.single(zero, jnp.ones(H)), swimmer.single(zero, jnp.ones(H).at[0].set(0)), atol=1e-7)
    returns = jnp.array([[1., -100., 4., 1000.]])
    assert np.array_equal(probabilities(returns, 0., 1.), np.full((1, 4), .25))
    many = jnp.tile(returns, (8192, 1))
    parents = sample_parents(many, jax.random.PRNGKey(19), 0., 1.)
    assert np.max(abs(np.bincount(np.asarray(parents).ravel(), minlength=4) / parents.size - .25)) < .015
    assert np.array_equal(g, mutate(g, jax.random.PRNGKey(4), 0., .1))
    mutated = mutate(g, jax.random.PRNGKey(4), 1., .1)
    assert np.allclose(np.linalg.norm(mutated[:, :D], axis=-1), 1.)
    assert np.allclose(np.linalg.norm(mutated[:, D:].reshape(3, L, -1), axis=-1), np.sqrt(D))
    fixture = [dict(s=s, pair_s=3., records=[dict(t=0, divergence=dict.fromkeys(LEVELS, 0.)),
                    dict(t=20, divergence=dict.fromkeys(LEVELS, value))]) for s, value in ((0., .3), (3., .15))]
    add_omega(fixture)
    assert fixture[1]["t_star"] == 20 and set(fixture[1]["omega"].values()) == {.5}
    fixture[0]["records"][1]["divergence"]["x0"] = .2
    add_omega(fixture)
    assert fixture[1]["omega"] is None and fixture[1]["t_star"] is None
    example = example_unit(np.array([[2., 0.], [-3., 0.], [0., 0.]]), np.ones(3, dtype=bool), 10.)
    assert example["qualifies"] and example["essential_lineage"] == 0 and example["dispensable_lineage"] == 2
    example = example_unit(np.zeros((2, H)), np.ones(2, dtype=bool), 10.)
    assert not example["qualifies"] and example["essential_lineage"] != example["dispensable_lineage"]
    print(f"check ok ({perf_counter() - start:.1f}s): deterministic swimmer, P=178, vmap, CKA, ablation, uniform neutral sampling, norms, omega, combine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "quick", "run", "combine", "figure"), nargs="?", default="run")
    parser.add_argument("--input", type=Path, nargs="+", help="Input JSON(s); combine defaults to the three seed 0/1 runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    for name in ("N", "K", "steps", "T-adapt", "N-adapt", "B", "T", "every", "probe-count"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--s", nargs="+", type=float, dest="strengths")
    parser.add_argument("--s-adapt", type=float)
    parser.add_argument("--u", type=float)
    parser.add_argument("--sigma", type=float)
    args = vars(parser.parse_args())
    command, path, output = args.pop("command"), args.pop("input"), args.pop("output")
    if command == "check":
        check()
    elif command == "figure":
        if path and len(path) != 1:
            parser.error("figure accepts exactly one --input JSON")
        figure(path[0] if path else OUTPUT / "neuroevo.json")
    elif command == "combine":
        try:
            combine(path, output)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    else:
        options = dict(N=16, K=2, steps=50, T_adapt=10, N_adapt=16, B=10, T=10, every=5) if command == "quick" else {}
        options.update({k: v for k, v in args.items() if v is not None})
        run(**options, quick=command == "quick", output=output)
