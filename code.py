"""Experiment F: selection for an efficient neural code, then drift beneath it.

Run: python code.py [check | quick | run | figure] [--workers 15] [--seed 0]
Natural-scene contrasts are represented by a truncated Student-t (default, df=3,
scale=0.25, so its central density is resolved by the 64-point contrast grid)
or Laplace density on 64 points; no image dataset is used. The endpoint-normalized
CDF is Laughlin's small-noise reference, not the exact finite-noise capacity.
Information is in nats, computed as H(r)-H(r|c) on a sampled Gaussian response
grid (spacing <= sigma_n/4, tails extend eight noise SD beyond [0,1]).

The 63 increments are proportional to exp(Phi @ (gain * unit(top_state))),
where Phi contains the first 16 orthonormal DCT-II modes, including the constant.
Thus only phenotype direction matters, as in drift.py's cosine phenotype.
The projected optimum z_star = Phi.T @ log(p[1:]) represents the CDF almost
exactly. A fixed gain is |z_star| / mean ancestral top-state norm, estimated once
from independent normalized L=3 genomes; it is never adapted during evolution.
Because unit(top_state) has norm one, the coefficient radius is gain itself.
The constant mode can absorb unused radius without changing the response,
making the projected optimum reachable whenever |z_star[1:]| <= gain.
Fitness uses the larger of CDF information and the best direct z optimization,
not a random-readout ceiling. Calibration starts at sigma_n=0.02 and lowers it
if necessary until the CDF is within 1% of both the best z optimization and an
unrestricted monotone-code diagnostic. These are numerical, not certified optima.

Genomes, linear development, normalization, per-site mutation, cosine state/map
divergence and omega come from drift.py. Wright-Fisher sampling is its same
batched multinomial step, with log weights shifted to avoid underflow. Each
adaptation replicate starts with an independent random clonal genome. All grid
cells share one readout/environment per seed. Forks repeat the first adapted
population, with selected/neutral arms sharing that population and RNG stream;
there is no neutral re-adaptation. Response divergence is pairwise mean squared
difference of lineage mean functions, averaged over contrast points (not cosine).
Last-quarter averages are finite-run estimates, not evidence of stationarity.
Quick uses K=4 for adaptation and forks; full uses K=8 and K=32 respectively.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from drift import normalize, forward, mutate, unit, divergence, add_omega

OUTPUT = Path(__file__).resolve().parent / "output"


class Channel:
    def __init__(self, seed=0, distribution="student", scale=0.25, df=3,
                 sigma_n=0.02, bins=64, d=16, samples_per_sigma=4, gain=None):
        if min(scale, df, sigma_n, samples_per_sigma) <= 0 or bins < 3 or not 1 <= d < bins:
            raise ValueError("Require positive scale, df, sigma_n and 1 <= d < bins")
        self.params = dict(seed=seed, distribution=distribution, scale=scale,
                           df=df, sigma_n=sigma_n, bins=bins, d=d,
                           samples_per_sigma=samples_per_sigma)
        self.c = np.linspace(-1, 1, bins)
        if distribution == "laplace":
            p = np.exp(-np.abs(self.c) / scale)
        elif distribution == "student":
            p = (1 + (self.c / scale)**2 / df)**(-(df + 1) / 2)
        else:
            raise ValueError(distribution)
        self.p = p / p.sum()
        cdf = np.cumsum(self.p)
        self.star = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        # DCT-II evaluated on the ordered increment grid c[1:]; exact orthonormality.
        self.Phi = np.cos(np.pi * (np.arange(bins - 1)[:, None] + 0.5)
                          * np.arange(d) / (bins - 1)) * np.sqrt(2 / (bins - 1))
        self.Phi[:, 0] /= np.sqrt(2)
        self.z_star = self.Phi.T @ np.log(self.p[1:])
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7122]))
        ancestors = normalize(rng.normal(size=(4096, d + 3 * d * d)), d)
        self.ancestor_top_norm = float(np.linalg.norm(forward(ancestors, d)[-1], axis=-1).mean())
        self.gain = float(np.linalg.norm(self.z_star) / self.ancestor_top_norm if gain is None else gain)
        if not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be finite and positive")
        self.params["gain"] = self.gain
        self.requested_sigma_n = sigma_n
        self.projected_response = self.coefficient_response(self.z_star)
        self.projected_z = self.phenotype_for_coefficients(self.z_star)
        self.set_noise(sigma_n)

    def set_noise(self, sigma_n):
        self.sigma_n = sigma_n
        self.params["sigma_n"] = sigma_n
        n = int(np.ceil((1 + 16 * sigma_n) * self.params["samples_per_sigma"] / sigma_n)) + 1
        self.r = np.linspace(-8 * sigma_n, 1 + 8 * sigma_n, n)
        self.cdf_information = float(self.information(self.star))
        self.I_max = self.cdf_information

    def response(self, z):
        return self.coefficient_response(self.gain * unit(np.asarray(z)))

    def coefficient_response(self, coefficients):
        # Softmax is a stable exponential followed by increment normalization.
        a = softmax(np.asarray(coefficients) @ self.Phi.T, axis=-1)
        f = np.concatenate([np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1)], axis=-1)
        return f / f[..., -1:]

    def phenotype_for_coefficients(self, coefficients):
        """Put coefficients on the fixed-radius sphere using the constant gauge."""
        z = np.array(coefficients, copy=True)
        radius2 = self.gain**2 - z[1:] @ z[1:]
        if radius2 < -1e-10:
            raise ValueError("gain is too small to represent the projected CDF")
        z[0] = -np.sqrt(max(radius2, 0))
        return z / self.gain

    def information(self, f, gradient=False):
        """Exact discrete-grid MI; batches cap temporary arrays at ~16 MB."""
        f = np.asarray(f)
        flat = f.reshape(-1, len(self.c))
        values, gradients = [], []
        for start in range(0, len(flat), 32):
            fs = flat[start:start + 32]
            delta = (self.r - fs[..., None]) / self.sigma_n
            log_kernel = -0.5 * delta**2
            q = np.exp(log_kernel)
            norm = q.sum(-1, keepdims=True)
            q /= norm
            mix = self.p @ q
            log_mix = np.log(np.maximum(mix, 1e-300))
            h_cond = (self.p * (np.log(norm[..., 0]) - (q * log_kernel).sum(-1))).sum(-1)
            values.append(-(mix * log_mix).sum(-1) - h_cond)
            if gradient:
                score = delta / self.sigma_n
                score -= (q * score).sum(-1, keepdims=True)
                log_ratio = log_kernel - np.log(norm) - log_mix[:, None, :]
                gradients.append(self.p * (q * score * log_ratio).sum(-1))
        result = np.concatenate(values).reshape(f.shape[:-1])
        if gradient:
            return result, np.concatenate(gradients).reshape(f.shape)
        return result

    def objective_z(self, z):
        direction = unit(z)
        increments = softmax(self.Phi @ (self.gain * direction))
        f = self.response(z)
        info, grad = self.information(f, gradient=True)
        tail = np.cumsum(grad[:0:-1])[::-1]
        da = (tail - grad @ f) * increments
        dc = da @ self.Phi
        dz = self.gain / np.linalg.norm(z) * (dc - (dc @ direction) * direction)
        return -float(info), -dz

    def optimize_direct(self):
        """Finite-noise diagnostic: optimize all interior monotone responses."""
        n = len(self.c) - 2

        def objective(interior):
            value, grad = self.information(np.r_[0, interior, 1], gradient=True)
            return -float(value), -grad[1:-1]

        result = minimize(objective, self.star[1:-1], jac=True, method="SLSQP",
                          bounds=[(0, 1)] * n,
                          constraints=[dict(type="ineq", fun=lambda x: np.diff(x),
                                            jac=lambda x: np.diff(np.eye(n), axis=0))],
                          options=dict(ftol=1e-11, maxiter=500))
        information = -float(result.fun)
        gain = information - self.cdf_information
        return dict(information=information, response=np.r_[0, result.x, 1].tolist(),
                    success=bool(result.success), gain_nats=gain,
                    relative_gap=gain / information, tolerance_relative=0.01,
                    cdf_near_optimal=bool(gain / information <= 0.01))

    def calibrate(self, starts=6):
        """Calibrate noise and a CDF-based reference, retaining numerical diagnostics."""
        rng = np.random.default_rng(self.params["seed"] + 7123)
        uniform = np.zeros(self.Phi.shape[1])
        uniform[0] = -1
        guesses = [uniform, self.projected_z]
        guesses += list(unit(rng.normal(size=(max(0, starts - 2), self.Phi.shape[1]))))
        constraint = dict(type="eq", fun=lambda z: z @ z - 1, jac=lambda z: 2 * z)
        noise_trials = []
        for attempt in range(16):
            results = [minimize(self.objective_z, z, jac=True, method="SLSQP",
                                constraints=[constraint],
                                options=dict(ftol=1e-11, maxiter=600)) for z in guesses]
            best = min(results, key=lambda r: r.fun)
            direct = self.optimize_direct()
            best_information = -float(best.fun)
            gap = max(0, 1 - self.cdf_information / max(best_information, direct["information"]))
            noise_trials.append(dict(sigma_n=self.sigma_n, cdf_information=self.cdf_information,
                                     best_z_information=best_information,
                                     direct_information=direct["information"], relative_gap=gap))
            if not best.success or not direct["success"]:
                raise RuntimeError("Reference optimization did not converge")
            if gap <= 0.01:
                break
            if attempt == 15:
                raise RuntimeError("Could not find noise with CDF within 1% of direct optima")
            self.set_noise(self.sigma_n * 0.8)
        self.I_max = max(self.cdf_information, best_information)
        self.best_z = unit(best.x)
        projected_information = float(self.information(self.projected_response))
        self.calibration = dict(I_max=self.I_max, cdf_information=self.cdf_information,
                                best_z_information=best_information,
                                best_cdf_efficiency=best_information / self.cdf_information,
                                best_z=self.best_z.tolist(), best_response=self.response(self.best_z).tolist(),
                                z_star=self.z_star.tolist(), projected_z=self.projected_z.tolist(),
                                projected_information=projected_information,
                                projected_cdf_efficiency=projected_information / self.cdf_information,
                                projection_residual=1 - projected_information / self.cdf_information,
                                cdf_fit_ks=float(np.max(np.abs(self.projected_response - self.star))),
                                log_density_rmse=float(np.sqrt(np.mean((self.Phi @ self.z_star - np.log(self.p[1:]))**2))),
                                ancestor_top_norm=self.ancestor_top_norm,
                                requested_sigma_n=self.requested_sigma_n, noise_trials=noise_trials,
                                direct_optimization=direct,
                                starts=[dict(information=-float(r.fun), success=bool(r.success),
                                             gradient_norm=float(np.linalg.norm(r.jac))) for r in results])
        return self.calibration

    def fitness(self, information, s):
        return np.exp(-s * (1 - np.asarray(information) / self.I_max))

    def export(self):
        return dict(**self.params, contrasts=self.c.tolist(), probabilities=self.p.tolist(),
                    response_grid=self.r.tolist(), readout=self.Phi.tolist(), f_star=self.star.tolist(),
                    **self.calibration)


def step(g, info, channel, s, d, u, sigma, rng):
    """drift.step's Wright-Fisher draw and mutation, with coding fitness."""
    K, N = g.shape[:2]
    logw = s * (info / channel.I_max - 1) if s else np.zeros((K, N))
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    cdf = np.cumsum(w, axis=1)
    draws = rng.random((K, N, 1)) * cdf[:, -1:, None]
    parents = (draws > cdf[:, None, :]).sum(-1)
    inherited = g[np.arange(K)[:, None], parents]
    info = info[np.arange(K)[:, None], parents].copy()
    g = mutate(inherited.copy(), d, u, sigma, rng)
    # Unmutated offspring inherit exactly the same information; no approximation.
    if s:
        changed = np.any(g != inherited, axis=-1)
        if changed.any():
            info[changed] = channel.information(channel.response(forward(g[changed], d)[-1]))
    return g, info


def measure(g, channel, d, info=None):
    states = forward(g, d)
    f = channel.response(states[-1])
    if info is None:
        info = channel.information(f)
    means = f.mean(axis=1)
    efficiency = (info / channel.I_max).mean(axis=1)
    ks = np.abs(means - channel.star).max(axis=1)
    return dict(efficiency=efficiency.tolist(), mean_efficiency=float(efficiency.mean()),
                cdf_efficiency=float(info.mean() / channel.cdf_information),
                ks=ks.tolist(), mean_ks=float(ks.mean()), mean_response=means.mean(axis=0).tolist(),
                replicate_responses=means.tolist())


def drift_measure(g, channel, d, info=None):
    record = measure(g, channel, d, info)
    states = unit(np.stack(forward(g, d), axis=-2))
    means = unit(states.mean(axis=1))
    maps = g[..., d:].reshape(*g.shape[:2], -1, d * d)
    responses = channel.response(forward(g, d)[-1]).mean(axis=1)
    K = len(g)
    # Pairwise squared distance identity; use centered values to avoid cancellation.
    response_divergence = 2 * K / (K - 1) * np.mean((responses - responses.mean(axis=0))**2)
    return dict(**record, state_divergence=divergence(means).tolist(),
                map_divergence=divergence(unit(maps.mean(axis=1))).tolist(),
                within_variation=np.clip(1 - (states * means[:, None]).sum(-1), 0, 2).mean(axis=(0, 1)).tolist(),
                response_divergence=float(response_divergence))


def adapt(config, channel):
    start = perf_counter()
    N, s, K, L, d = [config[k] for k in ("N", "s", "K", "L", "d")]
    initial, evolution = np.random.SeedSequence([config["seed"], N, int(s), 1]).spawn(2)
    rng = np.random.default_rng(initial)
    ancestors = normalize(rng.normal(size=(K, d + L * d * d)), d)
    g = np.repeat(ancestors[:, None], N, axis=1)
    info = np.repeat(channel.information(channel.response(forward(ancestors, d)[-1]))[:, None], N, axis=1)
    rng = np.random.default_rng(evolution)
    records = []
    for t in range(config["T_adapt"] + 1):
        if t % config["every"] == 0 or t == config["T_adapt"]:
            records.append(dict(t=t, **measure(g, channel, d, info)))
        if t < config["T_adapt"]:
            g, info = step(g, info, channel, s, d, config["u"], config["sigma"], rng)
    tail = [r for r in records if r["t"] >= 0.75 * config["T_adapt"]]
    efficiencies = np.array([r["efficiency"] for r in tail])
    replicate_means = efficiencies.mean(axis=0)
    replicate_responses = np.mean([r["replicate_responses"] for r in tail], axis=0)
    middle = len(tail) // 2
    equilibrium = dict(N=N, s=s, Ns=N * s, mean=float(replicate_means.mean()),
                       replicate_sd=float(replicate_means.std(ddof=1)),
                       temporal_sd=float(efficiencies.std(axis=0).mean()),
                       replicate_means=replicate_means.tolist(),
                       tail_change=float(efficiencies[middle:].mean() - efficiencies[:middle].mean()) if middle else 0,
                       cdf_efficiency=float(replicate_means.mean() * channel.I_max / channel.cdf_information),
                       mean_ks=float(np.mean([r["mean_ks"] for r in tail])),
                       mean_response=replicate_responses.mean(axis=0).tolist(),
                       response_sd=replicate_responses.std(axis=0, ddof=1).tolist())
    row = dict(**config, records=records, equilibrium=equilibrium, seconds=perf_counter() - start)
    return row, g[0] if N == 100 and s in (100, 1000) else None


def fork(config, population, channel):
    start = perf_counter()
    g = np.repeat(population[None], config["K"], axis=0)
    info = channel.information(channel.response(forward(g, config["d"])[-1]))
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 91]))
    records = []
    for t in range(config["T"] + 1):
        if t % config["every"] == 0 or t == config["T"]:
            record = drift_measure(g, channel, config["d"], info if config["s"] else None)
            record["mean_fitness"] = float(channel.fitness(info, config["s"]).mean()) if config["s"] else 1.0
            records.append(dict(t=t, **record))
        if t < config["T"]:
            g, info = step(g, info, channel, config["s"], config["d"], config["u"], config["sigma"], rng)
    return dict(**config, records=records, seconds=perf_counter() - start)


def worker(job):
    kind, config, channel, population = job
    return adapt(config, channel) if kind == "adapt" else fork(config, population, channel)


def thresholds(table):
    result = {}
    for threshold in (0.95, 0.99):
        by_N = {}
        for N in sorted({r["N"] for r in table}):
            rows = sorted((r for r in table if r["N"] == N), key=lambda r: r["Ns"])
            hit = next((i for i, r in enumerate(rows) if r["mean"] > threshold), None)
            by_N[str(N)] = dict(first_exceeds=rows[hit]["Ns"] if hit is not None else None,
                                lower_sample=rows[hit - 1]["Ns"] if hit else None,
                                tested_max=rows[-1]["Ns"])
        hits = [r["first_exceeds"] for r in by_N.values() if r["first_exceeds"] is not None]
        result[str(threshold)] = dict(first_observed=min(hits) if hits else None, by_N=by_N)
    return result


def collapse(table):
    """Descriptive comparison only at exactly matched Ns, without extrapolation."""
    groups = [[r for r in table if r["Ns"] == ns] for ns in sorted({r["Ns"] for r in table})]
    comparisons = [dict(Ns=g[0]["Ns"], spread=max(r["mean"] for r in g) - min(r["mean"] for r in g))
                   for g in groups if len(g) > 1]
    spread = max((r["spread"] for r in comparisons), default=None)
    return dict(comparisons=comparisons, tolerance=0.02, max_spread=spread,
                statement=("No matched N*s values to assess collapse." if spread is None else
                           "Curves approximately collapse at matched N*s (within 0.02)." if spread <= 0.02 else
                           "Curves do not collapse within 0.02 at matched N*s."))


def estimate_runtime(rows, drift_rows, workers=15):
    """Measured per-individual-generation costs; greedy 15-worker scheduling."""
    jobs = []
    for N in (10, 30, 100, 300):
        for s in (1, 3, 10, 30, 100, 300, 1000):
            near = min(rows, key=lambda r: abs(np.log(r["N"] / N)) + abs(np.log(r["s"] / s)))
            jobs.append(near["seconds"] * N / near["N"] * 8 / near["K"] * 20000 / near["T_adapt"])
    forks = [r["seconds"] * 32 / r["K"] * 20000 / r["T"] for r in drift_rows]

    def schedule(costs):
        loads = np.zeros(workers)
        for cost in sorted(costs, reverse=True):
            loads[np.argmin(loads)] += cost
        return float(loads.max())

    return dict(workers=workers, seconds=schedule(jobs) + schedule(forks),
                cpu_seconds=sum(jobs) + sum(forks),
                caveat="Linear N*K*T extrapolation from quick; staged adaptation/forks, no extra L. Hardware contention and convergence may change cost.")


def summary(data):
    env = data["environment"]
    print(f"sigma_n={env['sigma_n']:.6g}; CDF I={env['cdf_information']:.6f} nats; best z I={env['best_z_information']:.6f}; I_max={env['I_max']:.6f}")
    print(f"Projected CDF: {env['projected_cdf_efficiency']:.8%} of CDF information; residual={env['projection_residual']:.3g}; KS={env['cdf_fit_ks']:.6f}; gain={env['gain']:.6f}.")
    diagnostic = env["direct_optimization"]
    print(f"Unrestricted direct code I={diagnostic['information']:.6f}; improves on CDF by {diagnostic['gain_nats']:.4f} nats ({diagnostic['relative_gap']:.2%}); CDF within 1%: {diagnostic['cdf_near_optimal']}.")
    print("  N     s      N*s    efficiency +/- replicate SD   CDF eff.    KS    tail change")
    for r in data["equilibrium"]:
        print(f"{r['N']:3d} {r['s']:5g} {r['Ns']:8g}    {r['mean']:.4f} +/- {r['replicate_sd']:.4f}        {r['cdf_efficiency']:.4f}  {r['mean_ks']:.4f}  {r['tail_change']:+.4f}")
    print("Thresholds: first sampled N*s exceeding cutoff (NA = not reached)")
    for threshold, entry in data["Ns_crit"].items():
        print(f"  {threshold}: " + "; ".join(f"N={N}: {r['first_exceeds'] if r['first_exceeds'] is not None else 'NA'}" for N, r in entry["by_N"].items()))
    print(data["collapse"]["statement"])
    print("fork s   source s    t*    omega_0 ... omega_3                  response MSE")
    for r in data["drift"]:
        omega = " ".join(f"{v:.3f}" if v is not None else "NA" for v in r["omega"]) if r["omega"] is not None else "NA"
        print(f"{r['s']:6g} {r['source_s']:10g} {str(r['t_star']):>5}    {omega:37s} {r['records'][-1]['response_divergence']:.6f}")
    print(f"Elapsed {data['timings']['elapsed_seconds'] / 60:.2f} min; estimated full grid on 15 workers: {data['runtime_estimate']['seconds'] / 60:.1f} min (linear extrapolation).")


def main(quick=False, seed=0, workers=15, **environment):
    from tqdm import tqdm
    start = perf_counter()
    channel = Channel(seed=seed, **environment)
    channel.calibrate()
    calibration_seconds = perf_counter() - start
    print(f"sigma_n={channel.sigma_n:.6g}; reference={channel.I_max / channel.cdf_information:.4f} of CDF information; starting {'quick' if quick else 'full'} grid.", flush=True)
    common = dict(L=3, d=16, u=1e-3, sigma=0.1, seed=seed, every=200)
    configs = [dict(**common, N=N, s=s, K=4 if quick else 8, T_adapt=5000 if quick else 20000)
               for N in ((10, 100) if quick else (10, 30, 100, 300))
               for s in ((10, 100, 1000) if quick else (1, 3, 10, 30, 100, 300, 1000))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        adapted = list(tqdm(pool.map(worker, [("adapt", c, channel, None) for c in configs]), total=len(configs), desc="adaptation"))
        jobs = []
        for row, population in adapted:
            if population is None:
                continue
            for s in (row["s"], 0):
                config = dict(**common, N=100, s=s, source_s=row["s"], init="gaussian", tanh=False,
                              K=4 if quick else 32, T=4000 if quick else 20000)
                # add_omega matches on seed: each adapted fork gets a distinct ID.
                config["seed"] = int(np.random.SeedSequence([seed, int(row["s"]), 17]).generate_state(1)[0])
                jobs.append(("fork", config, channel, population))
        drift_rows = list(tqdm(pool.map(worker, jobs), total=len(jobs), desc="forks"))
    add_omega(drift_rows)
    rows = [r for r, _ in adapted]
    table = [r["equilibrium"] for r in rows]
    data = dict(quick=quick, params=dict(**common, workers=workers, adaptation_K=configs[0]["K"],
                                        T_adapt=configs[0]["T_adapt"], fork_K=4 if quick else 32,
                                        fork_T=4000 if quick else 20000),
                environment=channel.export(), adaptation=rows, equilibrium=table,
                Ns_crit=thresholds(table), collapse=collapse(table), drift=drift_rows,
                timings=dict(elapsed_seconds=perf_counter() - start, calibration_seconds=calibration_seconds,
                             adaptation_worker_seconds=sum(r["seconds"] for r in rows),
                             fork_worker_seconds=sum(r["seconds"] for r in drift_rows)),
                runtime_estimate=estimate_runtime(rows, drift_rows))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "code.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    summary(data)
    figure()


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401 (local figure style)

    data = json.loads((OUTPUT / "code.json").read_text())
    env = data["environment"]
    text_style = dict(fontsize=5.5, family="monospace", va="center")

    def labels(ax, items, gap=0.10):
        low, high = ax.get_ylim()
        placed = -gap
        ordered = sorted(items, key=lambda item: item[1])
        for i, (x, y, label, color) in enumerate(ordered):
            pos = np.clip((y - low) / (high - low), 0.04, 0.96 - gap * (len(items) - i - 1))
            pos = max(pos, placed + gap)
            placed = pos
            ax.annotate(label, xy=(x, y), xytext=(1.02, pos), textcoords="axes fraction", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.4), annotation_clip=False, **text_style)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "lines.linewidth": 1}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.065, right=0.88, bottom=0.22, top=0.85, wspace=1.05)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        c = np.array(env["contrasts"])
        ax.plot(c, env["f_star"], color="C0", linestyle="--")
        # Interior anchors distinguish curves whose endpoints necessarily coincide.
        anchor = int(0.62 * len(c))
        items = [(c[anchor], env["f_star"][anchor], "CDF", "C0")]
        candidates = sorted((r for r in data["equilibrium"] if r["N"] == 100), key=lambda r: r["s"])
        for i, r in enumerate([candidates[0], candidates[len(candidates) // 2], candidates[-1]], 1):
            f, sd = np.array(r["mean_response"]), np.array(r["response_sd"])
            # Mean curves can hide opposing lineage deviations at weak selection.
            ax.fill_between(c, np.maximum(0, f - sd), np.minimum(1, f + sd),
                            color=f"C{i}", alpha=0.12, linewidth=0)
            ax.plot(c, f, color=f"C{i}")
            items.append((c[anchor], f[anchor], f"s={r['s']:g}", f"C{i}"))
        ax.set(xlabel="Contrast", ylabel="Response", ylim=(0, 1.03), xlim=(-1, 1))
        ax.set_title("Mean +/- replicate SD", fontsize=6, family="monospace")
        labels(ax, items)
        ax = axes[1]
        items = []
        for i, N in enumerate(sorted({r["N"] for r in data["equilibrium"]})):
            rows = sorted((r for r in data["equilibrium"] if r["N"] == N), key=lambda r: r["Ns"])
            x, y, sd = [np.array([r[k] for r in rows]) for k in ("Ns", "mean", "replicate_sd")]
            ax.plot(x, y, ".-", color=f"C{i}")
            ax.fill_between(x, y - sd, y + sd, color=f"C{i}", alpha=0.12, linewidth=0)
            items.append((x[-1], y[-1], f"N={N}", f"C{i}"))
        for i, threshold in enumerate((0.95, 0.99), 4):
            ax.axhline(threshold, color=f"C{i}", linestyle=":", lw=0.6)
            items.append((ax.get_xlim()[1], threshold, f"{threshold:.2f}", f"C{i}"))
        ax.set(xscale="log", xlabel=r"$N\,s$", ylabel=r"Efficiency $I/I_{\max}$")
        labels(ax, items, gap=0.11)
        ax = axes[2]
        selected = next(r for r in data["drift"] if r["s"] == 1000)
        neutral = next(r for r in data["drift"] if r["s"] == 0 and r["source_s"] == 1000)
        ts = [r["t"] for r in selected["records"]]
        items = []
        for layer in range(4):
            y = [r["state_divergence"][layer] for r in selected["records"]]
            ax.plot(ts, y, color=f"C{layer}")
            items.append((ts[-1], y[-1], f"x{layer}", f"C{layer}"))
        for row, key, label, color, style in ((selected, "response_divergence", "f (MSE)", "C4", "-"),
                                             (neutral, "state_divergence", "neutral x3", "C5", "--")):
            y = [r[key][-1] if key == "state_divergence" else r[key] for r in row["records"]]
            ax.plot(ts, y, color=color, linestyle=style)
            items.append((ts[-1], y[-1], label, color))
        ax.set(xlabel="Generation after fork", ylabel="Divergence", ylim=(0, None))
        labels(ax, items, gap=0.12)
        ax.set_title("N=100, s=1000", fontsize=6, family="monospace")
        title = f"{'Quick' if data['quick'] else 'Full'}: DCT code, noise={env['sigma_n']:.4g}, reference/CDF={env['I_max'] / env['cdf_information']:.3f}"
        fig.suptitle(title, fontsize=7, family="monospace", y=0.99)
        fig.savefig(OUTPUT / "code.pdf")
        fig.savefig(OUTPUT / "code.png", dpi=200)
        plt.close(fig)


def check(seed=0, **environment):
    start = perf_counter()
    channel = Channel(seed=seed, **environment)
    calibration = channel.calibrate()
    rng = np.random.default_rng(121)
    z_batch = rng.normal(size=(5, 16))
    f = channel.response(z_batch)
    assert np.all(np.diff(f) > 0) and np.all(f[:, 0] == 0) and np.all(f[:, -1] == 1)
    assert abs(float(channel.information(np.full(64, 0.5)))) < 1e-12
    assert channel.cdf_information <= np.log(64)
    random_f = np.cumsum(rng.exponential(size=(8, 63)), axis=1)
    random_f = np.column_stack([np.zeros(8), random_f]) / random_f[:, -1:]
    assert np.max(channel.information(random_f)) <= channel.cdf_information + 1e-8
    assert np.allclose(channel.Phi.T @ channel.Phi, np.eye(16), atol=1e-12)
    assert np.allclose(f, channel.response(z_batch * np.arange(1, 6)[:, None]), atol=1e-12)
    assert np.allclose(channel.response(channel.projected_z), channel.projected_response, atol=1e-12)
    assert calibration["projection_residual"] < 0.01
    direct = calibration["direct_optimization"]
    assert direct["success"]
    assert direct["cdf_near_optimal"]
    assert channel.cdf_information / calibration["best_z_information"] >= 0.99
    assert direct["gain_nats"] >= -1e-8
    assert np.min(np.diff(direct["response"])) >= -1e-8
    assert direct["information"] <= -np.sum(channel.p * np.log(channel.p))
    assert np.isclose(float(channel.fitness(channel.I_max, 1000)), 1)
    assert channel.I_max <= np.log(64)
    assert np.isclose(np.linalg.norm(channel.best_z), 1)
    assert np.isclose(float(channel.information(channel.response(channel.best_z))), calibration["best_z_information"])
    assert np.isclose(channel.I_max, max(channel.cdf_information, calibration["best_z_information"]))
    # Analytic information/readout gradient, including normalization terms.
    z = rng.normal(size=16)
    _, grad = channel.objective_z(z)
    eps = 1e-5
    numeric = np.array([(channel.objective_z(z + eps * v)[0] - channel.objective_z(z - eps * v)[0]) / (2 * eps) for v in np.eye(16)])
    assert np.allclose(grad, numeric, atol=2e-8, rtol=1e-5)
    finer_params = dict(channel.params, samples_per_sigma=8)
    finer = Channel(**finer_params)
    assert np.allclose(channel.information(f), finer.information(f), atol=1e-7)
    config = dict(L=3, N=5, s=100, K=3, d=16, u=1e-3, sigma=0.1, seed=seed, every=2, T_adapt=8)
    row, _ = adapt(config, channel)
    population = normalize(rng.normal(size=(5, 16 + 3 * 16**2)), 16)
    fork_config = {k: v for k, v in config.items() if k != "T_adapt"}
    fork_config.update(T=8, init="gaussian", tanh=False, source_s=100)
    selected = fork(fork_config, population, channel)
    neutral = fork(dict(fork_config, s=0), population, channel)
    add_omega([selected, neutral])
    for arm in (selected, neutral):
        assert np.allclose(arm["records"][0]["state_divergence"], 0, atol=1e-12)
        assert arm["records"][0]["response_divergence"] < 1e-25
    assert selected["records"][0]["mean_response"] == neutral["records"][0]["mean_response"]
    # Cache agrees with recomputation after a selected generation.
    g = np.repeat(population[None], 3, axis=0)
    info = channel.information(channel.response(forward(g, 16)[-1]))
    g, cached = step(g, info, channel, 1000, 16, 1e-3, 0.1, rng)
    assert np.allclose(cached, channel.information(channel.response(forward(g, 16)[-1])), atol=1e-12)
    from drift import step as original_step
    neutral_g, _ = step(g, cached, channel, 0, 16, 1e-3, 0.1, np.random.default_rng(7))
    original_g = original_step(g, np.ones(16), 0, 16, 1e-3, 0.1, False, np.random.default_rng(7))
    assert np.array_equal(neutral_g, original_g)
    fs = channel.response(forward(g, 16)[-1]).mean(axis=1)
    pair_mse = np.mean([np.mean((fs[i] - fs[j])**2) for i in range(3) for j in range(i)])
    assert np.isclose(drift_measure(g, channel, 16)["response_divergence"], pair_mse)
    print(f"check ok ({perf_counter() - start:.1f}s): sigma_n={channel.sigma_n:.6g} (requested {channel.requested_sigma_n:g}); CDF I={channel.cdf_information:.6f}; best z I={calibration['best_z_information']:.6f}; I_max={channel.I_max:.6f}")
    print(f"Unrestricted direct I={direct['information']:.6f}; CDF gap={direct['relative_gap']:.2%}; within 1%: {direct['cdf_near_optimal']}.")
    print(f"Projected optimum I={calibration['projected_information']:.9f} ({calibration['projected_cdf_efficiency']:.8%} of CDF); residual={calibration['projection_residual']:.3g}; CDF-fit KS={calibration['cdf_fit_ks']:.6f}.")
    print(f"Fixed gain={channel.gain:.6f}; |z_star|={np.linalg.norm(channel.z_star):.6f}; mean ancestral top norm={channel.ancestor_top_norm:.6f}.")
    print("Noise trials: " + "; ".join(f"{trial['sigma_n']:.6g}: gap {trial['relative_gap']:.2%}" for trial in calibration["noise_trials"]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "quick", "figure", "run"), default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--distribution", choices=("laplace", "student"), default="student")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--df", type=float, default=3)
    parser.add_argument("--sigma-n", type=float, default=0.02)
    parser.add_argument("--gain", type=float, default=None, help="Fixed directional readout gain; default calibrated to ancestral scale")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    env = dict(distribution=args.distribution, scale=args.scale, df=args.df, sigma_n=args.sigma_n, gain=args.gain)
    if args.command == "check":
        check(seed=args.seed, **env)
    elif args.command == "figure":
        figure()
    else:
        main(quick=args.command == "quick", seed=args.seed, workers=args.workers, **env)
