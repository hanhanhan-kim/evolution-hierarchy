"""Experiment I2: annotation and simulated synapse-detection noise floors.

python noise.py check [--data /tmp/neuroevo-samples]
python noise.py run --data /root/data [--observed output/connectome.json]
python noise.py figure [output/noise.json] [--out output/noise]
Use --out output/noise-sample.json for sample data; its summary and figures use
the same stem. Experiment I and its outputs are never modified.

White A is the anterior subset, inferred from coverage: no numbered ventral-cord
motor neurons, unlike whole. Its neuronal chemical edges are contained in whole,
but some weights differ even on these endpoints. Use A, not the compiled whole,
for the N2U comparison. The supplied provenance identifies the anterior series
as N2U; the tables have no specimen/section metadata to independently verify it.
Normalize names and identify bilateral classes exactly as I, intersect individual
neurons BEFORE merging L/R, then call I's compare. Changed annotation conventions
and potentially differing section coverage make this an upper-bound proxy for
modern annotation noise, not a calibrated error estimate.

Each independent simulated copy has Binomial(weight, recall) true detections.
Conditional on their total D, add Poisson(D*(1-precision)/precision) false
positives: 95% distributed over reference edges proportional to reference weight,
5% uniformly over directed reference non-edges among the same nodes, including
self-pairs (I retains type self-edges). For a complete graph all false positives
go to existing edges. Poisson splitting implements this without individual
synapse arrays or dense adjacency matrices. Precision is a ratio of expected
counts, not a constraint on each realization. Node counts/identities stay fixed:
this model cannot estimate cell identification, segmentation or typing errors.

Simulation uses I's FW-R restricted to atlas-shared types, and I's merged CE8.
Thresholds are read from I's observed JSON. I's input normalization, common-edge
strength ranks, clipping and undefined-correlation behavior are unchanged. Floors
are replicate means; grid intervals are min/max of the 16 setting means, NOT CIs.
Central recall=precision=0.85 is simulated separately. Negative excess is retained.
Subtracting the same floor preserves category differences algebraically; these
descriptive residuals are not an additive decomposition of biological distance.
"""

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import connectome as experiment

OUTPUT = experiment.OUTPUT
LEVELS = ("cell_type", "connection", "strength")
GRID = (0.7, 0.8, 0.9, 0.95)


def load_reannotation(data):
    tables, audits = {}, {}
    for name, filename in (("A", "aconnectome_white_1986_A.csv"),
                           ("whole", "aconnectome_white_1986_whole.csv"),
                           ("Cook", "cel_n2u_chemical.csv")):
        path = experiment.locate(data, "worm/" + filename)
        # A has a space-separated header above tab-separated records.
        frame = pd.read_csv(path, sep="," if name == "Cook" else r"\s+")
        audit = dict(file=str(path), input_rows=len(frame))
        if name != "Cook":
            chemical = frame.type.astype("string").str.strip().str.lower().eq("chemical")
            audit["nonchemical_rows"] = int((~chemical).sum())
            frame = frame.loc[chemical].rename(columns={"pre": "source", "post": "target", "synapses": "weight"})
        frame = experiment.clean_edges(frame)
        audit["chemical_rows"] = len(frame)
        for col in ("source", "target"):
            frame[col] = experiment.worm_names(frame[col])
        names = pd.concat([frame.source, frame.target]).drop_duplicates()
        audit["excluded_names"] = sorted(names.loc[~experiment.neuronal(names)].tolist())
        audit["identified_names"] = sorted(names.loc[experiment.neuronal(names)].tolist())
        frame = frame.loc[experiment.neuronal(frame.source) & experiment.neuronal(frame.target)]
        tables[name], audits[name] = frame, audit
        audit.update(neuronal_edges=len(experiment.pair_totals(frame)), synapses=float(frame.weight.sum()))
    inventory = set().union(*(set(a["identified_names"]) for a in audits.values()))
    bilateral = {n[:-1] for n in inventory if n.endswith("L") and n[:-1] + "R" in inventory}
    mapping = {n: n[:-1] if n[-1:] in ("L", "R") and n[:-1] in bilateral else n for n in inventory}
    shared = sorted(set(audits["A"]["identified_names"]) & set(audits["Cook"]["identified_names"]))
    if not shared:
        raise ValueError("No shared White A / Cook neurons")
    nets = {}
    for name, frame in tables.items():
        sub = frame.loc[frame.source.isin(shared) & frame.target.isin(shared)]
        if sub.empty:
            raise ValueError(f"No {name} chemical edges over shared White A / Cook neurons")
        merged = sub.assign(source=sub.source.map(mapping), target=sub.target.map(mapping))
        nets[name] = experiment.network(pd.Series(1, index=sorted({mapping[n] for n in shared})),
                                        experiment.pair_totals(merged))
        audits[name].update(shared_neuron_edges=len(experiment.pair_totals(sub)),
                            merged_edges=len(nets[name]["edges"]),
                            dropped_names=sorted(set(audits[name]["identified_names"]) - set(shared)))
    a, whole = (experiment.pair_totals(tables[k]) for k in ("A", "whole"))
    common = a.index.intersection(whole.index)
    audit = dict(files=audits, shared_neurons=shared, shared_classes=nets["A"]["counts"].index.tolist(),
                 A_edges_in_whole=len(common), A_edges_absent_from_whole=len(a.index.difference(whole.index)),
                 A_whole_changed_weights=int((a.loc[common] != whole.loc[common]).sum()),
                 whole_only_neurons=sorted(set(audits["whole"]["identified_names"]) - set(audits["A"]["identified_names"])),
                 interpretation="A is anterior coverage consistent with N2U; whole adds ventral cord/tail/pharyngeal coverage. "
                                "N2U identity comes from supplied provenance, not specimen metadata in these tables.")
    return nets, audit


def reference(net, allowed=None):
    nodes = net["counts"].index.sort_values()
    if allowed is not None:
        nodes = nodes.intersection(allowed).sort_values()
    edges = net["edges"]
    pre, post = (nodes.get_indexer(edges.index.get_level_values(i)) for i in (0, 1))
    keep = (pre >= 0) & (post >= 0)
    ids, weights = pre[keep] * len(nodes) + post[keep], edges.to_numpy()[keep]
    if not len(weights) or not np.allclose(weights, np.rint(weights), rtol=0, atol=1e-9):
        raise ValueError("Simulation needs a nonempty graph with integer synapse counts")
    order = np.argsort(ids)
    return dict(nodes=nodes.tolist(), ids=ids[order], weights=np.rint(weights[order]).astype(np.int64))


def noisy_copy(ref, recall, precision, rng, nonedge_fraction=0.05):
    ids, weights, n = ref["ids"], ref["weights"], len(ref["nodes"])
    observed = rng.binomial(weights, recall)
    expected = observed.sum() * (1 - precision) / precision
    absent = n * n - len(ids)
    fraction = nonedge_fraction if absent else 0
    detected = observed + rng.poisson(expected * (1 - fraction) * weights / weights.sum())
    # Map uniformly sampled ranks in the complement to flat directed edge IDs.
    # This is O(edges + false positives) storage, even for thousands of types.
    size = rng.poisson(expected * fraction)
    ranks = rng.integers(absent, size=size) if size else np.empty(0, dtype=np.int64)
    extra = ranks + np.searchsorted(ids - np.arange(len(ids)), ranks, side="right")
    extra, counts = np.unique(extra, return_counts=True)
    keep = detected > 0
    return np.concatenate([ids[keep], extra]), np.concatenate([detected[keep], counts]).astype(float)


def pair_metrics(ref, a, b, threshold):
    ids = np.union1d(a[0], b[0])
    wa, wb = np.zeros(len(ids)), np.zeros(len(ids))
    wa[np.searchsorted(ids, a[0])], wb[np.searchsorted(ids, b[0])] = a[1], b[1]
    n = len(ref["nodes"])
    row = experiment.measure(ids // n, ids % n, wa, wb, np.ones(n, dtype=int), [threshold])[0]
    return dict(cell_type=dict(dissimilarity=0.0), connection=row["connection"], strength=row["strength"])


def estimate(values):
    valid = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return dict(mean=float(valid.mean()) if len(valid) else None,
                mcse=float(valid.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else None,
                replicate_interval=np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
                valid_replicates=len(valid))


def simulate(ref, threshold, repeats, rng, nonedge_fraction):
    settings = []
    for recall, precision in [*product(GRID, GRID), (0.85, 0.85)]:
        rows = [pair_metrics(ref, noisy_copy(ref, recall, precision, rng, nonedge_fraction),
                             noisy_copy(ref, recall, precision, rng, nonedge_fraction), threshold)
                for _ in range(repeats)]
        levels = {k: {metric: estimate([row[k][metric] for row in rows])
                      for metric in rows[0][k]} for k in LEVELS}
        settings.append(dict(recall=recall, precision=precision, levels=levels))
    bounds = {}
    for k in LEVELS:
        means = [s["levels"][k]["dissimilarity"]["mean"] for s in settings[:-1]]
        valid = [v for v in means if v is not None]
        bounds[k] = [min(valid), max(valid)] if valid else None
    return dict(reference_nodes=len(ref["nodes"]), reference_edges=len(ref["ids"]),
                reference_synapses=int(ref["weights"].sum()), threshold=threshold, settings=settings,
                central={k: settings[-1]["levels"][k]["dissimilarity"]["mean"] for k in LEVELS},
                grid_range=bounds)


def increasing(values):
    return bool(values[0] < values[1] < values[2]) if all(v is not None for v in values) else None


def excess(rows, simulations, annotation):
    adjusted = []
    for row in rows:
        sim = simulations[row["system"]]
        levels = {}
        for k in LEVELS:
            observed, floor, bounds = row["levels"][k]["dissimilarity"], sim["central"][k], sim["grid_range"][k]
            values = dict(observed=observed, simulated_floor=floor,
                          simulated_excess=observed - floor if observed is not None and floor is not None else None,
                          simulated_excess_interval=[observed - bounds[1], observed - bounds[0]]
                          if observed is not None and bounds is not None else None)
            if row["system"] == "worm":
                floor = annotation["levels"][k]["dissimilarity"]
                values.update(reannotation_floor=floor,
                              reannotation_excess=observed - floor if observed is not None and floor is not None else None)
            levels[k] = values
        tests = {method: increasing([levels[k][method + "_excess"] for k in LEVELS])
                 for method in (("simulated", "reannotation") if row["system"] == "worm" else ("simulated",))}
        # Test each grid setting jointly, rather than combining unrelated endpoints.
        grid = [increasing([levels[k]["observed"] - s["levels"][k]["dissimilarity"]["mean"]
                            if levels[k]["observed"] is not None and s["levels"][k]["dissimilarity"]["mean"] is not None
                            else None for k in LEVELS]) for s in sim["settings"][:-1]]
        adjusted.append(dict(name=row["name"], system=row["system"], category=row["category"],
                             threshold=row["threshold"], levels=levels, strictly_increasing=tests,
                             increasing_grid_settings=sum(v is True for v in grid),
                             valid_grid_settings=sum(v is not None for v in grid)))
    return adjusted


def contrasts(rows):
    results = []
    for system, label, higher, lower in (
            ("fly", "between-animal > within-animal", ["between-animal"], ["within-animal"]),
            ("worm", "between-animal > within-animal", ["within-CE", "within-PP"], ["within-animal"]),
            ("worm", "between-species > within-species", ["between-species"], ["within-CE", "within-PP"])):
        for method in (("simulated", "reannotation") if system == "worm" else ("simulated",)):
            levels = {}
            for k in LEVELS:
                means = []
                for categories in (higher, lower):
                    values = [r["levels"][k][method + "_excess"] for r in rows
                              if r["system"] == system and r["category"] in categories]
                    means.append(estimate(values)["mean"])
                difference = means[0] - means[1] if all(m is not None for m in means) else None
                levels[k] = dict(higher_mean=means[0], lower_mean=means[1], difference=difference,
                                 holds=difference > 0 if difference is not None else None)
            results.append(dict(system=system, question=label, method=method, levels=levels))
    return results


def category_means(rows):
    results = []
    for system, category in sorted({(r["system"], r["category"]) for r in rows}):
        selected = [r for r in rows if r["system"] == system and r["category"] == category]
        for method in (("simulated", "reannotation") if system == "worm" else ("simulated",)):
            values = {k: estimate([r["levels"][k][method + "_excess"] for r in selected])["mean"] for k in LEVELS}
            results.append(dict(system=system, category=category, method=method, comparisons=len(selected),
                                excess=values, strictly_increasing=increasing(list(values.values()))))
    return results


def summary(payload):
    def number(value):
        return "NA" if value is None else f"{value:.4f}"

    def triple(values):
        return "/".join(number(v) for v in values)

    annotation = payload["reannotation"]
    audit = annotation["audit"]
    lines = ["Experiment I2: measurement-error floors (cell type / connection / strength)",
             f"Data: {payload['data']}; observed: {payload['observed_path']} ({payload['observed_data']})",
             f"Replicates per setting={payload['repeats']}; seed={payload['seed']}; sample={payload['sample']}",
             f"White A/Cook: {len(audit['shared_neurons'])} shared neurons, {len(audit['shared_classes'])} merged classes; "
             + triple(annotation["comparison"]["levels"][k]["dissimilarity"] for k in LEVELS),
             f"A/whole audit: {audit['A_edges_in_whole']} shared neuronal chemical edges; "
             f"{audit['A_edges_absent_from_whole']} A edges absent from whole; {audit['A_whole_changed_weights']} changed weights.",
             audit["interpretation"], "", "Simulation means (grid plus central; MC uncertainty retained in JSON)",
             " r     p       fly cell/connection/strength       worm cell/connection/strength"]
    for fly, worm in zip(payload["simulations"]["fly"]["settings"], payload["simulations"]["worm"]["settings"]):
        lines.append(f"{fly['recall']:.2f}  {fly['precision']:.2f}      "
                     + "      ".join(triple(s["levels"][k]["dissimilarity"]["mean"] for k in LEVELS) for s in (fly, worm)))
    lines += ["", "Excess = observed - floor; intervals span grid means (not confidence intervals)."]
    for row in payload["comparisons"]:
        lines.append(f"{row['name']:13} observed " + triple(row["levels"][k]["observed"] for k in LEVELS))
        for method, test in row["strictly_increasing"].items():
            lines.append(f"  {method:13} " + triple(row["levels"][k][method + "_excess"] for k in LEVELS)
                         + f"; cell < connection < strength: {test}")
        bounds = [row["levels"][k]["simulated_excess_interval"] for k in LEVELS]
        lines.append("  grid interval " + "/".join("NA" if v is None else f"[{v[0]:.4f},{v[1]:.4f}]" for v in bounds)
                     + f"; increasing at {row['increasing_grid_settings']}/{row['valid_grid_settings']} valid grid settings")
    lines += ["", "Category mean excess: cell / connection / strength (unweighted comparison means)."]
    for row in payload["category_means"]:
        lines.append(f"{row['system']} {row['category']} {row['method']}: "
                     + triple(row["excess"][k] for k in LEVELS)
                     + f"; cell < connection < strength: {row['strictly_increasing']}")
    lines += ["", "Category contrasts: unweighted means of available comparisons, not significance tests."]
    for row in payload["contrasts"]:
        lines.append(f"{row['system']} {row['method']}: {row['question']} (cell/connection/strength): "
                     + "/".join(str(row["levels"][k]["holds"]) for k in LEVELS)
                     + "; differences=" + triple(row["levels"][k]["difference"] for k in LEVELS))
    lines += ["", *payload["notes"]]
    return "\n".join(lines) + "\n"


def run(data="/root/data", out=OUTPUT / "noise.json", observed=OUTPUT / "connectome.json",
        seed=0, repeats=100, nonedge_fraction=0.05):
    start = perf_counter()
    out, observed = Path(out), Path(observed)
    original = json.loads(observed.read_text())
    sample = "sample" in str(data).lower()
    if sample and out.stem == "noise":
        raise ValueError("Sample data require an alternate output stem, e.g. noise-sample.json")
    if repeats < 2 or not 0 <= nonedge_fraction <= 1:
        raise ValueError("Require repeats >= 2 and 0 <= nonedge_fraction <= 1")
    thresholds = {}
    for system in ("fly", "worm"):
        values = {r["threshold"] for r in original["comparisons"] if r["system"] == system}
        if len(values) != 1 or not all(np.isfinite(t) and t > 0 for t in values):
            raise ValueError(f"Require one finite positive primary {system} threshold in observed JSON")
        thresholds[system] = values.pop()
    rng = np.random.default_rng(seed)
    nets, audit = load_reannotation(data)
    annotation = experiment.compare(nets["A"], nets["Cook"], "White A/Cook N2U", "worm", "reannotation",
                                    [thresholds["worm"]], thresholds["worm"], rng, resamples=0)
    whole = experiment.compare(nets["whole"], nets["Cook"], "White whole/Cook (sensitivity only)", "worm", "compiled",
                               [thresholds["worm"]], thresholds["worm"], rng, resamples=0)
    fly, allowed = experiment.load_fly(data)
    worm, _, _ = experiment.load_worm(data)
    simulations = {}
    for system, ref in (("fly", reference(fly["FW-R"], allowed)), ("worm", reference(worm["CE8"]))):
        simulations[system] = simulate(ref, thresholds[system], repeats, rng, nonedge_fraction)
        print(f"Simulated {system}: {len(ref['nodes'])} nodes, {len(ref['ids'])} edges", flush=True)
    rows = excess(original["comparisons"], simulations, annotation)
    notes = [
        "White A/Cook uses identical-tissue provenance supplied with the task; tables alone cannot establish section equivalence. "
        "1986/2019 annotation conventions differ: treat this as an upper-bound proxy for modern pure annotation noise.",
        "White whole/Cook on the same neurons is saved as sensitivity only, never used as the floor: whole compiles animals.",
        "Names, neuronal filter, L/R classes, shared-node metrics and strength clipping follow connectome.py. "
        "White/Cook intersect individual neurons before L/R merging; worm cell dissimilarity is zero by construction.",
        f"Model: independent Binomial(w,r) detections; conditional expected false positives D*(1-p)/p; "
        f"Poisson allocation with {1-nonedge_fraction:.1%} proportional to reference edge weight and {nonedge_fraction:.1%} "
        "uniform over directed reference non-edges (self-pairs included). If no non-edges exist, use existing edges only.",
        "Synapse-only simulation holds neurons and type counts fixed. Zero cell-type floor does not establish error-free cell identification.",
        "Grid r,p = 0.7,0.8,0.9,0.95; central r=p=0.85. User-supplied literature ranges: Buhmann et al. 2021 "
        "automated fly synapses roughly 0.7-0.9; Scheffer et al. 2020 proofread T-bars/PSDs about 0.8-0.9. "
        "These motivate sensitivity settings, not validated independent edge-error rates. No network/literature retrieval performed.",
        "Manual worm EM annotation error rates are not well established; the same grid is exploratory.",
        "Grid bounds span setting means, not sampling CIs. JSON separately stores Monte Carlo SE and replicate percentiles. "
        "Undefined strength correlations stay null; valid replicate counts accompany all means.",
        "Floors use one FW-R and one CE8 reference; transferring them to other coverage, species or within-side subnetworks is an assumption. "
        "FW-R/HB uses the requested common-quality floor, not separate calibrated FlyWire and hemibrain error models.",
        "Excess is a signed descriptive subtraction, not a calibrated biological distance. The three metrics have different scales. "
        "Do not interpret negative excess as negative biological variation.",
        "A common per-system floor cancels from category differences, at every grid setting. "
        "Worm between-animal here means within-CE/within-PP pairs; between-species is compared with those within-species pairs.",
        "Observed values are read from the saved experiment I JSON, not recomputed. Its bootstrap uncertainty is not propagated into excess intervals."]
    if sample:
        notes.append("SAMPLE RUN: fly noise uses the small sample graph; observed values retain their saved provenance. "
                     "Fly excess is a pipeline illustration until the full pod run. Worm input files are complete per supplied provenance.")
    payload = dict(data=str(Path(data).resolve()), observed_path=str(observed.resolve()), observed_data=original["data"],
                   sample=sample, seed=seed, repeats=repeats, nonedge_fraction=nonedge_fraction,
                   references=dict(fly=dict(name="FW-R", **fly["FW-R"]["audit"]),
                                   worm=dict(name="CE8", **worm["CE8"]["audit"])),
                   reannotation=dict(audit=audit, comparison=annotation, whole_sensitivity=whole),
                   simulations=simulations, comparisons=rows, category_means=category_means(rows),
                   contrasts=contrasts(rows), notes=notes,
                   seconds=perf_counter() - start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    report = summary(payload)
    out.with_name(out.stem + "-summary.txt").write_text(report)
    print(report, end="")
    print(f"Wrote {out} ({payload['seconds']:.1f}s)")
    return payload


def figure(path=OUTPUT / "noise.json", out=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401 (applies the local figure style)

    path = Path(path)
    out = Path(out) if out is not None else path.with_suffix("")
    payload = json.loads(path.read_text())
    if payload["sample"] and out.stem == "noise":
        raise ValueError("Sample figures require an alternate output stem")
    xs = np.arange(3)
    annotation = [payload["reannotation"]["comparison"]["levels"][k]["dissimilarity"] for k in LEVELS]

    def band(ax, system, color, label):
        sim = payload["simulations"][system]
        bounds = np.array([sim["grid_range"][k] if sim["grid_range"][k] is not None else [np.nan, np.nan] for k in LEVELS])
        ax.fill_between(xs, bounds[:, 0], bounds[:, 1], color=color, alpha=0.2, lw=0)
        ax.plot(xs, [sim["central"][k] for k in LEVELS], color=color, ls=":", label=label)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 6, "lines.linewidth": 1, "savefig.bbox": None}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.81, wspace=0.35)
        for ax, letter, title in zip(axes, "ABC", ("Noise floors", "Fly", "Worm")):
            experiment.panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set(ylim=(-0.035, 1.035), yticks=[0, 0.5, 1], xlim=(-0.1, 2.1),
                   xticks=xs, xticklabels=["cell type", "connection", "strength"])
            ax.tick_params(axis="x", rotation=25)
            ax.set_title(title, fontsize=7)
        axes[0].set_ylabel("Dissimilarity")
        band(axes[0], "fly", "C0", "fly simulated")
        band(axes[0], "worm", "C1", "worm simulated")
        axes[0].plot(xs, annotation, "o", color="C2", ms=3, label="White A/Cook")
        band(axes[1], "fly", "C3", "simulated floor")
        for i, row in enumerate(r for r in payload["comparisons"] if r["system"] == "fly"):
            axes[1].plot(xs, [row["levels"][k]["observed"] for k in LEVELS], color=f"C{i}",
                         ls="--" if row["category"] == "within-animal" else "-", label=row["name"])
        axes[2].fill_between(xs, 0, np.array(annotation, dtype=float), color="C4", alpha=0.12, lw=0)
        axes[2].plot(xs, annotation, ":o", color="C4", ms=2, label="White A/Cook")
        for i, (category, label) in enumerate((("within-animal", "L/R"), ("within-CE", "within CE"),
                                               ("within-PP", "within PP"), ("between-species", "CE/PP"))):
            selected = [r for r in payload["comparisons"] if r["system"] == "worm" and r["category"] == category]
            if not selected:
                continue
            values = np.array([[r["levels"][k]["observed"] for k in LEVELS] for r in selected], dtype=float)
            style = "--" if category == "within-animal" else "-"
            for y in values:
                axes[2].plot(xs, y, color=f"C{i}", lw=0.45, alpha=0.3, ls=style)
            mean = [estimate(values[:, j])["mean"] for j in range(3)]
            axes[2].plot(xs, mean, color=f"C{i}", lw=1.5, ls=style, label=label)
        for ax in axes:
            ax.legend(loc="upper left", fontsize=4.8, frameon=False, handlelength=1.5, labelspacing=0.25)
        if payload["sample"]:
            fig.text(0.5, 0.98, "Sample fly floors; saved I observations; complete worm inputs", ha="center", va="top",
                     fontsize=6, family="monospace")
        fig.text(0.5, 0.01, "Bands: grid range. Dotted: central floor. Worm thin lines: pairs; thick: category means.",
                 ha="center", fontsize=5.3, family="monospace")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=200)
        plt.close(fig)
    print(f"Wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


def check(data="/tmp/neuroevo-samples"):
    rng = np.random.default_rng(13)
    n = 30
    ids = np.arange(n * n)[np.arange(n * n) % 3 != 0]
    ref = dict(nodes=list(range(n)), ids=ids, weights=rng.integers(1, 9, size=len(ids)))
    same = pair_metrics(ref, noisy_copy(ref, 1, 1, rng), noisy_copy(ref, 1, 1, rng), 1)
    assert all(abs(same[k]["dissimilarity"]) < 1e-12 for k in LEVELS)
    means = []
    for recall in (1, 0.95, 0.9, 0.8, 0.7):
        values = [pair_metrics(ref, noisy_copy(ref, recall, 1, rng), noisy_copy(ref, recall, 1, rng), 1)
                  ["connection"]["dissimilarity"] for _ in range(100)]
        means.append(np.mean(values))
    assert np.all(np.diff(means) > 0), means
    # Complement sampling adds only genuine non-edges and preserves exact metrics.
    a, b = noisy_copy(ref, 0.8, 0.7, rng, 1), noisy_copy(ref, 0.8, 0.7, rng, 1)
    for edge_ids, weights in (a, b):
        assert len(np.unique(edge_ids)) == len(edge_ids) and np.all((edge_ids >= 0) & (edge_ids < n*n))
        assert np.all(weights > 0)
    def net(copy):
        edge_ids, weights = copy
        index = pd.MultiIndex.from_arrays([edge_ids // n, edge_ids % n], names=["source", "target"])
        return experiment.network(pd.Series(1, index=range(n)), pd.Series(weights, index=index))
    direct = experiment.compare(net(a), net(b), "synthetic", "fly", "test", [3], 3, rng, resamples=0)
    sparse = pair_metrics(ref, a, b, 3)
    assert all(np.isclose(sparse[k]["dissimilarity"], direct["levels"][k]["dissimilarity"]) for k in LEVELS)
    nets, audit = load_reannotation(data)
    assert audit["shared_neurons"] and all(len(net["edges"]) > 0 for net in nets.values())
    assert nets["A"]["counts"].index.equals(nets["Cook"]["counts"].index)
    print(f"check ok: perfect detection, monotonic mean presence error {np.round(means, 4)}, "
          f"sparse metrics match I, chemical White/Cook shared neurons={len(audit['shared_neurons'])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("check")
    checker.add_argument("--data", default="/tmp/neuroevo-samples")
    runner = commands.add_parser("run")
    runner.add_argument("--data", default="/root/data")
    runner.add_argument("--out", type=Path, default=OUTPUT / "noise.json")
    runner.add_argument("--observed", type=Path, default=OUTPUT / "connectome.json")
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--repeats", type=int, default=100)
    runner.add_argument("--nonedge-fraction", type=float, default=0.05)
    renderer = commands.add_parser("figure")
    renderer.add_argument("json", nargs="?", type=Path, default=OUTPUT / "noise.json")
    renderer.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check(args.data)
    elif args.command == "figure":
        figure(args.json, args.out)
    else:
        run(args.data, args.out, args.observed, args.seed, args.repeats, args.nonedge_fraction)
