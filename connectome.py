"""Experiment I: variation in cell identity, wiring and synaptic strength.

python connectome.py check
python connectome.py run --data /root/data [--out output/connectome.json]
python connectome.py figure [output/connectome.json] [--out output/connectome]
An alternate JSON stem also names its summary and figures; samples never need
the final output names. All external tables are read with pandas, never executed.

FlyWire sides use neurons' soma-side annotations and within-side edges (including
type self-edges), not neuropil side. Neuropils are summed to neuron pairs before
type aggregation. All comparisons use exact hemibrain_type/type matches present
in both atlases, then in both members. Hemibrain is the supplied partial volume;
truncation and hemisphere asymmetry remain confounds, not corrected noise.

Worm _roi denotes a reconstruction/region label, not a new homolog: the samples
contain both X and X_roi, with different skeleton IDs. We strip only this suffix
and sum their edges; this assumes the fragments are complementary. The grouped
exports duplicate the edge exports, so only one is read. In the raw sample tables,
count equals the number of nonmissing Sections records and sum equals summed
Sections: use count (synapse records), not section extent. Keep c/C chemical rows.
Retain conservative neuron spellings, excluding glia, muscles and uncertain or
descriptive labels; normalize motor-neuron zero padding. Merge L/R only when both
counterpart names occur in the pooled neuron inventory (AVL, AQR, PVR stay intact).
Unsided names such as ASK join the corresponding bilateral class. Main comparisons
use these classes; within-animal baselines use bilateral neurons' within-side
edges only. Presence comes from edge endpoints: isolated neurons are unobservable.
After pairwise intersection worm level 1 is zero by construction; dropped names
are coverage information, never evolutionary differences.

Strength uses common above-threshold edges, but postsynaptic input includes ALL
positive edges among shared nodes, including subthreshold and noncommon edges.
SDs use ddof=0; constant ranks or fewer than two common edges give null correlation.
Raw 1-rho ranges over [0,2], so the requested [0,1] figure scale clips it at 1;
both values are saved. These three dissimilarities are operationally different
metrics, not a calibrated common biological distance.

95% percentile CIs use 200 paired node bootstraps: sample n shared nodes with
replacement, retain directed edges with multiplicity m_pre*m_post, recompute
input normalization (m_pre weights) and weighted ranks in each resample. Thresholds
apply to original edge counts, not bootstrap multiplicities. This preserves node
dependence, uses sparse edge arrays, and never expands a dense neuron matrix.
CIs describe sampling of nodes, not uncertainty from independent animal replicates.
"""

import json
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "output"


def locate(data, *names):
    for name in names:
        path = Path(data) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Under {data}, expected one of: {', '.join(names)}")


def clean_edges(frame):
    """Validate before summing: negative/unknown weights must not disappear."""
    if frame[["source", "target"]].isna().any().any():
        raise ValueError("Missing edge endpoint")
    frame = frame.copy()
    frame["weight"] = pd.to_numeric(frame.weight, errors="raise").astype(float)
    if not np.isfinite(frame.weight).all() or (frame.weight < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    return frame.loc[frame.weight > 0, ["source", "target", "weight"]]


def pair_totals(frame):
    return frame.groupby(["source", "target"], sort=False, observed=True).weight.sum()


def network(counts, edges, **audit):
    return dict(counts=counts.sort_index(), edges=edges, audit=audit)


def type_edges(pairs, labels):
    """Map integer neuron IDs once and group vectorially; no loop over edges."""
    pre, post = pairs.index.get_level_values(0), pairs.index.get_level_values(1)
    keep = pre.isin(labels.index) & post.isin(labels.index)
    frame = pd.DataFrame(dict(source=pd.Categorical(pre[keep].map(labels)),
                              target=pd.Categorical(post[keep].map(labels)),
                              weight=pairs.to_numpy()[keep]))
    return pair_totals(frame)


def load_fly(data):
    annotation = locate(data, "flywire/Supplemental_file1_neuron_annotations.tsv",
                        "flywire/annotations_sample.tsv")
    connection = locate(data, "flywire/proofread_connections_783.feather",
                        "flywire/connections_sample.feather")
    hb = "hemibrain/exported-traced-adjacencies-v1.2/"
    neurons = locate(data, hb + "traced-neurons.csv", "hemibrain/traced-neurons.csv")
    edges = locate(data, hb + "traced-total-connections.csv",
                   "hemibrain/traced-total-connections.csv")
    fw = pd.read_csv(annotation, sep="\t", usecols=["root_id", "side", "hemibrain_type"],
                     dtype={"root_id": "uint64", "side": "string", "hemibrain_type": "string"})
    hn = pd.read_csv(neurons, usecols=["bodyId", "type"],
                     dtype={"bodyId": "uint64", "type": "string"})
    fw.hemibrain_type = fw.hemibrain_type.str.strip().replace("", pd.NA)
    hn.type = hn.type.str.strip().replace("", pd.NA)
    fw.side = fw.side.str.strip().str.lower()
    if fw.root_id.duplicated().any() or hn.bodyId.duplicated().any():
        raise ValueError("Duplicate neuron IDs in annotations")
    allowed = pd.Index(fw.hemibrain_type.dropna().unique()).intersection(hn.type.dropna().unique())
    # Project just three columns; discard irrelevant neurons before the large groupby.
    table = pd.read_feather(connection, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    table.columns = ["source", "target", "weight"]
    table = clean_edges(table)
    total_rows = len(table)
    ids = fw.loc[fw.hemibrain_type.isin(allowed) & fw.side.isin(["left", "right"]), "root_id"]
    table = table.loc[table.source.isin(ids) & table.target.isin(ids)]
    pairs = pair_totals(table)
    retained_rows = len(table)
    del table
    nets = {}
    for side in ("left", "right"):
        cells = fw.loc[fw.side.eq(side) & fw.hemibrain_type.notna()]
        labels = cells.loc[cells.hemibrain_type.isin(allowed)].set_index("root_id").hemibrain_type
        nets[f"FW-{side[0].upper()}"] = network(
            cells.hemibrain_type.value_counts(), type_edges(pairs, labels),
            files=[str(annotation), str(connection)], side=side,
            neurons=len(cells), untyped_neurons=int((fw.side.eq(side) & fw.hemibrain_type.isna()).sum()),
            positive_edge_rows=total_rows, atlas_matched_edge_rows=retained_rows,
            atlas_matched_neuron_pairs=len(pairs))
    del pairs
    # CSV chunks keep the full hemibrain table out of memory; grouping is additive.
    labels = hn.loc[hn.type.isin(allowed)].set_index("bodyId").type
    chunks, total_rows = [], 0
    for chunk in pd.read_csv(edges, usecols=["bodyId_pre", "bodyId_post", "weight"],
                             dtype={"bodyId_pre": "uint64", "bodyId_post": "uint64"},
                             chunksize=1_000_000):
        chunk = clean_edges(chunk.rename(columns={"bodyId_pre": "source", "bodyId_post": "target"}))
        total_rows += len(chunk)
        chunk = chunk.loc[chunk.source.isin(labels.index) & chunk.target.isin(labels.index)]
        chunks.append(type_edges(pair_totals(chunk), labels))
    typed = pd.concat(chunks).groupby(level=[0, 1], sort=False, observed=True).sum()
    nets["HB"] = network(hn.type.value_counts(), typed, files=[str(neurons), str(edges)],
                         neurons=int(hn.type.notna().sum()), untyped_neurons=int(hn.type.isna().sum()),
                         positive_edge_rows=total_rows)
    return nets, allowed


def worm_names(names):
    names = names.astype("string").str.strip().str.replace(r"_roi$", "", regex=True)
    return names.str.replace(r"^(DA|DB|DD|VA|VB|VC|VD|AS)0+(\d+)$", r"\1\2", regex=True)


def neuronal(names):
    valid = names.str.fullmatch(r"(?:[A-Z]{3}[DLVR]{0,2}|IL[12][DLVR]{1,2}|(?:DA|DB|DD|VA|VB|VC|VD|AS)\d+)", na=False)
    return valid & ~names.str.startswith(("GLR", "UNK", "HMC"), na=False)


def load_worm(data):
    files = dict(CE7="witvliet_2020_7_ad_chem.csv", CE8="witvliet_2020_8_ad_chem.csv",
                 N2U="cel_n2u_chemical.csv", S14="s14_syn_edges_05_24_24.csv",
                 S15="s15_syn_edges_05_24_24.csv")
    tables, audits = {}, {}
    for name, filename in files.items():
        path = locate(data, "worm/" + filename, "worm/grouped_" + filename)
        frame = pd.read_csv(path)
        audit = dict(file=str(path), input_rows=len(frame))
        if "Synapse_Type" in frame:
            chemical = frame.Synapse_Type.astype("string").str.strip().str.lower().eq("c").fillna(False)
            audit.update(nonchemical_rows=int((~chemical).sum()), weight_column="count")
            frame = frame.loc[chemical].rename(columns={"Source": "source", "Target": "target", "count": "weight"})
        else:
            audit.update(nonchemical_rows=0, weight_column="weight")
        frame = clean_edges(frame)
        raw_names = pd.concat([frame.source, frame.target]).astype("string")
        audit["roi_endpoint_rows"] = int(raw_names.str.endswith("_roi").sum())
        for col in ("source", "target"):
            frame[col] = worm_names(frame[col])
        names = pd.concat([frame.source, frame.target]).drop_duplicates()
        audit["excluded_names"] = sorted(names.loc[~neuronal(names)].tolist())
        keep = neuronal(frame.source) & neuronal(frame.target)
        audit["excluded_edge_rows"] = int((~keep).sum())
        # Count identified neurons even if all their partners were excluded.
        audit["identified_names"] = sorted(names.loc[neuronal(names)].tolist())
        tables[name], audits[name] = frame.loc[keep], audit
    inventory = set().union(*(set(a["identified_names"]) for a in audits.values()))
    bilateral = {n[:-1] for n in inventory if n.endswith("L") and n[:-1] + "R" in inventory}
    mapping = {n: n[:-1] if n[-1:] in ("L", "R") and n[:-1] in bilateral else n for n in inventory}
    nets, baselines = {}, {}
    for name, frame in tables.items():
        audit = audits[name]
        merged = frame.assign(source=frame.source.map(mapping), target=frame.target.map(mapping))
        classes = pd.Index(sorted({mapping[n] for n in audit["identified_names"]}))
        nets[name] = network(pd.Series(1, index=classes), pair_totals(merged), **audit)
        sides = []
        for side in ("L", "R"):
            names = {n for n in audit["identified_names"] if n.endswith(side) and n[:-1] in bilateral}
            sub = frame.loc[frame.source.isin(names) & frame.target.isin(names)]
            sub = sub.assign(source=sub.source.map(mapping), target=sub.target.map(mapping))
            sides.append(network(pd.Series(1, index=sorted(mapping[n] for n in names)), pair_totals(sub)))
        baselines[name] = sides
    return nets, baselines, sorted(bilateral)


def weighted_ranks(values, weights):
    """Average ranks exactly as if each observation were repeated weights times."""
    _, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    return (np.cumsum(mass) - (mass - 1) / 2)[inverse]


def strength(x, y, weights):
    if not len(x):
        return dict(sd_log2_ratio=None, spearman=None, one_minus_spearman=None, dissimilarity=None)
    logs = np.log2(x / y)
    sd = float(np.sqrt(np.average((logs - np.average(logs, weights=weights)) ** 2, weights=weights)))
    rx, ry = weighted_ranks(x, weights), weighted_ranks(y, weights)
    rx -= np.average(rx, weights=weights)
    ry -= np.average(ry, weights=weights)
    denom = np.sqrt(np.sum(weights * rx**2) * np.sum(weights * ry**2))
    rho = float(np.clip(np.sum(weights * rx * ry) / denom, -1, 1)) if denom > 0 else None
    raw = 1 - rho if rho is not None else None
    return dict(sd_log2_ratio=sd, spearman=rho, one_minus_spearman=raw,
                dissimilarity=float(np.clip(raw, 0, 1)) if raw is not None else None)


def measure(pre, post, a, b, multiplicity, thresholds):
    mass = multiplicity[pre] * multiplicity[post]
    keep = mass > 0
    # Duplicated target copies have identical input; only source copies multiply it.
    ia = np.bincount(post, weights=a * multiplicity[pre], minlength=len(multiplicity))
    ib = np.bincount(post, weights=b * multiplicity[pre], minlength=len(multiplicity))
    na = np.divide(a, ia[post], out=np.zeros_like(a), where=ia[post] > 0)
    nb = np.divide(b, ib[post], out=np.zeros_like(b), where=ib[post] > 0)
    rows = []
    for threshold in thresholds:
        ea, eb = (a >= threshold) & keep, (b >= threshold) & keep
        common, union = ea & eb, ea | eb
        ncommon, nunion = int(mass[common].sum()), int(mass[union].sum())
        jaccard = ncommon / nunion if nunion else (1.0 if len(multiplicity) else None)
        rows.append(dict(threshold=threshold,
                         connection=dict(jaccard=jaccard, dissimilarity=1 - jaccard if jaccard is not None else None,
                                         edges_a=int(mass[ea].sum()), edges_b=int(mass[eb].sum()),
                                         common_edges=ncommon, union_edges=nunion),
                         strength=dict(common_edges=ncommon, **strength(na[common], nb[common], mass[common]))))
    return rows


def interval(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(ci95=np.quantile(values, [0.025, 0.975]).tolist() if values else None,
                valid_resamples=len(values))


def compare(a, b, name, system, category, thresholds, primary, rng, resamples=200, allowed=None):
    ca, cb = a["counts"], b["counts"]
    shared = ca.index.intersection(cb.index).sort_values()
    if allowed is not None:
        shared = shared.intersection(allowed).sort_values()
    dropped = {k: c.index.difference(shared).tolist() for k, c in (("a", ca), ("b", cb))}
    counts = dict(shared=len(shared), available_a=len(ca), available_b=len(cb),
                  dropped_a=len(dropped["a"]), dropped_b=len(dropped["b"]))
    table = pd.concat([a["edges"].rename("a"), b["edges"].rename("b")], axis=1).fillna(0)
    pre = shared.get_indexer(table.index.get_level_values(0))
    post = shared.get_indexer(table.index.get_level_values(1))
    keep = (pre >= 0) & (post >= 0)
    pre, post = pre[keep], post[keep]
    wa, wb = table.a.to_numpy(dtype=float)[keep], table.b.to_numpy(dtype=float)[keep]
    logs = np.log2(ca.reindex(shared).to_numpy() / cb.reindex(shared).to_numpy())
    unmatched = int((np.abs(logs) > np.log2(1.5)).sum()) if system == "fly" else 0
    cell = dict(dissimilarity=unmatched / len(shared) if len(shared) else None,
                unmatched=unmatched, matched=len(shared) - unmatched, items=len(shared),
                sd_log2_ratio=float(logs.std()) if len(logs) and system == "fly" else None,
                fraction_over_1_5=unmatched / len(shared) if len(shared) and system == "fly" else None,
                tolerance=1.5 if system == "fly" else "presence",
                counts_a=ca.reindex(shared).astype(int).to_dict(),
                counts_b=cb.reindex(shared).astype(int).to_dict())
    rows = measure(pre, post, wa, wb, np.ones(len(shared), dtype=int), thresholds)
    samples = [[] for _ in thresholds]
    if len(shared):
        for _ in range(resamples):
            m = np.bincount(rng.integers(len(shared), size=len(shared)), minlength=len(shared))
            for sample, row in zip(samples, measure(pre, post, wa, wb, m, thresholds)):
                sample.append(row)
    for row, sample in zip(rows, samples):
        for level, metrics in (("connection", ("jaccard", "dissimilarity")),
                               ("strength", ("sd_log2_ratio", "spearman", "one_minus_spearman", "dissimilarity"))):
            row[level]["bootstrap"] = {metric: interval([r[level][metric] for r in sample]) for metric in metrics}
    selected = next(row for row in rows if row["threshold"] == primary)
    return dict(name=name, system=system, category=category, counts=counts, dropped=dropped,
                shared_items=shared.tolist(), threshold=primary,
                levels=dict(cell_type=cell, connection=selected["connection"], strength=selected["strength"]),
                thresholds=rows)


def summary(payload):
    def number(x):
        return "NA" if x is None else f"{x:.3f}"

    def ci(level):
        bounds = level["bootstrap"]["dissimilarity"]["ci95"]
        return "[NA]" if bounds is None else f"[{bounds[0]:.3f},{bounds[1]:.3f}]"

    lines = ["Experiment I: shared-node dissimilarity (95% paired node bootstrap CIs)",
             f"Data: {payload['data']}; resamples={payload['resamples']}; seed={payload['seed']}",
             "Pair          category          shared drop A/B  thr    cell    connection [95% CI]    strength [95% CI]      SD(count) SD(weight)  rho"]
    for r in payload["comparisons"]:
        c, e, w = (r["levels"][k] for k in ("cell_type", "connection", "strength"))
        n = r["counts"]
        drop = f"{n['dropped_a']}/{n['dropped_b']}"
        lines.append(f"{r['name']:13} {r['category']:17} {n['shared']:5} {drop:>8} {r['threshold']:4g} "
                     f"{number(c['dissimilarity']):>7} {number(e['dissimilarity']):>7} {ci(e):17} "
                     f"{number(w['dissimilarity']):>7} {ci(w):17} "
                     f"{number(c['sd_log2_ratio']):>9} {number(w['sd_log2_ratio']):>10} {number(w['spearman']):>6}")
    lines += ["", "Fly threshold sensitivity: pair / threshold / 1-Jaccard [95% CI] / common,union edges"]
    for r in payload["comparisons"]:
        if r["system"] == "fly":
            for t in r["thresholds"]:
                e = t["connection"]
                lines.append(f"{r['name']:13} {t['threshold']:4g} {number(e['dissimilarity']):>7} {ci(e):17} "
                             f"{e['common_edges']},{e['union_edges']}")
    lines += ["", *payload["notes"]]
    return "\n".join(lines) + "\n"


def run(data="/root/data", out=OUTPUT / "connectome.json", seed=0, resamples=200,
        fly_threshold=5, worm_threshold=1):
    start = perf_counter()
    out = Path(out)
    rng = np.random.default_rng(seed)
    fly, allowed = load_fly(data)
    worm, baselines, bilateral = load_worm(data)
    comparisons = []
    thresholds = sorted({1, 3, 5, 10, fly_threshold})
    for a, b, category in (("FW-L", "FW-R", "within-animal"),
                            ("FW-R", "HB", "between-animal"), ("FW-L", "HB", "between-animal")):
        comparisons.append(compare(fly[a], fly[b], a + "/" + b, "fly", category,
                                   thresholds, fly_threshold, rng, resamples, allowed))
    for name, (a, b) in baselines.items():
        comparisons.append(compare(a, b, name + " L/R", "worm", "within-animal",
                                   [worm_threshold], worm_threshold, rng, resamples))
    ce, pp = ["CE7", "CE8", "N2U"], ["S14", "S15"]
    pairs = [(a, b, "within-CE") for a, b in combinations(ce, 2)]
    pairs += [("S14", "S15", "within-PP")]
    pairs += [(a, b, "between-species") for a, b in product(ce, pp)]
    for a, b, category in pairs:
        comparisons.append(compare(worm[a], worm[b], a + "/" + b, "worm", category,
                                   [worm_threshold], worm_threshold, rng, resamples))
    notes = ["Shared-node restriction excludes missing reconstruction; dropped A/B are node counts, with identities in JSON.",
             "Worm cell dissimilarity is zero by construction after intersection, not evidence of conservation.",
             "Worm count measures synapse records; sum measures section extent. _roi aliases are summed; bilateral homologs are merged.",
             "Within-animal worm baselines compare bilateral within-side subnetworks; main worm graphs include all merged homolog edges.",
             "Hemibrain truncation, P. pacificus coverage (especially s15), and biological side asymmetry remain confounds.",
             "Strength input totals use all edges within shared nodes. SD uses ddof=0; undefined correlation is NA/null.",
             "Figure strength is clip(1-rho,0,1); raw 1-rho and rho are retained. Metrics are not biologically calibrated.",
             "CIs: paired node resampling, recomputed input totals and weighted tied ranks; 2.5/97.5 percentiles of valid replicates.",
             "Worm: CE7/CE8/N2U = C. elegans; S14/S15 = P. pacificus. Category means in the figure are descriptive."]
    payload = dict(data=str(Path(data).resolve()), seed=seed, resamples=resamples,
                   fly_threshold=fly_threshold, worm_threshold=worm_threshold,
                   fly_atlas_shared_types=len(allowed), worm_bilateral_classes=bilateral,
                   datasets={k: dict(nodes=len(n["counts"]), edges=len(n["edges"]), **n["audit"])
                             for k, n in {**fly, **worm}.items()},
                   notes=notes, comparisons=comparisons, seconds=perf_counter() - start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    text = summary(payload)
    out.with_name(out.stem + "-summary.txt").write_text(text)
    print(text, end="")
    print(f"Wrote {out} ({payload['seconds']:.1f}s)")
    return payload


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(path=OUTPUT / "connectome.json", out=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401 (applies the local figure style)

    path = Path(path)
    out = Path(out) if out is not None else path.with_suffix("")
    payload = json.loads(path.read_text())
    rows = payload["comparisons"]

    def ys(row):
        return np.array([row["levels"][k]["dissimilarity"] for k in ("cell_type", "connection", "strength")], dtype=float)

    def bars(ax, xs, levels, color, alpha=0.7):
        for x, level in zip(xs, levels):
            ci = level["bootstrap"]["dissimilarity"]["ci95"]
            if ci is not None:
                ax.vlines(x, *ci, color=color, lw=0.65, alpha=alpha)
                ax.plot([x, x], ci, "_", color=color, ms=3, alpha=alpha)

    def labels(ax, items):
        items = sorted((y, text, color) for y, text, color in items if y is not None and np.isfinite(y))
        previous = 0
        for i, (y, text, color) in enumerate(items):
            pos = max(previous + 0.09, min(y, 0.94 - 0.09 * (len(items) - i - 1)))
            previous = pos
            ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"),
                        xytext=(1.03, pos), textcoords="axes fraction", color=color,
                        fontsize=5.3, family="monospace", va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.4))

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 6, "lines.linewidth": 1, "savefig.bbox": None}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.065, right=0.89, bottom=0.25, top=0.81, wspace=1.0)
        for ax, letter in zip(axes, "ABC"):
            panel_letter(ax, letter)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set(ylim=(-0.035, 1.035), yticks=[0, 0.5, 1])
        for ax in axes[:2]:
            ax.set(xticks=[0, 1, 2], xticklabels=["cell type", "connection", "strength"], xlim=(-0.1, 2.05))
            ax.tick_params(axis="x", rotation=25)
        axes[0].set_ylabel("Dissimilarity")
        axes[0].set_title(f"Fly (threshold {payload['fly_threshold']:g})", fontsize=7)
        axes[1].set_title("Worm", fontsize=7)
        axes[2].set(xlabel="Synapse threshold", xticks=[1, 3, 5, 10], xlim=(0.6, 10.3))
        axes[2].set_title("Fly connections", fontsize=7)
        ends_a, ends_c = [], []
        for i, row in enumerate(r for r in rows if r["system"] == "fly"):
            color, style = f"C{i}", "--" if row["category"] == "within-animal" else "-"
            y = ys(row)
            axes[0].plot(range(3), y, color=color, ls=style, marker=".", ms=3)
            bars(axes[0], [1, 2], [row["levels"][k] for k in ("connection", "strength")], color)
            ends_a.append((y[-1], row["name"], color))
            ts = row["thresholds"]
            x, y = [t["threshold"] for t in ts], [t["connection"]["dissimilarity"] for t in ts]
            axes[2].plot(x, y, color=color, ls=style, marker=".", ms=3)
            bars(axes[2], x, [t["connection"] for t in ts], color)
            ends_c.append((y[-1], row["name"], color))
        labels(axes[0], ends_a)
        labels(axes[2], ends_c)
        ends = []
        for i, (category, label) in enumerate((("within-animal", "L/R"), ("within-CE", "within CE"),
                                               ("within-PP", "within PP"), ("between-species", "CE/PP"))):
            selected = [r for r in rows if r["system"] == "worm" and r["category"] == category]
            if not selected:
                continue
            color, style = f"C{i}", "--" if category == "within-animal" else "-"
            for row in selected:
                axes[1].plot(range(3), ys(row), color=color, lw=0.45, alpha=0.35, ls=style)
                bars(axes[1], [1, 2], [row["levels"][k] for k in ("connection", "strength")], color, 0.2)
            values = np.array([ys(r) for r in selected])
            valid = np.isfinite(values).sum(axis=0)
            mean = np.divide(np.nansum(values, axis=0), valid, out=np.full(3, np.nan), where=valid > 0)
            axes[1].plot(range(3), mean, color=color, lw=1.7, ls=style)
            ends.append((mean[-1], label, color))
        labels(axes[1], ends)
        if "sample" in str(payload["data"]).lower():
            fig.text(0.5, 0.98, "Sample pipeline check", ha="center", va="top", fontsize=7, family="monospace")
        fig.text(0.5, 0.01, "Worm: CE = C. elegans; PP = P. pacificus. Bars: 95% node bootstrap CI.",
                 ha="center", fontsize=5.5, family="monospace")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=200)
        plt.close(fig)
    print(f"Wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


def check():
    rng = np.random.default_rng(13)
    names = pd.Index(list("ABCDEFGH"))
    pre, post = np.where(~np.eye(len(names), dtype=bool))
    frame = pd.DataFrame(dict(source=names[pre], target=names[post], weight=rng.integers(10, 100, len(pre))))
    net = network(pd.Series(np.arange(1, 9), index=names), pair_totals(frame))

    def test(other, system="fly"):
        return compare(net, other, "synthetic", system, "test", [1, 3, 5, 10], 5, rng, 20)

    same = test(net)
    assert all(abs(level["dissimilarity"]) < 1e-12 for level in same["levels"].values())
    shuffled = network(net["counts"], pair_totals(frame.assign(weight=rng.permutation(frame.weight))))
    perm = test(shuffled)
    assert perm["levels"]["connection"]["dissimilarity"] == 0
    assert perm["levels"]["strength"]["dissimilarity"] > 0
    scaled = test(network(net["counts"], net["edges"] * 0.58))
    assert scaled["levels"]["strength"]["sd_log2_ratio"] < 1e-12
    parts = pd.DataFrame(dict(source=[1, 1, 1], target=[2, 2, 3], neuropil=["X", "Y", "X"], weight=[2, 4, 8]))
    assert pair_totals(parts).to_dict() == {(1, 2): 6, (1, 3): 8}
    assert type_edges(pair_totals(parts), pd.Series({1: "A", 2: "B", 3: "B"})).iloc[0] == 14
    assert test(net, "worm")["levels"]["cell_type"]["dissimilarity"] == 0
    subset = network(net["counts"].iloc[:-1], net["edges"])
    restricted = test(subset)
    assert restricted["counts"]["dropped_a"] == 1 and restricted["counts"]["dropped_b"] == 0
    assert restricted["dropped"]["a"] == ["H"]
    assert restricted["levels"]["connection"]["dissimilarity"] == 0
    # Sparse node multiplicities equal explicit resampling on both matrix axes.
    m = np.array([2, 0, 1, 1])
    a, b = np.arange(1, 17, dtype=float).reshape(4, 4), np.arange(16, 0, -1, dtype=float).reshape(4, 4)
    pre, post = np.indices((4, 4)).reshape(2, -1)
    idx = np.repeat(np.arange(4), m)
    sparse = measure(pre, post, a.ravel(), b.ravel(), m, [1, 10])
    expanded = measure(pre, post, a[np.ix_(idx, idx)].ravel(), b[np.ix_(idx, idx)].ravel(), np.ones(4, dtype=int), [1, 10])
    for sa, sb in zip(sparse, expanded):
        for level in ("connection", "strength"):
            for metric in sa[level]:
                x, y = sa[level][metric], sb[level][metric]
                assert (x is None and y is None) or np.isclose(x, y), (level, metric, x, y)
    # Node bootstrap ranks agree with explicit repetition, including ties.
    from scipy.stats import spearmanr
    x, y, w = np.array([1, 1, 3, 4]), np.array([4, 2, 2, 1]), np.array([2, 3, 1, 4])
    assert np.isclose(strength(x, y, w)["spearman"], spearmanr(np.repeat(x, w), np.repeat(y, w)).statistic)
    assert strength(np.ones(3), np.ones(3), np.ones(3))["spearman"] is None
    assert neuronal(pd.Series(["AVAL", "IL1DL", "VB1", "GLRDL", "AVAL?", "BWM-DL01"])).tolist() == [True]*3 + [False]*3
    assert worm_names(pd.Series(["AVAL_roi", "VB01", "AVL"])).tolist() == ["AVAL", "VB1", "AVL"]
    json.dumps(same, allow_nan=False)
    print("check ok: identity, shuffled weights, neuropil sums, scale invariance, weighted ranks and worm names")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    runner = commands.add_parser("run")
    runner.add_argument("--data", default="/root/data")
    runner.add_argument("--out", type=Path, default=OUTPUT / "connectome.json")
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--resamples", type=int, default=200)
    runner.add_argument("--fly-threshold", type=float, default=5)
    runner.add_argument("--worm-threshold", type=float, default=1)
    renderer = commands.add_parser("figure")
    renderer.add_argument("json", nargs="?", type=Path, default=OUTPUT / "connectome.json")
    renderer.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "figure":
        figure(args.json, args.out)
    else:
        if args.resamples < 1 or not all(np.isfinite(t) and t > 0 for t in (args.fly_threshold, args.worm_threshold)):
            parser.error("Require resamples >= 1 and finite positive thresholds")
        run(args.data, args.out, args.seed, args.resamples, args.fly_threshold, args.worm_threshold)
