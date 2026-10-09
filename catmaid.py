"""Experiment I3: FAFB CATMAID versus FlyWire v783, the same fly reconstructed twice.

python catmaid.py check
python catmaid.py fetch --cache /root/data/catmaid-cache --bock /root/data/bocklab
python catmaid.py match --cache /root/data/catmaid-cache --data /root/data/flywire
python catmaid.py run --cache /root/data/catmaid-cache --data /root/data/flywire
python catmaid.py figure [output/catmaid.json]

Bock's matrix is KC x PN. Resolve embedded name numbers through the bouton table,
not by blindly subtracting one: many skeletons were merged/reassigned. The modal
+1 naming rule is only a flagged fallback. The Zheng annotation includes non-PNs;
fetch every target, but analyze only the matrix's PN -> complete-KC population.

Use I's input normalization, shared above-threshold edges, population SD and
clipped 1-rho, retaining raw 1-rho too. The primary floor is TYPE-level, threshold
5, calyx-only when annotated; also report neurons, thresholds 1/3/5, all neuropils
and the live connectivity alternative. Cell counts are conditioned on successful,
compatible, one-to-one matches: their zero floor is NOT a test of cell recovery or
typing accuracy. Missing/ambiguous/incompatible matches are coverage, not noise.

Negative excess is retained; subtraction is descriptive, not noise deconvolution.
One random-like PN->KC circuit cannot establish a universal whole-brain floor.
The Buhmann 96% at 5 synapses is a supplied contextual benchmark; pair accuracy
includes true negatives and must not be compared directly to Jaccard similarity.

External text is parsed only as JSON/CSV/TSV, never imported or executed. Local
FlyWire Feather tables use Arrow. Network responses and numeric lookups are cached
under --cache; interrupted runs resume. No fafbseg code is imported or copied.
Matching additionally requires --allow-numeric-binary: the specified Spine wire
protocol and CloudVolume cannot operate under a literal JSON/CSV-only restriction.
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import connectome as experiment
import noise

OUTPUT = experiment.OUTPUT
LEVELS = noise.LEVELS
BASE = "https://fafb.catmaid.virtualflybrain.org"
SPINE = "https://services.itanna.io/app/transform-service"
SEGMENTATION = "precomputed://gs://flywire_v141_m783"
MATRIX = "190319-440RDKC_PNfirst16comm_last22blue.csv"
BOUTONS = "201001_bouton_claw_table.csv"
LOG = logging.getLogger("catmaid")
NOTES = [
    "One same-brain PN->KC calyx circuit, with random-like connectivity; transferring its floor to the whole brain is an assumption.",
    "Cell-type/count floor is conditional on compatible one-to-one matches and is zero by construction; cell recovery and typing errors are unmeasured.",
    "Rejected matches are selection/coverage losses, not biological differences. Both reconstructions may share errors; neither is ground truth.",
    "Strength uses shared above-threshold edges and input normalization over all matched PN inputs, not every input to a KC.",
    "Excess is observed minus floor (negative values retained), not an additive decomposition or statistical noise removal.",
    "Subtracting a common floor preserves between-minus-within differences algebraically; the three metrics are not a calibrated common distance.",
    "Buhmann et al. 2021: supplied benchmark 96% calyx pair classification at 5 synapses; pair accuracy includes true negatives, unlike Jaccard.",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def csrf_headers(base, token):
    if not token:
        raise ValueError("CATMAID did not provide a csrftoken cookie")
    return {"X-CSRFToken": token, "Referer": base.rstrip("/") + "/"}


class Remote:
    """Serial, resumable network operations; cache keys include request content."""

    def __init__(self, cache, delay=0.15, retries=5, base=BASE):
        import requests
        if delay <= 0 or retries < 1:
            raise ValueError("Use a positive delay and at least one attempt")
        self.cache, self.delay, self.retries = Path(cache), delay, retries
        self.base, self.session = base.rstrip("/"), requests.Session()
        self.cache.mkdir(parents=True, exist_ok=True)

    def attempt(self, label, call):
        for attempt in range(self.retries):
            time.sleep(self.delay * min(2 ** attempt, 16))
            try:
                return call()
            except Exception as error:
                LOG.warning("%s attempt %d/%d: %s", label, attempt + 1, self.retries, error)
                if attempt + 1 == self.retries:
                    raise

    def csrf(self):
        def get_cookie():
            LOG.info("GET %s/ (CSRF bootstrap)", self.base)
            response = self.session.get(self.base + "/", timeout=(20, 180))
            response.raise_for_status()
            (self.cache / "csrf-landing.html").write_bytes(response.content)
            name, token = next(((name, value) for name, value in self.session.cookies.items()
                                if name.startswith("csrftoken")), (None, None))
            headers = csrf_headers(self.base, token)
            # Landing HTML is cached as inert bytes and never parsed or executed.
            write_json(self.cache / "csrf.json", dict(status=response.status_code, name=name, token=token))
            return headers
        return self.attempt("CSRF bootstrap", get_cookie)

    def request(self, method, url, data=None, catmaid=False, decoder=None):
        body = data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode()
        key = hashlib.sha256(method.encode() + url.encode() + body).hexdigest()
        path = self.cache / "http" / (key + ".json")
        if path.exists():
            LOG.info("Cache %s %s [%s]", method, url, key[:10])
            return read_json(path)["response"]
        def get_response():
            headers = {}
            if catmaid and method == "POST":
                token = next((value for name, value in self.session.cookies.items()
                              if name.startswith("csrftoken")), None)
                headers = csrf_headers(self.base, token) if token else self.csrf()
            time.sleep(self.delay)  # Also separate a just-completed CSRF GET from its POST.
            LOG.info("%s %s [%s]", method, url, key[:10])
            response = self.session.request(method, url, data=data, headers=headers, timeout=(20, 180))
            if response.status_code == 403 and catmaid and method == "POST":
                self.session.cookies.clear()
            response.raise_for_status()
            value = decoder(response) if decoder else response.json()
            if isinstance(value, dict) and value.get("error"):
                raise ValueError(str(value["error"])[:300])
            write_json(path, dict(method=method, url=url, request_sha256=key, response=value))
            return value
        return self.attempt(f"{method} {url}", get_response)

    @contextmanager
    def cloud_transport(self):
        """Pace/retry each HTTP send inside CloudVolume, including metadata GETs.

        Install once around the entire worker pool: nested per-call patches race.
        The wrapper also logs optional 404s without changing their meaning.
        """
        import requests
        from unittest.mock import patch
        original = requests.sessions.Session.send
        def send(session, request, **kwargs):
            def transmit():
                LOG.info("CloudVolume %s %s", request.method, request.url)
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = (20, 180)
                response = original(session, request, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                return response
            return self.attempt(f"CloudVolume {request.method} {request.url}", transmit)
        with patch.object(requests.sessions.Session, "send", send):
            yield


def targets(payload):
    rows = []
    for entity in payload["entities"]:
        if entity["type"] != "neuron":
            continue
        for skid in entity["skeleton_ids"]:
            rows.append(dict(skeleton_id=int(skid), name=str(entity["name"])))
    frame = pd.DataFrame(rows)
    if frame.empty or frame.skeleton_id.duplicated().any():
        raise ValueError("Empty or duplicate annotation skeleton IDs")
    return frame


def nodes(payload):
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
        raise ValueError("Expected compact-detail list with node table at index zero")
    table = payload[0]
    if not table or any(len(row) < 6 for row in table):
        raise ValueError("Empty or short compact-detail node row")
    xyz = np.array([row[3:6] for row in table], dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite skeleton coordinates")
    return xyz


def connectivity(payload, direction="outgoing"):
    rows = []
    for partner, record in payload[direction].items():
        for source, bins in record["skids"].items():
            counts = np.asarray(bins, dtype=float)
            if counts.shape != (5,) or not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError("Expected five nonnegative confidence-bin synapse counts")
            pre, post = (int(source), int(partner)) if direction == "outgoing" else (int(partner), int(source))
            rows.append((pre, post, float(counts.sum())))
    return experiment.clean_edges(pd.DataFrame(rows, columns=["source", "target", "weight"]))


def name_number(name):
    found = re.search(r"(?<!\S)(\d+)(?!\S)", str(name))
    if not found:
        raise ValueError(f"No standalone neuron number: {name}")
    return int(found[1])


def kc_class(name):
    text = str(name).lower().replace("α", "a").replace("β", "b").replace("γ", "g")
    if re.search(r"kc[ _-]*(g|y|gamma)", text):
        return "KCg"
    if "kc" in text and ("a'b'" in text or "apbp" in text or "alpha'" in text):
        return "KCapbp"
    if "kc" in text and ("ab" in text or "alpha" in text):
        return "KCab"
    return ""


def annotation_class(name):
    """Type extra annotation targets conservatively; never call all 145 PNs."""
    kind = kc_class(name)
    if kind:
        return "KC", kind
    if name.startswith("Uniglomerular"):
        found = re.search(r"\b[DV][A-Z]?\d*[dlmv]?\b", name)
        if found:
            return "PN", found[0]
    return "annotation-only", ""


def bock(directory):
    directory = Path(directory)
    table = pd.read_csv(directory / BOUTONS)
    matrix = pd.read_csv(directory / MATRIX, index_col=0)
    values = matrix.to_numpy(dtype=float)
    if matrix.shape != (440, 113) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Expected finite nonnegative 440 x 113 Bock matrix, got {matrix.shape}")
    roster, audit = [], {}
    for role, names in (("PN", matrix.columns), ("KC", matrix.index)):
        prefix = role.lower()
        unique = table[[prefix + "_skid", prefix + "_names"]].drop_duplicates().copy()
        unique["number"] = unique[prefix + "_names"].map(name_number)
        relevant = unique.loc[unique.number.isin([name_number(n) for n in names])]
        offsets = relevant.number - relevant[prefix + "_skid"]
        modal = int(offsets.mode().iloc[0])
        lookup = relevant.groupby("number")[prefix + "_skid"].agg(lambda x: sorted(set(map(int, x))))
        audit[role] = dict(explicit_names=len(relevant), modal_name_minus_skeleton=modal,
                           obey_modal=int(offsets.eq(modal).sum()), exceptions=int(offsets.ne(modal).sum()), fallback=[])
        for name in names:
            number = name_number(name)
            candidates = lookup.get(number, [])
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous name number {number}: {candidates}")
            skid = candidates[0] if candidates else number - modal
            if not candidates:
                audit[role]["fallback"].append(dict(name=name, skeleton_id=skid))
            if role == "PN":
                kinds = table.loc[table.pn_skid.eq(skid), "pn_type"].dropna().unique()
                if len(kinds) != 1:
                    raise ValueError(f"Ambiguous PN type for {skid}: {kinds}")
                kind = str(kinds[0])
            else:
                kind = kc_class(name)
                if not kind:
                    raise ValueError(f"Unknown KC class: {name}")
            roster.append(dict(skeleton_id=skid, name=name, role=role, catmaid_type=kind,
                               id_source="bouton" if candidates else "modal_name_rule"))
    cells = pd.DataFrame(roster)
    if cells.skeleton_id.duplicated().any():
        raise ValueError("Bock matrix maps multiple cells onto one skeleton")
    pns = cells.loc[cells.role.eq("PN"), "skeleton_id"].to_numpy()
    kcs = cells.loc[cells.role.eq("KC"), "skeleton_id"].to_numpy()
    r, c = np.nonzero(values)
    edges = pd.DataFrame(dict(source=pns[c], target=kcs[r], weight=values[r, c]))
    return cells, edges, audit


def agreement(a, b, possible):
    table = pd.concat([experiment.pair_totals(a).rename("a"), experiment.pair_totals(b).rename("b")], axis=1).fillna(0)
    equal = table.a.eq(table.b)
    return dict(possible_pairs=int(possible), positive_union_pairs=len(table),
                exact_positive_union=float(equal.mean()) if len(table) else None,
                exact_all_pairs=float(1 - (~equal).sum() / possible) if possible else None,
                total_a=float(table.a.sum()), total_b=float(table.b.sum()),
                mismatch_examples=table.loc[~equal].reset_index().head(20).to_dict("records"))


def fetch(cache, bock_dir, base=BASE, delay=0.15, retries=5, batch=25):
    cache = Path(cache)
    if batch < 1:
        raise ValueError("batch must be positive")
    remote = Remote(cache, delay, retries, base)
    raw = remote.request("POST", base.rstrip("/") + "/1/annotations/query-targets",
                         {"annotated_with": "11490509", "types": "neuron"}, catmaid=True)
    write_json(cache / "pn_targets.json", raw)
    annotated = targets(raw)
    cells, edges, rule = bock(bock_dir)
    write_csv(cache / "bock-neurons.csv", cells)
    write_csv(cache / "bock-edges.csv", edges)
    write_json(cache / "name-rule.json", rule)
    # Cache the original supplied tables as data, never executable resources.
    for filename in (MATRIX, BOUTONS):
        destination = cache / "bock" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(bock_dir) / filename).read_bytes())
    extra = annotated.loc[~annotated.skeleton_id.isin(cells.skeleton_id)].copy()
    classes = [annotation_class(name) for name in extra.name]
    extra["role"] = [role for role, kind in classes]
    extra["catmaid_type"] = [kind for role, kind in classes]
    extra["id_source"] = "annotation"
    roster = pd.concat([cells, extra], ignore_index=True)
    write_csv(cache / "neurons.csv", roster)
    LOG.info("Annotation: %d targets (not all PNs); matrix: %d PNs, 440 KCs; union %d skeletons",
             len(annotated), int(cells.role.eq("PN").sum()), len(roster))
    for i, skid in enumerate(roster.skeleton_id):
        payload = remote.request("GET", f"{base.rstrip('/')}/1/skeletons/{skid}/compact-detail?with_connectors=false", catmaid=True)
        count = len(nodes(payload))
        write_json(cache / "skeletons" / f"{skid}.json", payload)
        LOG.info("Skeleton %d/%d: %s (%d nodes)", i + 1, len(roster), skid, count)
    parts = []
    ids = roster.skeleton_id.tolist()
    for start in range(0, len(ids), batch):
        fields = {f"source_skeleton_ids[{i}]": str(s) for i, s in enumerate(ids[start:start + batch])}
        fields["boolean_op"] = "OR"
        payload = remote.request("POST", f"{base.rstrip('/')}/1/skeletons/connectivity", fields, catmaid=True)
        write_json(cache / "connectivity" / f"{start}.json", payload)
        part = connectivity(payload)
        # The public endpoint returns all partners; restrict explicitly here.
        parts.append(part.loc[part.source.isin(ids[start:start + batch]) & part.target.isin(ids)])
        LOG.info("Connectivity sources %d..%d; restricted to %d fetched neurons", start, min(start + batch, len(ids)), len(ids))
    all_edges = pd.concat(parts, ignore_index=True)
    write_csv(cache / "connectivity.csv", all_edges)
    pn, kc = cells.loc[cells.role.eq("PN"), "skeleton_id"], cells.loc[cells.role.eq("KC"), "skeleton_id"]
    restricted = all_edges.loc[all_edges.source.isin(pn) & all_edges.target.isin(kc)]
    audit = dict(annotation_targets=len(annotated), fetched_skeletons=len(roster), name_rule=rule,
                 matrix_vs_live=agreement(edges, restricted, len(pn) * len(kc)))
    write_json(cache / "fetch.json", audit)
    LOG.info("Bock vs live connectivity: %s", json.dumps(audit["matrix_vs_live"]))
    return audit


def majority(ids):
    """Like skid_to_id, discard zero before computing top share and vote margin."""
    counts = Counter(int(value) for value in ids if int(value) != 0)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    root, top = ordered[0] if ordered else (0, 0)
    second = ordered[1][1] if len(ordered) > 1 else 0
    return dict(root_id=str(root), confidence=top / total if total else 0.,
                margin=(top - second) / total if total else 0., nonzero=total, sampled=len(ids),
                nonzero_fraction=total / len(ids) if len(ids) else 0.)


def annotations(data):
    path = experiment.locate(data, "Supplemental_file1_neuron_annotations.tsv", "flywire/Supplemental_file1_neuron_annotations.tsv")
    frame = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    if not {"root_id", "cell_type"}.issubset(frame) or frame.root_id.duplicated().any():
        raise ValueError("Expected unique root_id and cell_type in v783 annotations")
    if not frame.root_id.str.fullmatch(r"[1-9]\d*").all():
        raise ValueError("Invalid annotation root ID")
    return frame.set_index("root_id")


def compatible(role, cat_type, fw_type):
    if role == "KC":
        return kc_class(fw_type) == cat_type
    if role == "PN":
        # Boundary is essential: VA1 must not silently match VA1d or VA1v.
        tokens = re.findall(r"(?<![A-Za-z0-9])(?:D|V)[A-Z]?\d*[dlmv]?(?=[_+ /-]|$)", str(fw_type))
        return cat_type in tokens or (cat_type in ("VP1", "VP2", "VP3") and
                                      any(t.startswith(cat_type) for t in tokens))
    return False


def qualify(frame, ann, confidence=0.5, margin=0.2):
    if not 0 <= confidence <= 1 or not 0 <= margin <= 1:
        raise ValueError("Confidence and margin must be in [0,1]")
    frame = frame.copy()
    scores = frame[["confidence", "margin"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Invalid match scores")
    frame["root_id"] = frame.root_id.astype(str)
    frame["flywire_type"] = frame.root_id.map(ann.cell_type).fillna("")
    if "side" in ann:
        frame["flywire_side"] = frame.root_id.map(ann.side).fillna("")
    frame["in_annotations"] = frame.root_id.isin(ann.index)
    frame["compatible"] = [compatible(r.role, r.catmaid_type, r.flywire_type) for r in frame.itertuples()]
    candidate = frame.confidence.ge(confidence) & frame.margin.ge(margin) & frame.root_id.ne("0")
    # Flag all colliding high-confidence mappings, even if one is incompatible.
    conflicts = frame.loc[candidate].groupby("root_id").skeleton_id.agg(list)
    conflicts = conflicts.loc[conflicts.map(len).gt(1)]
    frame["conflict"] = candidate & frame.root_id.isin(conflicts.index)
    frame["accepted"] = candidate & frame.in_annotations & frame.compatible & ~frame.conflict
    frame["reason"] = np.select([~candidate, ~frame.in_annotations, frame.conflict, ~frame.compatible],
                                 ["low-confidence", "absent-v783", "root-collision", "incompatible-type"], default="accepted")
    eligible = frame.role.isin(["PN", "KC"])
    audit = dict(total=len(frame), eligible=int(eligible.sum()), accepted=int(frame.accepted.sum()),
                 match_rate=float(frame.loc[eligible, "accepted"].mean()) if eligible.any() else None,
                 reasons=frame.reason.value_counts().to_dict(),
                 conflicts={root: list(map(int, skids)) for root, skids in conflicts.items()})
    return frame, audit


def compare_networks(a, b, cells, resolution, threshold=5):
    labels = cells.set_index("skeleton_id").catmaid_type
    # Prefix roles to prevent accidental PN/KC label collisions.
    labels = cells.set_index("skeleton_id").role + ":" + labels
    if resolution == "type":
        counts = labels.value_counts()
        ea, eb = [experiment.type_edges(experiment.pair_totals(e), labels) for e in (a, b)]
    else:
        counts = pd.Series(1, index=labels.index)
        ea, eb = [experiment.pair_totals(e) for e in (a, b)]
    row = experiment.compare(experiment.network(counts, ea), experiment.network(counts, eb),
                             "CATMAID/FlyWire", "fly", "same-brain", [1, 3, 5], threshold,
                             np.random.default_rng(0), resamples=0)
    row["resolution"] = resolution
    row["levels"]["cell_type"]["conditional_on_compatible_matches"] = True
    row["type_inventory"] = dict(catmaid=labels.value_counts().to_dict(), flywire=labels.value_counts().to_dict(),
                                 presence_jaccard=1.0, counts_identical_by_selection=True)
    return row


def paired_counts(a, b):
    return pd.concat([experiment.pair_totals(a).rename("catmaid"), experiment.pair_totals(b).rename("flywire")], axis=1).fillna(0)


def pair_statistics(table, possible):
    both = table.catmaid.gt(0) & table.flywire.gt(0)
    ratios = table.loc[both, "flywire"] / table.loc[both, "catmaid"]
    rows = []
    for threshold in (1, 3, 5):
        a, b = table.catmaid.ge(threshold), table.flywire.ge(threshold)
        tp, fp, fn = int((a & b).sum()), int((~a & b).sum()), int((a & ~b).sum())
        tn = possible - tp - fp - fn
        rows.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                         pair_accuracy=(tp + tn) / possible if possible else None))
    return dict(median_flywire_over_catmaid=float(ratios.median()) if len(ratios) else None,
                ratio_common_positive_pairs=len(ratios), total_catmaid=float(table.catmaid.sum()),
                total_flywire=float(table.flywire.sum()),
                total_ratio=float(table.flywire.sum() / table.catmaid.sum()) if table.catmaid.sum() else None,
                classification=rows)


def adjusted_comparisons(observed, floors):
    rows = []
    for row in observed["comparisons"]:
        if row["system"] != "fly" or row["category"] not in ("within-animal", "between-animal"):
            continue
        primary = next((r for r in floors["thresholds"] if r["threshold"] == row["threshold"]), None)
        if primary is None:
            raise ValueError(f"No measured floor for I's threshold {row['threshold']}")
        values = {}
        for level in LEVELS:
            floor = (floors["levels"][level] if level == "cell_type" else primary[level])["dissimilarity"]
            value = row["levels"][level]["dissimilarity"]
            values[level] = dict(observed=value, floor=floor, excess=value - floor if value is not None and floor is not None else None)
        rows.append(dict(name=row["name"], category=row["category"], threshold=row["threshold"], levels=values,
                         strictly_increasing=noise.increasing([values[k]["excess"] for k in LEVELS])))
    contrasts = []
    for between in [r for r in rows if r["category"] == "between-animal"]:
        for within in [r for r in rows if r["category"] == "within-animal"]:
            differences = {k: between["levels"][k]["excess"] - within["levels"][k]["excess"]
                           if between["levels"][k]["excess"] is not None and within["levels"][k]["excess"] is not None else None for k in LEVELS}
            contrasts.append(dict(between=between["name"], within=within["name"], difference=differences,
                                  between_greater_all_levels=all(v > 0 for v in differences.values())
                                  if all(v is not None for v in differences.values()) else None))
    return rows, contrasts


def summary(payload):
    lines = ["I3: same FAFB brain, CATMAID versus FlyWire v783" + (" [SYNTHETIC QUICK TEST]" if payload["synthetic"] else ""),
             f"Accepted matrix neurons: {payload['coverage']['PN']} PNs, {payload['coverage']['KC']} KCs",
             f"Primary floor: {payload['primary']} / type / threshold 5",
             "Cell floor is conditional, zero by selection; cell recovery/typing noise is not estimated."]
    for source, variants in payload["networks"].items():
        for region, result in variants.items():
            stats = result["statistics"]
            lines.append(f"{source}/{region}: median FW/CAT={stats['median_flywire_over_catmaid']}; total ratio={stats['total_ratio']}")
            for resolution in ("neuron", "type"):
                row = result[resolution]
                for t in row["thresholds"]:
                    lines.append(f"  {resolution}, t={t['threshold']}: cell={row['levels']['cell_type']['dissimilarity']}; "
                                 f"connection={t['connection']['dissimilarity']}; strength 1-rho={t['strength']['one_minus_spearman']}; "
                                 f"SD log2={t['strength']['sd_log2_ratio']}")
            accuracy = next(r for r in stats["classification"] if r["threshold"] == 5)["pair_accuracy"]
            lines.append(f"  Neuron-pair accuracy at 5={accuracy}; supplied Buhmann benchmark=0.96 (different evaluation population).")
    for row in payload["comparisons"]:
        values = [row["levels"][k]["excess"] for k in LEVELS]
        lines.append(f"{row['name']} excess cell/connection/strength={values}; strictly increasing={row['strictly_increasing']}")
    for row in payload["contrasts"]:
        lines.append(f"{row['between']} > {row['within']} at all levels: {row['between_greater_all_levels']}; delta={row['difference']}")
    lines.extend(["Matching: " + json.dumps(payload["matching"]), "Fetch audit: " + json.dumps(payload["fetch_audit"])])
    lines.append("Conditional verdict: " + json.dumps(payload["verdict"]))
    lines.extend(payload["notes"])
    return "\n".join(lines) + "\n"


def run(cache, data="/root/data/flywire", matches=OUTPUT / "catmaid-matches.csv", out=OUTPUT / "catmaid.json",
        observed=OUTPUT / "connectome.json", noise_path=OUTPUT / "noise.json", calyx="CA_R",
        confidence=0.5, margin=0.2, synthetic=False):
    cache, out = Path(cache), Path(out)
    if synthetic and out.stem == "catmaid":
        raise ValueError("Synthetic runs require an alternate output stem")
    cells = pd.read_csv(cache / "bock-neurons.csv", keep_default_na=False)
    population = cells.copy()
    candidate = pd.read_csv(matches, dtype={"root_id": "string"}, keep_default_na=False)
    if candidate.skeleton_id.duplicated().any():
        raise ValueError("Duplicate skeleton IDs in matches")
    # The cached Bock roster, not an editable matches CSV, owns analysis roles/types.
    for column in ("role", "catmaid_type"):
        authoritative = candidate.skeleton_id.map(cells.set_index("skeleton_id")[column])
        candidate.loc[authoritative.notna(), column] = authoritative.loc[authoritative.notna()]
    # Revalidate types and collisions against supplied v783 annotations, not saved acceptance flags.
    candidate, audit = qualify(candidate, annotations(data), confidence, margin)
    selected = candidate.loc[candidate.accepted & candidate.skeleton_id.isin(cells.skeleton_id)]
    cells = cells.loc[cells.skeleton_id.isin(selected.skeleton_id)]
    if not cells.role.eq("PN").any() or not cells.role.eq("KC").any():
        raise ValueError("Need accepted PN and KC matches")
    roots = selected.set_index("root_id").skeleton_id
    roots.index = pd.Index([int(root) for root in roots.index], dtype="uint64")
    pn, kc = cells.loc[cells.role.eq("PN"), "skeleton_id"], cells.loc[cells.role.eq("KC"), "skeleton_id"]
    path = experiment.locate(data, "proofread_connections_783.feather", "flywire/proofread_connections_783.feather", "connections.csv")
    if path.suffix == ".feather":
        import pyarrow as pa
        with pa.memory_map(str(path), "r") as stream:
            schema = pa.ipc.open_file(stream).schema.names
        columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count"] + (["neuropil"] if "neuropil" in schema else [])
        fw = pd.read_feather(path, columns=columns)
    else:
        fw = pd.read_csv(path, dtype={"pre_pt_root_id": "string", "post_pt_root_id": "string"})
    fw = fw.rename(columns={"pre_pt_root_id": "source", "post_pt_root_id": "target", "syn_count": "weight"})
    available = sorted(fw.neuropil.dropna().astype(str).unique()) if "neuropil" in fw else []
    for column in ("source", "target"):
        fw[column] = fw[column].astype("uint64")
    fw = fw.loc[fw.source.isin(roots.index) & fw.target.isin(roots.index)].copy()
    for column in ("source", "target"):
        fw[column] = fw[column].map(roots)
    fw = fw.loc[fw.source.isin(pn) & fw.target.isin(kc)].copy()
    experiment.clean_edges(fw)  # Validate before any grouping/filter could hide bad weights.
    fw.weight = pd.to_numeric(fw.weight)
    regions = {"all": fw}
    if calyx in available:
        regions["calyx"] = fw.loc[fw.neuropil.eq(calyx)]
    elif "neuropil" in fw:
        LOG.warning("No %s in edge table (available: %s); calyx-only floor unavailable", calyx, available)
    sources = {"bock": pd.read_csv(cache / "bock-edges.csv")}
    if (cache / "connectivity.csv").exists():
        sources["connectivity"] = pd.read_csv(cache / "connectivity.csv")
    results = {}
    for source, edges in sources.items():
        edges = experiment.clean_edges(edges)
        edges = edges.loc[edges.source.isin(pn) & edges.target.isin(kc)]
        results[source] = {}
        for region, frame in regions.items():
            frame = experiment.clean_edges(frame)
            table = paired_counts(edges, frame)
            results[source][region] = dict(statistics=pair_statistics(table, len(pn) * len(kc)),
                                           neuron=compare_networks(edges, frame, cells, "neuron"),
                                           type=compare_networks(edges, frame, cells, "type"))
            if source == "bock":
                results[source][region]["pairs"] = table.reset_index().to_dict("records")
    primary = "calyx" if "calyx" in regions else "all"
    floor = results["bock"][primary]["type"]
    observations, noises = read_json(observed), read_json(noise_path)
    comparisons, contrasts = adjusted_comparisons(observations, floor)
    def all_known(values):
        return all(values) if values and all(v is not None for v in values) else None
    verdict = dict(hierarchy_increases_in_all_comparisons=all_known([r["strictly_increasing"] for r in comparisons]),
                   between_exceeds_within_at_every_level=all_known([r["between_greater_all_levels"] for r in contrasts]),
                   interpretation="Descriptive conditional excess, not established biological noise removal.")
    payload = dict(synthetic=synthetic, cache=str(cache), data=str(data), matches=str(matches), observed=str(observed),
                   noise=str(noise_path), primary=primary, calyx=calyx, available_neuropils=available,
                   coverage={role: int(cells.role.eq(role).sum()) for role in ("PN", "KC")}, matching=audit,
                   population_coverage={role: dict(available=int(population.role.eq(role).sum()),
                                                   accepted=int(cells.role.eq(role).sum()),
                                                   fraction=float(cells.role.eq(role).sum() / population.role.eq(role).sum()))
                                        for role in ("PN", "KC")},
                   fetch_audit=read_json(cache / "fetch.json") if (cache / "fetch.json").exists() else None,
                   networks=results, comparisons=comparisons, contrasts=contrasts, verdict=verdict,
                   raw_type_counts=dict(catmaid=cells.catmaid_type.value_counts().to_dict(),
                                        flywire=selected.flywire_type.value_counts().to_dict()),
                   reference_floors=dict(worm=noises["reannotation"]["comparison"]["levels"],
                                         simulated=noises["simulations"]["fly"], noise_sample=noises.get("sample", False)),
                   notes=NOTES)
    write_json(out, payload)
    out.with_name(out.stem + "-summary.txt").write_text(summary(payload))
    LOG.info("Wrote %s and summary", out)
    return payload


def panel_letter(ax, letter):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=11, family="monospace",
            fontweight="semibold", va="bottom", ha="left")


def figure(path=OUTPUT / "catmaid.json", out=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    if not hasattr(cm, "get_cmap"):
        cm.get_cmap = plt.get_cmap
    import plotting  # noqa: F401

    payload = read_json(path)
    out = Path(out) if out else Path(path).with_suffix("")
    if payload["synthetic"] and out.stem == "catmaid":
        raise ValueError("Synthetic figures require an alternate output stem")
    result = payload["networks"]["bock"][payload["primary"]]
    xs = np.arange(3)
    floor = np.array([result["type"]["levels"][k]["dissimilarity"] for k in LEVELS], dtype=float)
    style = dict(fontsize=5.3, family="monospace")
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 6, "lines.linewidth": 1, "savefig.bbox": None}):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.8, wspace=0.45)
        for ax, letter, title in zip(axes, "ABC", ("Same-brain counts", "Noise floors (types)", "Fly comparisons (types)")):
            panel_letter(ax, letter)
            ax.set_title(title, fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[0]
        pairs = pd.DataFrame(result["pairs"])
        if not pairs.empty:
            keep = pairs.catmaid.gt(0) & pairs.flywire.gt(0)
            x, y = pairs.loc[keep, "catmaid"], pairs.loc[keep, "flywire"]
            ax.scatter(x, y, s=5, alpha=0.35, color="C0", linewidths=0, rasterized=True)
            maximum = max(10, float(pairs[["catmaid", "flywire"]].max().max()) * 1.2)
            line = np.array([1., maximum])
            ax.plot(line, line, color="C1", lw=0.8)
            ratio = result["statistics"]["median_flywire_over_catmaid"]
            if ratio is not None:
                ax.plot(line, ratio * line, color="C2", ls="--", lw=0.8)
                ax.text(0.02, 0.87, f"median ratio {ratio:.2g}", va="top", color="C2", transform=ax.transAxes, **style)
            ax.text(0.02, 0.98, "identity", va="top", color="C1", transform=ax.transAxes, **style)
            ax.text(0.98, 0.02, f"{int((~keep).sum())} zero-sided omitted", ha="right", transform=ax.transAxes, **style)
            ax.set(xscale="log", yscale="log", xlim=(0.8, maximum), ylim=(0.8, maximum))
        ax.set(xlabel="CATMAID synapses", ylabel="FlyWire synapses")
        for ax in axes[1:]:
            ax.set(xticks=xs, xticklabels=["cell type*", "connection", "strength"],
                   xlim=(-0.12, 2.12), ylim=(-0.035, 1.15), yticks=[0, 0.5, 1])
            ax.tick_params(axis="x", rotation=20)
        axes[1].set_ylabel("Dissimilarity")
        sim = payload["reference_floors"]["simulated"]
        bounds = np.array([sim["grid_range"][k] if sim["grid_range"][k] is not None else [np.nan, np.nan] for k in LEVELS])
        axes[1].fill_between(xs, bounds[:, 0], bounds[:, 1], color="C2", alpha=0.2, lw=0)
        axes[1].plot(xs, [sim["central"][k] for k in LEVELS], ":", color="C2")
        worm = [payload["reference_floors"]["worm"][k]["dissimilarity"] for k in LEVELS]
        axes[1].plot(xs, worm, "s-", color="C1", ms=2)
        axes[1].plot(xs, floor, "o-", color="C0", ms=2)
        for i, (label, color) in enumerate((("fly measured", "C0"), ("White/Cook", "C1"), ("fly simulated band", "C2"))):
            axes[1].text(0.03, 0.97 - i * 0.085, label, color=color, transform=axes[1].transAxes, **style)
        axes[2].fill_between(xs, 0, floor, color="C0", alpha=0.18, lw=0)
        axes[2].plot(xs, floor, ":", color="C0")
        for i, row in enumerate(payload["comparisons"]):
            color = f"C{i + 3}"
            axes[2].plot(xs, [row["levels"][k]["observed"] for k in LEVELS], color=color,
                         ls="--" if row["category"] == "within-animal" else "-")
            axes[2].text(0.03, 0.97 - i * 0.085, row["name"], color=color, transform=axes[2].transAxes, **style)
        axes[2].text(0.03, 0.08, "shading: fly floor", color="C0", transform=axes[2].transAxes, **style)
        if payload["synthetic"]:
            fig.text(0.5, 0.98, "SYNTHETIC MATCHES / EDGES — pipeline test only", ha="center", va="top", **style)
        fig.text(0.5, 0.015, "*Cell floor conditional on matching. One calyx circuit; transfer to whole brain assumed.", ha="center", **style)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=200)
        plt.close(fig)
    LOG.info("Wrote %s.pdf and .png", out)


def sample_nodes(payload, count=50):
    if count < 1:
        raise ValueError("sample must be positive")
    xyz = nodes(payload)
    # Uniform deterministic sampling across the entire arbor, independent of response row order.
    order = np.argsort([int(row[0]) for row in payload[0]])
    take = np.random.default_rng(1985).choice(len(order), min(count, len(order)), replace=False)
    return xyz[order[np.sort(take)]]


def transform_points(remote, xyz, spine=SPINE):
    """Implement the documented Spine wire protocol, not the GPL implementation.

    Only fixed little-endian float32 arrays are decoded, with an exact byte count.
    Coordinates sent are floor(nm / service voxel_size); returned dx/dy are base
    4-nm voxels even at mip 4. Z is unchanged. Cache decoded arrays as JSON.
    """
    info = remote.request("GET", spine.rstrip("/") + "/info")
    dataset = info["flywire_v1_inverse"]
    voxel = np.asarray(dataset["voxel_size"], dtype=float)
    if voxel.shape != (3,) or not np.isfinite(voxel).all() or (voxel <= 0).any() or 4 not in dataset["scales"]:
        raise ValueError("Unexpected flywire_v1_inverse metadata; need positive voxel_size and mip 4")
    body = np.floor_divide(xyz, voxel).astype("<f4").tobytes(order="C")
    def decode(response):
        if len(response.content) != len(xyz) * 2 * 4:
            raise ValueError(f"Spine returned {len(response.content)} bytes; expected {len(xyz) * 8}")
        offsets = np.frombuffer(response.content, dtype="<f4").reshape(-1, 2)
        # JSON null records failed transforms; do not turn failures into zero offsets.
        return [[float(v) if np.isfinite(v) else None for v in row] for row in offsets]
    offsets = remote.request("POST", spine.rstrip("/") +
                             "/transform/dataset/flywire_v1_inverse/s/4/values_binary/format/array_float_Nx3",
                             body, decoder=decode)
    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (len(xyz), 2):
        raise ValueError("Cached Spine offsets have wrong shape")
    transformed = xyz.astype(np.float32)
    transformed[:, :2] += offsets.astype(np.float32) * 4
    return transformed


def segmentation_points(volume, points):
    """Read unique mip-0 voxel coordinates in one batch, or once per chunk."""
    points = list(dict.fromkeys(tuple(map(int, point)) for point in points))
    if not points:
        return {}
    scattered = getattr(volume, "scattered_points", None)
    if callable(scattered):
        labels = scattered(points)
        return {point: int(labels[point]) for point in points}

    # Older CloudVolume versions: align to the chunk grid's voxel offset, and
    # clip edge chunks to the volume bounds. Never fetch a whole skeleton bbox.
    offset = np.asarray(volume.voxel_offset, dtype=np.int64)
    size = np.asarray(volume.chunk_size, dtype=np.int64)
    lower = np.asarray(volume.bounds.minpt, dtype=np.int64)
    upper = np.asarray(volume.bounds.maxpt, dtype=np.int64)
    groups = {}
    for point in points:
        xyz = np.asarray(point, dtype=np.int64)
        if (xyz < lower).any() or (xyz >= upper).any():
            raise ValueError(f"Segmentation voxel outside volume bounds: {point}")
        start = tuple(offset + ((xyz - offset) // size) * size)
        groups.setdefault(start, []).append(point)
    labels = {}
    for start, group in groups.items():
        lo = np.maximum(start, lower)
        hi = np.minimum(np.asarray(start) + size, upper)
        chunk = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])
        for point in group:
            labels[point] = int(chunk[tuple(np.asarray(point) - lo)].item())
    return labels


def point_roots(cache, skid, voxels, lookup):
    """Keep per-coordinate results, with one atomic JSON checkpoint per skeleton.

    Read legacy per-point files too, so an interrupted old run is reusable.
    Coordinates key the cache independently of sample size, order, or duplicates.
    """
    directory = Path(cache) / "point-roots-v783"
    path = directory / "skeletons" / f"{skid}.json"
    saved = read_json(path) if path.exists() else {}
    if saved and saved.get("segmentation") != SEGMENTATION:
        raise ValueError(f"Unexpected segmentation in {path}")
    roots = dict(saved.get("roots", {}))
    points = [tuple(map(int, point)) for point in voxels]
    keys = ["_".join(map(str, point)) for point in points]
    missing = {}
    for point, key in zip(points, keys):
        if key in roots or point in missing:
            continue
        legacy = directory / (key + ".json")
        if legacy.exists():
            roots[key] = str(int(read_json(legacy)["root_id"]))
        else:
            missing[point] = key
    if missing:
        labels = lookup(list(missing))
        roots.update({key: str(int(labels[point])) for point, key in missing.items()})
    if not saved or roots != saved.get("roots"):
        write_json(path, dict(segmentation=SEGMENTATION, roots=roots))
    return [int(roots[key]) for key in keys]


def match(cache, data="/root/data/flywire", out=OUTPUT / "catmaid-matches.csv", sample=50,
          confidence=0.5, margin=0.2, delay=0.15, retries=5, spine=SPINE, allow_numeric_binary=False,
          workers=16):
    if not allow_numeric_binary:
        raise ValueError("Spine and CloudVolume require fixed numeric binary formats. "
                         "Use --allow-numeric-binary to permit float32 offsets and segmentation chunks; no code is executed.")
    if not 0 <= confidence <= 1 or not 0 <= margin <= 1:
        raise ValueError("Confidence and margin must be in [0,1]")
    if sample < 1 or workers < 1:
        raise ValueError("sample and workers must be positive")
    from cloudvolume import CloudVolume  # Only the pod's match step needs this package.

    cache, out = Path(cache), Path(out)
    remote = Remote(cache, delay, retries)
    roster = pd.read_csv(cache / "neurons.csv", keep_default_na=False)
    if roster.skeleton_id.duplicated().any():
        raise ValueError("Duplicate skeleton IDs in roster")
    ann = annotations(data)
    cells = roster.to_dict("records")
    records = [None] * len(cells)
    failures = []
    prepared = []

    def failed(i, error):
        skid = int(cells[i]["skeleton_id"])
        LOG.exception("Failed skeleton %s; continuing, rerun resumes cached work", skid)
        failures.append(dict(skeleton_id=skid, error=str(error)))
        records[i] = dict(**cells[i], **majority([]), transformed=0, error=str(error))

    # Keep Spine's session, caching and rate limiting serial and unchanged.
    # Finish transforms before installing the CloudVolume-only HTTP wrapper.
    for i, cell in enumerate(cells):
        skid = int(cell["skeleton_id"])
        try:
            points = sample_nodes(read_json(cache / "skeletons" / f"{skid}.json"), sample)
            transformed = transform_points(remote, points, spine)
            valid = np.isfinite(transformed).all(axis=1)
            voxels = np.floor_divide(transformed[valid], [16, 16, 40]).astype(np.int64)
            if (voxels < 0).any():
                raise ValueError("Negative segmentation voxel coordinate")
            prepared.append((i, len(points), voxels))
        except Exception as error:
            failed(i, error)
    LOG.info("Prepared %d/%d skeletons; segmentation lookup with %d workers",
             len(prepared), len(cells), workers)

    # CloudVolume has mutable caches/configuration; conservatively isolate each
    # worker's instance and chunk cache. Point-root JSONs remain shared on disk.
    local = threading.local()
    def lookup(points):
        if not hasattr(local, "volume"):
            # Public HTTPS alias avoids Google credential discovery and CAVE.
            url = "precomputed://https://storage.googleapis.com/flywire_v141_m783"
            volume = remote.attempt("CloudVolume public v783 metadata", lambda: CloudVolume(
                url, mip=0, cache=str(cache / "cloudvolume-v783" / threading.current_thread().name),
                progress=False, parallel=False, bounded=True, fill_missing=False))
            if not np.array_equal(volume.resolution, [16, 16, 40]):
                raise ValueError(f"Unexpected segmentation resolution: {volume.resolution}")
            local.volume = volume
        return remote.attempt(f"v783 batch of {len(points)} voxels",
                              lambda: segmentation_points(local.volume, points))

    def match_skeleton(i, sampled, voxels):
        roots = point_roots(cache, int(cells[i]["skeleton_id"]), voxels, lookup)
        vote = majority(roots)
        vote.update(sampled=sampled, transformed=len(voxels),
                    nonzero_fraction=vote["nonzero"] / sampled)
        return dict(**cells[i], **vote, error="")

    started = time.monotonic()
    completed, processed_points = 0, 0
    with remote.cloud_transport(), ThreadPoolExecutor(max_workers=workers, thread_name_prefix="catmaid-match") as pool:
        pending = {pool.submit(match_skeleton, i, sampled, voxels): (i, len(voxels))
                   for i, sampled, voxels in prepared}
        for future in as_completed(pending):
            i, count = pending[future]
            try:
                records[i] = future.result()
                processed_points += count
            except Exception as error:
                failed(i, error)
            completed += 1
            # Only the coordinator writes CSVs; preserve roster order regardless
            # of completion order. Acceptance is recomputed globally at the end.
            write_csv(cache / "match-progress.csv", pd.DataFrame([r for r in records if r is not None]))
            if completed % 25 == 0 or completed == len(prepared):
                elapsed = max(time.monotonic() - started, 1e-9)
                LOG.info("Match progress %d/%d skeletons: %d points, %.1f points/s (including cached), %d failures",
                         sum(r is not None for r in records), len(cells), processed_points,
                         processed_points / elapsed, len(failures))
    result, audit = qualify(pd.DataFrame(records), ann, confidence, margin)
    write_csv(out, result)
    audit.update(failures=failures, confidence=confidence, margin=margin, sample=sample, workers=workers,
                 segmentation=SEGMENTATION, confidence_denominator="nonzero successful point lookups")
    write_json(out.with_suffix(".json"), audit)
    LOG.info("Match audit: %s", json.dumps(audit))
    if failures:
        raise RuntimeError(f"{len(failures)} skeletons failed; partial results saved to {out}. Rerun to resume.")
    return result


def quick_inputs(cache, bock_dir, samples):
    """Real Bock neurons and counts, synthetic roots, annotations and FlyWire edges."""
    cache = Path(cache)
    cells, edges, rule = bock(bock_dir)
    kcs = cells.loc[cells.role.eq("KC")].head(18)
    connected = edges.loc[edges.target.isin(kcs.skeleton_id)]
    pns = cells.loc[cells.role.eq("PN") & cells.skeleton_id.isin(connected.source)].head(12)
    chosen = pd.concat([pns, kcs]).copy()
    edges = edges.loc[edges.source.isin(pns.skeleton_id) & edges.target.isin(kcs.skeleton_id)]
    write_csv(cache / "bock-neurons.csv", cells)
    _, full, _ = bock(bock_dir)
    write_csv(cache / "bock-edges.csv", full)
    write_csv(cache / "neurons.csv", chosen)
    write_json(cache / "name-rule.json", rule)
    write_json(cache / "pn_targets.json", read_json(Path(samples) / "catmaid/pn_targets.json"))
    roots = {int(s): str(720575940000000000 + i) for i, s in enumerate(chosen.skeleton_id)}
    chosen["root_id"] = chosen.skeleton_id.map(roots)
    chosen["confidence"], chosen["margin"] = 0.9, 0.8
    chosen["flywire_type"] = [r.catmaid_type + "_adPN" if r.role == "PN" else
                              {"KCg": "KCg-m", "KCab": "KCab-c", "KCapbp": "KCapbp-m"}[r.catmaid_type]
                              for r in chosen.itertuples()]
    annotations_frame = chosen[["root_id", "flywire_type"]].rename(columns={"flywire_type": "cell_type"}).assign(side="right")
    data = cache / "flywire"
    data.mkdir(parents=True, exist_ok=True)
    annotations_frame.to_csv(data / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
    qualified, audit = qualify(chosen, annotations_frame.set_index("root_id"))
    write_csv(cache / "catmaid-matches-quick.csv", qualified)
    rng = np.random.default_rng(3)
    fw = edges.copy()
    fw["weight"] = rng.poisson(fw.weight.to_numpy() * 1.4)
    # Add a false-positive pair and a second neuropil row for summation/filtering.
    nonedges = [(int(p), int(k)) for p in pns.skeleton_id for k in kcs.skeleton_id
                if not ((edges.source == p) & (edges.target == k)).any()]
    if nonedges:
        fw = pd.concat([fw, pd.DataFrame([dict(source=nonedges[0][0], target=nonedges[0][1], weight=7)])])
    fw["neuropil"] = "CA_R"
    outside = fw.head(3).copy().assign(weight=3, neuropil="LH_R")
    fw = pd.concat([fw, outside], ignore_index=True)
    for column in ("source", "target"):
        fw[column] = fw[column].map(roots).astype("uint64")
    fw = fw.rename(columns={"source": "pre_pt_root_id", "target": "post_pt_root_id", "weight": "syn_count"})
    fw.to_feather(data / "proofread_connections_783.feather")
    # Exercise the connectivity alternative with actual parsed API rows too.
    live = pd.concat([full, connectivity(read_json(Path(samples) / "catmaid/conn_27295.json"))], ignore_index=True)
    live = live.groupby(["source", "target"], as_index=False).weight.sum()
    write_csv(cache / "connectivity.csv", live)
    write_json(cache / "fetch.json", dict(synthetic=True, name_rule=rule,
                                          matrix_vs_live=agreement(full, live.loc[live.source.isin(cells.loc[cells.role.eq('PN'), 'skeleton_id']) &
                                                                                 live.target.isin(cells.loc[cells.role.eq('KC'), 'skeleton_id'])], 440 * 113)))
    LOG.info("Synthetic inputs: %s; match audit %s", cache, audit)
    return cache


def check(samples="/tmp/neuroevo-samples", bock_dir="/tmp/neuroevo-ref/bocklab", quick_cache=None):
    import requests
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    samples = Path(samples)
    assert csrf_headers(BASE + "/", "test") == {"X-CSRFToken": "test", "Referer": BASE + "/"}
    try:
        csrf_headers(BASE, None)
        raise AssertionError("Missing CSRF token accepted")
    except ValueError:
        pass
    target = targets(read_json(samples / "catmaid/pn_targets.json"))
    assert len(target) == 145 and int(target.iloc[0].skeleton_id) == 16
    raw_nodes = read_json(samples / "catmaid/skel_27295.json")
    xyz = nodes(raw_nodes)
    assert xyz.shape == (10216, 3)
    assert np.allclose(xyz[0], [444160.12, 140584.42, 196880.])
    assert sample_nodes(raw_nodes).shape == (50, 3)
    assert sample_nodes(raw_nodes, 200).shape == (200, 3)
    conn = read_json(samples / "catmaid/conn_27295.json")
    edges = connectivity(conn)
    assert edges.loc[edges.target.eq(12070), "weight"].item() == 19
    assert edges.loc[edges.target.eq(11146), "weight"].item() == 25
    assert edges.weight.sum() == sum(sum(r["skids"]["27295"]) for r in conn["outgoing"].values())
    assert connectivity(conn, "incoming").target.eq(27295).all()
    cells, matrix, rule = bock(bock_dir)
    assert len(cells) == 553 and len(matrix) > 0
    assert rule["PN"]["modal_name_minus_skeleton"] == rule["KC"]["modal_name_minus_skeleton"] == 1
    assert (rule["PN"]["obey_modal"], rule["PN"]["exceptions"]) == (96, 17)
    assert (rule["KC"]["obey_modal"], rule["KC"]["exceptions"]) == (388, 51)
    assert len(rule["KC"]["fallback"]) == 1
    assert cells.loc[cells.name.eq("KCy 16943 BH"), "skeleton_id"].item() == 3104789
    vote = majority([0, 21, 21, 21, 22])
    assert vote["root_id"] == "21" and vote["confidence"] == .75 and vote["margin"] == .5
    assert majority([0, 0])["root_id"] == "0" and majority([1, 2])["margin"] == 0
    assert majority([720575940000000001])["root_id"] == "720575940000000001"
    assert compatible("PN", "VA4", "VA4_adPN") and not compatible("PN", "VA1", "VA1d_adPN")
    assert compatible("KC", "KCapbp", "KCapbp-m") and not compatible("KC", "KCg", "KCab-c")
    sample_cells = pd.DataFrame(dict(skeleton_id=[1, 2, 3, 4], role=["PN", "PN", "KC", "KC"], catmaid_type=["VA4", "DL1", "KCg", "KCab"]))
    a = pd.DataFrame(dict(source=[1, 2, 1, 2], target=[3, 3, 4, 4], weight=[8., 16., 32., 64.]))
    for resolution in ("neuron", "type"):
        identical = compare_networks(a, a, sample_cells, resolution)
        assert all(abs(identical["levels"][k]["dissimilarity"]) < 1e-12 for k in LEVELS)
        scaled = compare_networks(a, a.assign(weight=a.weight * 3), sample_cells, resolution)
        assert scaled["levels"]["strength"]["sd_log2_ratio"] < 1e-12
        assert scaled["levels"]["strength"]["dissimilarity"] < 1e-12
    stats = pair_statistics(paired_counts(a, a), 4)
    assert stats["classification"][-1]["pair_accuracy"] == 1 and stats["median_flywire_over_catmaid"] == 1
    small = a.assign(weight=[1., 3., 5., 8.])
    altered = small.assign(weight=[0., 4., 4., 8.])
    thresholds = compare_networks(small, altered, sample_cells, "neuron")["thresholds"]
    assert np.allclose([r["connection"]["dissimilarity"] for r in thresholds], [.25, 0, .5])
    empty = compare_networks(a, a.iloc[:0], sample_cells, "neuron")
    assert empty["levels"]["connection"]["dissimilarity"] == 1
    assert empty["levels"]["strength"]["dissimilarity"] is None
    ann = pd.DataFrame(dict(root_id=["10", "20", "30"], cell_type=["VA4_adPN", "DL1_adPN", "KCg-m"])).set_index("root_id")
    candidates = sample_cells.assign(root_id=["10", "20", "30", "30"], confidence=1., margin=1.)
    accepted, audit = qualify(candidates, ann)
    assert audit["accepted"] == 2 and audit["conflicts"] == {"30": [3, 4]}
    assert not accepted.loc[accepted.root_id.eq("30"), "accepted"].any()
    # Exercise the actual concurrent match path without CloudVolume or network.
    with TemporaryDirectory() as directory, patch("time.sleep"):
        cache = Path(directory)
        batches = []
        instances = []
        barrier = threading.Barrier(2)
        class BatchedVolume:
            resolution = [16, 16, 40]
            def __init__(self, *args, **kwargs):
                self.owner = threading.get_ident()
                instances.append(self)
            def scattered_points(self, points):
                assert self.owner == threading.get_ident()
                batches.append(points)
                barrier.wait(timeout=5)  # Both skeletons must run concurrently.
                return {point: point[0] * 10 for point in reversed(points)}
            def __getitem__(self, key):
                raise AssertionError("Batched path made a scalar read")
        write_csv(cache / "neurons.csv", sample_cells.iloc[:2])
        for skid in (1, 2):
            write_json(cache / "skeletons" / f"{skid}.json",
                       [[[n, -1, 0, skid * 16, y * 16, 40] for n, y in enumerate((1, 2, 2))]])
        with patch.dict("sys.modules", cloudvolume=SimpleNamespace(CloudVolume=BatchedVolume)), \
                patch.dict(match.__globals__, annotations=lambda data: ann,
                           transform_points=lambda remote, points, spine: points):
            result = match(cache, out=cache / "matches.csv", workers=2, retries=1, allow_numeric_binary=True)
            assert len(batches) == 2 and all(len(batch) == 2 for batch in batches)
            assert len(instances) == 2
            assert result.root_id.tolist() == ["10", "20"] and result.accepted.all()
            assert result.nonzero.tolist() == [3, 3]  # Duplicate points still vote.
            files = list((cache / "point-roots-v783").rglob("*.json"))
            assert len(files) == 2  # One cache write/file per skeleton.
            resumed = match(cache, out=cache / "matches.csv", workers=2, allow_numeric_binary=True)
            assert len(batches) == 2 and len(instances) == 2
            assert resumed.root_id.tolist() == result.root_id.tolist()
        # Legacy caches, changed samples, zeros and 64-bit labels retain exact values.
        large = 720575940000000001
        write_json(cache / "point-roots-v783" / "9_8_7.json", dict(root_id=str(large)))
        extra_batches = []
        def extra_lookup(points):
            extra_batches.append(points)
            return {point: 0 for point in points}
        assert point_roots(cache, 1, [(9, 8, 7), (1, 1, 1), (4, 5, 6)], extra_lookup) == [large, 10, 0]
        assert extra_batches == [[(4, 5, 6)]]
        assert point_roots(cache, 1, [(4, 5, 6), (9, 8, 7)], extra_lookup) == [0, large]
        assert len(extra_batches) == 1
        assert point_roots(cache, 3, [], extra_lookup) == [] and len(extra_batches) == 1
    # Old versions download each chunk once, including offset and edge chunks.
    class ChunkVolume:
        voxel_offset = [3, 5, 7]
        chunk_size = [4, 4, 4]
        bounds = SimpleNamespace(minpt=voxel_offset, maxpt=[9, 9, 11])
        def __init__(self):
            self.reads = []
        def __getitem__(self, slices):
            self.reads.append(slices)
            x, y, z = np.meshgrid(*(np.arange(s.start, s.stop) for s in slices), indexing="ij")
            return (x + 10 * y + 100 * z).astype(np.uint64)[..., None] + np.uint64(large)
    volume = ChunkVolume()
    points = [(3, 5, 7), (6, 8, 10), (8, 8, 10), (3, 5, 7)]
    labels = segmentation_points(volume, points)
    assert labels == {p: large + p[0] + 10 * p[1] + 100 * p[2] for p in points}
    assert len(volume.reads) == 2 and volume.reads[-1][0] == slice(7, 9)
    assert segmentation_points(volume, []) == {} and len(volume.reads) == 2
    # A fake session exercises POST headers, bootstrap, transient failures, and resume.
    with TemporaryDirectory() as directory, patch("time.sleep"):
        remote = Remote(directory)
        events = []
        class Reply:
            status_code = 200
            content = b"<html></html>"
            def raise_for_status(self):
                pass
            def json(self):
                return {"ok": True}
        class Session(requests.Session):
            def get(self, url, **kwargs):
                events.append("csrf")
                domain = "fafb.catmaid.virtualflybrain.org"
                self.cookies.set("unrelated", "ignored", domain=domain, path="/")
                self.cookies.set("csrftoken_fafb", "fake", domain=domain, path="/")
                self.cookies.set("csrftoken_other", "later", domain=domain, path="/")
                return Reply()
            def request(self, method, url, **kwargs):
                prepared = self.prepare_request(requests.Request(method, url, headers=kwargs["headers"]))
                assert "csrftoken_fafb=fake" in prepared.headers["Cookie"]
                events.append(kwargs["headers"])
                if len(events) == 2:
                    raise OSError("synthetic transient failure")
                return Reply()
        remote.session = Session()
        url = BASE + "/1/test"
        assert remote.request("POST", url, {"x": "1"}, catmaid=True) == {"ok": True}
        assert events[1] == csrf_headers(BASE, "fake") and len(events) == 3
        assert events[2] == csrf_headers(BASE, "fake")
        assert read_json(Path(directory) / "csrf.json") == dict(status=200, name="csrftoken_fafb", token="fake")
        remote.request("POST", url, {"x": "1"}, catmaid=True)
        assert len(events) == 3
        # Synthetic numeric bytes only: verify exact wire shape and nm offset units offline.
        class SpineMock:
            def request(self, method, url, data=None, decoder=None):
                if method == "GET":
                    return {"flywire_v1_inverse": {"voxel_size": [4, 4, 40], "scales": [4]}}
                assert url.endswith("/s/4/values_binary/format/array_float_Nx3")
                assert np.array_equal(np.frombuffer(data, dtype="<f4"), [1, 2, 3])
                response = Reply()
                response.content = np.array([[2, -3]], dtype="<f4").tobytes()
                return decoder(response)
        assert np.array_equal(transform_points(SpineMock(), np.array([[7., 11., 139.]])), [[15., -1., 139.]])
    if quick_cache:
        quick_inputs(quick_cache, bock_dir, samples)
    LOG.info("check passed: fixtures, name exceptions, CSRF/retry/cache, matching, units, metrics")
    print("catmaid check: passed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    checker = sub.add_parser("check")
    checker.add_argument("--samples", default="/tmp/neuroevo-samples")
    checker.add_argument("--bock", dest="bock_dir", default="/tmp/neuroevo-ref/bocklab")
    checker.add_argument("--quick-cache", type=Path)
    fetcher = sub.add_parser("fetch")
    fetcher.add_argument("--bock", dest="bock_dir", type=Path, default=Path("/root/data/bocklab"))
    fetcher.add_argument("--base", default=BASE)
    fetcher.add_argument("--batch", type=int, default=25)
    matcher = sub.add_parser("match")
    matcher.add_argument("--sample", type=int, default=50)
    matcher.add_argument("--workers", type=int, default=16)
    matcher.add_argument("--spine", default=SPINE)
    matcher.add_argument("--allow-numeric-binary", action="store_true")
    matcher.add_argument("--out", type=Path, default=OUTPUT / "catmaid-matches.csv")
    runner = sub.add_parser("run")
    runner.add_argument("--matches", type=Path, default=OUTPUT / "catmaid-matches.csv")
    runner.add_argument("--out", type=Path, default=OUTPUT / "catmaid.json")
    runner.add_argument("--observed", type=Path, default=OUTPUT / "connectome.json")
    runner.add_argument("--noise", dest="noise_path", type=Path, default=OUTPUT / "noise.json")
    runner.add_argument("--calyx", default="CA_R")
    runner.add_argument("--synthetic", action="store_true")
    for command in (fetcher, matcher, runner):
        command.add_argument("--cache", type=Path, required=True)
    for command in (fetcher, matcher):
        command.add_argument("--delay", type=float, default=.15)
        command.add_argument("--retries", type=int, default=5)
    for command in (matcher, runner):
        command.add_argument("--data", type=Path, default=Path("/root/data/flywire"))
        command.add_argument("--confidence", type=float, default=.5)
        command.add_argument("--margin", type=float, default=.2)
    renderer = sub.add_parser("figure")
    renderer.add_argument("path", nargs="?", type=Path, default=OUTPUT / "catmaid.json")
    renderer.add_argument("--out", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if command in ("fetch", "match"):
        args["cache"].mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(args["cache"] / (command + ".log"))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
    dict(check=check, fetch=fetch, match=match, run=run, figure=figure)[command](**args)
