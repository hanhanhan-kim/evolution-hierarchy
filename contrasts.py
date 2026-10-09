"""Natural-image local contrasts for experiment F2 (Laughlin 1981 reference).

Run: python contrasts.py [check | run] [--images /root/data/vanhateren]
     [--sigma 8] [--output output/contrasts.json]
For the two local examples use --images /tmp/neuroevo-samples/vanhateren
--output output/contrasts-sample.json; these are not a population estimate.
Laughlin measured horizontal scans with a fly-relevant photometer in lakeside
scenes. Van Hateren & van der Schaaf (1998) provide luminance-calibrated scenes
from the Netherlands, so this is an approximate match. We pool horizontal
scans after computing a 2D Gaussian local mean (sigma in pixels, about one
arcminute/pixel); sigma=8 is a spatial-scale proxy, not a fitted fly receptive
field. Sensitivity uses half and twice the requested sigma.

IML is headerless big-endian uint16, height 1024 by width 1536, read ONLY with
numpy.fromfile. Zero and saturated pixels are omitted from both means and
samples. By default each image's maximum is conservatively treated as saturated
(the samples have a common ceiling of 6282); --saturation overrides this with
a known ceiling. A four-sigma border is removed. Histogram centres equal F's
64 contrast points; Voronoi bins have tails folded into the end bins. Moments
are reported both before clipping and on the discrete grid. KS compares the
two grid CDFs, not independent-pixel significance. Pixels are correlated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from code import Channel, OUTPUT

SHAPE = (1024, 1536)
GRID = np.linspace(-1, 1, 64)
EDGES = np.r_[-np.inf, (GRID[1:] + GRID[:-1]) / 2, np.inf]


def read_image(path):
    if path.stat().st_size != 2 * np.prod(SHAPE):
        raise ValueError(f"{path}: expected exactly {2 * np.prod(SHAPE)} bytes")
    pixels = np.fromfile(path, dtype=">u2", count=int(np.prod(SHAPE)) + 1)
    if pixels.size != np.prod(SHAPE):
        raise ValueError(f"{path}: incomplete or oversized image")
    return pixels.reshape(SHAPE).astype(np.float64)


def local_contrasts(image, sigma=8, saturation=None):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    border = int(np.ceil(4 * sigma))
    if 2 * border >= min(image.shape):
        raise ValueError("sigma must be positive and leave an interior after 4-sigma cropping")
    ceiling = float(image.max() if saturation is None else saturation)
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("saturation must be positive")
    valid = (image > 0) & (image < ceiling)
    weight = gaussian_filter(valid.astype(float), sigma, truncate=4)
    total = gaussian_filter(np.where(valid, image, 0), sigma, truncate=4)
    mean = total / np.maximum(weight, np.finfo(float).tiny)
    interior = np.zeros(image.shape, dtype=bool)
    interior[border:-border, border:-border] = True
    keep = interior & valid & (mean > 0)
    return (image[keep] - mean[keep]) / mean[keep], ceiling


def moments(raw):
    m1, m2, m3, m4 = raw
    variance = max(0., m2 - m1**2)
    return dict(mean=float(m1), variance=float(variance),
                skewness=float((m3 - 3 * m1 * m2 + 2 * m1**3) / variance**1.5) if variance else None,
                kurtosis=float((m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / variance**2) if variance else None)


def grid_moments(p):
    return moments(np.array([p @ GRID**i for i in range(1, 5)]))


def extract(paths, scales=(4, 8, 16), saturation=None):
    hist = np.zeros((len(scales), 64), dtype=np.int64)
    powers = np.zeros((len(scales), 4))
    clipped = np.zeros((len(scales), 2), dtype=np.int64)
    images = [[] for _ in scales]
    for path in paths:
        image = read_image(path)
        for j, sigma in enumerate(scales):
            values, ceiling = local_contrasts(image, sigma, saturation)
            hist[j] += np.histogram(values, EDGES)[0]
            powers[j] += [np.sum(values**i) for i in range(1, 5)]
            clipped[j] += [np.sum(values < -1), np.sum(values > 1)]
            images[j].append(dict(path=str(path.resolve()), pixels=int(values.size), saturation=ceiling))
    standin = Channel().p
    rows = []
    for j, sigma in enumerate(scales):
        count = int(hist[j].sum())
        if not count:
            raise ValueError("No valid contrast samples")
        p = hist[j] / count
        rows.append(dict(sigma_pixels=sigma, window="2D Gaussian, truncate=4 sigma",
                         border_pixels=int(np.ceil(4 * sigma)), pixel_count=count, images=images[j],
                         probabilities=p.tolist(), counts=hist[j].tolist(),
                         moments=moments(powers[j] / count), grid_moments=grid_moments(p),
                         fraction_clipped=float(clipped[j].sum() / count),
                         fraction_clipped_low=float(clipped[j, 0] / count),
                         fraction_clipped_high=float(clipped[j, 1] / count),
                         comparison=dict(ks_distance=float(np.max(np.abs(np.cumsum(p - standin)))),
                                         measured_grid_kurtosis=grid_moments(p)["kurtosis"],
                                         standin_grid_kurtosis=grid_moments(standin)["kurtosis"])))
    return rows


def check():
    image = np.full((80, 90), 100.)
    image[35, 35] = 0
    image[35, 36] = 65535
    values, _ = local_contrasts(image, 2, 65535)
    assert np.max(np.abs(values)) < 1e-12
    assert len(values) == (80 - 16) * (90 - 16) - 2
    h = np.histogram([-4, -1, 0, 1, 8], EDGES)[0]
    assert h.sum() == 5 and h[0] == h[-1] == 2
    p = np.ones(64) / 64
    assert np.isclose(grid_moments(p)["mean"], 0)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.iml"
        path.write_bytes(b"not an image")
        try:
            read_image(path)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed image accepted")
        original = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
        original.astype(">u2").tofile(path)
        assert np.array_equal(read_image(path), original)
    print("check ok: endian/size, invalid pixels, local mean, borders, tail bins")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"), nargs="?", default="run")
    parser.add_argument("--images", type=Path, default=Path("/root/data/vanhateren"))
    parser.add_argument("--sigma", type=float, default=8)
    parser.add_argument("--saturation", type=float)
    parser.add_argument("--output", type=Path, default=OUTPUT / "contrasts.json")
    args = parser.parse_args()
    if args.command == "check":
        check()
        return
    paths = sorted(args.images.glob("*.iml"))
    if not paths:
        parser.error(f"No .iml files in {args.images}")
    if len(paths) <= 2 and args.output.name == "contrasts.json":
        parser.error("Use --output output/contrasts-sample.json for a two-image sample")
    rows = extract(paths, (args.sigma / 2, args.sigma, args.sigma * 2), args.saturation)
    data = dict(schema=1, dataset="van Hateren linearized IML", sample=len(paths) <= 2,
                contrasts=GRID.tolist(), **rows[1], sensitivity=[rows[0], rows[2]],
                saturation_policy="per-image maximum (conservative)" if args.saturation is None else "explicit ceiling",
                note="Approximate Laughlin spatial contrast proxy; correlated pixels; not photometer replication.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    for row in rows:
        print(f"sigma={row['sigma_pixels']:g}px: {row['pixel_count']:,} pixels; "
              f"clipped={row['fraction_clipped']:.3%}; KS={row['comparison']['ks_distance']:.4f}; "
              f"kurtosis raw/grid={row['moments']['kurtosis']:.3f}/{row['grid_moments']['kurtosis']:.3f}")
    print(args.output)


if __name__ == "__main__":
    main()
