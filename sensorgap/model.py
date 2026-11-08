"""
SensorGap model: a Gaussian process over NYC that returns an estimate AND a calibrated uncertainty everywhere.

Design choices (each one is something we can defend to a judge):
  - Inputs are x/y in kilometres (local projection), so the length scale reads as "how far a reading is informative".
  - Each sensor type gets its own noise level: reference monitors are trusted more than corrected PurpleAirs.
    sklearn takes this as a per-point `alpha` (noise variance added to the diagonal).
  - PurpleAirs read low next to monitors (~1.6 ug/m3 on average, varying by hour). lowcost_offset() measures that
    gap every hour; shifting PurpleAirs by it was tested and made held-out error worse, so it is off by default
    (CORRECT_LOWCOST_BIAS) and the gap is reported instead.
  - Matern(nu=1.5) kernel: rougher than the default RBF, which suits pollution near local sources.
  - Hyperparameters (signal size, length scale) are re-fit on every snapshot, so the model widens its reach on
    regional smoke days and narrows it on local-source days.

Usage:
    python -m sensorgap.model data/snapshots/snapshot_<UTC>.csv
        -> prints the fitted kernel and writes outputs/grid_<UTC>.csv (lat, lon, mean, std, lo95, hi95)
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 40.49, 40.92, -74.26, -73.70
LAT0, LON0 = 40.7, -74.0

NOISE_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
NOISE_SD = {"reference": 1.0, "lowcost": 3.0}


def to_xy_km(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - LON0) * 111.320 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110.574
    return np.column_stack([x, y])


MIN_LENGTH_SCALE_KM = 1.0


PAIR_KM = 1.0
MIN_PAIRS = 3
CORRECT_LOWCOST_BIAS = False


def lowcost_offset(df):
    df = df.dropna(subset=["lat", "lon", "pm25"])
    ref, lc = df[df["noise_class"] == "reference"], df[df["noise_class"] == "lowcost"]
    if ref.empty or lc.empty:
        return 0.0, 0
    a, b = to_xy_km(ref["lat"], ref["lon"]), to_xy_km(lc["lat"], lc["lon"])
    near = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)) <= PAIR_KM
    lc_vals = lc["pm25"].to_numpy(float)
    gaps = [r - lc_vals[row].mean() for r, row in zip(ref["pm25"].to_numpy(float), near) if row.any()]
    if len(gaps) < MIN_PAIRS:
        return 0.0, len(gaps)
    return float(np.median(gaps)), len(gaps)


def correct_lowcost(df):
    offset, n = lowcost_offset(df)
    out = df.copy()
    out.loc[out["noise_class"] == "lowcost", "pm25"] = out.loc[out["noise_class"] == "lowcost", "pm25"] + offset
    out.attrs["lowcost_offset"], out.attrs["lowcost_pairs"] = offset, n
    return out


def make_kernel():
    return ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
        length_scale=3.0, length_scale_bounds=(MIN_LENGTH_SCALE_KM, 100.0), nu=1.5)


def _fit_fixed(X, y, classes, noise_sd, restarts, seed):
    y_std = float(np.std(y)) or 1.0
    alpha = (np.array([noise_sd[c] for c in classes]) / y_std) ** 2
    gp = GaussianProcessRegressor(kernel=make_kernel(), alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=restarts, random_state=seed)
    gp.fit(X, y)
    gp.noise_sd_ = dict(noise_sd)
    return gp


def learn_noise(df, correct_bias=None):
    df = df.dropna(subset=["lat", "lon", "pm25"])
    if CORRECT_LOWCOST_BIAS if correct_bias is None else correct_bias:
        df = correct_lowcost(df)
    X, y, classes = to_xy_km(df["lat"], df["lon"]), df["pm25"].to_numpy(float), df["noise_class"].to_numpy()
    best = None
    for ref, lc in itertools.product(NOISE_GRID, NOISE_GRID):
        if lc < ref:
            continue
        gp = _fit_fixed(X, y, classes, {"reference": ref, "lowcost": lc}, restarts=0, seed=0)
        lml = gp.log_marginal_likelihood_value_ - len(y) * np.log(gp._y_train_std)
        if best is None or lml > best[0]:
            best = (lml, {"reference": ref, "lowcost": lc})
    return best[1]


def fit(df, noise_sd=None, restarts=3, seed=0, correct_bias=None):
    df = df.dropna(subset=["lat", "lon", "pm25"])
    if CORRECT_LOWCOST_BIAS if correct_bias is None else correct_bias:
        df = correct_lowcost(df)
    if noise_sd is None:
        noise_sd = learn_noise(df, correct_bias=False)
    unknown = set(df["noise_class"]) - set(noise_sd)
    if unknown:
        raise ValueError(f"No noise level for sensor types {unknown}")
    X = to_xy_km(df["lat"], df["lon"])
    gp = _fit_fixed(X, df["pm25"].to_numpy(float), df["noise_class"].to_numpy(), noise_sd, restarts, seed)
    gp.lowcost_offset_ = df.attrs.get("lowcost_offset", 0.0)
    gp.lowcost_pairs_ = df.attrs.get("lowcost_pairs", 0)
    return gp


def predict(gp, lat, lon):
    mean, std = gp.predict(to_xy_km(lat, lon), return_std=True)
    return mean, std


def make_grid(spacing_km=0.5):
    dlat = spacing_km / 110.574
    dlon = spacing_km / (111.320 * np.cos(np.radians(LAT0)))
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, dlat)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, dlon)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return LA.ravel(), LO.ravel()


def load_sd_scale():
    f = ROOT / "outputs" / "calibration.json"
    return float(json.loads(f.read_text())["sd_scale"]) if f.exists() else 1.0


def predict_grid(gp, spacing_km=0.5, sd_scale=1.0):
    lat, lon = make_grid(spacing_km)
    mean, std = predict(gp, lat, lon)
    std = std * sd_scale
    return pd.DataFrame({
        "lat": lat, "lon": lon,
        "mean": mean.round(2), "std": std.round(2),
        "lo95": np.clip(mean - 1.96 * std, 0, None).round(2),
        "hi95": (mean + 1.96 * std).round(2),
    })


def describe(gp):
    k = gp.kernel_
    return {
        "signal_sd_scaled": float(np.sqrt(k.k1.constant_value)),
        "length_scale_km": float(k.k2.length_scale),
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", help="a data/snapshots/snapshot_<UTC>.csv file")
    p.add_argument("--spacing-km", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_csv(args.snapshot)
    gp = fit(df)
    info = describe(gp)
    print(f"Fitted on {len(df)} sensors ({df['noise_class'].value_counts().to_dict()})")
    off, n_pairs = lowcost_offset(df)
    print(f"PurpleAir reads {off:+.2f} ug/m3 vs monitors within {PAIR_KM:g} km ({n_pairs} pairs); "
          + ("corrected" if gp.lowcost_pairs_ else "not corrected (CORRECT_LOWCOST_BIAS is off)"))
    print(f"Learned noise sd (ug/m3): {gp.noise_sd_}")
    print(f"Length scale: {info['length_scale_km']:.2f} km   kernel: {gp.kernel_}")

    k = load_sd_scale()
    print(f"Calibration: widening bands x{k:.2f}" + ("" if k != 1.0 else " (run sensorgap.validate to measure it)"))
    grid = predict_grid(gp, args.spacing_km, sd_scale=k)
    out = ROOT / "outputs" / f"grid_{Path(args.snapshot).stem.replace('snapshot_', '')}.csv"
    out.parent.mkdir(exist_ok=True)
    grid.to_csv(out, index=False)
    print(f"Wrote {len(grid)} grid cells to {out}")
    print(f"Band half-width (1.96*std): median {1.96 * grid['std'].median():.1f}, "
          f"max {1.96 * grid['std'].max():.1f} ug/m3")


if __name__ == "__main__":
    main()
