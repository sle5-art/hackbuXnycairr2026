"""
SensorGap model: a Gaussian process over NYC that returns an estimate AND a calibrated uncertainty everywhere.

Design choices (each one is something we can defend to a judge):
  - Inputs are x/y in kilometres (local projection), so the length scale reads as "how far a reading is informative".
  - Each sensor type gets its own noise level: reference monitors are trusted more than corrected PurpleAirs.
    sklearn takes this as a per-point `alpha` (noise variance added to the diagonal).
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

# hitting a bound is expected on very clean, flat days; it isn't an error worth printing hundreds of times
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]

# NYC-area bounding box (same as ingest)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 40.49, 40.92, -74.26, -73.70
LAT0, LON0 = 40.7, -74.0  # projection origin

# Measurement noise (standard deviation, ug/m3) per sensor type is LEARNED from each snapshot:
# we try every pair on this grid (low-cost never quieter than reference) and keep the one with the highest
# marginal likelihood. NOISE_SD is only the fallback when a caller passes noise explicitly.
NOISE_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
NOISE_SD = {"reference": 1.0, "lowcost": 3.0}


def to_xy_km(lat, lon):
    """Equirectangular projection: accurate to well under 1% across a city."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - LON0) * 111.320 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110.574
    return np.column_stack([x, y])


# Shortest length scale the model may learn. NYC sensors are typically ~1.2 km from their nearest neighbour,
# so the data can't tell apart scales below ~1 km. Without this floor, one hotspot (e.g. Midtown West reading 3x its
# surroundings) can collapse the scale to a few hundred metres and flatten the whole map.
MIN_LENGTH_SCALE_KM = 1.0


def make_kernel():
    return ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
        length_scale=3.0, length_scale_bounds=(MIN_LENGTH_SCALE_KM, 100.0), nu=1.5)


def _fit_fixed(X, y, classes, noise_sd, restarts, seed):
    # normalize_y=True standardises y before fitting and sklearn adds alpha in that standardised space,
    # so convert our noise (in ug/m3) to the same scale
    y_std = float(np.std(y)) or 1.0
    alpha = (np.array([noise_sd[c] for c in classes]) / y_std) ** 2
    gp = GaussianProcessRegressor(kernel=make_kernel(), alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=restarts, random_state=seed)
    gp.fit(X, y)
    gp.noise_sd_ = dict(noise_sd)
    return gp


def learn_noise(df):
    """Pick per-type noise levels by maximum marginal likelihood over NOISE_GRID."""
    df = df.dropna(subset=["lat", "lon", "pm25"])
    X, y, classes = to_xy_km(df["lat"], df["lon"]), df["pm25"].to_numpy(float), df["noise_class"].to_numpy()
    best = None
    for ref, lc in itertools.product(NOISE_GRID, NOISE_GRID):
        if lc < ref:
            continue
        gp = _fit_fixed(X, y, classes, {"reference": ref, "lowcost": lc}, restarts=0, seed=0)
        # normalize_y scales the likelihood; add back the log-Jacobian so different noise levels compare fairly
        lml = gp.log_marginal_likelihood_value_ - len(y) * np.log(gp._y_train_std)
        if best is None or lml > best[0]:
            best = (lml, {"reference": ref, "lowcost": lc})
    return best[1]


def fit(df, noise_sd=None, restarts=3, seed=0):
    """Fit a GP to one snapshot. df needs lat, lon, pm25, noise_class.
    noise_sd=None learns the noise levels from the data; pass a dict to fix them."""
    df = df.dropna(subset=["lat", "lon", "pm25"])
    if noise_sd is None:
        noise_sd = learn_noise(df)
    unknown = set(df["noise_class"]) - set(noise_sd)
    if unknown:
        raise ValueError(f"No noise level for sensor types {unknown}")
    X = to_xy_km(df["lat"], df["lon"])
    return _fit_fixed(X, df["pm25"].to_numpy(float), df["noise_class"].to_numpy(), noise_sd, restarts, seed)


def predict(gp, lat, lon):
    """Mean and standard deviation of the underlying PM2.5 field (not of a noisy sensor reading)."""
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
    """Band-widening factor measured by validate.py on held-out monitors (1.0 if not run yet)."""
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
        "signal_sd_scaled": float(np.sqrt(k.k1.constant_value)),  # in units of normalized y
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
