import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sensorgap import model as m

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    (9.0, "Good"),
    (35.4, "Moderate"),
    (55.4, "Unhealthy for Sensitive Groups"),
    (125.4, "Unhealthy"),
    (225.4, "Very Unhealthy"),
    (np.inf, "Hazardous"),
]
_UPPER = np.array([c[0] for c in CATEGORIES])
_NAMES = np.array([c[1] for c in CATEGORIES])


def category(pm25):
    c = np.floor(np.asarray(pm25, dtype=float) * 10 + 1e-9) / 10
    return _NAMES[np.searchsorted(_UPPER, c, side="left")]


def nowcast_weights(hourly):
    h = np.clip(hourly, 0, None)
    cmin = np.nanmin(h, axis=0)
    cmax = np.nanmax(h, axis=0)
    w = np.where(cmax > 0, np.maximum(cmin / np.where(cmax > 0, cmax, 1), 0.5), 1.0)
    powers = w[None, :] ** np.arange(hourly.shape[0])[:, None]
    powers = np.where(np.isnan(hourly), 0.0, powers)
    return powers / powers.sum(axis=0)


def nowcast(hourly_mean, hourly_std):
    a = nowcast_weights(hourly_mean)
    mean = np.nansum(a * hourly_mean, axis=0)
    std = np.nansum(a * hourly_std, axis=0)
    return mean, std


def label(lower, upper):
    cat_lower = category(lower)
    cat_upper = category(upper)
    confident = cat_lower == cat_upper
    text = np.where(confident, cat_lower, np.char.add(np.char.add(cat_lower.astype(str), " to "),
                                                        np.char.add(cat_upper.astype(str), ", not enough data")))
    return text, confident


def _stamp(path, prefix):
    return datetime.strptime(path.stem.replace(prefix, ""), "%Y%m%dT%H%M%SZ")


def make_missing_grids(hours=12):
    snaps = sorted((ROOT / "data" / "snapshots").glob("snapshot_*.csv"))
    if not snaps:
        return
    newest = _stamp(snaps[-1], "snapshot_")
    by_hour = {}
    for s in snaps:
        lag = int(round((newest - _stamp(s, "snapshot_")).total_seconds() / 3600))
        if lag < hours:
            by_hour[lag] = s
    sd_scale = m.load_sd_scale()
    for s in by_hour.values():
        out = ROOT / "outputs" / s.name.replace("snapshot_", "grid_")
        if not out.exists():
            print(f"Making missing map {out.name}")
            out.parent.mkdir(exist_ok=True)
            m.predict_grid(m.fit(pd.read_csv(s)), sd_scale=sd_scale).to_csv(out, index=False)


def load_hourly_grids(paths, hours=12):
    stamped = sorted((_stamp(p, "grid_"), p) for p in paths)
    newest_t = stamped[-1][0]
    newest = stamped[-1][1]
    base = pd.read_csv(newest)[["lat", "lon"]]
    mean = np.full((hours, len(base)), np.nan)
    std = np.full((hours, len(base)), np.nan)
    for item in stamped:
        t = item[0]
        p = item[1]
        lag = int(round((newest_t - t).total_seconds() / 3600))
        g = pd.read_csv(p)
        if lag >= hours or len(g) != len(base):
            continue
        mean[lag] = g["mean"].to_numpy()
        std[lag] = g["std"].to_numpy()
    return base, mean, std, newest


def make_labels(base, mean, std):
    if np.isnan(mean[:3]).sum(axis=0).max() > 1:
        print("Fewer than 2 of the last 3 hours available: labelling the latest hour only (not a NowCast).")
        mean = mean[:1]
        std = std[:1]
    nowcast_result = nowcast(mean, std)
    nc = nowcast_result[0]
    nc_std = nowcast_result[1]
    lower = np.clip(nc - 1.96 * nc_std, 0, None)
    upper = nc + 1.96 * nc_std
    label_result = label(lower, upper)
    text = label_result[0]
    confident = label_result[1]
    return base.assign(nowcast=nc.round(2), lo95=lower.round(2), hi95=upper.round(2), label=text, confident=confident)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=12)
    args = p.parse_args()

    make_missing_grids(args.hours)
    paths = list((ROOT / "outputs").glob("grid_*.csv"))
    if not paths:
        raise SystemExit("No snapshots yet: run python -m sensorgap.ingest first.")
    grids = load_hourly_grids(paths, args.hours)
    base = grids[0]
    mean = grids[1]
    std = grids[2]
    newest = grids[3]
    n_hours = int((~np.isnan(mean[:, 0])).sum())
    out = make_labels(base, mean, std)
    path = ROOT / "outputs" / newest.name.replace("grid_", "labels_")
    out.to_csv(path, index=False)

    print(f"Averaged {n_hours} hourly map(s) ending {newest.stem.replace('grid_', '')}")
    print(f"Cells with a confident label: {out['confident'].mean():.0%} of {len(out)}")
    print(out["label"].value_counts().to_string())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
