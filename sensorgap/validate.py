"""
SensorGap validation: are the 95% bands honest?

Two things happen here:

1. Compare guessed noise levels against noise levels learned from each snapshot (model.learn_noise).

2. Hold-out test on reference monitors.
   For each reference monitor in each snapshot: remove it AND every sensor within `--buffer-km` of it
   (so clustered neighbours can't make the test look easy), fit the GP on the rest, predict the monitor.
   A z-score inside +/-1.96 means the 95% band contained the real reading. Honest bands contain it ~95% of the time.
   We also compare against the naive baseline "uncertainty = distance to nearest sensor".

Usage:
    python -m sensorgap.validate                 # normal: noise learned once per hour (fast)
    python -m sensorgap.validate --last 3        # only the 3 most recent hours (quick check while coding);
                                                 # leaves calibration.json / validation_summary.json untouched
                                                 # and writes its other files with a _last3 suffix
    python -m sensorgap.validate --strict        # noise re-learned for every hidden monitor (slow, fully clean)
                                                 # -> run this once before presenting
Outputs:
    outputs/validation_holdout_*.csv     one row per held-out reading
    outputs/reliability_curve.png        expected vs observed coverage (the headline slide)
    outputs/calibration.json             band-widening factor, used by model.py
    outputs/validation_summary.json      headline numbers, so the dashboard can show them instantly
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensorgap import model as m

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
Z_FOR = {lv: z for lv, z in zip(LEVELS, [0.674, 0.994, 1.282, 1.645, 1.960, 2.576])}


def _dist_km(a_lat, a_lon, b_lat, b_lon):
    a = m.to_xy_km(a_lat, a_lon)
    b = m.to_xy_km(b_lat, b_lon)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def holdout(hist, noise_sd=None, buffer_km=1.0, strict=False):
    rows = []
    stamps = sorted(hist["snapshot_utc"].unique())
    for i, (stamp, snap) in enumerate(hist.groupby("snapshot_utc")):
        if strict:
            print(f"  strict: snapshot {i + 1}/{len(stamps)}", flush=True)
        snap = snap.dropna(subset=["lat", "lon", "pm25"]).reset_index(drop=True)
        noise = noise_sd or (None if strict else m.learn_noise(snap))
        refs = snap.index[snap["noise_class"] == "reference"]
        for k in refs:
            target = snap.loc[k]
            d = _dist_km([target["lat"]], [target["lon"]], snap["lat"], snap["lon"])[0]
            train = snap[d > buffer_km]
            if len(train) < 10:
                continue
            fold_noise = noise or m.learn_noise(train)
            gp = m.fit(train, noise_sd=fold_noise, restarts=0)
            mean, std = m.predict(gp, [target["lat"]], [target["lon"]])
            pred_sd = float(np.sqrt(std[0] ** 2 + fold_noise["reference"] ** 2))
            rows.append({
                "snapshot_utc": stamp, "site_id": target["site_id"], "source": target["source"],
                "observed": target["pm25"], "predicted": float(mean[0]), "pred_sd": pred_sd,
                "z": float((target["pm25"] - mean[0]) / pred_sd),
                "nearest_train_km": float(d[d > buffer_km].min()),
                "noise_ref": fold_noise["reference"], "noise_lowcost": fold_noise["lowcost"],
            })
    return pd.DataFrame(rows)


def coverage_table(res):
    return pd.DataFrame({
        "expected": LEVELS,
        "observed": [float((res["z"].abs() <= Z_FOR[lv]).mean()) for lv in LEVELS],
    })


def baseline_comparison(res):
    err = (res["observed"] - res["predicted"]).abs()
    return {
        "spearman_err_vs_gp_sd": float(err.corr(res["pred_sd"], method="spearman")),
        "spearman_err_vs_distance": float(err.corr(res["nearest_train_km"], method="spearman")),
    }


def dedupe_snapshots(hist, min_gap_min=30):
    counts = hist.groupby("snapshot_utc").size()
    times = pd.to_datetime(pd.Series(counts.index), format="%Y%m%dT%H%M%SZ").sort_values()
    clusters, cur = [], []
    for t in times:
        if cur and (t - cur[-1]).total_seconds() >= min_gap_min * 60:
            clusters.append(cur)
            cur = []
        cur.append(t)
    clusters.append(cur)
    keep = [max(c, key=lambda t: (counts[t.strftime("%Y%m%dT%H%M%SZ")], t)).strftime("%Y%m%dT%H%M%SZ")
            for c in clusters]
    dropped = len(counts) - len(keep)
    if dropped:
        print(f"Skipping {dropped} snapshot(s) taken within {min_gap_min} min of a fuller one")
    return hist[hist["snapshot_utc"].isin(keep)]


def calibrate_scale(res, level=0.95):
    return max(float(np.quantile(res["z"].abs(), level)) / Z_FOR[level], 1.0)


def time_split_calibration(res):
    stamps = sorted(res["snapshot_utc"].unique())
    if len(stamps) < 2:
        return None
    cut = stamps[: len(stamps) // 2]
    fit_part, test_part = res[res["snapshot_utc"].isin(cut)], res[~res["snapshot_utc"].isin(cut)]
    k = calibrate_scale(fit_part)
    widened = test_part.assign(z=test_part["z"] / k)
    return k, coverage_table(test_part), coverage_table(widened), len(fit_part), len(test_part)


def diagnose(res):
    by_src = res.assign(err=res["observed"] - res["predicted"]).groupby("source").agg(
        n=("z", "size"),
        inside_95=("z", lambda z: float((z.abs() <= 1.96).mean())),
        mean_error=("err", "mean"),
    )
    print("\nBy source (mean_error > 0 means the monitor reads HIGHER than the map predicts):")
    print(by_src.round(2).to_string())
    worst = res.reindex(res["z"].abs().sort_values(ascending=False).index).head(8)
    print("\nBiggest misses:")
    print(worst[["snapshot_utc", "site_id", "source", "observed", "predicted", "pred_sd", "z",
                 "nearest_train_km"]].round(2).to_string(index=False))


def plot_reliability(tables, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0.4, 1], [0.4, 1], color="#999", lw=1, ls="--", label="perfectly honest")
    for label, tab in tables.items():
        ax.plot(tab["expected"], tab["observed"], marker="o", lw=2, label=label)
    ax.set_xlabel("Band says it contains the truth this often")
    ax.set_ylabel("How often it actually did (held-out monitors)")
    ax.set_xlim(0.4, 1.01)
    ax.set_ylim(0.4, 1.01)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Are SensorGap's uncertainty bands honest?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default=str(ROOT / "data" / "snapshots" / "history.csv"))
    p.add_argument("--buffer-km", type=float, default=1.0)
    p.add_argument("--strict", action="store_true",
                   help="re-learn noise levels inside every test, without the hidden monitor (slow, no leakage)")
    p.add_argument("--last", type=int, default=0, help="only use the N most recent snapshots")
    p.add_argument("--pa-offset", action="store_true",
                   help="test the hourly PurpleAir offset correction (model.CORRECT_LOWCOST_BIAS) for this run")
    args = p.parse_args()
    if args.pa_offset:
        m.CORRECT_LOWCOST_BIAS = True
        print("PurpleAir offset correction: ON for this run")

    hist = dedupe_snapshots(pd.read_csv(args.history))
    if args.last:
        keep = sorted(hist["snapshot_utc"].unique())[-args.last:]
        hist = hist[hist["snapshot_utc"].isin(keep)]
    mode = "strict" if args.strict else "normal"
    print(f"{hist['snapshot_utc'].nunique()} snapshots, {len(hist)} readings, mode: {mode}")

    learned_label = "learned noise (strict)" if args.strict else "learned noise"
    variants = {"guessed noise (ref 1, low-cost 3)": dict(m.NOISE_SD), learned_label: None}

    tables, out = {}, ROOT / "outputs"
    tag = f"_last{args.last}" if args.last else ""
    out.mkdir(exist_ok=True)
    for label, noise in variants.items():
        res = holdout(hist, noise, args.buffer_km, strict=args.strict and noise is None)
        if res.empty:
            raise SystemExit("No hold-out predictions were possible. Need more sensors per snapshot.")
        tab = coverage_table(res)
        tables[label] = tab
        cmp_ = baseline_comparison(res)
        print(f"\n[{label}] {len(res)} held-out readings")
        print(tab.round(3).to_string(index=False))
        print(f"error vs GP sd (Spearman): {cmp_['spearman_err_vs_gp_sd']:.2f}   "
              f"error vs distance: {cmp_['spearman_err_vs_distance']:.2f}")
        if noise is None:
            print("learned noise (ref, low-cost) -> how often:",
                  res[["noise_ref", "noise_lowcost"]].value_counts().to_dict())
        res.to_csv(out / f"validation_holdout_{label.split()[0]}{tag}.csv", index=False)
        if noise is None:
            learned = res

    diagnose(learned)
    split = time_split_calibration(learned)
    if split:
        k, before, after, n_fit, n_test = split
        print(f"\nCalibration: widen bands x{k:.2f} (fit on earlier {n_fit} readings, tested on later {n_test})")
        print(pd.DataFrame({"expected": LEVELS, "test before": before["observed"],
                            "test after": after["observed"]}).round(3).to_string(index=False))
        tables["learned + calibrated (later half)"] = after
        k_all = calibrate_scale(learned)
        if args.last:
            print(f"sd_scale would be {k_all:.2f}; not saved (--last is a partial run)")
        else:
            (out / "calibration.json").write_text(json.dumps({"sd_scale": round(k_all, 3), "n": len(learned)}))
            print(f"Saved sd_scale={k_all:.2f} (from all {len(learned)} readings) to outputs/calibration.json; "
                  f"model.py applies it to the map.")

    cov = coverage_table(learned).set_index("expected")["observed"]
    cmp_ = baseline_comparison(learned)
    summary = {
        "mode": mode,
        "run_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
        "snapshots": int(hist["snapshot_utc"].nunique()),
        "held_out_readings": int(len(learned)),
        "coverage_at_95": round(float(cov.loc[0.95]), 3),
        "coverage_at_68": round(float(cov.loc[0.68]), 3),
        "coverage_at_95_after_calibration_later_half": round(float(split[2].set_index("expected")["observed"].loc[0.95]), 3) if split else None,
        "sd_scale": round(calibrate_scale(learned), 3),
        "spearman_error_vs_gp_sd": round(cmp_["spearman_err_vs_gp_sd"], 2),
        "spearman_error_vs_distance": round(cmp_["spearman_err_vs_distance"], 2),
    }
    if args.last:
        print("Headline numbers not saved (--last is a partial run)")
    else:
        (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
        print("Saved headline numbers to outputs/validation_summary.json")

    curve = out / f"reliability_curve{tag}.png"
    plot_reliability(tables, curve)
    print(f"\nWrote {curve}")


if __name__ == "__main__":
    main()
