"""
SensorGap in the terminal: collect, calculate and explain everything the map shows, step by step.

It runs the real pipeline (the same functions and output files the map reads), so after it finishes a browser
refresh of the map shows exactly these numbers.

Run from the repo root:
    python -m app.walkthrough                      # newest snapshot already on disk
    python -m app.walkthrough --live               # pull a fresh snapshot from the 3 feeds first (needs .env keys)
    python -m app.walkthrough --pause              # wait for Enter between steps (for presenting)
    python -m app.walkthrough --at 40.758,-73.992  # also explain one point, like clicking the map
    python -m app.walkthrough --all                # print every sensor and site, not just the top rows

Writes (same files as the individual modules):
    outputs/grid_<UTC>.csv, outputs/labels_<UTC>.csv, outputs/placement.csv, outputs/placement_summary.json
    outputs/map_data.csv     every land cell on the map: estimate, band, label, distance to the nearest sensors
"""

import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np
import pandas as pd
import shapely

from app import dashboard as D
from sensorgap import labels as L
from sensorgap import model as M
from sensorgap import placement as P

ROOT, OUT = D.ROOT, D.OUT
SNAPS = ROOT / "data" / "snapshots"
UNIT = "ug/m3"  # plain ASCII so every Windows terminal shows it
W = 88

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


class Show:
    def __init__(self, pause, show_all):
        self.pause, self.all, self.n = pause, show_all, 0

    def step(self, title, why):
        if self.pause and self.n:
            input("\n  [Enter for the next step] ")
        self.n += 1
        print("\n" + "=" * W + f"\n STEP {self.n}  {title}\n" + "=" * W)
        print(f" Why: {why}\n")

    def say(self, text=""):
        print(f"  {text}")

    def table(self, df, rows=10):
        body = df if self.all or len(df) <= rows else df.head(rows)
        print("\n".join("  " + line for line in body.to_string(index=False).splitlines()))
        if len(body) < len(df):
            print(f"  ... {len(df) - len(body)} more rows (use --all)")


def nearest(points, sensors):
    """For each point: distance (km) to the nearest sensor, its id/source, and to the nearest reference monitor."""
    xy_p = M.to_xy_km(points["lat"], points["lon"])
    xy_s = M.to_xy_km(sensors["lat"], sensors["lon"])
    d = np.hypot(xy_p[:, None, 0] - xy_s[None, :, 0], xy_p[:, None, 1] - xy_s[None, :, 1])
    i = d.argmin(axis=1)
    ref = (sensors["noise_class"] == "reference").to_numpy()
    return pd.DataFrame({
        "nearest_sensor_km": d[np.arange(len(d)), i].round(2),
        "nearest_sensor": sensors["site_id"].to_numpy()[i],
        "nearest_sensor_source": sensors["source"].to_numpy()[i],
        "nearest_reference_km": d[:, ref].min(axis=1).round(2) if ref.any() else np.nan,
    }, index=points.index)


def quiet(fn, *a):
    """Run a module function without its own prints (we narrate instead)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="pull a fresh snapshot from the feeds first")
    p.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    p.add_argument("--all", action="store_true", help="print every row")
    p.add_argument("--at", help="explain one point, e.g. 40.758,-73.992")
    args = p.parse_args()
    show = Show(args.pause, args.all)

    # 1 -------------------------------------------------------------------------------------------- collect
    show.step("COLLECT the readings",
              "the map is only as good as its inputs, so first: what did we measure, where, and from whom?")
    if args.live:
        from sensorgap import ingest
        show.say("Pulling PurpleAir, NYC Health and AirNow right now ...")
        argv, sys.argv = sys.argv, ["ingest"]
        try:
            ingest.main()
        finally:
            sys.argv = argv
    snap_path = P.latest_snapshot()
    snap = pd.read_csv(snap_path).dropna(subset=["lat", "lon", "pm25"])
    show.say(f"Snapshot: {snap_path.relative_to(ROOT)}  ({len(snap)} sensors)")
    by = snap.groupby(["source", "noise_class"])["pm25"].agg(["count", "min", "median", "max"]).round(1)
    show.say("")
    show.table(by.reset_index().rename(columns={"noise_class": "trust"}))
    show.say("")
    show.say("PurpleAir readings already passed EPA's A/B check (channels agree within 5 ug/m3 or 61%)")
    show.say("and EPA's humidity correction: PM2.5 = 0.524*PA - 0.0862*RH + 5.75 (Barkjohn et al. 2021).")
    show.say("")
    show.table(snap[["site_id", "source", "noise_class", "lat", "lon", "pm25"]]
               .sort_values(["noise_class", "pm25"], ascending=[False, False]))

    # 2 -------------------------------------------------------------------------------------------- fit
    show.step("FIT the model (Gaussian process), live",
              "turn ~94 scattered readings into an estimate AND an uncertainty everywhere in the city")
    t0 = time.time()
    gp = M.fit(snap)
    info = M.describe(gp)
    show.say(f"Fitted in {time.time() - t0:.1f} s on {len(snap)} sensors.")
    show.say(f"Learned noise (how much one reading is trusted, lower = more): "
             f"reference {gp.noise_sd_['reference']} {UNIT}, PurpleAir {gp.noise_sd_['lowcost']} {UNIT}")
    show.say(f"Learned length scale: {info['length_scale_km']:.2f} km "
             f"(how far a reading stays informative; floor {M.MIN_LENGTH_SCALE_KM} km = typical sensor spacing)")
    show.say(f"Kernel: {gp.kernel_}")

    # 3 -------------------------------------------------------------------------------------------- calibrate
    sd_scale = M.load_sd_scale()
    show.step("CALIBRATE the uncertainty",
              "a band is only useful if it is honest, so we tested it on monitors the model never saw")
    vpath = OUT / "validation_summary.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        show.say(f"From the last strict test ({v['run_utc']} UTC, precomputed; rerun: python -m sensorgap.validate --strict):")
        show.say(f"  hid each reference monitor + every sensor within 1 km, refit, predicted it: "
                 f"{v['held_out_readings']} readings over {v['snapshots']} hours")
        show.say(f"  {v['coverage_at_95']:.0%} fell inside the 95% band -> bands widened x{v['sd_scale']:.2f} "
                 "to make them honest")
    show.say(f"Applying band widening x{sd_scale:.2f} to every map below.")
    hold = D.load()["holdout"]
    if hold is not None:
        worst = (hold.assign(miss=hold["observed"] - hold["predicted"], az=hold["z"].abs())
                 .groupby(["site_id", "source"]).agg(hours=("z", "size"), observed=("observed", "mean"),
                                                     predicted=("predicted", "mean"), miss=("miss", "mean"),
                                                     mean_abs_z=("az", "mean"))
                 .sort_values("mean_abs_z", ascending=False).round(2).reset_index())
        worst.insert(1, "name", worst["site_id"].map(D.SITE_NAMES).fillna(""))
        show.say("")
        show.say("Biggest misses when hidden (the red ring on the map is the top row):")
        show.table(worst, rows=5)

    # 4 -------------------------------------------------------------------------------------------- this hour's map
    show.step("MAP this hour on a 0.5 km grid",
              "every square gets an estimate and a 95% band = estimate +/- 1.96 x calibrated standard deviation")
    grid = M.predict_grid(gp, sd_scale=sd_scale)
    grid_path = OUT / snap_path.name.replace("snapshot_", "grid_")
    grid.to_csv(grid_path, index=False)
    hw = (grid["hi95"] - grid["lo95"]) / 2
    show.say(f"{len(grid)} squares -> {grid_path.relative_to(ROOT)}")
    show.say(f"Estimate {grid['mean'].min():.1f} to {grid['mean'].max():.1f} {UNIT}; "
             f"band half-width +/-{hw.min():.2f} (next to sensors) to +/-{hw.max():.2f} (far from all of them)")

    # 5 -------------------------------------------------------------------------------------------- labels
    show.step("LABEL with NowCast, and refuse when unsure",
              "AQI isn't defined on single hours; and we only say 'Good' if the WHOLE band sits inside Good")
    quiet(L.make_missing_grids, 12)
    base, mean, std, newest = L.load_hourly_grids(list(OUT.glob("grid_*.csv")), 12)
    hours = int((~np.isnan(mean[:, 0])).sum())
    lab = quiet(L.make_labels, base, mean, std)
    lab_path = OUT / newest.name.replace("grid_", "labels_")
    lab.to_csv(lab_path, index=False)
    land = D.land_of(D.load()["boroughs"])
    lab = lab[shapely.contains_xy(land, lab["lon"], lab["lat"])].copy()
    lab["halfwidth"] = ((lab["hi95"] - lab["lo95"]) / 2).round(2)
    show.say(f"NowCast over {hours} hourly map(s) -> {lab_path.relative_to(ROOT)}")
    show.say(f"EPA 2024 breakpoints: Good 0-9.0, Moderate 9.1-35.4, USG 35.5-55.4 {UNIT}")
    show.say(f"{len(lab)} squares are on NYC land (water and NJ are dropped using the borough shapes).")
    counts = lab["label"].value_counts()
    for name, n in counts.items():
        show.say(f"  {n:5d} squares  {name}")
    show.say(f"Confident label on {lab['confident'].mean():.0%} of land; the rest get grey stripes on the map.")

    # 6 -------------------------------------------------------------------------------------------- distances
    show.step("MEASURE how far every square is from a sensor",
              "uncertainty should grow with distance from sensors; here is the evidence, square by square")
    lab = lab.join(nearest(lab, snap))
    cols = ["lat", "lon", "nowcast", "lo95", "hi95", "halfwidth", "label", "nearest_sensor_km",
            "nearest_sensor_source", "nearest_reference_km"]
    show.say("Least certain squares on land:")
    show.table(lab.sort_values("halfwidth", ascending=False)[cols], rows=5)
    show.say("")
    show.say("Most certain squares on land:")
    show.table(lab.sort_values("halfwidth")[cols], rows=5)
    rho = lab["halfwidth"].corr(lab["nearest_sensor_km"], method="spearman")
    show.say("")
    show.say(f"Rank correlation between band width and distance to nearest sensor: {rho:.2f}")
    map_path = OUT / "map_data.csv"
    lab[cols + ["confident", "nearest_sensor"]].to_csv(map_path, index=False)
    show.say(f"Every land square, with all of these columns -> {map_path.relative_to(ROOT)}")

    # 7 -------------------------------------------------------------------------------------------- placement
    show.step("RECOMMEND 10 new sensor sites, live",
              "put sensors where they cut uncertainty most for adults with asthma (CDC PLACES 2025 x adults)")
    t0 = time.time()
    argv, sys.argv = sys.argv, ["placement", str(snap_path)]
    try:
        quiet(P.main)
    finally:
        sys.argv = argv
    sites = pd.read_csv(OUT / "placement.csv", dtype={"tract": str, "rank_if_population_only": "Int64"})
    summary = json.loads((OUT / "placement_summary.json").read_text())
    show.say(f"Greedy search over {summary['n_tracts']} census-tract centroids, at least "
             f"{summary['min_spacing_km']:g} km apart, in {time.time() - t0:.1f} s.")
    show.say("Each pick = the site whose new PurpleAir shrinks asthma-weighted uncertainty most; then repeat.")
    sites = sites.assign(borough=sites["tract"].str[2:5].map(D.BOROUGH)).join(nearest(sites, snap))
    here = lab.reset_index(drop=True)
    idx = [int(((here["lat"] - r.lat) ** 2 + (here["lon"] - r.lon) ** 2).idxmin()) for r in sites.itertuples()]
    sites["band_now"] = here.loc[idx, "halfwidth"].to_numpy()
    show.say("")
    show.table(sites[["rank", "borough", "tract", "lat", "lon", "asthma_pct", "adults", "band_now",
                      "nearest_sensor_km", "nearest_reference_km", "rank_if_population_only"]]
               .rename(columns={"rank_if_population_only": "pop_only_rank"}).astype({"pop_only_rank": "string"})
               .fillna("-"), rows=10)
    hi, lo = summary["band_halfwidth_high_asthma_tracts"], summary["band_halfwidth_other_tracts"]
    show.say("")
    show.say(f"Weighting by headcount instead picks {summary['picks_shared_with_population_only']} of the same "
             f"{summary['k']} sites: who you weight by decides where the city looks.")
    show.say(f"Band half-width after the {summary['k']} sites: highest-asthma quarter of tracts "
             f"+/-{hi[0]} -> +/-{hi[1]}, all other tracts +/-{lo[0]} -> +/-{lo[1]} {UNIT}")

    # 8 -------------------------------------------------------------------------------------------- one point
    if args.at:
        lat, lon = (float(x) for x in args.at.split(","))
        show.step(f"EXPLAIN one point ({lat}, {lon})", "the same reasoning the map shows when you click")
        text = D.explain({"labels": lab, "snapshot": snap}, land, lat, lon)
        for line in (text or "That point is on water or outside NYC.").replace("**", "").split("  \n"):
            show.say(line)

    print("\n" + "=" * W)
    print(" Done. Refresh the map: it now shows exactly these numbers.")
    print("=" * W)


if __name__ == "__main__":
    main()
