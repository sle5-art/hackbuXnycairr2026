"""
Tonight vs June 7, 2023: the same SensorGap pipeline on a clean night and on NYC's worst smoke day.

For each day it fits the model on every hour of a 12-hour window, blends the hours with EPA NowCast, labels the
map (refusing where the band crosses a category line), runs the held-out honesty test on the last hour, and
picks 10 new sensor sites. Then it prints the two days side by side.

Both days use the same settings, including one widened noise range (see SMOKE_NOISE_GRID below).

Run from the repo root (no internet needed; the smoke-day data ships in data/static/replay_20230607/):
    python -m app.compare            # add --pause to step through it, --all for the full tables

Writes outputs/replay_20230607/: labels_*.csv, placement.csv, map.html (open in a browser)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import shapely

from app import dashboard as D
from app.walkthrough import Show, nearest, quiet
from sensorgap import labels as L
from sensorgap import model as M
from sensorgap import placement as P
from sensorgap import validate as V

ROOT, OUT = D.ROOT, D.OUT
REPLAY_DIR = ROOT / "data" / "static" / "replay_20230607"
REPLAY_OUT = OUT / "replay_20230607"

SMOKE_NOISE_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


def last_hours(paths, hours=12):
    stamped = sorted((L._stamp(p, "snapshot_"), p) for p in paths)
    newest = stamped[-1][0]
    by_lag = {}
    for t, p in stamped:
        lag = int(round((newest - t).total_seconds() / 3600))
        if lag < hours:
            by_lag[lag] = p
    return [by_lag[k] for k in sorted(by_lag, reverse=True)]


def run_day(name, snap_paths, land, tracts, show):
    t0 = time.time()
    snaps = [pd.read_csv(p).dropna(subset=["lat", "lon", "pm25"]) for p in snap_paths]
    sd_scale = M.load_sd_scale()
    means, stds, gps = [], [], []
    for s in snaps:
        gp = M.fit(s, restarts=0)
        gps.append(gp)
        g = M.predict_grid(gp, sd_scale=sd_scale)
        means.append(g["mean"].to_numpy())
        stds.append(g["std"].to_numpy())
    base = g[["lat", "lon"]]
    lab = quiet(L.make_labels, base, np.array(means[::-1]), np.array(stds[::-1]))
    on_land = shapely.contains_xy(land, lab["lon"], lab["lat"])
    lab = lab[on_land].copy()
    lab["halfwidth"] = ((lab["hi95"] - lab["lo95"]) / 2).round(1)
    last, gp = snaps[-1], gps[-1]

    hist = last.assign(snapshot_utc=last["snapshot_utc"].astype(str))
    held = V.holdout(hist, buffer_km=1.0)
    cover = float((held["z"].abs() <= 1.96).mean()) if len(held) else np.nan

    xy = M.to_xy_km(tracts["lat"], tracts["lon"])
    _, cov = gp.predict(xy, return_cov=True)
    cov *= sd_scale ** 2
    w = (tracts["adults"] * tracts["asthma_pct"] / 100).to_numpy()
    picks, _, _ = P.greedy(cov, w, 10, gp.noise_sd_["lowcost"] ** 2, xy, 1.0)
    sites = tracts.iloc[picks].assign(rank=range(1, 11))
    sites = sites.assign(borough=sites["tract"].str[2:5].map(D.BOROUGH))

    stats = {
        "Readings, last hour (ug/m3)": f"{last['pm25'].min():.0f} to {last['pm25'].max():.0f}",
        "Learned noise, reference (ug/m3)": f"{gp.noise_sd_['reference']:g}",
        "Learned length scale (km)": f"{M.describe(gp)['length_scale_km']:.1f}",
        "NowCast estimate on land (ug/m3)": f"{lab['nowcast'].min():.0f} to {lab['nowcast'].max():.0f}",
        "95% band half-width (ug/m3)": f"+/-{lab['halfwidth'].min():g} to +/-{lab['halfwidth'].max():g}",
        "Squares with a confident label": f"{lab['confident'].mean():.0%} of {len(lab)}",
        "Hidden monitors inside 95% band": f"{cover:.0%} of {len(held)}" if len(held) else "n/a",
    }
    hours = f"{len(snaps)} hour{'s' if len(snaps) > 1 else ''}"
    show.say(f"{name}: {hours}, {len(last)} sensors in the last hour, done in {time.time() - t0:.1f} s")
    return stats, lab, sites, last, held


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pause", action="store_true")
    p.add_argument("--all", action="store_true", help="also print the full tables")
    args = p.parse_args()
    show = Show(args.pause, args.all)
    M.NOISE_GRID = SMOKE_NOISE_GRID

    land = D.land_of(D.load()["boroughs"])
    tracts = P.load_tracts()
    tonight_paths = last_hours(list((ROOT / "data" / "snapshots").glob("snapshot_*.csv")))
    smoke_paths = sorted(REPLAY_DIR.glob("snapshot_*.csv"))
    if not smoke_paths:
        raise SystemExit("No replay data: run python -m app.replay once (needs internet).")

    show.step("RUN the same pipeline on two days",
              "a map that only works on clean nights isn't worth much; test it on NYC's worst air day")
    show.say("Tonight: the hourly collector's snapshots. June 7, 2023: NYC Health + AirNow archives,")
    show.say("reference monitors only, 12 hours ending 18:00 UTC (2 pm in New York), when the smoke peaked.")
    show.say("Same code and settings for both days.")
    show.say("")
    a, lab_a, sites_a, _, _ = run_day("Tonight", tonight_paths, land, tracts, show)
    b, lab_b, sites_b, last_b, held_b = run_day("June 7, 2023", smoke_paths, land, tracts, show)

    show.step("SIDE BY SIDE", "same code, same settings, two very different days")
    table = pd.DataFrame({"": list(a), "Tonight": list(a.values()), "June 7, 2023": list(b.values())})
    show.table(table, rows=100)

    show.step("WHAT THE LABELS SAY on the smoke day",
              "this is the refuse-to-label rule doing its job: the band crosses a category line, so we say so")
    for name, n in lab_b["label"].value_counts().items():
        show.say(f"  {n:5d} squares  {name}")
    show.say("")
    show.say("EPA 2024 breakpoints: Unhealthy 55.5-125.4, Very Unhealthy 125.5-225.4, Hazardous 225.5+ ug/m3")
    if show.all:
        show.say("NowCast sits below the 2 pm readings because the morning was 50-140 ug/m3: EPA's 12-hour weighting")
        show.say("leans hard on the latest hours when air changes fast, but doesn't jump all the way in one hour.")
    lab_b = lab_b.join(nearest(lab_b, last_b))
    cols = ["lat", "lon", "nowcast", "lo95", "hi95", "halfwidth", "label", "nearest_sensor_km"]
    if show.all:
        show.say("")
        show.table(lab_b.sort_values("halfwidth", ascending=False)[cols])

    show.step("HONESTY on the smoke day",
              "hide each monitor (and anything within 1 km), refit without it, and check the band caught it")
    if len(held_b):
        held_b = held_b.assign(name=held_b["site_id"].map(D.SITE_NAMES).fillna(""),
                               inside=np.where(held_b["z"].abs() <= 1.96, "yes", "NO"))
        share = (held_b["inside"] == "yes").mean()
        show.say(f"{share:.0%} of {len(held_b)} hidden monitors fell inside their 95% band at the peak hour.")
        show.say("About 95% is the target: much lower = bands too narrow (overconfident), 100% = a bit wide (cautious).")
        if show.all:
            show.say("")
            show.table(held_b[["site_id", "source", "observed", "predicted", "pred_sd", "inside"]]
                       .round(1).sort_values("observed", ascending=False))

    show.step("WHERE THE NEW SENSORS GO on each day",
              "placement depends on where sensors are AND on how far a reading reaches that day")
    both = sites_a[["rank", "borough", "tract"]].merge(sites_b[["rank", "borough", "tract"]], on="rank",
                                                      suffixes=("_tonight", "_smoke"))
    if show.all:
        show.table(both)
    show.say(f"{len(set(sites_a['tract']) & set(sites_b['tract']))} of the 10 recommended sites are the same on both days:")
    show.say("in smoke a reading only reaches about half as far, so the gaps worth filling move.")

    REPLAY_OUT.mkdir(parents=True, exist_ok=True)
    lab_b.to_csv(REPLAY_OUT / f"labels_{smoke_paths[-1].stem[9:]}.csv", index=False)
    sites_b.assign(rank_if_population_only=pd.NA).to_csv(REPLAY_OUT / "placement.csv",
                                                                              index=False)
    m = D.build_map({"labels": lab_b, "boroughs": D.load()["boroughs"], "snapshot": last_b, "holdout": None,
                     "placement": sites_b.assign(rank_if_population_only=pd.NA)}, "aqi")
    m.save(str(REPLAY_OUT / "map.html"))
    (REPLAY_OUT / "summary.json").write_text(json.dumps({"tonight": a, "june_7_2023": b}, indent=2))
    print("\n" + "=" * 88)
    print(f" Smoke-day map: {(REPLAY_OUT / 'map.html').relative_to(ROOT)} (open in a browser)")
    print("=" * 88)


if __name__ == "__main__":
    main()
