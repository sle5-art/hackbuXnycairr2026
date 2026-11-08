import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sensorgap import model as m

ROOT = Path(__file__).resolve().parents[1]
PLACES_URL = "https://data.cdc.gov/resource/cwsq-ngmh.json"
NYC_COUNTIES = ("36005", "36047", "36061", "36081", "36085")
TRACTS_CSV = ROOT / "data" / "static" / "places_asthma_nyc.csv"


def load_tracts():
    if TRACTS_CSV.exists():
        return pd.read_csv(TRACTS_CSV, dtype={"tract": str})
    where = f"measureid='CASTHMA' AND countyfips in({','.join(repr(c) for c in NYC_COUNTIES)})"
    r = requests.get(PLACES_URL, params={"$where": where, "$limit": 5000}, timeout=60)
    r.raise_for_status()
    df = pd.json_normalize(r.json())
    df = pd.DataFrame({
        "tract": df["locationname"],
        "lon": df["geolocation.coordinates"].str[0],
        "lat": df["geolocation.coordinates"].str[1],
        "population": pd.to_numeric(df["totalpopulation"], errors="coerce"),
        "adults": pd.to_numeric(df["totalpop18plus"], errors="coerce"),
        "asthma_pct": pd.to_numeric(df.get("data_value"), errors="coerce"),
    }).dropna()
    TRACTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACTS_CSV, index=False)
    return df


def greedy(cov, weights, k, noise_var, xy=None, min_km=0.0):
    cov = cov.copy()
    picks = []
    gains = []
    blocked = np.zeros(len(cov), dtype=bool)
    for _ in range(k):
        gain = (weights @ cov ** 2) / (np.diag(cov) + noise_var)
        gain[picks] = -np.inf
        gain[blocked] = -np.inf
        s = int(np.argmax(gain))
        picks.append(s)
        gains.append(float(gain[s]))
        cov -= np.outer(cov[:, s], cov[s, :]) / (cov[s, s] + noise_var)
        if min_km > 0:
            blocked |= np.hypot(*(xy - xy[s]).T) < min_km
    return picks, gains, cov


def latest_snapshot():
    snaps = sorted((ROOT / "data" / "snapshots").glob("snapshot_*.csv"))
    if not snaps:
        raise SystemExit("No data/snapshots/snapshot_*.csv yet: run python -m sensorgap.ingest first.")
    return snaps[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", nargs="?", help="default: newest data/snapshots/snapshot_*.csv")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--min-spacing-km", type=float, default=1.0, help="no two new sites closer than this; 0 = off")
    args = p.parse_args()

    snap = Path(args.snapshot) if args.snapshot else latest_snapshot()
    gp = m.fit(pd.read_csv(snap))
    tracts = load_tracts()
    sd_scale = m.load_sd_scale()
    xy = m.to_xy_km(tracts["lat"], tracts["lon"])
    prediction = gp.predict(xy, return_cov=True)
    cov = prediction[1]
    cov *= sd_scale ** 2
    noise_var = gp.noise_sd_["lowcost"] ** 2

    w_asthma = (tracts["adults"] * tracts["asthma_pct"] / 100).to_numpy()
    w_pop = tracts["population"].to_numpy()
    asthma_result = greedy(cov, w_asthma, args.k, noise_var, xy, args.min_spacing_km)
    picks = asthma_result[0]
    gains = asthma_result[1]
    cov_after = asthma_result[2]
    pop_result = greedy(cov, w_pop, args.k, noise_var, xy, args.min_spacing_km)
    picks_pop = pop_result[0]

    pop_rank = {s: i + 1 for i, s in enumerate(picks_pop)}
    out = tracts.iloc[picks].assign(
        rank=range(1, args.k + 1),
        weighted_var_cut=np.round(gains, 1),
        rank_if_population_only=pd.array([pop_rank.get(s) for s in picks], dtype="Int64"),
    )[["rank", "lat", "lon", "tract", "asthma_pct", "adults", "weighted_var_cut", "rank_if_population_only"]]
    (ROOT / "outputs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "outputs" / "placement.csv", index=False)

    hw_before = 1.96 * np.sqrt(np.clip(np.diag(cov), 0, None))
    hw_after = 1.96 * np.sqrt(np.clip(np.diag(cov_after), 0, None))
    top = tracts["asthma_pct"] >= tracts["asthma_pct"].quantile(0.75)
    summary = {
        "snapshot": snap.name,
        "k": args.k,
        "min_spacing_km": args.min_spacing_km,
        "n_tracts": len(tracts),
        "picks_shared_with_population_only": len(set(picks) & set(picks_pop)),
        "band_halfwidth_high_asthma_tracts": [round(float(hw_before[top].mean()), 2),
                                              round(float(hw_after[top].mean()), 2)],
        "band_halfwidth_other_tracts": [round(float(hw_before[~top].mean()), 2),
                                        round(float(hw_after[~top].mean()), 2)],
    }
    (ROOT / "outputs" / "placement_summary.json").write_text(json.dumps(summary, indent=2))

    print(out.to_string(index=False))
    print(f"\n{summary['picks_shared_with_population_only']} of {args.k} sites are also picked when weighting by "
          f"population only")
    before = summary["band_halfwidth_high_asthma_tracts"][0]
    after = summary["band_halfwidth_high_asthma_tracts"][1]
    print(f"Band half-width, highest-asthma quarter of tracts: {before} -> {after} ug/m3")
    before = summary["band_halfwidth_other_tracts"][0]
    after = summary["band_halfwidth_other_tracts"][1]
    print(f"Band half-width, other tracts:                     {before} -> {after} ug/m3")
    print("Wrote outputs/placement.csv and outputs/placement_summary.json")


if __name__ == "__main__":
    main()
