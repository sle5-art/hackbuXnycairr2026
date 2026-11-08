"""
SensorGap ingestion: one snapshot of NYC PM2.5 from two sources.

  1. PurpleAir outdoor sensors (needs PURPLEAIR_KEY in a .env file next to this script)
     - QC: keep a reading if channels A and B agree within 5 ug/m3 OR within 61% (Barkjohn et al. 2021)
     - EPA US-wide correction: PM2.5 = 0.524*PA_cf1 - 0.0862*RH + 5.75
  2. NYC DOHMH real-time PM2.5 monitors (NYCCAS, no key)
     - data:     https://raw.githubusercontent.com/nychealth/nyccas-data/refs/heads/main/portal/view.csv (last 7 days)
     - stations: https://raw.githubusercontent.com/nychealth/nyccas-data/main/portal/station-new.csv

  3. AirNow regulatory monitors (state DEC sites), if AIRNOW_KEY is in .env

Keys go in <repo>/.env:
    PURPLEAIR_KEY=...
    AIRNOW_KEY=...

Usage (from the repo root):
    pip install -r requirements.txt
    python -m sensorgap.ingest --inspect      # print the city CSV's columns + first rows
    python -m sensorgap.ingest                # write one snapshot
    python -m sensorgap.ingest --loop 3600    # write a snapshot every hour

Output:
    data/snapshots/snapshot_<UTC>.csv   source, site_id, lat, lon, pm25, obs_time_utc, noise_class
    data/snapshots/history.csv          every snapshot appended (for validation later)
    data/raw/                           every raw API response, kept for reproducibility
"""

import argparse
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "snapshots"
RAW = ROOT / "data" / "raw"

BBOX = dict(nwlng=-74.26, nwlat=40.92, selng=-73.70, selat=40.49)

PA_URL = "https://api.purpleair.com/v1/sensors"
PA_FIELDS = "latitude,longitude,last_seen,humidity,pm2.5_cf_1_a,pm2.5_cf_1_b"

CITY_URL = "https://raw.githubusercontent.com/nychealth/nyccas-data/refs/heads/main/portal/view.csv"
STATION_URL = "https://raw.githubusercontent.com/nychealth/nyccas-data/main/portal/station-new.csv"

AIRNOW_URL = "https://www.airnowapi.org/aq/data/"

UA = {"User-Agent": "SensorGap-hackathon/0.1 (non-commercial research)"}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        b = env.read_bytes()
        text = b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8-sig")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def barkjohn_qc(a, b):
    diff = (a - b).abs()
    mean = (a + b) / 2
    pct = (diff / mean.where(mean > 0)).fillna(0) * 100
    return (diff < 5) | (pct < 61)


def barkjohn_correct(pa_cf1, rh):
    return (0.524 * pa_cf1 - 0.0862 * rh + 5.75).clip(lower=0)


def fetch_purpleair(key, max_age_s, stamp):
    params = {"api_key": key, "location_type": 0, "fields": PA_FIELDS, "max_age": max_age_s, **BBOX}
    r = requests.get(PA_URL, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    js = r.json()
    (RAW / f"purpleair_raw_{stamp}.json").write_text(json.dumps(js), encoding="utf-8")

    df = pd.DataFrame(js["data"], columns=js["fields"])
    n_raw = len(df)
    df = df.dropna(subset=["pm2.5_cf_1_a", "pm2.5_cf_1_b", "humidity", "latitude", "longitude"])
    ok = barkjohn_qc(df["pm2.5_cf_1_a"], df["pm2.5_cf_1_b"])
    df = df[ok].copy()
    pa = (df["pm2.5_cf_1_a"] + df["pm2.5_cf_1_b"]) / 2
    df["pm25"] = barkjohn_correct(pa, df["humidity"])

    out = pd.DataFrame({
        "source": "purpleair",
        "site_id": "PA" + df["sensor_index"].astype(str),
        "lat": df["latitude"],
        "lon": df["longitude"],
        "pm25": df["pm25"].round(2),
        "obs_time_utc": pd.to_datetime(df["last_seen"], unit="s", utc=True),
        "noise_class": "lowcost",
    })
    print(f"PurpleAir: {n_raw} returned, {len(out)} kept after missing-data + A/B QC "
          f"(max_age={max_age_s}s)")
    return out


def get_city_raw(stamp=None):
    r = requests.get(CITY_URL, headers=UA, timeout=60)
    r.raise_for_status()
    if stamp:
        (RAW / f"city_raw_{stamp}.csv").write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content))


def get_stations():
    r = requests.get(STATION_URL, headers=UA, timeout=60)
    r.raise_for_status()
    st = pd.read_csv(io.BytesIO(r.content))
    return st[st["SiteID"].notna() & (st["loc_col"] != "DEC_Avg")]


def find_time_col(df):
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    raise ValueError(f"No timestamp column found in {list(df.columns)}")


def fetch_city(stamp):
    raw = get_city_raw(stamp)
    st = get_stations()
    tcol = find_time_col(raw)
    t = pd.to_datetime(raw[tcol], errors="coerce")
    if t.dt.tz is None:
        t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    raw[tcol] = t.dt.tz_convert("UTC")

    wide_cols = [c for c in st["loc_col"] if c in raw.columns]
    if wide_cols:
        long = raw.melt(id_vars=[tcol], value_vars=wide_cols, var_name="loc_col", value_name="pm25")
        long = long.merge(st, on="loc_col")
    else:
        id_col = next((c for c in raw.columns
                       if c.lower() in ("siteid", "site_id", "site", "loc_col", "sitename")), None)
        val_col = next((c for c in raw.columns if any(k in c.lower() for k in ("value", "pm25", "pm2.5", "pm"))
                        and c != id_col), None)
        if not id_col or not val_col:
            raise ValueError("Unknown city CSV layout. Run `python ingest.py --inspect` and send the output.")
        long = raw.rename(columns={val_col: "pm25"})
        key = "SiteID" if id_col.lower() in ("siteid", "site_id", "site") else "loc_col"
        long = long.rename(columns={id_col: key}).merge(st, on=key)

    long = long.dropna(subset=["pm25", tcol])
    latest = long.sort_values(tcol).groupby("SiteID").tail(1)
    out = pd.DataFrame({
        "source": "nyc_dohmh",
        "site_id": latest["SiteID"],
        "lat": latest["Latitude"],
        "lon": latest["Longitude"],
        "pm25": pd.to_numeric(latest["pm25"], errors="coerce").round(2),
        "obs_time_utc": latest[tcol],
        "noise_class": "reference",
    }).dropna(subset=["pm25"])
    print(f"NYC monitors: {len(out)} sites with a latest reading "
          f"(newest {out['obs_time_utc'].max()}, oldest {out['obs_time_utc'].min()})")
    return out


def fetch_airnow(key, stamp):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    params = {
        "startDate": (now - pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H"),
        "endDate": now.strftime("%Y-%m-%dT%H"),
        "parameters": "PM25",
        "BBOX": f"{BBOX['nwlng']},{BBOX['selat']},{BBOX['selng']},{BBOX['nwlat']}",
        "dataType": "C",
        "format": "application/json",
        "verbose": 1,
        "monitorType": 0,
        "API_KEY": key,
    }
    r = requests.get(AIRNOW_URL, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    (RAW / f"airnow_raw_{stamp}.json").write_bytes(r.content)
    df = pd.DataFrame(r.json())
    if df.empty:
        print("AirNow: 0 readings returned")
        return df
    df["UTC"] = pd.to_datetime(df["UTC"], utc=True)
    df = df[df["Value"] >= 0]
    site = df["FullAQSCode"] if "FullAQSCode" in df else df["SiteName"]
    df = df.assign(site=site).sort_values("UTC").groupby("site").tail(1)
    out = pd.DataFrame({
        "source": "airnow",
        "site_id": "AQS" + df["site"].astype(str),
        "lat": df["Latitude"],
        "lon": df["Longitude"],
        "pm25": df["Value"].round(2),
        "obs_time_utc": df["UTC"],
        "noise_class": "reference",
    })
    print(f"AirNow: {len(out)} monitors with a reading in the last 3 hours")
    return out


def snapshot(key, max_age_s, airnow_key=None):
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    parts = []
    for name, fn in (("PurpleAir", lambda: fetch_purpleair(key, max_age_s, stamp)),
                     ("NYC monitors", lambda: fetch_city(stamp)),
                     ("AirNow", lambda: fetch_airnow(airnow_key, stamp) if airnow_key else None)):
        try:
            part = fn()
            if part is not None and len(part):
                parts.append(part)
        except Exception as e:
            print(f"WARNING: {name} failed: {e}")
    if not parts:
        print("No data this round.")
        return
    snap = pd.concat(parts, ignore_index=True)
    snap.insert(0, "snapshot_utc", stamp)
    snap.to_csv(DATA / f"snapshot_{stamp}.csv", index=False)
    hist = DATA / "history.csv"
    snap.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    print(f"Wrote {len(snap)} rows to data/snapshots/snapshot_{stamp}.csv\n"
          f"{snap.groupby('source')['pm25'].describe().round(1)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true", help="show the city CSV layout and exit")
    p.add_argument("--loop", type=int, default=0, help="seconds between snapshots (0 = once)")
    p.add_argument("--max-age", type=int, default=3600,
                   help="skip PurpleAir sensors not heard from in this many seconds")
    args = p.parse_args()

    if args.inspect:
        raw = get_city_raw()
        print("columns:", list(raw.columns))
        print(raw.head(5).to_string())
        print(raw.tail(3).to_string())
        return

    load_env()
    key = os.environ.get("PURPLEAIR_KEY")
    if not key:
        raise SystemExit("Put PURPLEAIR_KEY=your-key in a .env file in the repo root")
    airnow_key = os.environ.get("AIRNOW_KEY")
    if not airnow_key:
        print("Note: no AIRNOW_KEY in .env, skipping state monitors")

    while True:
        snapshot(key, args.max_age, airnow_key)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
