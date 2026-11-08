"""
Replay data for June 7, 2023, the day Canadian wildfire smoke gave NYC its worst air on record.

Builds one snapshot per hour, in the same columns as sensorgap.ingest, from two public archives:
    NYC Health real-time monitors: github.com/nychealth/nyccas-data  hist/csv/2023/6.csv + hist/csv/location.csv
    AirNow hourly files (state + NJ reference monitors): files.airnowtech.org (public S3 mirror)
Only reference monitors: PurpleAir history is paid per sensor, and 2023's sensors aren't today's.

The snapshots are small and saved in data/static/replay_20230607/, so the demo needs no internet.
Rebuild them (needs internet) from the repo root:
    python -m app.replay
"""

import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from app.dashboard import ROOT

DAY = "2023-06-07"
PEAK_UTC = datetime(2023, 6, 7, 18, tzinfo=timezone.utc)
HOURS = 12
DIR = ROOT / "data" / "static" / "replay_20230607"
NYCCAS = "https://raw.githubusercontent.com/nychealth/nyccas-data/main/hist/csv"
AIRNOW = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/{d:%Y}/{d:%Y%m%d}/{name}"
BOX = dict(lat=(40.49, 40.92), lon=(-74.26, -73.70))
COLS = ["snapshot_utc", "source", "site_id", "lat", "lon", "pm25", "obs_time_utc", "noise_class"]


def get(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def in_box(df):
    return df[df["lat"].between(*BOX["lat"]) & df["lon"].between(*BOX["lon"])]


def city_hours(hours):
    obs = pd.read_csv(io.BytesIO(get(f"{NYCCAS}/2023/6.csv")), parse_dates=["ObservationTimeUTC"])
    loc = pd.read_csv(io.BytesIO(get(f"{NYCCAS}/location.csv")), parse_dates=["StartTime", "EndTime"])
    out = []
    for t in hours:
        naive = t.replace(tzinfo=None)
        here = loc[(loc["StartTime"] <= naive) & (loc["EndTime"] >= naive)].drop_duplicates("SiteID")
        o = obs[obs["ObservationTimeUTC"] == naive].merge(here, on="SiteID")
        out.append(pd.DataFrame({"t": t, "source": "nyc_dohmh", "site_id": o["SiteID"], "lat": o["Latitude"],
                                 "lon": o["Longitude"], "pm25": o["Value"]}))
    return pd.concat(out)


def airnow_hours(hours):
    sites = pd.read_csv(io.BytesIO(get(AIRNOW.format(d=hours[-1], name="Monitoring_Site_Locations_V2.dat"))),
                        sep="|", encoding="latin1", dtype=str).drop_duplicates("StationID")
    sites = sites.assign(lat=sites["Latitude"].astype(float), lon=sites["Longitude"].astype(float))
    out = []
    for t in hours:
        h = pd.read_csv(io.BytesIO(get(AIRNOW.format(d=t, name=f"HourlyData_{t:%Y%m%d%H}.dat"))), sep="|",
                        header=None, encoding="latin1", dtype={2: str},
                        names=["date", "time", "aqsid", "name", "gmt", "param", "unit", "value", "agency"])
        pm = h[h["param"] == "PM2.5"].merge(sites[["StationID", "lat", "lon"]], left_on="aqsid", right_on="StationID")
        out.append(pd.DataFrame({"t": t, "source": "airnow", "site_id": "AQS" + pm["aqsid"], "lat": pm["lat"],
                                 "lon": pm["lon"], "pm25": pm["value"].astype(float)}))
    return pd.concat(out)


def main():
    hours = [PEAK_UTC - timedelta(hours=h) for h in reversed(range(HOURS))]
    df = in_box(pd.concat([city_hours(hours), airnow_hours(hours)]))
    df = df[df["pm25"] >= 0]
    DIR.mkdir(parents=True, exist_ok=True)
    for t, g in df.groupby("t"):
        stamp = f"{t:%Y%m%dT%H%M%SZ}"
        g.assign(snapshot_utc=stamp, obs_time_utc=f"{t:%Y-%m-%d %H:%M:%S}+00:00", noise_class="reference")[COLS] \
            .to_csv(DIR / f"snapshot_{stamp}.csv", index=False)
        print(f"{stamp}: {len(g)} monitors, {g['pm25'].min():.0f}-{g['pm25'].max():.0f} ug/m3")
    print(f"Wrote {df['t'].nunique()} hourly snapshots to {DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
