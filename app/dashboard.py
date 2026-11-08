"""
SensorGap map: where NYC's PM2.5 map is sure, where it is guessing, and where the next sensors should go.

Reads only files that are already computed, so it opens in about a second:
    outputs/labels_<UTC>.csv          (python -m sensorgap.labels)
    outputs/placement.csv             (python -m sensorgap.placement --k 10)
    outputs/validation_holdout_learned.csv   (python -m sensorgap.validate --strict)
    data/snapshots/snapshot_<UTC>.csv (python -m sensorgap.ingest)
    data/static/nyc_boroughs.geojson  (NYC DCP borough boundaries, clipped to the shoreline)
Slide figures (reliability curve, placement table, fairness chart) live in app/slide_figures.py.

Run from the repo root:
    python -m streamlit run app/dashboard.py
"""

import json
from pathlib import Path

import branca.colormap as bcm
import folium
import numpy as np
import pandas as pd
import shapely
from branca.element import MacroElement, Template
from folium.utilities import mercator_transform
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# EPA AQI colours, upper edge of each PM2.5 category in ug/m3 (2024 breakpoints, same as sensorgap/labels.py)
AQI = [(9.0, "#00e400", "Good"), (35.4, "#ffff00", "Moderate"), (55.4, "#ff7e00", "Unhealthy for Sensitive Groups"),
       (125.4, "#ff0000", "Unhealthy"), (225.4, "#8f3f97", "Very Unhealthy"), (500.0, "#7e0023", "Hazardous")]
BOROUGH = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
SOURCE = {"nyc_dohmh": "NYC Health monitor", "airnow": "State (DEC) monitor via AirNow", "purpleair": "PurpleAir"}
SITE_NAMES = {"36061NY09929": "Midtown West"}  # names the city feed doesn't carry
PX = 8  # image pixels per grid cell (room for the "not enough data" stripes)
UNIT = "µg/m³"
PIN = "#c51b7d"
LAYERS = {"How unsure": "unsure", "PM2.5 estimate": "pm25", "AQI label": "aqi"}


def latest(pattern, folder=OUT):
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load():
    """Everything the map shows, as plain data. Missing files come back as None."""
    def read(path, fn):
        return fn(path) if path and path.exists() else None

    labels_path = latest("labels_*.csv")
    return {
        "labels_path": labels_path,
        "labels": read(labels_path, pd.read_csv),
        "placement": read(OUT / "placement.csv", lambda p: pd.read_csv(p, dtype={"tract": str})),
        "holdout": read(OUT / "validation_holdout_learned.csv", pd.read_csv),
        "snapshot": read(latest("snapshot_*.csv", ROOT / "data" / "snapshots"), pd.read_csv),
        "boroughs": read(ROOT / "data" / "static" / "nyc_boroughs.geojson", lambda p: json.loads(p.read_text())),
    }


def land_of(boroughs):
    return unary_union([shape(f["geometry"]) for f in boroughs["features"]])


def to_raster(labels, land):
    """Grid columns -> 2-D arrays (row 0 = north) plus a land mask and the image bounds."""
    lats, lons = np.sort(labels["lat"].unique()), np.sort(labels["lon"].unique())
    def grid(col):
        return labels.pivot(index="lat", columns="lon", values=col).reindex(index=lats, columns=lons).values[::-1]
    g = {c: grid(c) for c in ["nowcast", "lo95", "hi95"]}
    g["confident"] = grid("confident").astype(bool)
    g["halfwidth"] = (g["hi95"] - g["lo95"]) / 2
    lon_g, lat_g = np.meshgrid(lons, lats[::-1])
    g["land"] = shapely.contains_xy(land, lon_g, lat_g)
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    g["bounds"] = [[lats[0] - dlat / 2, lons[0] - dlon / 2], [lats[-1] + dlat / 2, lons[-1] + dlon / 2]]
    return g


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def paint(colors, g, stripes, alpha=170):
    """colors: (rows, cols, 3) per cell -> RGBA image; water transparent; optional grey stripes on unsure cells."""
    img = np.repeat(np.repeat(colors, PX, 0), PX, 1)
    a = np.repeat(np.repeat(np.where(g["land"], alpha, 0), PX, 0), PX, 1)
    if stripes:
        yy, xx = np.indices(a.shape)
        img[np.repeat(np.repeat(~g["confident"], PX, 0), PX, 1) & ((xx + yy) % PX < 2)] = 90
    return np.dstack([img, a]).astype(np.uint8)


def ramp(values, land, colors, caption):
    v = values[land]
    cmap = bcm.LinearColormap(colors, vmin=float(v.min()), vmax=float(v.max()), caption=caption)
    return np.array([[hex_to_rgb(cmap(x)[:7]) for x in row] for row in np.nan_to_num(values)]), cmap


def layer_image(g, layer):
    """Image and legend for one layer: 'unsure' (band half-width), 'pm25' (NowCast value), 'aqi' (category)."""
    if layer == "unsure":
        colors, cmap = ramp(g["halfwidth"], g["land"], ["#f2f0f7", "#9e9ac8", "#3f007d"],
                            f"How unsure: ± half-width of the 95% band ({UNIT})")
    elif layer == "pm25":
        colors, cmap = ramp(g["nowcast"], g["land"], ["#fff5eb", "#fd8d3c", "#7f2704"],
                            f"PM2.5, 12-hour NowCast ({UNIT})")
    else:
        idx = np.searchsorted([u for u, _, _ in AQI], np.floor(np.nan_to_num(g["nowcast"]) * 10 + 1e-9) / 10)
        n = max(3, int(idx[g["land"]].max()) + 1)  # legend shows up to the worst category on the map
        edges = [0.0] + [u for u, _, _ in AQI[:n]]
        cmap = bcm.StepColormap([c for _, c, _ in AQI[:n]], index=edges, vmin=0, vmax=edges[-1],
                                caption=f"AQI label ({UNIT}) · stripes: not enough data")
        colors = np.array([hex_to_rgb(AQI[i][1]) for i in idx.ravel()]).reshape(*idx.shape, 3)
    return paint(colors, g, stripes=layer == "aqi"), cmap


def to_web_mercator(img, bounds):
    """Stretch the image rows to web-Mercator, keeping exact colours. folium's own mercator_project returns floats,
    which folium then rescales channel by channel, shifting every colour (EPA's maroon came out pink)."""
    out = mercator_transform(img, (bounds[0][0], bounds[1][0]))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def biggest_miss(holdout):
    """Held-out site the map got most wrong on average (by |z|), with its observed range and mean prediction."""
    by = holdout.assign(az=holdout["z"].abs()).groupby("site_id")
    site = by["az"].mean().idxmax()
    rows = holdout[holdout["site_id"] == site]
    return {"site_id": site, "name": SITE_NAMES.get(site, site), "hours": len(rows),
            "observed": (rows["observed"].min(), rows["observed"].max()), "predicted": rows["predicted"].mean()}


def km_between(lat, lon, lats, lons):
    """Distance in km from one point to many (flat-earth approximation; fine at city scale)."""
    dy = (np.asarray(lats) - lat) * 111.0
    dx = (np.asarray(lons) - lon) * 111.0 * np.cos(np.radians(lat))
    return np.hypot(dx, dy)


KEY = Template("""
{% macro script(this, kwargs) %}
var key = L.control({position: 'bottomleft'});
key.onAdd = function () {
  var d = L.DomUtil.create('div');
  d.style.cssText = 'background:rgba(255,255,255,.92);padding:8px 10px;border-radius:4px;font:12px/1.7 system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.3)';
  d.innerHTML = `
    <div><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#222"/></svg> Reference monitor (city or state)</div>
    <div><svg width="14" height="14"><circle cx="7" cy="7" r="3.5" fill="#fff" stroke="#777"/></svg> PurpleAir low-cost sensor</div>
    <div><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="none" stroke="#d7191c" stroke-width="2.5"/></svg> Biggest miss when hidden</div>
    <div><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{{ this.pin }}" stroke="#fff" stroke-width="1.5"/></svg> Suggested new sensor (rank)</div>`;
  return d;
};
key.addTo({{ this._parent.get_name() }});
{% endmacro %}
""")


def add_key(m):
    el = MacroElement()
    el._template = KEY
    el.pin = PIN
    m.add_child(el)


def build_map(d, layer="unsure"):
    m = folium.Map(location=[40.70, -73.95], zoom_start=11, tiles=None, control_scale=True, zoom_snap=0.25)
    folium.TileLayer("Esri.WorldGrayCanvas", control=False).add_to(m)
    land = land_of(d["boroughs"])
    g = to_raster(d["labels"], land)
    img, cmap = layer_image(g, layer)
    folium.raster_layers.ImageOverlay(to_web_mercator(img, g["bounds"]), bounds=g["bounds"], name="map",
                                      control=False).add_to(m)
    cmap.add_to(m)
    x0, y0, x1, y1 = land.bounds
    m.fit_bounds([[y0, x0], [y1, x1]])

    outlines = folium.FeatureGroup("Borough outlines")
    folium.GeoJson(d["boroughs"], style_function=lambda _: {"color": "#555", "weight": 1, "fill": False},
                   tooltip=folium.GeoJsonTooltip(["BoroName"], labels=False)).add_to(outlines)
    outlines.add_to(m)

    if d["snapshot"] is not None:
        refs, cheap = folium.FeatureGroup("Reference monitors"), folium.FeatureGroup("PurpleAir sensors")
        for r in d["snapshot"].itertuples():
            ref = r.noise_class == "reference"
            folium.CircleMarker([r.lat, r.lon], radius=5 if ref else 3.5, weight=1, color="#222" if ref else "#777",
                                fill=True, fill_color="#222" if ref else "#fff", fill_opacity=0.9,
                                tooltip=f"{SOURCE.get(r.source, r.source)}: {r.pm25:.1f} {UNIT} this hour"
                                ).add_to(refs if ref else cheap)
        cheap.add_to(m)
        refs.add_to(m)

        miss = biggest_miss(d["holdout"]) if d["holdout"] is not None else None
        hit = d["snapshot"][d["snapshot"]["site_id"] == miss["site_id"]] if miss else []
        if len(hit):
            o, p = miss["observed"], miss["predicted"]
            fg = folium.FeatureGroup("Biggest miss")
            folium.CircleMarker([hit.lat.iloc[0], hit.lon.iloc[0]], radius=11, weight=3, color="#d7191c", fill=False,
                                tooltip=folium.Tooltip(f"{miss['name']}: reads {o[0]:.0f}-{o[1]:.0f}, map says ~{p:.0f}",
                                                       permanent=True, direction="right")).add_to(fg)
            fg.add_to(m)

    if d["placement"] is not None:
        fg = folium.FeatureGroup("Suggested new sensors")
        for r in d["placement"].itertuples():
            shared = pd.notna(r.rank_if_population_only)
            folium.Marker([r.lat, r.lon], icon=folium.DivIcon(
                html=f'<div style="background:{PIN};color:#fff;border:2px solid #fff;border-radius:50%;width:24px;'
                     f'height:24px;text-align:center;font:600 12px/20px system-ui,sans-serif;'
                     f'font-variant-numeric:tabular-nums;box-shadow:0 1px 3px rgba(0,0,0,.45)">{r.rank}</div>',
                icon_size=(24, 24), icon_anchor=(12, 12)),
                tooltip=f"Suggested sensor #{r.rank}: {BOROUGH.get(r.tract[2:5], '')} tract {r.tract}, "
                        f"adult asthma {r.asthma_pct:.1f}%, {r.adults:,} adults"
                        + (" (population-only weighting also picks it)" if shared else "")).add_to(fg)
        fg.add_to(m)

    add_key(m)
    folium.LayerControl(collapsed=False, position="topleft").add_to(m)
    return m


def explain(d, land, lat, lon):
    """Plain-language readout for a clicked point, or None if it is on water or outside NYC."""
    if not shapely.contains_xy(land, lon, lat):
        return None
    lab = d["labels"]
    c = lab.loc[((lab["lat"] - lat) ** 2 + (lab["lon"] - lon) ** 2).idxmin()]
    lines = [f"**{c['label']}** · estimate {c['nowcast']:.1f} {UNIT}, 95% band {c['lo95']:.1f} to {c['hi95']:.1f}"]
    if c["confident"]:
        lo, (hi, _, name) = next((lo, a) for lo, a in zip([0.0] + [u for u, _, _ in AQI], AQI) if c["hi95"] <= a[0])
        lines.append(f"Labelled because the whole band sits inside {name} ({lo:g} to {hi:g} {UNIT}).")
    else:
        lines.append("No label: the band crosses an AQI boundary, so the map can't say which category is true.")
    s = d["snapshot"]
    if s is not None:
        km = km_between(c["lat"], c["lon"], s["lat"], s["lon"])
        i = int(np.argmin(km))
        ref = (s["noise_class"] == "reference").values
        near_ref = f"; nearest reference monitor {km[ref].min():.1f} km" if ref.any() else ""
        lines.append(f"Nearest sensor {km[i]:.1f} km away ({SOURCE.get(s['source'].iloc[i], s['source'].iloc[i])})"
                     f"{near_ref}.")
    return "  \n".join(lines)


def main():
    import streamlit as st
    from streamlit_folium import st_folium

    st.set_page_config(page_title="SensorGap · NYC PM2.5 map", layout="wide")
    d = st.cache_data(_load_keyed)(_files_key())
    if d["labels"] is None or d["boroughs"] is None:
        st.error("No outputs/labels_*.csv or data/static/nyc_boroughs.geojson yet. "
                 "Run `python -m sensorgap.labels` from the repo root first.")
        return

    stamp = d["labels_path"].stem.split("_", 1)[1]
    st.title("How sure is NYC's air-quality map?")
    top = st.columns([3, 2])
    top[0].caption(f"PM2.5, 12-hour NowCast ending {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} "
                   f"{stamp[9:11]}:{stamp[11:13]} UTC. Pink pins: where 10 new sensors would cut uncertainty most "
                   "for adults with asthma. Click any spot on land.")
    layer = top[1].radio("Colour the map by", list(LAYERS), horizontal=True, label_visibility="collapsed")

    m = build_map(d, LAYERS[layer])
    click = st_folium(m, height=720, use_container_width=True, returned_objects=["last_clicked"])
    pt = (click or {}).get("last_clicked")
    text = explain(d, land_of(d["boroughs"]), pt["lat"], pt["lng"]) if pt else None
    if text:
        st.info(text)

    st.caption("**Preliminary:** a few hours of data. Concentrations are hourly maps combined by NowCast; "
               "PurpleAir uses the EPA US-wide correction; asthma rates are CDC PLACES model estimates. "
               "Advisory only, not regulatory.")


def _files_key():
    # names and modification times of the inputs, so re-running labels/placement/validate shows up on refresh
    files = [latest("labels_*.csv"), OUT / "placement.csv", OUT / "validation_holdout_learned.csv",
             latest("snapshot_*.csv", ROOT / "data" / "snapshots")]
    return tuple((str(f), f.stat().st_mtime) for f in files if f and f.exists())


def _load_keyed(key):
    return load()


if __name__ == "__main__":
    main()
