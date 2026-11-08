import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

from app import dashboard as D


def fake():
    lats, lons = np.linspace(40.6, 40.8, 5), np.linspace(-74.0, -73.8, 4)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    n = la.size
    labels = pd.DataFrame({"lat": la.ravel(), "lon": lo.ravel(), "nowcast": np.full(n, 8.5),
                           "lo95": np.full(n, 7.0), "hi95": np.linspace(9.0, 10.0, n)})
    labels["confident"] = labels["hi95"] <= 9.0
    labels["label"] = np.where(labels["confident"], "Good", "Good to Moderate, not enough data")
    west = box(-74.01, 40.59, -73.9, 40.81)  # west half is land, east half is water
    boroughs = {"type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {"BoroName": "Test"}, "geometry": mapping(west)}]}
    snapshot = pd.DataFrame({"source": ["nyc_dohmh", "purpleair"], "site_id": ["A", "B"], "lat": [40.7, 40.6],
                             "lon": [-74.0, -73.95], "pm25": [5.0, 3.0], "noise_class": ["reference", "lowcost"]})
    return {"labels": labels, "boroughs": boroughs, "snapshot": snapshot, "holdout": None, "placement": None}


def test_water_is_transparent_and_unsure_cells_are_striped():
    d = fake()
    g = D.to_raster(d["labels"], D.land_of(d["boroughs"]))
    assert g["land"][:, :2].all() and not g["land"][:, 2:].any()
    img, _ = D.layer_image(g, "aqi")
    assert img.shape == (5 * D.PX, 4 * D.PX, 4)
    assert (img[:, 3 * D.PX:, 3] == 0).all()  # water
    unsure = np.repeat(np.repeat(~g["confident"] & g["land"], D.PX, 0), D.PX, 1)
    assert (img[unsure][:, :3] == 90).all(axis=1).any()  # stripes drawn on unsure land cells
    assert not (img[~unsure & (img[..., 3] > 0)][:, :3] == 90).all(axis=1).any()  # and nowhere else


def test_map_builds_for_every_layer():
    d = fake()
    for layer in D.LAYERS.values():
        html = D.build_map(d, layer).get_root().render()
        assert "imageOverlay" in html and "Reference monitor (city or state)" in html


def test_click_readout_explains_the_label():
    d = fake()
    land = D.land_of(d["boroughs"])
    assert D.explain(d, land, 40.7, -73.85) is None  # water
    sure = D.explain(d, land, 40.6, -74.0)  # first cell: band 7.0 to 9.0, inside Good
    assert "whole band sits inside Good" in sure and "Nearest sensor" in sure
    unsure = D.explain(d, land, 40.8, -73.933)
    assert "No label" in unsure


def test_walkthrough_nearest_distances():
    from app.walkthrough import nearest
    sensors = pd.DataFrame({"site_id": ["R", "P"], "source": ["nyc_dohmh", "purpleair"], "lat": [40.70, 40.71],
                            "lon": [-74.00, -74.00], "noise_class": ["reference", "lowcost"]})
    pts = pd.DataFrame({"lat": [40.712], "lon": [-74.00]})
    out = nearest(pts, sensors)
    assert out["nearest_sensor"].iloc[0] == "P"
    assert abs(out["nearest_sensor_km"].iloc[0] - 0.22) < 0.02  # 0.002 deg of latitude
    assert abs(out["nearest_reference_km"].iloc[0] - 1.33) < 0.02


def test_map_image_keeps_exact_colours():
    img = np.zeros((40, 30, 4), dtype=np.uint8)
    img[..., :3] = D.hex_to_rgb("#7e0023")  # EPA Hazardous maroon; folium's own projection turned it pink
    img[..., 3] = 170
    out = D.to_web_mercator(img, [[40.49, -74.26], [40.92, -73.70]])
    assert out.dtype == np.uint8
    assert {tuple(px) for px in out.reshape(-1, 4)} == {(126, 0, 35, 170)}


def test_colour_scale_is_fixed_not_stretched():
    # a 0.5 ug/m3 spread on a calm hour must not span the whole colour ramp
    d = fake()
    g = D.to_raster(d["labels"], D.land_of(d["boroughs"]))
    _, cmap = D.layer_image(g, "unsure")
    assert cmap.vmin == 0 and cmap.vmax == 5.0
    _, cmap = D.layer_image(g, "pm25")
    assert cmap.vmin == 0 and cmap.vmax == 35.4
    g["nowcast"] = g["nowcast"] * 10  # smoke-day values move to a wider, still fixed, scale
    _, cmap = D.layer_image(g, "pm25")
    assert cmap.vmax == 125.4
