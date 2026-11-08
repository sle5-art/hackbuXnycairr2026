import numpy as np
import pandas as pd

from sensorgap import model as m


def _synthetic(seed=1):
    rng = np.random.default_rng(seed)
    n_lc, n_ref = 86, 26
    lat = rng.uniform(40.55, 40.88, n_lc + n_ref)
    lon = rng.uniform(-74.15, -73.75, n_lc + n_ref)
    cls = np.array(["lowcost"] * n_lc + ["reference"] * n_ref)

    def field(la, lo):
        xy = m.to_xy_km(la, lo)
        return 8 + 3 * np.sin(xy[:, 0] / 6) + 12 * np.exp(-((xy[:, 0] - 2) ** 2 + (xy[:, 1] + 3) ** 2) / 4)

    y = field(lat, lon) + rng.normal(0, np.where(cls == "lowcost", 3.0, 1.0))
    return pd.DataFrame(dict(lat=lat, lon=lon, pm25=y, noise_class=cls)), field


def test_band_covers_true_field_about_95_percent():
    df, field = _synthetic()
    grid = m.predict_grid(m.fit(df), spacing_km=1.0)
    truth = field(grid["lat"].to_numpy(), grid["lon"].to_numpy())
    inside = (truth >= grid["mean"] - 1.96 * grid["std"]) & (truth <= grid["mean"] + 1.96 * grid["std"])
    assert 0.85 <= inside.mean() <= 1.0


def test_uncertainty_grows_away_from_sensors():
    df, _ = _synthetic()
    gp = m.fit(df)
    _, near = m.predict(gp, df["lat"].iloc[:1], df["lon"].iloc[:1])
    _, far = m.predict(gp, [40.92], [-74.26])
    assert far[0] > near[0]


def test_purpleair_offset_is_measured_and_applied():
    import numpy as np
    import pandas as pd
    from sensorgap import model as m
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(40.6, 40.8, 6), rng.uniform(-74.0, -73.85, 6)
    ref = pd.DataFrame({"lat": lat, "lon": lon, "pm25": 8.0, "noise_class": "reference"})
    lc = pd.DataFrame({"lat": lat + 0.002, "lon": lon, "pm25": 6.0, "noise_class": "lowcost"})
    df = pd.concat([ref, lc], ignore_index=True)
    off, n = m.lowcost_offset(df)
    assert n == 6 and abs(off - 2.0) < 1e-9
    fixed = m.correct_lowcost(df)
    assert (fixed.loc[fixed["noise_class"] == "lowcost", "pm25"] == 8.0).all()
    assert (fixed.loc[fixed["noise_class"] == "reference", "pm25"] == 8.0).all()
    assert m.lowcost_offset(df.iloc[[0, 1, 6, 7]])[0] == 0.0
