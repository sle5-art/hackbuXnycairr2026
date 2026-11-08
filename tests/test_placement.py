import numpy as np
import pandas as pd

from sensorgap import model as m
from sensorgap import placement as P


def _gp_and_sites():
    rng = np.random.default_rng(0)
    lat = rng.uniform(40.6, 40.85, 40)
    lon = rng.uniform(-74.1, -73.95, 40)
    df = pd.DataFrame(dict(lat=lat, lon=lon, pm25=8 + rng.normal(0, 1, 40), noise_class="lowcost"))
    gp = m.fit(df, noise_sd={"reference": 1.0, "lowcost": 2.0}, restarts=0)
    sites = pd.DataFrame(dict(lat=np.repeat(np.linspace(40.6, 40.85, 6), 6),
                              lon=np.tile(np.linspace(-74.1, -73.75, 6), 6)))
    prediction = gp.predict(m.to_xy_km(sites["lat"], sites["lon"]), return_cov=True)
    cov = prediction[1]
    return sites, cov


def test_first_pick_goes_where_uncertainty_is():
    setup = _gp_and_sites()
    sites = setup[0]
    cov = setup[1]
    result = P.greedy(cov, np.ones(len(sites)), 1, 4.0)
    picks = result[0]
    assert sites["lon"].iloc[picks[0]] > -73.95


def test_weights_steer_the_pick():
    setup = _gp_and_sites()
    sites = setup[0]
    cov = setup[1]
    w = np.where((sites["lon"] > -73.8) & (sites["lat"] < 40.65), 100.0, 1.0)
    result = P.greedy(cov, w, 1, 4.0)
    picks = result[0]
    assert sites["lon"].iloc[picks[0]] > -73.85 and sites["lat"].iloc[picks[0]] < 40.7


def test_greedy_never_repeats_and_variance_only_falls():
    setup = _gp_and_sites()
    sites = setup[0]
    cov = setup[1]
    result = P.greedy(cov, np.ones(len(sites)), 5, 4.0)
    picks = result[0]
    gains = result[1]
    after = result[2]
    assert len(set(picks)) == 5
    assert np.all(np.diag(after) <= np.diag(cov) + 1e-9)
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))


def test_min_spacing_keeps_picks_apart():
    setup = _gp_and_sites()
    sites = setup[0]
    cov = setup[1]
    xy = m.to_xy_km(sites["lat"], sites["lon"])
    w = np.where((sites["lon"] > -73.8) & (sites["lat"] < 40.65), 100.0, 1.0)
    result = P.greedy(cov, w, 4, 4.0, xy, min_km=8.0)
    picks = result[0]
    d = np.hypot(*(xy[picks][:, None] - xy[picks][None]).transpose(2, 0, 1))
    assert d[np.triu_indices(4, 1)].min() >= 8.0
