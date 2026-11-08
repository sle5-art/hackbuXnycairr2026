import numpy as np
from sensorgap import labels as L


def test_breakpoints_2024():
    vals = [0.0, 9.0, 9.04, 9.1, 35.4, 35.5, 55.4, 55.5, 125.4, 125.5, 225.4, 225.5]
    assert list(L.category(vals)) == [
        "Good", "Good", "Good", "Moderate", "Moderate", "Unhealthy for Sensitive Groups",
        "Unhealthy for Sensitive Groups", "Unhealthy", "Unhealthy", "Very Unhealthy", "Very Unhealthy", "Hazardous"]


def test_nowcast_steady_air_is_plain_average():
    hourly = np.full((12, 1), 7.0)
    hourly_std = np.full((12, 1), 1.0)
    result = L.nowcast(hourly, hourly_std)
    mean = result[0]
    std = result[1]
    assert np.isclose(mean[0], 7.0) and np.isclose(std[0], 1.0)


def test_nowcast_changing_air_weights_recent_hours():
    hourly = np.array([[50.0]] + [[5.0]] * 11)
    hourly_std = np.ones((12, 1))
    result = L.nowcast(hourly, hourly_std)
    mean = result[0]
    expected = (50 + 5 * sum(0.5 ** i for i in range(1, 12))) / sum(0.5 ** i for i in range(12))
    assert np.isclose(mean[0], expected)


def test_nowcast_skips_missing_hours():
    hourly = np.array([[8.0], [np.nan], [8.0]] + [[np.nan]] * 9)
    hourly_std = np.ones((12, 1))
    result = L.nowcast(hourly, hourly_std)
    mean = result[0]
    assert np.isclose(mean[0], 8.0)


def test_label_only_when_band_inside_one_category():
    lower = np.array([2.0, 6.0])
    upper = np.array([8.5, 12.0])
    result = L.label(lower, upper)
    text = result[0]
    confident = result[1]
    assert list(text) == ["Good", "Good to Moderate, not enough data"]
    assert list(confident) == [True, False]
