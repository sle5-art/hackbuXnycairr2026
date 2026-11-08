import pandas as pd
from sensorgap.ingest import barkjohn_correct, barkjohn_qc


def test_correction_known_value():
    assert abs(barkjohn_correct(pd.Series([10.0]), pd.Series([50.0]))[0] - 6.68) < 1e-9


def test_correction_never_negative():
    assert barkjohn_correct(pd.Series([0.0]), pd.Series([100.0]))[0] == 0


def test_qc_keeps_agreeing_channels():
    assert barkjohn_qc(pd.Series([54.6]), pd.Series([54.7]))[0]


def test_qc_drops_disagreeing_channels():
    assert not barkjohn_qc(pd.Series([30.0]), pd.Series([2.0]))[0]


def test_qc_keeps_small_absolute_gap_at_low_levels():
    assert barkjohn_qc(pd.Series([0.2]), pd.Series([0.0]))[0]
