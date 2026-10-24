import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("run", Path(__file__).parents[1] / "src" / "run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_features_do_not_see_future():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2020-01-01", periods=150)
    df = pd.DataFrame({"Close": 100 * np.exp(rng.normal(0, 0.01, 150).cumsum()), "Volume": rng.integers(100, 200, 150)}, index=dates)
    before = m.features(df, "X")
    changed = df.copy()
    changed.loc[dates[100]:, "Close"] *= 5
    after = m.features(changed, "X")
    pd.testing.assert_frame_equal(before.loc[:dates[99], m.FEATURES], after.loc[:dates[99], m.FEATURES])


def test_split_purges_boundary_label():
    dates = pd.bdate_range("2022-12-01", periods=50)
    data = pd.DataFrame({"label_end": pd.Series(dates, index=dates).shift(-1)}, index=dates)
    train, test = m.split_rows(data, pd.Timestamp("2023-01-02"), pd.Timestamp("2024-01-01"))
    assert train.label_end.max() < test.index.min()
    assert pd.Timestamp("2022-12-30") not in train.index


def test_risk_free_conversion_and_paired_bootstrap():
    assert np.isclose((1 + m.RF_DAILY) ** 252 - 1, 0.065)
    x = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 100))
    report = m.bootstrap(pd.DataFrame({"a": x, "same": x}), draws=100)
    assert report["strategy_minus_benchmark_sharpe_95_ci"]["same"]["ci"] == [0.0, 0.0]


def test_missing_feature_slot_stays_cash():
    dates = pd.bdate_range("2023-01-02", periods=2)
    pred = pd.DataFrame({"ticker": ["A", "A", "B"], "p": [1.0, 1.0, 1.0]}, index=[dates[0], dates[1], dates[0]])
    ret = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, 0.2]}, index=dates)
    strategy, ew, pos, turnover = m.portfolio(pred, 0, ret)
    assert strategy.iloc[1] == 0.05
    assert np.isclose(ew.iloc[1], 0.15)
    assert pos.loc[dates[1], "B"] == 0


def test_bootstrap_same_seed_reproduces_report():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"s": rng.normal(0.001, 0.02, 300), "b": rng.normal(0.0005, 0.02, 300)})
    assert m.bootstrap(df, draws=50) == m.bootstrap(df, draws=50)  # seeded: identical report


def test_zero_volume_rows_are_dropped_from_features():
    dates = pd.bdate_range("2020-01-01", periods=80)
    close = pd.Series(np.linspace(100, 120, 80), index=dates)
    vol = pd.Series([1000.0] * 80, index=dates)
    vol.iloc[60] = 0  # suspended-style zero-volume day must not flow into volume features
    df = pd.DataFrame({"Close": close, "Volume": vol}, index=dates)
    out = m.features(df, "X")
    assert dates[60] not in out.index
    assert dates[59] in out.index


def test_compact_svg_collapses_whitespace(tmp_path):
    p = tmp_path / "x.svg"
    p.write_text('<svg>\n   <rect   x="1"/>\n\n</svg>\n')
    m.compact_svg(p)
    txt = p.read_text()
    assert "\n" not in txt.strip() and "  " not in txt and txt.endswith("\n")
