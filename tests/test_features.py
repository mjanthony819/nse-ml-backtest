"""features() builds the model matrix: features must be causal, labels may look ahead
exactly one day (fwd_ret, label_end), and incomplete rows must be dropped, never kept."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("run", Path(__file__).parents[1] / "src" / "run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def frame(n=120, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": rng.integers(1e5, 1e7, n).astype(float)}, index=idx)


def test_feature_columns_are_causal():
    """Changing the future must leave every earlier feature value untouched."""
    df = frame()
    cut = 90
    base = m.features(df, "X.NS")
    mod = df.copy()
    mod.iloc[cut:, mod.columns.get_indexer(["Open", "High", "Low", "Close", "Volume"])] *= 4.0
    alt = m.features(mod, "X.NS")
    common = base.index[base.index < mod.index[cut - 1]]
    pd.testing.assert_frame_equal(base.loc[common, m.FEATURES], alt.loc[common, m.FEATURES])


def test_labels_look_ahead_exactly_one_day():
    df = frame()
    x = m.features(df, "X.NS")
    px = df.Close
    for d in x.index[:50]:
        nxt = px.index[px.index.get_loc(d) + 1]
        assert x.loc[d, "label_end"] == nxt
        expected = px.loc[nxt] / px.loc[d] - 1
        assert x.loc[d, "fwd_ret"] == pytest.approx(expected)
        assert x.loc[d, "y"] == int(expected > 0)


def test_incomplete_rows_are_dropped():
    df = frame()
    x = m.features(df, "X.NS")
    assert df.index[-1] not in x.index            # last day has no next-day label
    assert len(x) < len(df)                        # warm-up rows dropped too
    assert not x[m.FEATURES + ["fwd_ret", "label_end"]].isna().any().any()
    assert x.index.name == "date" and (x["ticker"] == "X.NS").all()
