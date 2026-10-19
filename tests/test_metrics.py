import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("run", Path(__file__).parents[1] / "src" / "run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rsi_known_extremes():
    up = pd.Series(range(1, 40), dtype=float)  # strictly rising: no down moves
    assert m.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    mixed = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19], dtype=float)
    r = m.rsi(mixed, 14).iloc[-1]
    assert 0 < r < 100


def test_metrics_drawdown_is_measured_from_initial_wealth():
    ret = pd.Series([-0.1, -0.1])
    res = m.metrics(ret)
    assert res["Total return %"] == pytest.approx(-19.0)
    assert res["Max DD %"] == pytest.approx(-19.0)  # peak clips at 1, so the path counts from day one


def test_metrics_sharpe_sign_follows_mean_excess_return():
    rng = np.random.default_rng(3)
    good = pd.Series(rng.normal(0.002, 0.01, 500))   # mean above the daily risk-free
    bad = pd.Series(rng.normal(-0.002, 0.01, 500))
    assert m.sharpe(good) > 0 > m.sharpe(bad)
    assert m.sharpe(good, 0) > m.sharpe(good)  # zero-rf Sharpe exceeds the 6.5%-rf one


def test_accuracy_test_counts_and_baseline():
    pred = pd.DataFrame({"p": [0.9, 0.8, 0.2, 0.4], "y": [1, 1, 0, 0]})
    acc = m.accuracy_test(pred)
    assert acc["correct"] == 4 and acc["n"] == 4 and acc["accuracy"] == 1.0
    assert acc["test_up_rate"] == 0.5 and acc["majority_class_accuracy"] == 0.5
    all_up = pd.DataFrame({"p": [0.9, 0.9], "y": [1, 1]})
    assert m.accuracy_test(all_up)["accuracy"] == 1.0


def test_portfolio_charges_cost_on_entry_and_exit():
    dates = pd.bdate_range("2023-01-02", periods=2)
    pred = pd.DataFrame({"ticker": ["A", "A"], "p": [1.0, 0.0]}, index=dates)
    ret = pd.DataFrame({"A": [0.1, 0.1]}, index=dates)
    strategy, ew, pos, turnover = m.portfolio(pred, 0.001, ret)
    assert strategy.iloc[0] == pytest.approx(0.1 - 0.001)  # day 1: long, pays to enter
    assert strategy.iloc[1] == pytest.approx(-0.001)       # day 2: flat, pays to exit
    assert turnover["A"].tolist() == [1.0, 1.0]
