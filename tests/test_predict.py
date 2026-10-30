"""predict() is the walk-forward engine: fold boundaries, per-fold label purging and the
fold audit metadata are what make the out-of-sample results interpretable, so pin them."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("run", Path(__file__).parents[1] / "src" / "run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dataset(n=1150, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    x = pd.DataFrame(rng.normal(0, 1, (n, len(m.FEATURES))), columns=m.FEATURES, index=dates)
    x["fwd_ret"] = rng.normal(0, 0.01, n)
    x["label_end"] = pd.Series(dates, index=dates).shift(-1)
    x["ticker"] = "X.NS"
    x["y"] = (x.fwd_ret > 0).astype(int)
    x.index.name = "date"
    return x.dropna(subset=["label_end"])


def test_static_mode_is_a_single_fold_spanning_the_whole_test_period():
    pred, folds = m.predict(dataset(), "2023-01-01", "2025-06-30", "static")
    assert len(folds) == 1 and (pred.fold_start == "2023-01-01").all()
    f = folds[0]
    assert f["train_last_label"] < f["test_start"]  # purged: no training label touches the test period
    assert pred.index.min() >= pd.Timestamp("2023-01-01")
    assert pred.index.max() < pd.Timestamp("2025-06-30")
    assert pred["p"].between(0, 1).all()


def test_walk_forward_adds_yearly_boundaries_with_expanding_train():
    pred, folds = m.predict(dataset(), "2023-01-01", "2025-06-30", "walk_forward")
    assert len(folds) == 3
    assert [fs for fs in pred.fold_start.unique()] == ["2023-01-01", "2024-01-01", "2025-01-01"]
    for f in folds:
        assert f["train_last_label"] < f["test_start"] <= f["test_end"]
    # expanding window: every retrain sees strictly more rows
    assert [f["train_rows"] for f in folds] == sorted(f["train_rows"] for f in folds)
    assert len(set(f["train_rows"] for f in folds)) == 3
    # fold test windows tile the period without overlap
    assert not pred.index.duplicated().any()
    assert sum(f["test_rows"] for f in folds) == len(pred)


def test_shuffled_label_fits_are_seed_reproducible():
    data = dataset(n=600)
    a, _ = m.predict(data, "2023-01-01", "2023-06-30", "static", np.random.default_rng(42))
    b, _ = m.predict(data, "2023-01-01", "2023-06-30", "static", np.random.default_rng(42))
    assert a["p"].equals(b["p"])  # the null distribution depends on this being repeatable
