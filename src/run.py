"""Expanding-window NSE direction experiment. Educational, not investment advice."""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS"]
EXTRA = ["SBIN.NS", "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"]
FEATURES = [f"ret_{n}" for n in (1, 5, 10, 20)] + ["vol_10", "vol_20", "vol_chg_5", "vol_rel_20", "rsi_14", "rsi_5", "dist_ma50"]
RF_ANNUAL = 0.065
RF_DAILY = (1 + RF_ANNUAL) ** (1 / 252) - 1
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "nse"


def prices(ticker, start, end, refresh=False):
    """Cache the exact adjusted OHLCV snapshot; end is exclusive."""
    cache = ROOT / "data" / "cache" / f"{ticker}_{start}_{end}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True).dropna(subset=["Close"])
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No Yahoo data for {ticker}")
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(cache, float_format="%.17g")
    return pd.read_csv(cache, index_col=0, parse_dates=True).dropna(subset=["Close"])


def rsi(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)


def features(df, ticker):
    close, volume = df.Close, df.Volume.replace(0, np.nan)
    ret = close.pct_change(fill_method=None)
    x = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        x[f"ret_{n}"] = close.pct_change(n, fill_method=None)
    x["vol_10"] = ret.rolling(10).std()
    x["vol_20"] = ret.rolling(20).std()
    x["vol_chg_5"] = volume.pct_change(5, fill_method=None)
    x["vol_rel_20"] = volume / volume.rolling(20).mean()
    x["rsi_14"], x["rsi_5"] = rsi(close, 14), rsi(close, 5)
    x["dist_ma50"] = close / close.rolling(50).mean() - 1
    x["fwd_ret"] = ret.shift(-1)
    x["label_end"] = pd.Series(df.index, index=df.index).shift(-1)
    x["ticker"] = ticker
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["fwd_ret", "label_end"])
    x["y"] = (x.fwd_ret > 0).astype(int)
    x.index.name = "date"
    return x


def new_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
        min_samples_leaf=100, l2_regularization=1.0, random_state=0)


def split_rows(data, start, stop):
    # Purge any target touching the next test period. No future labels at retrain.
    train = data[(data.index < start) & (data.label_end < start)]
    test = data[(data.index >= start) & (data.index < stop)]
    assert len(train) and len(test)
    assert train.label_end.max() < test.index.min()
    return train, test


def predict(data, split, end, mode, rng=None):
    boundaries = [pd.Timestamp(split)]
    if mode == "walk_forward":
        boundaries += list(pd.date_range(f"{pd.Timestamp(split).year + 1}-01-01", end, freq="YS"))
    boundaries += [pd.Timestamp(end)]
    pieces, folds = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        train, test = split_rows(data, start, stop)
        y = train.y.to_numpy()
        if rng is not None:
            y = rng.permutation(y)
        model = new_model().fit(train[FEATURES], y)
        out = test.copy()
        out["p"] = model.predict_proba(test[FEATURES])[:, 1]
        out["fold_start"] = str(start.date())
        pieces.append(out)
        folds.append(dict(train_start=str(train.index.min().date()), train_last_signal=str(train.index.max().date()),
            train_last_label=str(train.label_end.max().date()), test_start=str(test.index.min().date()),
            test_end=str(test.index.max().date()), train_rows=len(train), test_rows=len(test)))
    return pd.concat(pieces).sort_index(kind="stable"), folds


def portfolio(pred, cost, stock_returns):
    ret = stock_returns.loc[pred.index.min():pred.index.max()].copy()
    if ret.isna().any().any():
        raise ValueError("Missing raw stock return: inspect provider calendar")
    pos = pred.assign(pos=(pred.p > 0.5).astype(float)).pivot(columns="ticker", values="pos").reindex(index=ret.index, columns=ret.columns).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strategy = (pos * ret - cost * turnover).mean(axis=1)
    equal_weight = ret.mean(axis=1).copy()
    equal_weight.iloc[0] -= cost
    return strategy, equal_weight, pos, turnover


def sharpe(ret, risk_free=RF_DAILY):
    return float((np.mean(ret) - risk_free) / np.std(ret, ddof=1) * np.sqrt(252))


def metrics(ret):
    eq = (1 + ret).cumprod()
    # Include initial wealth 1 when measuring drawdown.
    peak = eq.cummax().clip(lower=1)
    return {"CAGR %": float((eq.iloc[-1] ** (252 / len(ret)) - 1) * 100),
        "Ann. Sharpe (6.5% RF)": sharpe(ret), "Max DD %": float((eq / peak - 1).min() * 100),
        "Ann. Vol %": float(ret.std() * np.sqrt(252) * 100), "Total return %": float((eq.iloc[-1] - 1) * 100)}


def bootstrap(returns, draws=5000, block=20, seed=2026):
    """Paired circular moving-block bootstrap, common dates across every column."""
    a = returns.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(a)
    samples = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        x = a[idx]
        s = (x.mean(axis=0) - RF_DAILY) / x.std(axis=0, ddof=1) * np.sqrt(252)
        samples.append(s)
    samples = np.asarray(samples)
    report = {"method": "paired circular moving-block percentile bootstrap", "draws": draws,
        "block_trading_days": block, "seed": seed, "sharpe_95_ci": {}, "strategy_minus_benchmark_sharpe_95_ci": {}}
    for i, col in enumerate(returns.columns):
        report["sharpe_95_ci"][col] = np.quantile(samples[:, i], [0.025, 0.975]).tolist()
        if i:
            report["strategy_minus_benchmark_sharpe_95_ci"][col] = {
                "estimate": sharpe(a[:, 0]) - sharpe(a[:, i]),
                "ci": np.quantile(samples[:, 0] - samples[:, i], [0.025, 0.975]).tolist()}
    return report


def accuracy_test(pred):
    n = len(pred)
    k = int(((pred.p > 0.5).astype(int) == pred.y).sum())
    p_up = float(pred.y.mean())
    baseline = max(p_up, 1 - p_up)
    test = binomtest(k, n, baseline, alternative="two-sided")
    return {"correct": k, "n": n, "accuracy": k / n, "test_up_rate": p_up,
        "majority_class_accuracy": baseline, "two_sided_binomial_p_vs_majority": float(test.pvalue),
        "two_sided_binomial_p_vs_up_rate": float(binomtest(k, n, p_up).pvalue),
        "warning": "IID binomial diagnostic only: pooled tickers/days dependent; null rate estimated on test data. Not valid independent evidence of an edge."}


def compact_svg(path):
    path.write_text(re.sub(r"\s+", " ", path.read_text()).strip() + "\n")


def figures(returns, out):
    for name, curve, title, ylabel in [
        ("equity_curve", (1 + returns).cumprod(), "Out-of-sample wealth, net of signal costs", "Growth of 1"),
        ("drawdown", (lambda e: (e / e.cummax().clip(lower=1) - 1) * 100)((1 + returns).cumprod()), "Drawdown", "%")]:
        curve.plot(figsize=(10, 5), title=title)
        plt.ylabel(ylabel); plt.grid(alpha=0.3); plt.tight_layout()
        for ext in ("png", "svg"):
            plt.savefig(out / f"{name}.{ext}", dpi=130)
            if ext == "svg":
                compact_svg(out / f"{name}.{ext}")
        plt.close()
    # Importance is omitted: the old final model was applied to earlier folds, invalid for WF.


def run(args):
    out = ROOT / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    raw = {t: prices(t, args.start, args.end, args.refresh) for t in args.tickers}
    index_prices = prices("^NSEI", args.start, args.end, args.refresh)
    etf = prices("NIFTYBEES.NS", args.start, args.end, args.refresh)
    # Use shared observed closes, not mismatched holding periods or fabricated zero returns.
    common = index_prices.index.intersection(etf.index)
    union = common
    for df in raw.values():
        union = union.union(df.index)
        common = common.intersection(df.index)
    excluded_dates = [str(d.date()) for d in union.difference(common)]
    raw = {t: df.loc[common] for t, df in raw.items()}
    index_prices, etf = index_prices.loc[common], etf.loc[common]
    data = pd.concat([features(df, t) for t, df in raw.items()]).sort_index(kind="stable")
    stock_returns = pd.DataFrame({t: df.Close.pct_change(fill_method=None).shift(-1) for t, df in raw.items()})
    for mode in ("static", "walk_forward"):
        folder = out / mode
        folder.mkdir(exist_ok=True)
        pred, folds = predict(data, args.split, args.end, mode)
        strategy, ew, pos, turn = portfolio(pred, args.cost, stock_returns)
        returns = pd.DataFrame({"ML strategy": strategy, "EW daily-rebalanced stocks": ew})
        for name, df in [("Nifty BeES adjusted ETF proxy", etf), ("Nifty 50 price index (not TR)", index_prices)]:
            r = df.Close.pct_change(fill_method=None).shift(-1).reindex(returns.index)
            # A missing benchmark return is a failure, never a fabricated zero.
            if r.isna().any():
                raise ValueError(f"Missing benchmark returns: {name}")
            r.iloc[0] -= args.cost
            returns[name] = r
        m = pd.DataFrame({c: metrics(returns[c]) for c in returns})
        acc = accuracy_test(pred)
        m.loc["Directional accuracy %", "ML strategy"] = acc["accuracy"] * 100
        m.loc["Time in market %", "ML strategy"] = pos.to_numpy().mean() * 100
        m.loc["Avg daily signal turnover", "ML strategy"] = turn.to_numpy().mean()
        m.to_csv(folder / "metrics.csv", float_format="%.10g")
        (folder / "metrics.md").write_text(m.round(3).to_markdown() + "\n")
        returns.to_csv(folder / "daily_returns.csv", float_format="%.17g")
        pred[["ticker", "fwd_ret", "label_end", "y", "p", "fold_start"]].to_csv(folder / "predictions.csv", float_format="%.17g")
        stats = bootstrap(returns, args.bootstrap, args.block)
        stats["accuracy"] = acc
        stats["excluded_noncommon_dates"] = excluded_dates
        stats["folds"] = folds
        stats["period"] = {"first_signal": str(returns.index.min().date()), "last_signal": str(returns.index.max().date()),
            "last_label": str(pred.label_end.max().date()), "days": len(returns), "missing_feature_cash_slots": int(len(returns) * len(args.tickers) - len(pred))}
        # Null: shuffle only each training window's labels; genuine OOS features/returns stay fixed.
        null_path = folder / "null_sharpes.csv"
        null = pd.read_csv(null_path, float_precision="round_trip") if null_path.exists() and not args.refresh else pd.DataFrame(columns=["shuffle", "sharpe_excess", "sharpe_zero_rf"])
        for i in range(len(null), args.shuffles):
            shuffled, _ = predict(data, args.split, args.end, mode, np.random.default_rng(10000 + i))
            r, _, _, _ = portfolio(shuffled, args.cost, stock_returns)
            null.loc[len(null)] = [i, sharpe(r), sharpe(r, 0)]
            if i % 10 == 0:
                print(args.tag, mode, "shuffle", i, flush=True)
                null.to_csv(null_path, index=False, float_format="%.17g")
        null.to_csv(null_path, index=False, float_format="%.17g")
        if len(null):
            observed = sharpe(strategy)
            stats["shuffled_label_null"] = {"n": len(null), "observed_excess_sharpe": observed,
                "observed_zero_rf_sharpe": sharpe(strategy, 0),
                "null_excess_95_range": null.sharpe_excess.quantile([0.025, 0.975]).tolist(),
                "upper_tail_plus_one_p": (1 + int((null.sharpe_excess >= observed).sum())) / (len(null) + 1),
                "observed_percentile": float((null.sharpe_excess < observed).mean() * 100),
                "published_minus_0_05_percentile_in_zero_rf_null": float((null.sharpe_zero_rf < -0.05).mean() * 100)}
            plt.figure(figsize=(8, 4)); plt.hist(null.sharpe_excess, bins=25, alpha=0.75)
            plt.axvline(observed, color="red", label=f"Observed {observed:.3f}"); plt.legend()
            plt.xlabel("Annualized Sharpe above 6.5% RF"); plt.ylabel("Shuffled-label fits")
            plt.title(f"{args.tag}: {mode}, {len(null)} training-label shuffles"); plt.tight_layout()
            plt.savefig(folder / "null_distribution.png", dpi=130)
            plt.savefig(folder / "null_distribution.svg")
            compact_svg(folder / "null_distribution.svg")
            plt.close()
        (folder / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
        figures(returns, folder)
        print(args.tag, mode, "\n", m.round(3), "\n", json.dumps(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="+", default=MAIN)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--split", default="2023-01-01")
    p.add_argument("--end", default="2026-10-15", help="Exclusive fixed snapshot cutoff")
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--tag", default="main")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--shuffles", type=int, default=200)
    p.add_argument("--refresh", action="store_true")
    run(p.parse_args())
