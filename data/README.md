# Data snapshots

The script downloads adjusted daily OHLCV via yfinance into `data/cache/`.
Each filename includes ticker, start, and exclusive end date. Reruns reuse the
CSV exactly; `--refresh` replaces it. These provider data are not covered by
the code's MIT license and are not redistributed here. Check Yahoo's terms
before use or redistribution. `snapshot_manifest.json` hashes the local CSVs
used for the checked-in report. A fresh download can differ after revisions;
the manifest detects that rather than promising exact reproduction without
the original cache.

Snapshot: downloaded October 15, 2026; requested end October 15 exclusive.
NIFTYBEES had no usable close on October 14, so the shared-calendar sample
ends with October 12 signal / October 13 realized close. Other noncommon
provider dates are listed in each `statistics.json`. No missing benchmark
return is filled with zero. Prices are intersected across the universe and
both benchmarks before feature/return calculation so every return has the
same holding period. This can span more than one exchange session where
provider data are missing.

A zero-volume RELIANCE bar on March 18, 2025 (also in several other stocks)
invalidates volume features for subsequent windows. Such missing-feature
stock slots stay in cash, not silently reweighted into the remaining stock.
This is a data-quality policy, not a model feature change. Cash earns zero.
