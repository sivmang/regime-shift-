"""
Synthetic regime-switching market data.

Used for unit testing and as a notebook fallback when Yahoo Finance / FRED
are unreachable (e.g. offline environments, CI). NOT a substitute for the
real data sources listed in the project brief — swap in
data_ingestion.fetch_asset_prices / fetch_vix / fetch_macro for the real run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_market(n_days: int = 1500, seed: int = 0
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features, asset_returns):
      features      -> HMM input: return, vol_20d, vix  (SPY-driven)
      asset_returns -> daily returns for SPY, TLT, GLD
    Includes two simulated crisis bursts so the HMM has real regimes to find.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    vol = np.full(n_days, 0.008)
    mean = np.full(n_days, 0.0006)
    vol[400:460] = 0.035
    mean[400:460] = -0.004
    vol[1000:1050] = 0.03
    mean[1000:1050] = -0.003

    spy_ret = rng.normal(mean, vol)
    tlt_ret = rng.normal(0.0002, 0.005, n_days) + rng.normal(0, 0.003, n_days)
    gld_ret = rng.normal(0.0002, 0.006, n_days)

    asset_returns = pd.DataFrame({"SPY": spy_ret, "TLT": tlt_ret, "GLD": gld_ret}, index=dates)

    vix_proxy = pd.Series(
        50 * pd.Series(vol, index=dates).rolling(5).std().fillna(0.01) + 12,
        name="vix", index=dates,
    )

    features = pd.DataFrame({
        "return": spy_ret,
        "vol_20d": pd.Series(spy_ret, index=dates).rolling(20).std(),
    }, index=dates).join(vix_proxy).dropna()

    return features, asset_returns
