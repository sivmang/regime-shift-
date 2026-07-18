"""
Walk-forward backtest

Strict temporal test splits: on each rebalance date, the HMM is re-fit ONLY on data strictly before that date,
and the CVXPY optimizer uses only the moments estimated from that same training window.

Transaction friction is applied explicitly: every rebalance that changes
weights pays a cost proportional to total turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regime_detection import fit_and_decode
from .optimizer import estimate_moments, solve_portfolio


def walk_forward_backtest(features: pd.DataFrame,
                           asset_returns: pd.DataFrame,
                           rebalance_freq: str = "M", #monthly
                           train_window: int = 756, #days
                           min_train_size: int = 252, 
                           friction_bps: float = 7.5,
                           n_states: int = 3, #hmm states
                           return_col: str = "return") -> pd.DataFrame:
    """
    Run the full detect -> allocate -> hold -> cost loop.
    Returns
    DataFrame indexed by date with columns: regime, gross_return, cost,
    net_return, turnover, plus one weight_<asset> column per asset.
    """

    common_index = features.index.intersection(asset_returns.index)
    features = features.loc[common_index].sort_index() #HMM input features (return, vol, VIX)
    asset_returns = asset_returns.loc[common_index].sort_index() #daily returns of the allocatable assets

    rebalance_dates = features.resample(rebalance_freq).last().index
    rebalance_dates = [d for d in rebalance_dates if d in features.index]

    assets = asset_returns.columns.tolist()
    prev_weights = np.zeros(len(assets))

    records = []
    friction_rate = friction_bps / 1e4

    for i, reb_date in enumerate(rebalance_dates):
        loc = features.index.get_loc(reb_date)
        if isinstance(loc, slice):
            loc = loc.stop - 1

        train_start = max(0, loc - train_window) if train_window else 0
        if loc - train_start < min_train_size:
            continue  # not enough history yet

        train_features = features.iloc[train_start:loc + 1]
        train_asset_returns = asset_returns.iloc[train_start:loc + 1]

        # 1. Detect regime using data up to and including reb_date.
        try:
            model, regimes = fit_and_decode(train_features, n_states=n_states,
                                             return_col=return_col)
            current_regime = regimes.iloc[-1]
        except Exception:
            current_regime = records[-1]["regime"] if records else "Bear"

        # 2. Estimate mu/Sigma from the same training window, solve for weights
        mu, sigma = estimate_moments(train_asset_returns)
        weights = solve_portfolio(mu, sigma, current_regime)

        # 3. Turnover + transaction cost for moving from old to new weights
        turnover = np.abs(weights - prev_weights).sum()
        cost = turnover * friction_rate

        # 4. Hold these weights until the next rebalance date, accumulate return
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else features.index[-1]
        holding_period = asset_returns.loc[reb_date:next_date].iloc[1:]  # exclude reb_date itself
        gross_return = (holding_period @ weights).sum() if len(holding_period) else 0.0

        record = {
            "date": reb_date,
            "regime": current_regime,
            "turnover": turnover,
            "cost": cost,
            "gross_return": gross_return,
            "net_return": gross_return - cost,
        }
        for asset, w in zip(assets, weights):
            record[f"weight_{asset}"] = w
        records.append(record)

        prev_weights = weights

    result = pd.DataFrame(records).set_index("date")
    return result


def equity_curve(backtest_result: pd.DataFrame, return_col: str = "net_return") -> pd.Series:
    """Cumulative growth of $1 from a sequence of periodic returns."""
    return (1 + backtest_result[return_col]).cumprod()


def static_benchmark_returns(asset_returns: pd.DataFrame,
                              weights: dict[str, float],
                              rebalance_freq: str = "M") -> pd.Series:
    
    
    w = pd.Series(weights).reindex(asset_returns.columns).fillna(0.0).values
    rebalance_dates = asset_returns.resample(rebalance_freq).last().index
    rebalance_dates = [d for d in rebalance_dates if d in asset_returns.index]

    periodic_returns = []
    for i, d in enumerate(rebalance_dates):
        next_d = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else asset_returns.index[-1]
        period = asset_returns.loc[d:next_d].iloc[1:]
        periodic_returns.append((d, (period @ w).sum() if len(period) else 0.0))

    return pd.Series(dict(periodic_returns), name="benchmark_return")

