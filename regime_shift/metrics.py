"""
Performance tear sheet: Sharpe, Sortino, max drawdown, Calmar ratio,
and turnover analysis, comparable across the strategy and static benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualize_factor(freq: str = "M") -> int:
    return {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}.get(freq, 12)


def sharpe_ratio(returns: pd.Series, freq: str = "M", risk_free: float = 0.0) -> float:
    af = annualize_factor(freq)
    excess = returns - risk_free / af
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(af)


def sortino_ratio(returns: pd.Series, freq: str = "M", risk_free: float = 0.0) -> float:
    af = annualize_factor(freq)
    excess = returns - risk_free / af
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(af)


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series, freq: str = "M") -> float:
    af = annualize_factor(freq)
    n_periods = len(returns)
    if n_periods == 0:
        return np.nan
    cagr = (1 + returns).prod() ** (af / n_periods) - 1
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return cagr / abs(mdd)


def tear_sheet(strategy_returns: pd.Series,
               benchmark_returns: dict[str, pd.Series],
               turnover: pd.Series | None = None,
               freq: str = "M") -> pd.DataFrame:
    """
    Build a comparison table: strategy vs. one or more static benchmarks,
    on Sharpe, Sortino, max drawdown, Calmar, and (for the strategy) turnover.
    """
    series_map = {"Regime-Shift (strategy)": strategy_returns, **benchmark_returns}

    rows = []
    for name, series in series_map.items():
        series = series.dropna()
        row = {
            "Series": name,
            "Ann. Return": (1 + series).prod() ** (annualize_factor(freq) / len(series)) - 1 if len(series) else np.nan,
            "Sharpe": sharpe_ratio(series, freq=freq),
            "Sortino": sortino_ratio(series, freq=freq),
            "Max Drawdown": max_drawdown(series),
            "Calmar": calmar_ratio(series, freq=freq),
        }
        if name.startswith("Regime-Shift") and turnover is not None:
            row["Avg Turnover"] = turnover.mean()
            row["Total Cost Drag"] = turnover.sum() * 0  # placeholder, filled by caller with actual bps if desired
        rows.append(row)

    return pd.DataFrame(rows).set_index("Series")
