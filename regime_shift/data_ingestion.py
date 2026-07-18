"""
Data ingestion for Regime-Shift.

Pulls three families of data described in the project brief:
  1. Multi-asset daily returns (Yahoo Finance)
  2. CBOE VIX (Yahoo Finance ticker ^VIX)
  3. Macro proxies: CPI, 10yr yield, high-yield credit spread (FRED)
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
except ImportError:  # fredapi is optional if the user only wants price/VIX data
    Fred = None


DEFAULT_TICKERS = ["SPY", "TLT", "GLD"]  # equities, bonds, gold — the allocatable universe

FRED_SERIES = {
    "cpi": "CPIAUCSL",          # CPI, monthly -> inflation regime proxy
    "ten_year_yield": "DGS10",  # 10yr treasury yield, daily
    "hy_spread": "BAMLH0A0HYM2",  # high-yield credit spread, daily -> crisis proxy
}


def fetch_asset_prices(tickers: list[str] = DEFAULT_TICKERS,
                        start: str = "2005-01-01",
                        end: str | None = None) -> pd.DataFrame:
    """Daily adjusted close prices for the allocatable asset universe."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")
    return prices


def fetch_vix(start: str = "2005-01-01", end: str | None = None) -> pd.Series:
    """CBOE VIX close, used as a fast-moving fear-gauge feature for the HMM."""
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix = raw["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.name = "vix"
    return vix.dropna()


def fetch_macro(fred_api_key: str,
                 start: str = "2005-01-01",
                 end: str | None = None) -> pd.DataFrame:
    """CPI, 10yr yield, and HY credit spread from FRED, resampled to daily."""
    if Fred is None:
        raise ImportError("fredapi is not installed — pip install fredapi")

    fred = Fred(api_key=fred_api_key)
    series = {}
    for name, code in FRED_SERIES.items():
        s = fred.get_series(code, observation_start=start, observation_end=end)
        s.name = name
        series[name] = s

    macro = pd.concat(series.values(), axis=1)
    # Forward-fill so a monthly print (e.g. CPI) stays "known" until the next
    # print, but is never back-filled into days before it was released.
    macro = macro.asfreq("D").ffill()
    return macro


def build_feature_frame(prices: pd.DataFrame,
                         vix: pd.Series,
                         macro: pd.DataFrame | None = None,
                         primary_asset: str = "SPY",
                         vol_window: int = 20) -> pd.DataFrame:
    """
    Assemble the daily feature matrix used to fit the HMM:
    return, rolling realized volatility, VIX level, and (optionally) macro.

    Only uses information available up to and including each row's date —
    no centered rolling windows, no shifting data backward.
    """
    returns = prices.pct_change()

    frame = pd.DataFrame({
        "return": returns[primary_asset],
        "vol_{}d".format(vol_window): returns[primary_asset].rolling(vol_window).std(),
    })
    frame = frame.join(vix, how="left")

    if macro is not None:
        frame = frame.join(macro, how="left")

    frame = frame.dropna()
    return frame


def build_asset_return_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns for every asset in the allocatable universe."""
    return prices.pct_change().dropna(how="all")
