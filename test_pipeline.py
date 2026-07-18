import numpy as np
import pandas as pd
from regime_shift import regime_detection, optimizer, backtest, metrics

np.random.seed(0)
n_days = 1500
dates = pd.bdate_range("2018-01-01", periods=n_days)

# Simulate a regime-switching return series: mostly calm, with two crisis bursts
vol = np.full(n_days, 0.008)
mean = np.full(n_days, 0.0006)
vol[400:460] = 0.035
mean[400:460] = -0.004
vol[1000:1050] = 0.03
mean[1000:1050] = -0.003

spy_ret = np.random.normal(mean, vol)
tlt_ret = np.random.normal(0.0002, 0.005) + np.random.normal(0, 0.003, n_days)
gld_ret = np.random.normal(0.0002, 0.006, n_days)
asset_returns = pd.DataFrame({"SPY": spy_ret, "TLT": tlt_ret, "GLD": gld_ret}, index=dates)
vix_proxy = pd.Series(50 * pd.Series(vol, index=dates).rolling(5).std().fillna(0.01) + 12, name="vix")

features = pd.DataFrame({
    "return": spy_ret,
    "vol_20d": pd.Series(spy_ret, index=dates).rolling(20).std(),
}, index=dates).join(vix_proxy).dropna()

print("== Testing regime_detection ==")
model, regimes = regime_detection.fit_and_decode(features)
print(regimes.value_counts())

print("\n== Testing optimizer ==")
mu, sigma = optimizer.estimate_moments(asset_returns)
w = optimizer.solve_portfolio(mu, sigma, "Crisis")
print("Crisis weights:", dict(zip(asset_returns.columns, w.round(3))))
w = optimizer.solve_portfolio(mu, sigma, "Bull")
print("Bull weights:", dict(zip(asset_returns.columns, w.round(3))))

print("\n== Testing walk_forward_backtest ==")
result = backtest.walk_forward_backtest(features, asset_returns, rebalance_freq="ME",
                                         train_window=252, min_train_size=100)
print(result[["regime", "turnover", "cost", "net_return"]].head(8))
print(f"...{len(result)} total rebalances")

print("\n== Testing metrics / tear sheet ==")
bench = backtest.static_benchmark_returns(asset_returns, {"SPY": 0.6, "TLT": 0.4}, rebalance_freq="ME")
sheet = metrics.tear_sheet(result["net_return"], {"60/40 Static": bench}, turnover=result["turnover"], freq="M")
print(sheet)

print()
print("ALL PIPELINE STAGES RAN SUCCESSFULLY")
