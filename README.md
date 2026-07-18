# regime-shift-
HMM-based market regime detection, dynamic portfolio optimization; using yfinance, FRED API and CBOE VIX data sources

DELIVERABLES REQUESTED
- Trained HMM with regime labels + transition probabilities (jupyter notebook section 2)
- Walk-forward validation, annotated regime charts, equity curve visualisations (backtest)
- Performance tear sheet (backtest and metrics)
- Jupyter notebook: ingestion → detection → optimization → backtest → reporting
- This README


1. REQUIREMENTS
hmmlearn>=0.3.3
cvxpy>=1.9.0
numpy>=2.0
pandas>=2.0
scipy>=1.14
matplotlib>=3.8
yfinance>=1.5
fredapi>=0.5
nbformat>=5.10

2. Architecture
data_ingestion.py- pull prices (yfinance), VIX, macro (FRED); align to daily index
regime_detection.py- fit GaussianHMM, Viterbi-decode regimes, relabel by mean return
optimizer.py- CVXPY: objective changes by regime (Max Sharpe / Min Vol)
backtest.py- walk-forward loop: refit -> allocate -> hold -> apply friction
metrics.py- Sharpe, Sortino, max drawdown, Calmar, tear sheet
synthetic_data.py- regime-switching synthetic series for tests / offline demo

3. Setup
```bash
pip install -r requirements.txt
```
FRED API key (free, only needed for macro data): https://fred.stlouisfed.org/docs/api/api_key.html

4. Running

```python
from regime_shift import data_ingestion, backtest

prices = data_ingestion.fetch_asset_prices(start="2005-01-01")
vix = data_ingestion.fetch_vix(start="2005-01-01")
macro = data_ingestion.fetch_macro(fred_api_key="...", start="2005-01-01")
features = data_ingestion.build_feature_frame(prices, vix, macro)
asset_returns = data_ingestion.build_asset_return_frame(prices)

result = backtest.walk_forward_backtest(features, asset_returns,
                                         rebalance_freq="ME", train_window=252,
                                         friction_bps=7.5)
```

Or just open `notebook/regime_shift_pipeline.ipynb` — runs on synthetic data by default (`USE_LIVE_DATA = False`); flip to `True` + add a FRED key for the real 2005–present backtest.

5. Testing
```bash
python3 test_pipeline.py
```
Runs the full pipeline on synthetic data — no network or FRED key needed.

6. Reproducibility
- `random_state=42` fixed in `fit_regime_model`
- Windows sliced by row index, not calendar dates
- Sensitivity knobs: `friction_bps`, `train_window`, `rebalance_freq`
