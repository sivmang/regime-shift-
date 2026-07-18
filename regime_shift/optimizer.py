"""
Convex portfolio optimization with dynamic, regime-dependent objectives.

Implements the project's "Dynamic Constraint Mapping" goal: the optimizer's
objective function changes depending on which regime the HMM currently
believes we are in.

  Bull    -> Maximize risk-adjusted return (mean-variance utility)
  Bear    -> Balanced: reduce risk, keep some return-seeking
  Crisis  -> Minimize portfolio volatility only

All variants are convex (quadratic objective, linear constraints), which is
required for CVXPY's default solvers (ECOS/OSQP) to guarantee a global
optimum.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
import pandas as pd

REGIME_RISK_AVERSION = {
    "Bull": 1.0,     # lower risk-aversion -> tilts toward the return term
    "Bear": 5.0,      # higher risk-aversion -> more conservative
    "Crisis": None,   # None signals pure minimum-volatility (ignore mu entirely)
}


def estimate_moments(returns: pd.DataFrame, lookback: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample mean vector and covariance matrix from historical returns.
    If lookback is given, only the last `lookback` rows are used — this is
    what the walk-forward harness calls on each rolling training window.
    """
    window = returns.iloc[-lookback:] if lookback else returns
    mu = window.mean().values
    sigma = window.cov().values
    # Tiny ridge to keep Sigma strictly PSD for numerical stability
    sigma = sigma + np.eye(len(mu)) * 1e-8
    return mu, sigma


def solve_portfolio(mu: np.ndarray,
                     sigma: np.ndarray,
                     regime: str,
                     long_only: bool = True,
                     max_weight: float | None = None) -> np.ndarray:
    """
    Solve for portfolio weights under the objective mapped to `regime`.

    Bull/Bear: Maximize mu @ w - risk_aversion * w' Sigma w
    Crisis:    Minimize w' Sigma w   (expected return ignored entirely)
    """
    n = len(mu)
    w = cp.Variable(n)

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)

    risk_aversion = REGIME_RISK_AVERSION.get(regime, 3.0)

    if regime == "Crisis" or risk_aversion is None:
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    else:
        objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if w.value is None:
        # Fallback: equal weight if the solver fails on a given window
        return np.full(n, 1.0 / n)

    weights = np.clip(w.value, 0, None)
    weights = weights / weights.sum()  # renormalize for numerical safety
    return weights
