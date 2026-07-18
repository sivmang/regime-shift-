"""
Regime detection via Hidden Markov Model.

Fits a GaussianHMM on market features (return, volatility, VIX, ...) and
decodes the most likely hidden-state sequence with the Viterbi algorithm
(hmmlearn's .predict()). States are unlabeled integers by construction —
this module relabels them into human-readable regimes (Bull/Bear/Crisis)
by ranking each state's mean return, so labels stay consistent across
repeated refits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_ORDER = ["Crisis", "Bear", "Bull"]  # worst mean return -> best mean return


def fit_regime_model(features: pd.DataFrame,
                      n_states: int = 3,
                      return_col: str = "return",
                      n_iter: int = 1000,
                      random_state: int = 42,
                      covariance_type: str = "diag",
                      min_covar: float = 1e-3) -> GaussianHMM:
    """
    Fit a GaussianHMM on the given feature matrix (rows = trading days).

    covariance_type defaults to "diag" rather than "full": with only a few
    hundred rows per rolling training window and 3+ states, a full
    covariance matrix frequently goes singular (especially inside a
    low-variance regime), which crashes Baum-Welch with a Cholesky error.
    Diagonal covariance is the standard, numerically stable choice for
    regime-detection HMMs on daily financial data; min_covar adds a small
    floor to each variance as a second safety net.
    """
    X = features.values
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        min_covar=min_covar,
    )
    model.fit(X)
    return model


def label_map_by_mean_return(model: GaussianHMM,
                              features: pd.DataFrame,
                              return_col: str = "return") -> dict[int, str]:
    """
    hmmlearn assigns arbitrary integer IDs to states, and the ID<->meaning
    mapping can flip between refits. This ranks states by their fitted mean
    return (model.means_) and maps the worst -> "Crisis", best -> "Bull",
    so downstream code can always refer to regimes by name.
    """
    return_idx = list(features.columns).index(return_col)
    mean_returns = model.means_[:, return_idx]
    ranked_state_ids = np.argsort(mean_returns)  # ascending: worst return first

    n_states = model.n_components
    if n_states == 3:
        names = REGIME_ORDER
    else:
        # Generic fallback for n_states != 3: Regime_0 (worst) ... Regime_k (best)
        names = [f"Regime_{i}" for i in range(n_states)]

    return {state_id: name for state_id, name in zip(ranked_state_ids, names)}


def decode_regimes(model: GaussianHMM,
                    features: pd.DataFrame,
                    return_col: str = "return") -> pd.Series:
    """
    Run Viterbi decoding and return a Series of regime labels (str),
    indexed the same as `features`.
    """
    state_ids = model.predict(features.values)
    label_map = label_map_by_mean_return(model, features, return_col)
    labels = pd.Series([label_map[s] for s in state_ids],
                        index=features.index, name="regime")
    return labels


def fit_and_decode(features: pd.DataFrame,
                    n_states: int = 3,
                    return_col: str = "return",
                    **fit_kwargs) -> tuple[GaussianHMM, pd.Series]:
    """Convenience wrapper: fit the HMM and decode regimes in one call."""
    model = fit_regime_model(features, n_states=n_states, return_col=return_col, **fit_kwargs)
    regimes = decode_regimes(model, features, return_col=return_col)
    return model, regimes


def transition_matrix_report(model: GaussianHMM,
                              label_map: dict[int, str]) -> pd.DataFrame:
    """Human-readable transition probability matrix, rows/cols named by regime."""
    n = model.n_components
    names = [label_map[i] for i in range(n)]
    df = pd.DataFrame(model.transmat_, index=names, columns=names)
    return df
