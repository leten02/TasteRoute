"""Evaluation metrics.

Rating error: RMSE, MAE. Ranking metrics are added with the re-ranking stage.
"""
import numpy as np


def _errors(y_true, y_pred):
    """r - r_hat for two equal-length 1-D arrays.

    Raises instead of returning nonsense: shapes (n,) and (n, 1) would broadcast
    to an (n, n) matrix, and an empty input would give NaN.
    """
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if t.ndim != 1 or t.shape != p.shape:
        raise ValueError(f"y_true and y_pred must be 1-D with the same length, got {t.shape} and {p.shape}")
    if t.size == 0:
        raise ValueError("no ratings to evaluate")
    return t - p


def rmse(y_true, y_pred):
    """Root mean squared error: sqrt(mean((r - r_hat)^2))."""
    return float(np.sqrt(np.mean(_errors(y_true, y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error: mean(|r - r_hat|)."""
    return float(np.mean(np.abs(_errors(y_true, y_pred))))


def evaluate(model, df):
    """Score a fitted model on the ratings in df.

    Args:
        model: any recommender with predict(u, i) -> array of predicted stars.
        df: DataFrame with columns u, i, stars.

    Returns:
        dict with rmse, mae and n (number of ratings scored).
    """
    pred = model.predict(df.u.values, df.i.values)
    return {"rmse": rmse(df.stars.values, pred), "mae": mae(df.stars.values, pred), "n": len(df)}
