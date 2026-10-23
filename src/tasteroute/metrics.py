"""Evaluation metrics.

Rating error: RMSE, MAE. Ranking metrics are added with the re-ranking stage.
"""
import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error: sqrt(mean((r - r_hat)^2))."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error: mean(|r - r_hat|)."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(d)))


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
