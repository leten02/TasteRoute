"""Evaluation metrics.

Rating error: RMSE, MAE. Top-N ranking: Precision, Recall, HitRate, NDCG at K.
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


# --- Top-N ranking (one user at a time) -------------------------------------
# ranked: item ids in recommended order; relevant: set of the user's relevant
# test items (test rating >= 4). Scores are averaged over users by the caller.


def precision_at_k(ranked, relevant, k):
    """Share of the top-k recommendations that are relevant: |L ∩ Rel| / k."""
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked, relevant, k):
    """Share of the relevant items found in the top k: |L ∩ Rel| / |Rel|."""
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(ranked, relevant, k):
    """1 if at least one relevant item is in the top k, else 0."""
    return float(bool(set(ranked[:k]) & relevant))


def ndcg_at_k(ranked, relevant, k):
    """Normalized discounted cumulative gain with binary relevance.

    DCG = sum_{pos=1..k} rel_pos / log2(pos + 1); IDCG is the DCG of a list
    with all relevant items on top (at most k of them).
    """
    dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(ranked[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
