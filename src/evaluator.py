"""
evaluator.py
------------
Formal evaluation metrics for the PSX PNR recommender.

Two metrics computed at multiple K values:

    Precision@K
        Simple fraction of Top-K recommendations that match
        the user's sector. Easy to interpret and explain.

    NDCG@K (Normalized Discounted Cumulative Gain)
        Like Precision@K but position-aware — a relevant article
        at rank 1 is worth more than the same article at rank 10.
        Standard metric in information retrieval.

Both metrics treat "relevant" as: primary_tag == user.sector
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from user_profile import SyntheticUser


def precision_at_k(
    recommendations: pd.DataFrame,
    user: SyntheticUser,
    k: int
) -> float:
    """
    Fraction of the Top-K recommendations that match the user's sector.

    P@K = (# relevant articles in Top-K) / K

    Args:
        recommendations: Output of Recommender.recommend(), sorted by score
        user           : SyntheticUser (used for sector label)
        k              : Cutoff

    Returns:
        Float between 0 and 1
    """
    top_k = recommendations.head(k)
    hits  = (top_k['primary_tag'] == user.sector).sum()
    return round(hits / k, 4)


def dcg_at_k(recommendations: pd.DataFrame, user: SyntheticUser, k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    Relevant articles at the top contribute more than those at the bottom.

    DCG@K = sum( rel_i / log2(i + 1) ) for i in 1..K
    where rel_i = 1 if article i matches user sector, else 0
    """
    top_k = recommendations.head(k)
    dcg   = 0.0
    for i, (_, row) in enumerate(top_k.iterrows(), start=1):
        rel  = 1.0 if row['primary_tag'] == user.sector else 0.0
        dcg += rel / np.log2(i + 1)
    return dcg


def ndcg_at_k(
    recommendations: pd.DataFrame,
    user: SyntheticUser,
    k: int
) -> float:
    """
    Normalized DCG at K.

    NDCG@K = DCG@K / IDCG@K
    where IDCG is the best possible DCG (all relevant articles at the top).

    Returns float between 0 and 1. 1.0 = perfect ranking.
    """
    actual_dcg = dcg_at_k(recommendations, user, k)

    # Ideal case: as many relevant articles as possible at the top
    n_relevant = (recommendations['primary_tag'] == user.sector).sum()
    n_ideal    = min(n_relevant, k)

    if n_ideal == 0:
        return 0.0

    # Build ideal ranking: n_ideal relevant articles first
    ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_ideal + 1))

    return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0


def evaluate_config(
    user_recs: Dict[str, pd.DataFrame],
    users: List[SyntheticUser],
    config_name: str,
    k_values: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Compute Precision@K and NDCG@K for all users under one config.

    Args:
        user_recs  : {user_id: recommendations DataFrame}
        users      : List of SyntheticUser objects
        config_name: Label for this config (e.g. 'SBERT — no boost')
        k_values   : List of K cutoffs to evaluate

    Returns:
        DataFrame with one row per (user, k) combination
    """
    rows = []
    for user in users:
        recs = user_recs[user.user_id]
        for k in k_values:
            rows.append({
                'config':    config_name,
                'user_id':   user.user_id,
                'sector':    user.sector,
                'k':         k,
                'precision': precision_at_k(recs, user, k),
                'ndcg':      ndcg_at_k(recs, user, k),
            })
    return pd.DataFrame(rows)


def evaluate_all(
    all_recs: Dict[str, Dict[str, pd.DataFrame]],
    users_by_config: Dict[str, List[SyntheticUser]],
    k_values: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Run evaluation across all configs and all users.

    Args:
        all_recs       : {config_name: {user_id: recommendations}}
        users_by_config: {config_name: list of SyntheticUser}
        k_values       : K cutoffs to evaluate

    Returns:
        Full evaluation DataFrame
    """
    dfs = []
    for config_name, user_recs in all_recs.items():
        users = users_by_config[config_name]
        df    = evaluate_config(user_recs, users, config_name, k_values)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summary_table(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Mean Precision@K and NDCG@K per config and K value.
    This is the main comparison table for the report.
    """
    return (
        df_eval
        .groupby(['config', 'k'])[['precision', 'ndcg']]
        .mean()
        .round(4)
    )
