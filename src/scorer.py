"""
scorer.py
---------
Low-level scoring functions used by the recommender.

Three things happen here:
    1. Compute how close a user profile is to each article (cosine similarity)
    2. Give a small bonus to articles mentioning stocks that moved recently
    3. Add the two together into a final score
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def cosine_scores(
    profile_vector: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    How similar is this user to each article?
    Just cosine similarity between the profile vector and every article embedding.

    Returns one score per article, between -1 and 1.
    Higher = more relevant.
    """
    return cosine_similarity([profile_vector], embeddings)[0]


def recency_boost(
    df_articles: pd.DataFrame,
    df_stocks: pd.DataFrame,
    boost_weight: float = 0.2
) -> np.ndarray:
    """
    Give a small bonus to articles that mention a ticker
    which moved significantly on the latest trading day.

    The idea: if OGDC jumped 8% today and an article mentions OGDC,
    that article is probably more relevant right now than usual.

    The boost is normalised to [0, boost_weight] so it never
    overwhelms the base similarity score.
    """
    # Latest date in the stocks dataset
    latest_date = df_stocks['date'].max()
    latest      = df_stocks[df_stocks['date'] == latest_date].copy()

    # Build a ticker → abs(price change) dict for that day
    latest['abs_change'] = latest['change_pct'].abs()
    ticker_change = dict(zip(latest['symbol'].str.lower(), latest['abs_change']))

    def article_boost(headline: str) -> float:
        if not isinstance(headline, str):
            return 0.0
        headline_lower = headline.lower()
        # Take the biggest mover if multiple tickers are mentioned
        max_change = 0.0
        for ticker, change in ticker_change.items():
            if ticker in headline_lower:
                max_change = max(max_change, change)
        return max_change

    raw_boost = df_articles['headline'].apply(article_boost).values.astype(float)

    # Normalise so the biggest mover gets exactly boost_weight, others get less
    max_val = raw_boost.max()
    if max_val > 0:
        raw_boost = raw_boost / max_val

    return raw_boost * boost_weight


def combine_scores(
    similarity_scores: np.ndarray,
    boost_scores: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Merge similarity and boost into one final score.
    If no boost is passed, just return the similarity scores as-is.
    """
    if boost_scores is None:
        return similarity_scores
    return similarity_scores + boost_scores