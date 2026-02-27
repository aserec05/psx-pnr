"""
recommender.py
--------------
The actual recommender — takes a user profile vector and returns
the Top-K most relevant articles from the pool.

Works with both Word2Vec and SBERT embeddings since the logic
is the same regardless of the embedding model used.

Four configs are tested in notebook 08:
    - W2V,   no boost
    - W2V,   recency boost
    - SBERT, no boost
    - SBERT, recency boost
"""

import numpy as np
import pandas as pd
from typing import Optional

from user_profile import SyntheticUser
from scorer import cosine_scores, recency_boost, combine_scores


class Recommender:
    """
    Give it embeddings, an article pool, and optionally stock data,
    and it will rank articles for any user.

    Args:
        embeddings : Pre-computed article embeddings (n_articles, dim)
        df_articles: The article pool — needs a 'headline' column
        df_stocks  : PSX stocks data for the recency boost (optional)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        df_articles: pd.DataFrame,
        df_stocks: Optional[pd.DataFrame] = None
    ):
        self.embeddings  = embeddings
        self.df_articles = df_articles.reset_index(drop=True)
        self.df_stocks   = df_stocks

    def recommend(
        self,
        user: SyntheticUser,
        top_k: int = 10,
        use_recency_boost: bool = False,
        boost_weight: float = 0.2,
        exclude_read: bool = True
    ) -> pd.DataFrame:
        """
        Return the Top-K articles for this user.

        We compute cosine similarity between the user profile and all articles,
        optionally add a recency boost for articles mentioning moving stocks,
        and return the highest-scoring ones.

        Articles the user already read are excluded by default — no point
        recommending something they've seen.

        Returns a DataFrame with rank, headline, tag, similarity, boost, final score.
        """
        if user.profile_vector is None:
            raise ValueError(f"{user.user_id} has no profile vector yet")

        scores = cosine_scores(user.profile_vector, self.embeddings)

        boost = np.zeros(len(scores))
        if use_recency_boost and self.df_stocks is not None:
            boost = recency_boost(self.df_articles, self.df_stocks, boost_weight)

        final = combine_scores(scores, boost if use_recency_boost else None)

        results = self.df_articles.copy()
        results['article_id']  = results.index
        results['similarity']  = scores
        results['boost']       = boost
        results['final_score'] = final

        # Remove already-read articles
        if exclude_read and user.clicked_articles:
            results = results[~results['article_id'].isin(user.clicked_articles)]

        results = results.sort_values('final_score', ascending=False).head(top_k)
        results.insert(0, 'rank', range(1, len(results) + 1))

        return results[['rank', 'article_id', 'headline', 'primary_tag',
                        'similarity', 'boost', 'final_score']].reset_index(drop=True)


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    user: SyntheticUser
) -> dict:
    """
    Quick quality check on a set of recommendations.

    Three things we look at:
        - sector_precision : how many of the Top-K actually match the user's sector
        - diversity        : how many different tags appear in the Top-K
        - mean_similarity  : average cosine score (higher = more confident picks)
    """
    top_k         = len(recommendations)
    sector_hits   = (recommendations['primary_tag'] == user.sector).sum()
    unique_tags   = recommendations['primary_tag'].nunique()
    mean_sim      = recommendations['similarity'].mean()
    mean_final    = recommendations['final_score'].mean()

    return {
        'user_id':          user.user_id,
        'sector':           user.sector,
        'top_k':            top_k,
        'sector_precision': round(sector_hits / top_k, 4),
        'diversity':        unique_tags,
        'mean_similarity':  round(mean_sim, 4),
        'mean_final_score': round(mean_final, 4),
    }
