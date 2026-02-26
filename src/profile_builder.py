"""
profile_builder.py
------------------
Build user profile vectors from article embeddings.

User_Vector = mean(embeddings of clicked articles)

Each user ends up represented as a single vector in the same
embedding space as the articles — ready for cosine similarity.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from user_profile import SyntheticUser


def build_profile_vectors(
    users: List[SyntheticUser],
    embeddings: np.ndarray,
    df_tagged: pd.DataFrame
) -> List[SyntheticUser]:
    """
    Build a profile vector for each user by averaging the embeddings
    of their clicked articles.

    User_Vector = mean(embeddings[clicked_article_positions])

    Args:
        users     : List of SyntheticUser with populated clicked_articles
        embeddings: Numpy array of shape (n_articles, embedding_dim)
        df_tagged : Tagged pool DataFrame (used to map index to row position)

    Returns:
        Updated list of SyntheticUser with populated profile_vector
    """
    index_to_pos = {idx: pos for pos, idx in enumerate(df_tagged.index)}

    for user in users:
        valid_positions = [
            index_to_pos[idx]
            for idx in user.clicked_articles
            if idx in index_to_pos
        ]

        if not valid_positions:
            print(f"[Builder] {user.user_id}: no valid embeddings — using zero vector")
            user.profile_vector = np.zeros(embeddings.shape[1])
            continue

        clicked_embeddings  = embeddings[valid_positions]
        user.profile_vector = np.mean(clicked_embeddings, axis=0)

        print(f"[Builder] {user.user_id}: vector from {len(valid_positions)} articles, "
              f"dim={user.profile_vector.shape[0]}")

    return users


def save_profiles(users: List[SyntheticUser], path: str):
    """Save all user profile vectors to a .npz file."""
    data = {
        user.user_id: user.profile_vector
        for user in users
        if user.profile_vector is not None
    }
    np.savez(path, **data)
    print(f"[Builder] Saved {len(data)} profiles to {path}")


def load_profiles(path: str) -> Dict[str, np.ndarray]:
    """Load user profile vectors from a .npz file."""
    data = np.load(path)
    profiles = {k: data[k] for k in data.files}
    print(f"[Builder] Loaded {len(profiles)} profiles from {path}")
    return profiles
