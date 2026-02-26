"""
history_simulator.py
--------------------
Simulate reading history for synthetic users.

For each user: sample articles from their sector (+ a small noise fraction
from other sectors to mimic realistic browsing behaviour).
"""

import random
import numpy as np
import pandas as pd
from typing import List
from user_profile import SyntheticUser, SECTOR_KEYWORDS

# Sub-focus keyword mapping for Construction users
SUB_FOCUS_KEYWORDS = {
    "cement-heavy":         ["luck", "dgkc", "fccl", "maple leaf cement", "cement sector"],
    "infrastructure-heavy": ["infrastructure", "construction industry",
                             "capacity expansion", "housing scheme", "psdp"],
}


def simulate_reading_history(
    users: List[SyntheticUser],
    df_tagged: pd.DataFrame,
    n_clicks: int = 10,
    noise_ratio: float = 0.1,
    seed: int = 42
) -> List[SyntheticUser]:
    """
    Simulate reading history for each synthetic user.

    For each user:
        - Sample n_clicks * (1 - noise_ratio) articles from their sector
        - Sample n_clicks * noise_ratio articles from other sectors (realistic noise)
        - If user has a sub_focus, prioritise articles matching sub-focus keywords

    Args:
        users      : List of SyntheticUser objects
        df_tagged  : Combined tagged article pool with 'primary_tag' column
        n_clicks   : Total number of articles per user (8-12 recommended)
        noise_ratio: Fraction of clicks from other sectors
        seed       : Random seed for reproducibility

    Returns:
        Updated list of SyntheticUser with populated clicked_articles
    """
    random.seed(seed)
    np.random.seed(seed)

    for user in users:
        sector_pool = df_tagged[df_tagged["primary_tag"] == user.sector].index.tolist()
        other_pool  = df_tagged[df_tagged["primary_tag"] != user.sector].index.tolist()

        # Apply sub_focus weighting
        if user.sub_focus and user.sub_focus in SUB_FOCUS_KEYWORDS:
            focus_kws = SUB_FOCUS_KEYWORDS[user.sub_focus]
            focused = df_tagged[
                (df_tagged["primary_tag"] == user.sector) &
                df_tagged["headline"].str.lower().apply(
                    lambda h: any(kw in str(h) for kw in focus_kws)
                )
            ].index.tolist()
            if len(focused) >= n_clicks // 2:
                # Move focused articles to the front of the pool
                sector_pool = focused + [i for i in sector_pool if i not in focused]

        n_sector = max(1, int(n_clicks * (1 - noise_ratio)))
        n_noise  = n_clicks - n_sector

        sector_clicks = random.sample(sector_pool, min(n_sector, len(sector_pool)))
        noise_clicks  = random.sample(other_pool,  min(n_noise,  len(other_pool)))

        user.clicked_articles = sector_clicks + noise_clicks
        random.shuffle(user.clicked_articles)

        print(f"[History] {user.user_id} ({user.sector} / {user.sub_focus}): "
              f"{len(sector_clicks)} sector + {len(noise_clicks)} noise clicks")

    return users


def get_reading_log(
    users: List[SyntheticUser],
    df_tagged: pd.DataFrame
) -> pd.DataFrame:
    """
    Export the simulated reading history as a flat DataFrame.

    Returns:
        DataFrame with columns [user_id, sector, article_id, headline, primary_tag]
    """
    rows = []
    for user in users:
        for article_id in user.clicked_articles:
            if article_id in df_tagged.index:
                row = df_tagged.loc[article_id]
                rows.append({
                    "user_id":     user.user_id,
                    "sector":      user.sector,
                    "article_id":  article_id,
                    "headline":    row.get("headline", ""),
                    "primary_tag": row["primary_tag"]
                })
    return pd.DataFrame(rows)
