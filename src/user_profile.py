"""
user_profile.py
---------------
Core definitions for synthetic user profiling.

Contains:
    - SECTOR_KEYWORDS : interest dictionary (10 keywords per sector)
    - SyntheticUser   : user class with sector, sub_focus, history, vector
    - SYNTHETIC_USERS : pre-defined list of 4 synthetic PSX investors
"""

import numpy as np
from typing import List, Optional


# ─────────────────────────────────────────
# INTEREST DICTIONARY — 10 keywords per sector
# Word boundary matching (\b) used in article_tagger.py
# to avoid false positives (e.g. "pol" matching "police")
# ─────────────────────────────────────────

SECTOR_KEYWORDS = {
    "Construction": [
        # Tickers
        "luck", "dgkc", "fccl", "maple leaf cement",
        # Sector terms — added broader terms for better coverage
        "cement", "concrete", "construction", "infrastructure",
        "housing scheme", "psdp", "capacity expansion"
    ],
    "Banking": [
        # Tickers — use full names to avoid partial matches
        "mebl", "habib bank", "hbl", "united bank", "ubl",
        "national bank", "nbp", "state bank", "sbp",
        # News terms
        "policy rate", "interest rate", "net interest margin",
        "deposits", "dividend payout"
    ],
    "Energy": [
        # Tickers — full names added alongside abbreviations
        "ogdc", "oil and gas development",
        "ppl", "pakistan petroleum",
        "pso", "pakistan state oil",
        # "pol" removed — too short, matches "police', 'political' etc.
        # News terms
        "crude oil", "exploration", "refinery",
        "gas production", "fuel prices", "circular debt"
    ]
}


# ─────────────────────────────────────────
# SYNTHETIC USER CLASS
# ─────────────────────────────────────────

class SyntheticUser:
    """
    Represents a synthetic PSX investor with a sector interest,
    a simulated reading history, and a profile vector.

    Attributes:
        user_id         : Unique identifier (e.g. 'User_1')
        sector          : Primary sector ('Construction', 'Banking', 'Energy')
        sub_focus       : Optional sub-focus (e.g. 'cement-heavy')
        clicked_articles: List of article indices from reading history
        profile_vector  : Averaged embedding vector of clicked articles
    """

    def __init__(self, user_id: str, sector: str, sub_focus: Optional[str] = None):
        self.user_id          = user_id
        self.sector           = sector
        self.sub_focus        = sub_focus
        self.clicked_articles: List[int] = []
        self.profile_vector: Optional[np.ndarray] = None

    def __repr__(self):
        return (
            f"SyntheticUser(id={self.user_id}, sector={self.sector}, "
            f"sub_focus={self.sub_focus}, clicks={len(self.clicked_articles)})"
        )


# ─────────────────────────────────────────
# PRE-DEFINED SYNTHETIC USERS
# ─────────────────────────────────────────

SYNTHETIC_USERS = [
    SyntheticUser("User_1", "Construction", sub_focus="cement-heavy"),
    SyntheticUser("User_2", "Construction", sub_focus="infrastructure-heavy"),
    SyntheticUser("User_3", "Banking"),
    SyntheticUser("User_4", "Energy"),
]