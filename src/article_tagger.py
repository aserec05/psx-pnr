"""
article_tagger.py
-----------------
Tag articles with sector labels using word-boundary keyword matching.

Uses \b word boundaries to avoid false positives like:
    - "pol" matching "police" or "political"
    - "sbp" matching "isbp" or similar
    - "ppl" matching common words

Each article can receive multiple tags if it matches multiple sectors.
"""

import re
import pandas as pd
from typing import List
from user_profile import SECTOR_KEYWORDS


def _make_pattern(keyword: str) -> re.Pattern:
    """
    Compile a case-insensitive regex pattern with word boundaries
    for a given keyword. Multi-word keywords use simple containment.
    """
    # For multi-word keywords, no boundary needed — specific enough
    if " " in keyword:
        return re.compile(re.escape(keyword), re.IGNORECASE)
    # Single-word keywords: wrap in word boundaries
    return re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)


# Pre-compile all patterns at import time for efficiency
COMPILED_PATTERNS = {
    sector: [(kw, _make_pattern(kw)) for kw in keywords]
    for sector, keywords in SECTOR_KEYWORDS.items()
}


def match_sectors(text: str) -> List[str]:
    """
    Return list of sectors whose keywords match the given text.
    Uses word-boundary regex to avoid partial matches.
    """
    if not isinstance(text, str):
        return []
    matched = []
    for sector, patterns in COMPILED_PATTERNS.items():
        if any(pattern.search(text) for _, pattern in patterns):
            matched.append(sector)
    return matched


def tag_articles(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """
    Tag each article with matching sector labels using word-boundary
    keyword matching. An article can receive multiple tags.

    Args:
        df      : DataFrame with articles (CNH-PSX or Pakistan News)
        text_col: Column to run keyword matching on

    Returns:
        DataFrame with new columns:
            - tags        : list of all matched sectors
            - primary_tag : first matched sector, or 'Other'
    """
    df = df.copy()
    df["tags"]        = df[text_col].apply(match_sectors)
    df["primary_tag"] = df["tags"].apply(lambda t: t[0] if t else "Other")

    total  = len(df)
    tagged = (df["primary_tag"] != "Other").sum()
    print(f"[Tagger] {tagged}/{total} articles tagged ({tagged/total*100:.1f}%)")
    for sector in SECTOR_KEYWORDS:
        n = (df["primary_tag"] == sector).sum()
        print(f"  {sector}: {n}")
    print(f"  Other: {(df['primary_tag'] == 'Other').sum()}")

    return df


def combine_tagged_datasets(
    df_cnhpsx_tagged: pd.DataFrame,
    df_news_tagged: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine CNH-PSX and Pakistan News tagged datasets into a single
    article pool for reading history simulation.

    Args:
        df_cnhpsx_tagged: Tagged CNH-PSX DataFrame
        df_news_tagged  : Tagged Pakistan News DataFrame

    Returns:
        Combined DataFrame with columns:
            [headline, primary_tag, tags, source, text_for_embedding]
    """
    cnhpsx = df_cnhpsx_tagged[["headline", "headline_clean", "primary_tag", "tags"]].copy()
    cnhpsx["source"]             = "cnhpsx"
    cnhpsx["text_for_embedding"] = cnhpsx["headline_clean"]

    news = df_news_tagged.copy()
    news["headline"] = news.get("heading", news.get("headline", ""))
    news = news[["headline", "text_clean", "primary_tag", "tags"]].copy()
    news["source"]             = "pakistan_news"
    news["text_for_embedding"] = news["text_clean"]

    pool = pd.concat(
        [cnhpsx[["headline", "primary_tag", "tags", "source", "text_for_embedding"]],
          news[["headline",  "primary_tag", "tags", "source", "text_for_embedding"]]],
        ignore_index=True
    )

    print(f"[Pool] Combined pool: {len(pool)} articles")
    print(pool["primary_tag"].value_counts().to_string())
    return pool