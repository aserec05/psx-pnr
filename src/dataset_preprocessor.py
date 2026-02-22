"""
dataset_preprocessor.py
------------------------
Dataset-specific preprocessing functions for each data source.
Uses text_cleaner.py for generic cleaning steps.
"""

import pandas as pd
from text_cleaner import clean_text, filter_short_texts


# ─────────────────────────────────────────
# CNH-PSX (Mendeley)
# ─────────────────────────────────────────

def preprocess_cnhpsx(filepath: str, min_words: int = 3) -> pd.DataFrame:
    """
    Load and preprocess CNH-PSX Mendeley dataset (V1 or V2).
    
    Steps:
        - Load CSV
        - Rename columns to standard names
        - Drop duplicates on Headlines
        - Drop nulls
        - Clean headlines (remove brackets, lowercase, punctuation, stopwords, lemmatize)
        - Filter short headlines
        - Parse dates
    
    Args:
        filepath: Path to CNH-PSX CSV file
        min_words: Minimum number of words to keep a headline
    
    Returns:
        Cleaned DataFrame with columns: [date, headline, headline_clean, category]
    """
    df = pd.read_csv(filepath)

    # Rename to standard names
    df = df.rename(columns={
        'Publishing Date': 'date',
        'Headlines': 'headline',
        'Category': 'category'
    })

    # Keep only needed columns
    cols = ['date', 'headline', 'category']
    if 'Structural Hierarchy Description' in df.columns:
        df = df.rename(columns={'Structural Hierarchy Description': 'hierarchy'})
        cols.append('hierarchy')
    df = df[cols]

    # Drop nulls and duplicates
    df = df.dropna(subset=['headline'])
    df = df.drop_duplicates(subset=['headline'])

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Clean headlines
    df['headline_clean'] = df['headline'].apply(clean_text)

    # Filter too-short headlines
    df = df[df['headline_clean'].apply(lambda x: filter_short_texts(x, min_words))]

    df = df.reset_index(drop=True)
    print(f"[CNH-PSX] Loaded {len(df)} clean headlines from {filepath}")
    return df


# ─────────────────────────────────────────
# Pakistan News Headlines (Kaggle)
# ─────────────────────────────────────────

def preprocess_pakistan_news(filepath: str, min_words: int = 5) -> pd.DataFrame:
    """
    Load and preprocess Pakistan News Headlines dataset.
    Used primarily as Word2Vec training corpus.
    
    Steps:
        - Load CSV
        - Drop nulls on Story Heading and Story Excerpt
        - Drop duplicates on Story Excerpt
        - Combine Heading + Excerpt for richer text
        - Clean combined text
        - Filter short texts
    
    Args:
        filepath: Path to pakistan_news.csv
        min_words: Minimum number of words to keep a text
    
    Returns:
        Cleaned DataFrame with columns: [heading, excerpt, text_combined, text_clean]
    """
    df = pd.read_csv(filepath)

    # Drop nulls on key columns
    df = df.dropna(subset=['Story Heading', 'Story Excerpt'])

    # Drop duplicates on excerpt (main text)
    df = df.drop_duplicates(subset=['Story Excerpt'])

    # Rename
    df = df.rename(columns={
        'Story Heading': 'heading',
        'Story Excerpt': 'excerpt',
        'Section': 'section'
    })

    # Combine heading + excerpt for richer Word2Vec training
    df['text_combined'] = df['heading'].astype(str) + ' ' + df['excerpt'].astype(str)

    # Clean combined text (no stopword removal for Word2Vec — keeps more context)
    df['text_clean'] = df['text_combined'].apply(
        lambda x: clean_text(x, remove_stops=False, do_lemmatize=False)
    )

    # Filter short texts
    df = df[df['text_clean'].apply(lambda x: filter_short_texts(x, min_words))]

    df = df[['heading', 'excerpt', 'section', 'text_combined', 'text_clean']]
    df = df.reset_index(drop=True)
    print(f"[Pakistan News] Loaded {len(df)} clean articles from {filepath}")
    return df


# ─────────────────────────────────────────
# PSX Stock Market Data (Kaggle)
# ─────────────────────────────────────────

def preprocess_psx_stocks(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess PSX Stock Market Data 2017-2025.
    Used for optional recency weighting in the recommendation pipeline.
    
    Steps:
        - Load CSV
        - Parse dates
        - Fill NaN in CHANGE (%) with 0
        - Filter out rows with VOLUME = 0 (no trading activity)
        - Standardize column names
    
    Args:
        filepath: Path to psx_stocks.csv
    
    Returns:
        Cleaned DataFrame with columns: [date, symbol, open, high, low, close, change, change_pct, volume]
    """
    df = pd.read_csv(filepath)

    # Parse dates
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])

    # Fill NaN in CHANGE (%)
    df['CHANGE (%)'] = df['CHANGE (%)'].fillna(0)

    # Filter inactive rows (no volume, no price movement)
    df = df[df['VOLUME'] > 0]

    # Rename to standard names
    df = df.rename(columns={
        'DATE': 'date',
        'SYMBOL': 'symbol',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'CHANGE': 'change',
        'CHANGE (%)': 'change_pct',
        'VOLUME': 'volume',
        'LDCP': 'ldcp'
    })

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f"[PSX Stocks] Loaded {len(df)} trading rows from {filepath}")
    print(f"  Tickers: {df['symbol'].nunique()} | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def get_recent_movers(df_stocks: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Get the top N tickers with the biggest absolute price change on the latest date.
    Useful for recency weighting in the recommendation pipeline.
    
    Args:
        df_stocks: Preprocessed PSX stocks DataFrame
        top_n: Number of top movers to return
    
    Returns:
        DataFrame with top movers sorted by absolute change_pct
    """
    latest_date = df_stocks['date'].max()
    latest = df_stocks[df_stocks['date'] == latest_date].copy()
    latest['abs_change'] = latest['change_pct'].abs()
    top_movers = latest.nlargest(top_n, 'abs_change')[['symbol', 'close', 'change_pct', 'volume']]
    return top_movers.reset_index(drop=True)
