# PSX PNR — Pakistan Stock Exchange Personalized News Recommender

A personalized news recommendation system for PSX investors using NLP embeddings and user profiling.

> Université Claude Bernard Lyon 1 — Adam Muhammad Safi Ullah & Ceresa Thomas

---

## Project Structure

```
psxpnr/
├── notebooks/              # EDA, experiments, model comparisons
│   └── 01_eda_cnhpsx.ipynb # Start here — explore Mendeley dataset
├── src/                    # Clean pipeline modules
│   ├── preprocess.py       # Cleaning, tokenization, lemmatization
│   ├── embeddings.py       # Word2Vec / SBERT embedding generation
│   └── recommender.py      # Cosine similarity + Top-K ranking
├── data/
│   ├── raw/                # Original CSV files (not committed to git)
│   └── processed/          # Cleaned and embedded data
├── app.py                  # Streamlit interface
└── requirements.txt
```

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| CNH-PSX Categorized Financial News | [Mendeley](https://data.mendeley.com/datasets/mc4s7zvx9c/1) | Main news corpus |
| Pakistan News Headlines | [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistan-news-headlines) | Word2Vec training |
| PSX Stock Market Data 2017–2025 | [Kaggle](https://www.kaggle.com/datasets/fayaznoor10/pakistan-stock-market-data-20172025) | Optional recency weighting |

## Getting Started

```bash
pip install -r requirements.txt

# Place CSV files in data/raw/ then run:
jupyter notebook notebooks/01_eda_cnhpsx.ipynb
```

## Pipeline

1. **Data Collection** → Download CSVs, place in `data/raw/`
2. **EDA** → `notebooks/01_eda_cnhpsx.ipynb`
3. **Preprocessing** → `src/preprocess.py`
4. **Embeddings** → `src/embeddings.py` (Word2Vec vs SBERT)
5. **Recommendation** → `src/recommender.py` (cosine similarity + Top-K)
6. **Interface** → `app.py` (Streamlit)