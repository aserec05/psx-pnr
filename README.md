# PSX PNR — Pakistan Stock Exchange Personalized News Recommender

A personalized news recommendation system for PSX investors using NLP embeddings and user profiling.

> Université Claude Bernard Lyon 1 — Adam Muhammad Safi Ullah & Ceresa Thomas

---

## Project Status

This project is currently in active development. Here is where we stand:

| Step | Status |
|---|---|
| Data Collection | ✅ Done |
| EDA (all 3 datasets) | ✅ Done |
| Preprocessing | ✅ Done |
| Embeddings (Word2Vec + SBERT) | ✅ Done |
| Embedding Comparison | ✅ Done |
| Recommender Engine | 🔄 In progress |
| User Profiling | 🔄 In progress |
| Streamlit Interface | ⏳ Planned |
| Evaluation (Precision@K, NDCG) | ⏳ Planned |

---

## Project Structure

```
psx-pnr/
├── notebooks/                        # Start here — run these in order
│   ├── 01_eda_cnhpsx.ipynb           # EDA on CNH-PSX Mendeley dataset
│   ├── 02_eda_pakistan_news.ipynb    # EDA on Pakistan News Headlines
│   ├── 03_eda_psx_stocks.ipynb       # EDA on PSX Stock Market Data
│   ├── 04_preprocessing.ipynb        # Preprocessing pipeline for all datasets
│   └── 05_embeddings.ipynb           # Word2Vec + SBERT training and comparison
├── src/                              # Clean reusable modules (used by notebooks)
│   ├── text_cleaner.py               # Generic text cleaning functions
│   ├── dataset_preprocessor.py       # Dataset-specific preprocessing
│   ├── embeddings.py                 # Word2Vec and SBERT embedding generation
│   └── recommender.py                # (in progress) Top-K recommendation engine
├── data/
│   ├── raw/                          # Original CSV files (not committed to git)
│   └── processed/                    # Cleaned CSVs and saved embeddings (.npy, .model)
├── doc/                              # PDF exports of all notebooks + slides
├── app.py                            # (planned) Streamlit interface
└── requirements.txt
```

---

## Datasets

| Dataset | Source | Usage | Size after cleaning |
|---|---|---|---|
| CNH-PSX Categorized Financial News | [Mendeley](https://data.mendeley.com/datasets/mc4s7zvx9c/1) | Main news corpus for recommendation | 8 858 headlines |
| Pakistan News Headlines | [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistan-news-headlines) | Word2Vec training corpus | 25 912 articles |
| PSX Stock Market Data 2017–2025 | [Kaggle](https://www.kaggle.com/datasets/fayaznoor10/pakistan-stock-market-data-20172025) | Optional recency weighting | 813 588 rows, 891 tickers |

---

## Getting Started

### 1. Clone the repo and go into the project folder

```bash
cd psx-pnr
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / WSL
venv\Scripts\activate           # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The main libraries used are:

- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — cosine similarity, evaluation metrics
- **nltk** — stopword removal, lemmatization
- **gensim** — Word2Vec training
- **sentence-transformers** — SBERT pretrained models
- **streamlit** — web interface (planned)
- **jupyter** — notebooks
- **matplotlib / seaborn** — visualizations
- **tqdm** — progress bars

### 4. Download the datasets (if necessary)

Download the raw CSV files from the links above and place them in `data/raw/`:

```
data/raw/
├── CNH-PSX_Ver1.csv
├── CNH-PSX_Ver2.csv
├── pakistan_news.csv
└── psx_stocks.csv
```

### 5. Run the notebooks in order

```bash
jupyter notebook --no-browser
```

> On WSL, copy the `http://127.0.0.1:8888/?token=...` link into your Windows browser.

We recommend running the notebooks rather than the `src/` scripts directly — they include visualizations, outputs, and step-by-step explanations. The `src/` modules are the clean reusable code called by the notebooks.

---

## Key Findings So Far

### Preprocessing
- CNH-PSX headlines contained `['...']` artifacts that were cleaned
- 3 354 duplicate headlines removed from CNH-PSX (~27%)
- Pakistan News: 24 574 duplicates removed, date column partially unparseable — used text only for Word2Vec
- PSX Stocks: filtered rows with zero volume, 2 767 NaN filled in `CHANGE (%)`

### Embeddings

We compared 4 models on 3 tests (CNH-PSX headlines, Pakistan News sections, PSX ticker mentions):

| Model | CNH-PSX Δ | Stocks Δ | Notes |
|---|---|---|---|
| Word2Vec (clean) | **0.22** | — | Best for PSX-specific news |
| SBERT-MiniLM (raw) | 0.07 | 0.03 | Good balance, fast |
| SBERT-MPNet (raw) | -0.005 | **0.11** | Best for ticker matching |
| SBERT-Multilingual (raw) | -0.04 | 0.06 | Underperforms on this corpus |

**Key insight**: SBERT performs better on raw headlines than preprocessed ones — aggressive cleaning (stopword removal, lemmatization) removes context that SBERT needs. Word2Vec benefits from cleaning since it works word by word.

**Chosen strategy**: Word2Vec for the main recommendation engine, SBERT-MPNet for ticker-to-news linking.

---

## Documentation

All notebook outputs (EDA results, preprocessing summaries, embedding comparisons) are exported as PDFs in the `doc/` folder, along with the project slides.

---

## Pipeline Overview

```
Raw Data (CSV)
     ↓
Preprocessing (text_cleaner.py + dataset_preprocessor.py)
     ↓
Embeddings (embeddings.py) → .npy files saved in data/processed/
     ↓
User Profile (interests + watchlist)
     ↓
Cosine Similarity + Ranking
     ↓
Top-K News Recommendations
```

---

## Known Limitations

- CNH-PSX corpus only covers 2006–2017 — no recent news
- Headlines are short (~7 words on average) which limits embedding quality
- No full article text available — headlines only
- Synthetic user profiles used (no real user interaction data)