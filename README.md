# PSX PNR — Pakistan Stock Exchange Personalized News Recommender

A personalized news recommendation system for PSX investors using NLP embeddings and user profiling.

> Université Claude Bernard Lyon 1 — Adam Muhammad Safi Ullah & Ceresa Thomas

---

## Project Status

| Step | Status |
|---|---|
| Data Collection | ✅ Done |
| EDA (all 3 datasets) | ✅ Done |
| Preprocessing | ✅ Done |
| Embeddings (Word2Vec + SBERT) | ✅ Done |
| Embedding Comparison | ✅ Done |
| Article Tagging | ✅ Done |
| Reading History Simulation | ✅ Done |
| User Profile Vectors | ✅ Done |
| Recommender Engine | ✅ Done |
| Formal Evaluation (Precision@K, NDCG@K) | ✅ Done |
| Streamlit Interface | ⏳ Planned |

---

## Project Structure

```
psx-pnr/
├── notebooks/
│   ├── 01_eda_cnhpsx.ipynb
│   ├── 02_eda_pakistan_news.ipynb
│   ├── 03_eda_psx_stocks.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_embeddings.ipynb
│   ├── 06_tagging_history.ipynb
│   ├── 07_profile_vectors.ipynb
│   ├── 08_recommender.ipynb
│   └── 09_evaluation.ipynb
├── src/
│   ├── text_cleaner.py
│   ├── dataset_preprocessor.py
│   ├── embeddings.py
│   ├── user_profile.py
│   ├── article_tagger.py
│   ├── history_simulator.py
│   ├── profile_builder.py
│   ├── scorer.py
│   ├── recommender.py
│   └── evaluator.py
├── data/
│   ├── raw/                              # Original CSVs (not committed)
│   └── processed/
│       ├── *.csv                         # Cleaned datasets + article pool
│       ├── embeddings_pool_*.npy         # Pre-computed embeddings (LFS)
│       ├── user_profiles_*.npz           # User profile vectors (LFS)
│       ├── word2vec.model                # Trained W2V model (LFS)
│       ├── recommendations/              # Top-K CSVs from notebook 08
│       └── evaluation_results.csv        # Precision@K and NDCG@K results
├── doc/                                  # PDF exports of all notebooks
├── app.py                                # (planned) Streamlit interface
└── requirements.txt
```

---

## Datasets

| Dataset | Source | Usage | Size after cleaning |
|---|---|---|---|
| CNH-PSX Categorized Financial News | [Mendeley](https://data.mendeley.com/datasets/mc4s7zvx9c/1) | Main news corpus | 8 858 headlines |
| Pakistan News Headlines | [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistan-news-headlines) | Word2Vec training + article pool | 25 912 articles |
| PSX Stock Market Data 2017–2025 | [Kaggle](https://www.kaggle.com/datasets/fayaznoor10/pakistan-stock-market-data-20172025) | Recency weighting | 813 588 rows, 891 tickers |

---

## Getting Started

### 1. Clone and enter the project

```bash
cd psx-pnr
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / WSL
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries: **pandas**, **numpy**, **scikit-learn**, **nltk**, **gensim**, **sentence-transformers**, **streamlit**, **jupyter**, **matplotlib**, **seaborn**, **tqdm**

### 4. Download the datasets

Place raw CSV files in `data/raw/`:

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

We recommend running the notebooks rather than the `src/` scripts directly. The `src/` modules are clean reusable code called by the notebooks.

---

## Key Findings

### Embeddings (notebook 05)

| Model | CNH-PSX Δ | Stocks Δ | Notes |
|---|---|---|---|
| Word2Vec (clean) | **0.22** | — | Best for headline similarity |
| SBERT-MiniLM (raw) | 0.07 | 0.03 | Good balance, fast |
| SBERT-MPNet (raw) | -0.005 | **0.11** | Best for ticker matching |
| SBERT-Multi (raw) | -0.04 | 0.06 | Underperforms |

SBERT performs better on raw headlines — aggressive cleaning removes context it needs.

### User Profiling (notebook 07)

| Model | Same sector Δ | Notes |
|---|---|---|
| Word2Vec | -0.037 ❌ | Collapses all financial vocab together |
| SBERT-MPNet (raw) | **+0.175** ✅ | Clear sector separation |

**SBERT-MPNet chosen for user profiling and recommender.**

### Recommender Evaluation (notebooks 08 & 09)

| Config | P@5 | P@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| W2V — no boost | 0.45 | 0.45 | 0.42 | 0.45 |
| W2V — recency boost | 0.25 | 0.20 | 0.27 | 0.37 |
| **SBERT — no boost** | **0.55** | **0.55** | **0.63** | **0.70** |
| SBERT — recency boost | 0.50 | 0.50 | 0.50 | 0.50 |

**SBERT no boost is the best configuration across all metrics.**
W2V fails to separate Construction from other financial sectors due to vocabulary overlap.
The recency boost degrades results in both models — it favors active tickers regardless of user sector.

---

## Synthetic Users

| User | Sector | Sub-focus |
|---|---|---|
| User_1 | Construction | cement-heavy |
| User_2 | Construction | infrastructure-heavy |
| User_3 | Banking | — |
| User_4 | Energy | — |

---

## Large Files (Git LFS)

The following file types are tracked with Git LFS:

```
*.csv *.npy *.npz *.model
```

---

## Documentation

All notebook outputs are exported as PDFs in `doc/`, along with the project slides.

---

## Pipeline Overview

```
Raw Data (CSV)
     ↓
Preprocessing → data/processed/*.csv
     ↓
Embeddings (Word2Vec + SBERT) → data/processed/*.npy
     ↓
Article Tagging (keyword matching + word boundaries)
     ↓
Reading History Simulation (10 clicks/user, 10% noise)
     ↓
User Profile Vectors → data/processed/user_profiles_*.npz
     ↓
Cosine Similarity + Top-K Ranking (± recency boost)
     ↓
Evaluation (Precision@K, NDCG@K) → evaluation_results.csv
     ↓
[next] Streamlit Interface
```

---

## Known Limitations

- CNH-PSX corpus covers 2006–2017 only — no recent news
- Headlines are short (~7 words) which limits embedding quality
- No full article text — headlines only
- Synthetic user profiles — no real interaction data
- Construction sector underrepresented (66 articles vs 1 016 Banking, 773 Energy)
- Recency boost too generic — favors active tickers regardless of user sector
