"""
embeddings.py
-------------
Embedding generation for PSX PNR pipeline.
Supports:
    - Word2Vec (trained on local corpus)
    - SBERT multilingual (paraphrase-multilingual-MiniLM-L12-v2)
    - SBERT MiniLM English (all-MiniLM-L6-v2)
    - SBERT MPNet English (all-mpnet-base-v2)

Usage:
    from embeddings import train_word2vec, embed_corpus, load_sbert, SBERT_MODELS
"""

import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# WORD2VEC
# ─────────────────────────────────────────

def train_word2vec(
    texts: list,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
    save_path: str = None
):
    """
    Train a Word2Vec model on a list of texts.

    Args:
        texts: List of cleaned text strings
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency to include
        workers: Number of CPU threads
        epochs: Number of training epochs
        save_path: If provided, save model to this path

    Returns:
        Trained Word2Vec model
    """
    from gensim.models import Word2Vec

    print(f"[Word2Vec] Tokenizing {len(texts)} texts...")
    tokenized = [text.split() for text in texts if isinstance(text, str) and text.strip()]

    print(f"[Word2Vec] Training (vector_size={vector_size}, window={window}, epochs={epochs})...")
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )

    print(f"[Word2Vec] Vocabulary size: {len(model.wv.key_to_index)} words")

    if save_path:
        model.save(save_path)
        print(f"[Word2Vec] Model saved to {save_path}")

    return model


def get_word2vec_embedding(text: str, model) -> np.ndarray:
    """
    Get embedding for a text using Word2Vec (mean of word vectors).

    Args:
        text: Cleaned text string
        model: Trained Word2Vec model

    Returns:
        Numpy array of shape (vector_size,)
    """
    tokens = text.split() if isinstance(text, str) else []
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def load_word2vec(model_path: str):
    """Load a saved Word2Vec model."""
    from gensim.models import Word2Vec
    model = Word2Vec.load(model_path)
    print(f"[Word2Vec] Loaded from {model_path}")
    return model


# ─────────────────────────────────────────
# SBERT
# ─────────────────────────────────────────

# Available SBERT models for comparison
SBERT_MODELS = {
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # best for PSX (mixed language), 384-dim
    "minilm":       "all-MiniLM-L6-v2",                       # fast, English only, 384-dim
    "mpnet":        "all-mpnet-base-v2",                       # best quality, English only, 768-dim
}


def load_sbert(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Load a pretrained SBERT model.

    Recommended models (see SBERT_MODELS):
        - multilingual : paraphrase-multilingual-MiniLM-L12-v2 — best for PSX mixed-language news
        - minilm       : all-MiniLM-L6-v2                      — very fast, English only
        - mpnet        : all-mpnet-base-v2                      — highest quality, English only

    You can pass either the key (e.g. 'multilingual') or the full model name.

    Args:
        model_name: Key from SBERT_MODELS or full HuggingFace model name

    Returns:
        SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    # Allow shorthand keys
    resolved = SBERT_MODELS.get(model_name, model_name)
    print(f"[SBERT] Loading: {resolved}...")
    model = SentenceTransformer(resolved)
    print(f"[SBERT] Ready — embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def get_sbert_embedding(text: str, model) -> np.ndarray:
    """
    Get a single SBERT embedding for a text.

    Args:
        text: Text string
        model: Loaded SentenceTransformer model

    Returns:
        Numpy array of shape (embedding_dim,)
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text, show_progress_bar=False)


# ─────────────────────────────────────────
# CORPUS EMBEDDING
# ─────────────────────────────────────────

def embed_corpus(
    texts: list,
    method: str,
    model,
    save_path: str = None,
    batch_size: int = 64
) -> np.ndarray:
    """
    Embed an entire corpus of texts.

    Args:
        texts: List of text strings
        method: 'word2vec' or 'sbert'
        model: Trained Word2Vec or SBERT model
        save_path: If provided, save embeddings as .npy file
        batch_size: Batch size for SBERT (ignored for Word2Vec)

    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    print(f"[embed_corpus] Embedding {len(texts)} texts with {method}...")

    if method == 'word2vec':
        embeddings = np.array([
            get_word2vec_embedding(text, model)
            for text in tqdm(texts, desc="Word2Vec")
        ])

    elif method == 'sbert':
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'word2vec' or 'sbert'.")

    print(f"[embed_corpus] Done. Shape: {embeddings.shape}")

    if save_path:
        np.save(save_path, embeddings)
        print(f"[embed_corpus] Saved to {save_path}")

    return embeddings


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a .npy file."""
    embeddings = np.load(path)
    print(f"[load_embeddings] Loaded {embeddings.shape} from {path}")
    return embeddings
