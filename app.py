"""
app.py
------
PSX PNR — Streamlit demo app.

Run with:
    streamlit run app.py
"""

import sys
sys.path.append('src')

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from user_profile import SYNTHETIC_USERS, SECTOR_KEYWORDS

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PSX PNR",
    page_icon="📰",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load data (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_pool     = pd.read_csv('data/processed/article_pool.csv')
    reading_log = pd.read_csv('data/processed/reading_log.csv')
    emb_sbert   = np.load('data/processed/embeddings_pool_sbert_mpnet.npy')
    return df_pool, reading_log, emb_sbert

df_pool, reading_log, emb_sbert = load_data()

@st.cache_data
def load_profiles():
    data = np.load('data/processed/user_profiles_sbert_mpnet.npz')
    return {k: data[k] for k in data.files}

profiles = load_profiles()

# ─────────────────────────────────────────────
# Recommend function
# ─────────────────────────────────────────────
def recommend(profile_vector, read_ids, top_k=3):
    scores  = cosine_similarity([profile_vector], emb_sbert)[0]
    results = df_pool.copy()
    results['article_id']  = results.index
    results['score']       = scores
    results = results[~results['article_id'].isin(read_ids)]
    results = results.sort_values('score', ascending=False).head(top_k)
    results.insert(0, 'rank', range(1, len(results) + 1))
    return results[['rank', 'headline', 'primary_tag', 'score']].reset_index(drop=True)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("📰 PSX Personalized News Recommender")
st.caption("Université Claude Bernard Lyon 1 — Adam Muhammad Safi Ullah & Ceresa Thomas")
st.divider()

# Sidebar — user selection
with st.sidebar:
    st.header("Select User")
    user_labels = {
        u.user_id: f"{u.user_id} — {u.sector} ({u.sub_focus or 'general'})"
        for u in SYNTHETIC_USERS
    }
    selected_id = st.selectbox(
        "Synthetic user",
        options=list(user_labels.keys()),
        format_func=lambda x: user_labels[x]
    )
    st.divider()
    st.caption("SBERT-MPNet · No recency boost · Top-3")

# Get selected user
user = next(u for u in SYNTHETIC_USERS if u.user_id == selected_id)
user.clicked_articles = reading_log[
    reading_log['user_id'] == selected_id
]['article_id'].tolist()
user.profile_vector = profiles[selected_id]

# Two columns
col1, col2 = st.columns([1, 2])

# Left — user profile info
with col1:
    st.subheader(f"{user.user_id}")
    st.markdown(f"**Sector:** {user.sector}")
    st.markdown(f"**Sub-focus:** {user.sub_focus or '—'}")
    st.markdown(f"**Articles read:** {len(user.clicked_articles)}")

    st.markdown("**Reading history:**")
    history = reading_log[reading_log['user_id'] == selected_id]
    for _, row in history.iterrows():
        tag   = df_pool.loc[row['article_id'], 'primary_tag'] if row['article_id'] < len(df_pool) else '?'
        label = df_pool.loc[row['article_id'], 'headline'][:60] if row['article_id'] < len(df_pool) else '?'
        color = {"Construction": "🟡", "Banking": "🔵", "Energy": "🟠"}.get(tag, "⚪")
        st.markdown(f"{color} {label}...")

# Right — recommendations
with col2:
    st.subheader("Top-3 Recommendations")
    recs = recommend(user.profile_vector, user.clicked_articles, top_k=3)

    for _, row in recs.iterrows():
        tag   = row['primary_tag']
        color = {"Construction": "🟡", "Banking": "🔵", "Energy": "🟠"}.get(tag, "⚪")
        with st.container(border=True):
            st.markdown(f"**#{int(row['rank'])} {color} {tag}**")
            st.markdown(f"{row['headline']}")
            st.progress(float(row['score']), text=f"Score: {row['score']:.3f}")
