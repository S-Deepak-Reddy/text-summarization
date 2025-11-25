"""
summarization.py - TF-IDF, TextRank and embedding-based summarizers
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from src.preprocessing import sentences_from_text

def tfidf_sentence_summary(text: str, n_sentences: int = 3):
    sents = sentences_from_text(text)
    if not sents:
        return ""
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(sents)
    scores = X.sum(axis=1).A1
    top_idx = np.argsort(scores)[-n_sentences:][::-1]
    top_idx_sorted = sorted(top_idx)
    return " ".join([sents[i] for i in top_idx_sorted])

def textrank_summary(text: str, n_sentences: int = 3):
    sents = sentences_from_text(text)
    if not sents:
        return ""
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(sents)
    sim = cosine_similarity(X)
    g = nx.from_numpy_array(sim)
    scores = nx.pagerank(g)
    ranked = sorted(((scores[i], i) for i in scores), reverse=True)
    top_idx = [idx for (_, idx) in ranked[:n_sentences]]
    top_idx_sorted = sorted(top_idx)
    return " ".join([sents[i] for i in top_idx_sorted])

def embeddings_summary(text: str, n_sentences: int = 3, model_name: str = 'all-MiniLM-L6-v2'):
    sents = sentences_from_text(text)
    if not sents:
        return ""
    model = SentenceTransformer(model_name)
    emb = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    centroid = emb.mean(axis=0)
    sim = emb @ centroid
    top_idx = np.argsort(sim)[-n_sentences:][::-1]
    top_idx_sorted = sorted(top_idx)
    return " ".join([sents[i] for i in top_idx_sorted])
