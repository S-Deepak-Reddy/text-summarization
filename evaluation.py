"""
evaluation.py - ROUGE and keyword evaluation utilities
"""
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_rouge(system_summary: str, reference_summary: str):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, system_summary)
    out = {}
    for k, v in scores.items():
        out[k] = {'precision': v.precision, 'recall': v.recall, 'fmeasure': v.fmeasure}
    return out

_embed_model = None
def summary_cosine_similarity(system_summary: str, reference_summary: str, model_name='all-MiniLM-L6-v2'):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    emb_sys = _embed_model.encode(system_summary, convert_to_numpy=True, show_progress_bar=False)
    emb_ref = _embed_model.encode(reference_summary, convert_to_numpy=True, show_progress_bar=False)
    return float(cosine_similarity([emb_sys], [emb_ref])[0][0])

def keyword_precision_recall_f1(pred_keywords, gold_keywords):
    def norm(k):
        k = k.lower()
        k = re.sub(r'[^a-z0-9\s]', '', k)
        return k.strip()
    pred = set(norm(k) for k in pred_keywords)
    gold = set(norm(k) for k in gold_keywords)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}
