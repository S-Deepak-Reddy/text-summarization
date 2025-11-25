
"""
batch_evaluate.py
-----------------
Run summarization + keyword extraction + evaluation on a CSV dataset.
Place this file in: src/batch_evaluate.py

Usage inside Colab or locally:
-------------------------------
from src.batch_evaluate import run_batch_evaluation
run_batch_evaluation("path/to/your.csv", "results.csv")

Assumptions:
- The notebook already defines summarize_and_extract_keywords().
- If not, a TF‑IDF + RAKE fallback pipeline is used.
"""

import re, os, pandas as pd, numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from rapidfuzz import fuzz
except:
    fuzz = None


def normalize_gold_kw(raw):
    if pd.isna(raw):
        return []
    if isinstance(raw, str):
        parts = re.split(r"[;,\n]", raw)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(raw, (list,tuple)):
        return [str(p).strip() for p in raw if str(p).strip()]
    return [str(raw).strip()]


def fuzzy_keyword_eval(preds, golds, threshold=65):
    preds = [str(p).lower().strip() for p in preds]
    golds = [str(g).lower().strip() for g in golds]

    if len(preds)==0 and len(golds)==0:
        return {'precision':1.0,'recall':1.0,'f1':1.0,'matched':[]}

    # fallback exact match if rapidfuzz missing
    if fuzz is None:
        predset, goldset = set(preds), set(golds)
        tp = len(predset & goldset)
        prec = tp/len(predset) if predset else 0.0
        rec = tp/len(goldset) if goldset else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return {'precision':prec,'recall':rec,'f1':f1,'matched':list(predset & goldset)}

    matched_gold = set()
    tp = 0
    for p in preds:
        for g in golds:
            if g in matched_gold:
                continue
            if fuzz.partial_ratio(p, g) >= threshold:
                tp += 1
                matched_gold.add(g)
                break

    prec = tp / len(preds) if preds else 0.0
    rec = tp / len(golds) if golds else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {'precision':prec,'recall':rec,'f1':f1,'matched':list(matched_gold)}



# ROUGE
_rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
def compute_rouge(sys_sum, ref_sum):
    s = _rouge.score(ref_sum or "", sys_sum or "")
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge2_f": s["rouge2"].fmeasure,
        "rougeL_f": s["rougeL"].fmeasure
    }


# Embedding similarity (cached model)
_embed_model = None
def summary_cosine_similarity(sys_sum, ref_sum, model_name="all-MiniLM-L6-v2"):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    v_sys = _embed_model.encode(sys_sum or "", convert_to_numpy=True)
    v_ref = _embed_model.encode(ref_sum or "", convert_to_numpy=True)
    if np.linalg.norm(v_sys)==0 or np.linalg.norm(v_ref)==0:
        return 0.0
    return float(cosine_similarity([v_sys],[v_ref])[0][0])



# ---------------- FALLBACK PIPELINE (if notebook doesn't define it) ---------------
def fallback_sentences(text):
    import nltk
    from nltk.tokenize import sent_tokenize
    if not isinstance(text,str):
        return []
    return [s.strip() for s in sent_tokenize(text.replace("\n"," ")) if len(s.strip())>10]

def fallback_tfidf_summary(text, n_sentences=3):
    sents = fallback_sentences(text)
    if not sents:
        return ""
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vect.fit_transform(sents)
    scores = X.sum(axis=1).A1
    top = np.argsort(scores)[-min(n_sentences,len(sents)):][::-1]
    top = sorted(top)
    return " ".join([sents[i] for i in top])

def fallback_rake_keywords(text, top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text or "")
    ranked = r.get_ranked_phrases()
    return ranked[:top_n]

def fallback_pipeline(text, summary_type="tfidf", n_sentences=3, keyword_method="rake"):
    summary = fallback_tfidf_summary(text, n_sentences=n_sentences)
    kws = fallback_rake_keywords(text, top_n=10)
    return {"summary": summary, "keywords": kws}



# ---------------------- MAIN FUNCTION ----------------------
def run_batch_evaluation(csv_path, out_csv,
                         summary_method="tfidf",
                         n_sentences=3,
                         keyword_method="rake"):
    """
    Runs summarization + keyword extraction + metrics for each row.
    CSV must contain: sample_text, gold_summary, gold_keywords
    """

    df = pd.read_csv(csv_path)
    for col in ["sample_text","gold_summary","gold_keywords"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Determine if notebook-defined function exists
    use_existing = False
    try:
        summarize_and_extract_keywords  # noqa
        use_existing = True
    except:
        use_existing = False

    rows = []
    for idx, r in df.iterrows():
        text = r["sample_text"]
        gold_sum = r.get("gold_summary") or ""
        gold_kw = normalize_gold_kw(r.get("gold_keywords"))

        # Run whichever pipeline is available
        if use_existing:
            out = summarize_and_extract_keywords(
                text,
                summary_type=summary_method,
                n_sentences=n_sentences,
                keyword_method=keyword_method
            )
            sys_sum = out.get("summary","")
            sys_kws = out.get("keywords",[])
        else:
            out = fallback_pipeline(text, summary_method, n_sentences, keyword_method)
            sys_sum = out["summary"]
            sys_kws = out["keywords"]

        # flatten keyword output
        if isinstance(sys_kws, list) and sys_kws and isinstance(sys_kws[0], tuple):
            sys_kws = [t[0] for t in sys_kws]

        # evaluations
        rouge_scores = compute_rouge(sys_sum, gold_sum)
        emb_sim = summary_cosine_similarity(sys_sum, gold_sum)
        kw = fuzzy_keyword_eval(sys_kws, gold_kw, threshold=65)

        rows.append({
            "id": r.get("id", idx),
            "sys_summary": sys_sum,
            "gold_summary": gold_sum,
            "rouge1_f": rouge_scores["rouge1_f"],
            "rouge2_f": rouge_scores["rouge2_f"],
            "rougeL_f": rouge_scores["rougeL_f"],
            "embedding_cosine": emb_sim,
            "sys_keywords": "; ".join(sys_kws),
            "gold_keywords": "; ".join(gold_kw),
            "kw_precision": kw["precision"],
            "kw_recall": kw["recall"],
            "kw_f1": kw["f1"]
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    return out_df


# Optional helper to compute aggregated metrics separately
def aggregate_results(df):
    return {
        "rouge1_f_mean": df["rouge1_f"].mean(),
        "rouge2_f_mean": df["rouge2_f"].mean(),
        "rougeL_f_mean": df["rougeL_f"].mean(),
        "embedding_cosine_mean": df["embedding_cosine"].mean(),
        "kw_precision_mean": df["kw_precision"].mean(),
        "kw_recall_mean": df["kw_recall"].mean(),
        "kw_f1_mean": df["kw_f1"].mean()
    }
