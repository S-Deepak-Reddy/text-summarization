"""
keyword_extraction.py - RAKE, YAKE, and TF-IDF keyword extraction
"""
from rake_nltk import Rake
import yake
from sklearn.feature_extraction.text import TfidfVectorizer

def rake_keywords(text: str, top_n: int = 10):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked = r.get_ranked_phrases_with_scores()
    return [(phrase, score) for score, phrase in ranked][:top_n]

def yake_keywords(text: str, top_n: int = 10, max_ngram_size: int = 3):
    kw = yake.KeywordExtractor(n=max_ngram_size, top=top_n)
    return kw.extract_keywords(text)

def tfidf_top_terms(text: str, top_n: int = 10):
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform([text])
    features = vect.get_feature_names_out()
    scores = X.toarray().ravel()
    idx = scores.argsort()[-top_n:][::-1]
    return list(zip(features[idx], scores[idx]))
