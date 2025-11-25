"""
preprocessing.py - text cleaning and tokenization helpers
""" 
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^0-9A-Za-z.,;:!?\'"()\- ]+', '', text)
    return text

def sentences_from_text(text: str):
    text = clean_text(text)
    sents = sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip())>0]
