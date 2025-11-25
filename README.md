# Automatic Text Summarization and Keyword Extraction

**Project:** Automatic Text Summarization and Keyword Extraction using NLP  
**Techniques:** TF-IDF, TextRank, RAKE, YAKE, Sentence-Transformers embeddings

## Repository structure
```
Automatic-Text-Summarization-Keyword-Extraction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── Automatic_Text_Summarization_and_Keyword_Extraction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── summarization.py
│   ├── keyword_extraction.py
│   ├── evaluation.py
│   └── utils.py
├── data/
│   ├── sample_texts/
│   ├── gold_summaries/
│   └── gold_keywords/
│   └── src_sample_dataset/
├── results/
    └──keyword_extraction1.png
    └──sample_txt1.png
    └──txtrank_summ1.png
    └──UI_output1.png
    └──UI_output2.png
    └──eval_metrics.png
└── assets/
    └── README.md
    └── screenshots/
    	└── src_dataset.png
```

## How to run (Colab)
1. Open `notebook/Automatic_Text_Summarization_and_Keyword_Extraction.ipynb` in Google Colab.
2. Run the first cell to install dependencies and download NLTK/Spacy data.
3. Run all cells (Runtime → Restart and run all).

## Requirements
See `requirements.txt` for required Python packages.

## Notes
- Embedding models (sentence-transformers) will download the model on first run (~50-100MB).
- The `src/` folder contains modular code if you want to run locally or package the project.
