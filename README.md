# CS410_Project_CustomerReviewAnalyzerProducts
A customer-review analysis tool that transforms thousands of unstructured reviews into actionable product insights by extracting features, analyzing sentiment, and clustering similar complaints using modern NLP models like BERT and Sentence-BERT.

# Overview

This project processes numerous unstructured customer reviews and turns them into clear, prioritized insights that can guide product improvements. It extracts product features, evaluates the sentiment behind each one, and groups similar pieces of feedback using SBERT embeddings, keyword extraction, and clustering methods. The result is an easy way to spot recurring issues, understand what customers care about most, and identify high-impact themes based on real review data. Everything runs directly on the Amazon Polarity Dataset, so there's no need to download or manage your own data files.
All data is retrieved directly from the Amazon Polarity Dataset, so no manual data folder setup is required.

# Pipeline Diagram
        ┌──────────────────────────────┐
        │   Amazon Polarity Dataset    │
        │     (auto-downloaded)        │
        └───────────────┬──────────────┘
                        ▼
            ┌────────────────────┐
            │  Data Preprocessing│
            │ - Clean text       │
            │ - Tokenize         │
            │ - Sentence split   │
            └───────────┬────────┘
                        ▼
            ┌────────────────────┐
            │ Feature Extraction │
            │ - SBERT embeddings │
            │ - KeyBERT keywords │
            │ - Aspect merging   │
            └───────────┬────────┘
                        ▼
        ┌────────────────────────────────┐
        │   Sentiment Analysis           │
        │ - Transformer sentiment model  │
        │ - Polarity scoring             │
        └───────────┬────────────────────┘
                    ▼
        ┌────────────────────────────────┐
        │  Clustering & Theme Detection  │
        │ - SBERT embeddings             │
        │ - KMeans / HDBSCAN             │
        └───────────┬────────────────────┘
                    ▼
         ┌────────────────────────────────┐
         │ Prioritization & Insights      │
         │ - Frequency                    │
         │ - Negativity                   │
         │ - Recency (if used)            │
         └────────────────────────────────┘

# Setup
1. Install Dependencies: pip install -r requirements.txt
2. Go into src folder and Run the Analyzer python review_analyzer.py


This will call run_pipeline(), load the Amazon Polarity dataset, perform feature extraction, sentiment analysis, clustering, and generate prioritized insights automatically.

Packages Used
# NLP / ML
import nltk

# Download the stopwords corpus (one-time; cached under ~/.cache/nltk or ~/nltk_data)
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# SBERT for sentence embeddings
from sentence_transformers import SentenceTransformer

# Clustering / Vectorization
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Fuzzy matching for aspect normalization
from rapidfuzz import process, fuzz

# Keyword extraction
from keybert import KeyBERT

# Density-based clustering
import hdbscan

# Optional transformer-based sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

