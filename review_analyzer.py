import nltk

# Required for newer NLTK versions
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import numpy as np
import pandas as pd
import re

# -----------------------------
# 1. LOAD AMAZON POLARITY DATA
data = load_dataset("amazon_polarity")

df = pd.DataFrame({
    "label": data["train"]["label"],
    "review": data["train"]["content"]
})

# SAMPLE for speed (adjust as needed)
df = df.sample(5000, random_state=42).reset_index(drop=True)

# -----------------------------
# 2. SPLIT INTO SENTENCES
sentences = []
for review in df["review"]:
    for sent in sent_tokenize(review):
        if 10 < len(sent) < 300:
            sentences.append(sent.strip())

print("Total opinion sentences:", len(sentences))

# -----------------------------
# 3. EMBEDDINGS
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)

# -----------------------------
# 4. UMAP FOR CLUSTER SPREAD
umap_embeddings = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=50,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)

# -----------------------------
# 5. HDBSCAN FOR MANY CLUSTERS
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,      # Lower → more clusters
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='leaf'  # more granular clusters
)

labels = clusterer.fit_predict(umap_embeddings)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Clusters found:", n_clusters)

# Filter noise (-1)
cluster_df = pd.DataFrame({
    "sentence": sentences,
    "cluster": labels
})
cluster_df = cluster_df[cluster_df["cluster"] != -1]

# -----------------------------
# 6. SENTIMENT ANALYSIS
from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")

sentiments = sentiment_model(cluster_df["sentence"].tolist(), batch_size=32)
cluster_df["sentiment"] = [1 if r["label"] == "POSITIVE" else -1 for r in sentiments]
cluster_df["score"] = [r["score"] for r in sentiments]

# -----------------------------
# 7. KEYWORDS FOR EACH CLUSTER
kw_model = KeyBERT(model='all-mpnet-base-v2')

cluster_info = []

for c in sorted(cluster_df["cluster"].unique()):
    sub = cluster_df[cluster_df["cluster"] == c]

    # top keywords
    try:
        keywords = kw_model.extract_keywords(
            " ".join(sub["sentence"].tolist())[:5000],
            top_n=6
        )
        top_kw = [k[0] for k in keywords]
    except:
        top_kw = []

    avg_sent = sub["sentiment"].mean()
    neg_frac = np.mean(sub["sentiment"] == -1)
    freq = len(sub)

    # Priority score combines negativity, size
    priority = freq * (1 + neg_frac) * (1 - avg_sent)

    examples = sub["sentence"].head(3).tolist()

    cluster_info.append({
        "cluster": c,
        "size": freq,
        "avg_sentiment": avg_sent,
        "neg_frac": neg_frac,
        "priority_score": priority,
        "keywords": top_kw,
        "examples": examples
    })

# Sort clusters by importance
cluster_info = sorted(cluster_info, key=lambda x: x["priority_score"], reverse=True)

# -----------------------------
# 8. PRINT RESULTS
print("\n\n===========================")
print("  TOP PRIORITY CLUSTERS")
print("===========================\n")

for info in cluster_info[:10]:
    print(f"\n#{info['cluster']}  (Score: {info['priority_score']:.2f}, Size: {info['size']})")
    print(" Avg Sentiment:", round(info["avg_sentiment"], 3))
    print(" Negativity Fraction:", round(info["neg_frac"], 3))
    print(" Keywords:", info["keywords"])
    print(" Example Sentences:")
    for ex in info["examples"]:
        print("  -", ex)
