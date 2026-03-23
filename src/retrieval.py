import numpy as np
import faiss
from rank_bm25 import BM25Okapi

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def build_bm25(corpus):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)

def bm25_retrieve(query, bm25, k=5):
    scores = bm25.get_scores(query.split())
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx

def rrf_fusion(dense_idx, sparse_idx, k=5, c=60):
    combined = {}
    for rank, idx in enumerate(dense_idx):
        combined[idx] = combined.get(idx, 0) + 1/(c+rank+1)
    for rank, idx in enumerate(sparse_idx):
        combined[idx] = combined.get(idx, 0) + 1/(c+rank+1)
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]
