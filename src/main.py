from data_loader import load_data
from embedding import load_model, encode_texts
from retrieval import build_faiss_index, build_bm25, bm25_retrieve, rrf_fusion
from evaluation import compute_bti
from utils import expand_query

import numpy as np
import faiss

def main():

    # 🔹 Load data
    df = load_data("data/medquad_sample.csv")

    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    # 🔹 Ontology (same as your original code)
    ontology_synonyms = {
        "hypertension": ["high blood pressure"],
        "diabetes": ["high blood sugar"],
        "cancer": ["tumor"]
    }

    # 🔹 Expand queries
    expanded_questions = [
        " ".join(expand_query(q, ontology_synonyms)) for q in questions
    ]

    # 🔹 Load model
    model = load_model()

    # 🔹 Create embeddings
    q_emb = encode_texts(model, expanded_questions)
    a_emb = encode_texts(model, answers)

    # 🔹 Build FAISS index
    index = build_faiss_index(a_emb)

    # 🔹 Build BM25
    bm25 = build_bm25(answers)

    print("System ready! Running retrieval...")

    k = 3

    results = []

    for i, (query, q_vector) in enumerate(zip(questions, q_emb)):

        # FAISS search
        q_vector = q_vector.reshape(1, -1)
        faiss.normalize_L2(q_vector)

        D, I = index.search(q_vector, k)

        # BM25 search
        sparse_idx = bm25_retrieve(query, bm25, k)

        # RRF fusion
        hybrid_idx = rrf_fusion(I[0], sparse_idx, k)

        # Get top chunks
        top_chunks = [answers[j] for j in hybrid_idx]

        # Simple generation (same as your original)
        generated_answer = " ".join(top_chunks)

        # Confidence (from FAISS)
        confidence = float(np.mean(D[0])) if len(D[0]) else 0.0

        # Exact match (simple)
        exact_match = int(generated_answer.strip().lower() == answers[i].strip().lower())

        # Dummy attribution (for now)
        attribution = 0.8

        # BTI
        bti = compute_bti(attribution, confidence, exact_match)

        results.append({
            "question": query,
            "generated_answer": generated_answer,
            "confidence": confidence,
            "bti": bti
        })

    print("\nSample Output:\n")

    for r in results[:5]:
        print("Q:", r["question"])
        print("A:", r["generated_answer"])
        print("BTI:", round(r["bti"], 3))
        print("-" * 50)


if __name__ == "__main__":
    main()
