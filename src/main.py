from data_loader import load_data
from embedding import load_model, encode_texts
from retrieval import build_faiss_index, build_bm25, bm25_retrieve, rrf_fusion

def main():
    df = load_data("data/medquad.csv")

    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    model = load_model()

    q_emb = encode_texts(model, questions)
    a_emb = encode_texts(model, answers)

    index = build_faiss_index(a_emb)
    bm25 = build_bm25(answers)

    print("System ready!")

if __name__ == "__main__":
    main()
