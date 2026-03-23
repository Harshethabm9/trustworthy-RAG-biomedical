from sentence_transformers import SentenceTransformer

def load_model(device="cpu"):
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

def encode_texts(model, texts):
    return model.encode(texts, batch_size=32, convert_to_tensor=False)
