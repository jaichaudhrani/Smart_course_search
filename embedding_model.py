from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=True)
