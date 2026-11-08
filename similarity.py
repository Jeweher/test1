import requests
import numpy as np

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

text1 = "Artificial intelligence is transforming technology."
text2 = "AI is changing the world of technology."
text3 = "I like eating pizza."

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

print("Similar texts:", cosine_similarity(emb1, emb2))
print("Different texts:", cosine_similarity(emb1, emb3))