import json
import numpy as np
from google import genai

key = open("key.txt").read().strip()
client = genai.Client(api_key=key)

chunks = json.load(open("chunks.json", encoding="utf-8"))
texts = [c["text"] for c in chunks]

vectors = []
for i in range(0, len(texts), 10):
    batch = texts[i:i + 10]
    res = client.models.embed_content(
        model="gemini-embedding-001", contents=batch
    )
    vectors.extend([e.values for e in res.embeddings])
    print("done", len(vectors), "of", len(texts))

np.save("vectors.npy", np.array(vectors))
print("Index ready")