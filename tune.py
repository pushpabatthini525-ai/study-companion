import json
import numpy as np
from google import genai

client = genai.Client(api_key=open("key.txt").read().strip())
vectors = np.load("vectors.npy")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
qs = json.load(open("questions.json", encoding="utf-8"))
res = client.models.embed_content(model="gemini-embedding-001", contents=[q["question"] for q in qs])
v = np.array([e.values for e in res.embeddings])
v = v / np.linalg.norm(v, axis=1, keepdims=True)
best = (v @ vectors.T).max(axis=1)
print("in-scope min:", round(best.min(), 2), "| mean:", round(best.mean(), 2))
print("sorted lowest 5:", [round(x, 2) for x in sorted(best)[:5]])