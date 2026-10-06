import json
import numpy as np
from google import genai

client = genai.Client(api_key=open("key.txt").read().strip())
MODEL = "gemini-3.8-flash"
THRESHOLD = 0.62

chunks = json.load(open("chunks.json", encoding="utf-8"))
questions = json.load(open("questions.json", encoding="utf-8"))
vectors = np.load("vectors.npy")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

off_topic = [
    "Who is the prime minister of India?",
    "How do I cook biryani?",
    "What is the capital of France?",
    "What is SQL injection?",
    "Explain how B-tree indexing works",
    "What is a stored procedure in SQL?",
    "What are ACID properties in transactions?",
    "How does the TCP three-way handshake work?",
]

def embed(texts):
    res = client.models.embed_content(model="gemini-embedding-001", contents=texts)
    v = np.array([e.values for e in res.embeddings])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# ---- Part 1: retrieval on in-scope questions ----
sims = embed([q["question"] for q in questions]) @ vectors.T
hit1 = hit3 = answered = 0
mrr = 0.0
for i, q in enumerate(questions):
    order = list(np.argsort(sims[i])[::-1])
    gold = [j for j, c in enumerate(chunks) if c["source"] == q["source"]][0]
    rank = order.index(gold) + 1
    hit1 += rank == 1
    hit3 += rank <= 3
    mrr += 1 / rank
    answered += sims[i].max() >= THRESHOLD
n = len(questions)

# ---- Part 2: refusal on off-topic questions ----
off_best = (embed(off_topic) @ vectors.T).max(axis=1)
declined_retrieval = int((off_best < THRESHOLD).sum())
declined_llm = 0
for q, s in zip(off_topic, off_best):
    if s < THRESHOLD:
        continue
    top = np.argsort(embed([q]) @ vectors.T)[0][::-1][:3]
    ctx = "\n\n".join(chunks[j]["text"] for j in top)
    prompt = ("Answer using ONLY the excerpts. If they do not contain the answer, say exactly: "
              "'Not covered in the video.'\n\nExcerpts:\n" + ctx + "\n\nQuestion: " + q)
    try:
        t = client.models.generate_content(model=MODEL, contents=prompt).text
        declined_llm += "not covered in the video" in t.lower()
    except Exception as e:
        print("LLM check failed:", str(e)[:60])

m = len(off_topic)
results = {
    "in_scope_questions": n,
    "hit_at_1": round(hit1 / n, 3),
    "hit_at_3 (context recall)": round(hit3 / n, 3),
    "MRR": round(mrr / n, 3),
    "context_precision_at_3": round(hit3 / (3 * n), 3),
    "in_scope_answer_rate": round(answered / n, 3),
    "off_topic_questions": m,
    "refused_by_threshold": declined_retrieval,
    "refused_by_LLM_after_threshold": declined_llm,
    "total_refusal_rate": round((declined_retrieval + declined_llm) / m, 3),
}
json.dump(results, open("eval_results.json", "w"), indent=2)
for k, v in results.items():
    print(f"{k}: {v}")
for q, s in zip(off_topic, off_best):
    print(f"  {s:.2f}  {q}")