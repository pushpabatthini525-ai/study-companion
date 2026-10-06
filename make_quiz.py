import json, time, os
from google import genai
from google.genai import types

client = genai.Client(api_key=open("key.txt").read().strip())
MODEL = "gemini-3.5-flash-lite"
JSON_CFG = types.GenerateContentConfig(response_mime_type="application/json")
MAX_CHUNKS = 8   # okasari entha chunks process cheyyali

chunks = json.load(open("chunks.json", encoding="utf-8"))
if os.path.exists("quiz_state.json"):
    state = json.load(open("quiz_state.json", encoding="utf-8"))
else:
    state = {"done": [], "questions": [], "rejected": 0}

def save():
    json.dump(state, open("quiz_state.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(state["questions"], open("questions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

def ask_json(prompt):
    for _ in range(3):
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt, config=JSON_CFG)
            time.sleep(6)
            return json.loads(r.text)
        except Exception as e:
            msg = str(e)
            print("  retry:", msg[:70])
            time.sleep(60 if "429" in msg else 5)
    return None

def generate(chunk):
    prompt = (
        "From the lecture excerpt below, write 2 multiple-choice questions that can be "
        "answered ONLY from this excerpt. Return a JSON list. Each item: "
        '{"question": str, "options": [4 strings], "answer": index 0-3, '
        '"difficulty": "easy"|"medium"|"hard", "topic": short topic name}.\n\n'
        f"Excerpt:\n{chunk['text']}"
    )
    return ask_json(prompt)

def verify(q, chunk):
    prompt = (
        "Using ONLY the excerpt, pick the correct option. "
        'Reply JSON: {"answer": index 0-3}.\n\n'
        f"Excerpt:\n{chunk['text']}\n\nQuestion: {q['question']}\n"
        + "\n".join(f"{i}. {o}" for i, o in enumerate(q["options"]))
    )
    r = ask_json(prompt)
    return isinstance(r, dict) and r.get("answer") == q["answer"]

processed = 0
for n, c in enumerate(chunks):
    if n in state["done"]:
        continue
    if processed >= MAX_CHUNKS:
        break
    print(f"chunk {n+1}/{len(chunks)}")
    qs = generate(c)
    if qs is None:
        print("Quota/server problem. Progress saved. Run again later.")
        break
    for q in qs:
        try:
            ok = len(q["options"]) == 4 and verify(q, c)
        except Exception:
            ok = False
        if ok:
            q["source"] = c["source"]
            q["timestamp"] = c["timestamp"]
            state["questions"].append(q)
        else:
            state["rejected"] += 1
    state["done"].append(n)
    processed += 1
    save()

save()
print("Verified questions:", len(state["questions"]),
      "| Rejected:", state["rejected"],
      "| Chunks done:", len(state["done"]), "of", len(chunks))