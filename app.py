import json, os
import numpy as np
import streamlit as st
from google import genai

st.set_page_config(page_title="Study Companion", layout="wide")

client = genai.Client(api_key=open("key.txt").read().strip())
MODEL = "gemini-3.5-flash-lite"
THRESHOLD = 0.62
P0, T, S, G = 0.30, 0.15, 0.10, 0.25   # BKT parameters

chunks = json.load(open("chunks.json", encoding="utf-8"))
vectors = np.load("vectors.npy")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
questions = json.load(open("questions.json", encoding="utf-8")) if os.path.exists("questions.json") else []

def norm(t):
    return t.lower().strip()

def load_learner():
    if os.path.exists("learner.json"):
        return json.load(open("learner.json", encoding="utf-8"))
    return {"mastery": {}, "asked": [], "history": []}

def save_learner():
    json.dump(st.session_state.learner, open("learner.json", "w", encoding="utf-8"))

def bkt(p, correct):
    if correct:
        post = p * (1 - S) / (p * (1 - S) + (1 - p) * G)
    else:
        post = p * S / (p * S + (1 - p) * (1 - G))
    return post + (1 - post) * T

def pick():
    L = st.session_state.learner
    pool = [i for i in range(len(questions)) if i not in L["asked"]]
    if not pool:
        return None
    return min(pool, key=lambda i: L["mastery"].get(norm(questions[i]["topic"]), P0))

def retrieve(q, k=3):
    res = client.models.embed_content(model="gemini-embedding-001", contents=q)
    qv = np.array(res.embeddings[0].values)
    qv = qv / np.linalg.norm(qv)
    scores = vectors @ qv
    idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in idx]

def answer(q):
    hits = retrieve(q)
    if hits[0][1] < THRESHOLD:
        return "This topic is not covered in the video, so I can't answer it from the course material.", []
    ctx = "\n\n".join(f"[{n+1}] (time {c['timestamp']}) {c['text']}" for n, (c, s) in enumerate(hits))
    prompt = ("You are a study tutor. Answer using ONLY the lecture excerpts below. "
              "Cite excerpts like [1], [2]. If the excerpts do not contain the answer, say exactly: "
              "'Not covered in the video.' Do not use outside knowledge.\n\n"
              f"Excerpts:\n{ctx}\n\nQuestion: {q}")
    text = client.models.generate_content(model=MODEL, contents=prompt).text
    if "not covered in the video" in text.lower():
        return text, []
    return text, [(n + 1, c["timestamp"], c["source"]) for n, (c, s) in enumerate(hits)]

if "learner" not in st.session_state:
    st.session_state.learner = load_learner()
    st.session_state.cur = None
    st.session_state.result = None
    st.session_state.msgs = []

st.title("Study Companion: DBMS Normalization")
tab_chat, tab_quiz, tab_dash = st.tabs(["Tutor Chat", "Adaptive Quiz", "Dashboard"])

with tab_chat:
    for role, text in st.session_state.msgs:
        st.chat_message(role).markdown(text)
    q = st.chat_input("Ask a question about the lecture")
    if q:
        st.chat_message("user").markdown(q)
        try:
            text, srcs = answer(q)
        except Exception as e:
            text, srcs = f"Error: {str(e)[:150]}", []
        if srcs:
            text += "\n\n**Sources:**\n" + "\n".join(f"- [{n}] [{ts}]({url})" for n, ts, url in srcs)
        st.chat_message("assistant").markdown(text)
        st.session_state.msgs += [("user", q), ("assistant", text)]

with tab_quiz:
    L = st.session_state.learner
    if not questions:
        st.warning("No questions found. Run make_quiz.py first.")
    elif st.session_state.result:
        r = st.session_state.result
        (st.success if r["correct"] else st.error)("Correct!" if r["correct"] else f"Wrong. Correct answer: {r['right']}")
        st.markdown(f"Topic: **{r['topic']}** | Mastery now: **{r['p']:.0%}**")
        st.markdown(f"Review this part of the lecture: [{r['ts']}]({r['src']})")
        if st.button("Next question"):
            st.session_state.result = None
            st.session_state.cur = pick()
            st.rerun()
    else:
        if st.session_state.cur is None:
            st.session_state.cur = pick()
        cur = st.session_state.cur
        if cur is None:
            st.info("All questions done. Check the Dashboard tab.")
        else:
            qq = questions[cur]
            st.caption(f"Topic: {qq['topic']} | Difficulty: {qq['difficulty']}")
            choice = st.radio(qq["question"], qq["options"], index=None, key=f"q{cur}")
            if st.button("Submit"):
                if choice is None:
                    st.warning("Pick an option first.")
                else:
                    correct = qq["options"].index(choice) == qq["answer"]
                    t = norm(qq["topic"])
                    p = bkt(L["mastery"].get(t, P0), correct)
                    L["mastery"][t] = p
                    L["asked"].append(cur)
                    L["history"].append({"topic": t, "correct": correct})
                    save_learner()
                    st.session_state.result = {"correct": correct, "right": qq["options"][qq["answer"]],
                                               "topic": qq["topic"], "p": p,
                                               "ts": qq["timestamp"], "src": qq["source"]}
                    st.rerun()

with tab_dash:
    L = st.session_state.learner
    if not L["mastery"]:
        st.info("Take a quiz first. Mastery appears here.")
    else:
        st.subheader("Topic mastery")
        st.bar_chart(L["mastery"])
        weak = [t for t, p in L["mastery"].items() if p < 0.5]
        st.subheader("Weak topics")
        st.write(", ".join(weak) if weak else "None. Good job!")
        n = len(L["history"])
        c = sum(h["correct"] for h in L["history"])
        st.metric("Quiz accuracy", f"{c}/{n}")
    if st.button("Reset my progress"):
        st.session_state.learner = {"mastery": {}, "asked": [], "history": []}
        save_learner()
        st.session_state.cur = None
        st.session_state.result = None
        st.rerun()