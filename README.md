# Study Companion: Source-Grounded Tutor for Lecture Videos

Multimodal AI Hackathon 2026 (IIT Mandi), Track D: Personalized Tutoring & Adaptive Learning.

An AI study companion that turns a lecture video into a source-cited knowledge base, answers questions only from that video, and runs adaptive quizzes while tracking what the student knows.

Demo course material: a DBMS lecture on Normalization (1NF to BCNF).

## Features
- **Video ingestion:** YouTube transcript is split into ~60-second chunks. Every chunk keeps its timestamp and a clickable link (`youtu.be/<id>?t=<seconds>`).
- **Source-grounded tutor chat:** answers use only retrieved lecture excerpts, cite them as [1], [2], and link to the exact timestamp.
- **Refusal of off-material questions:** two layers. A similarity threshold (0.62) blocks unrelated queries, then the LLM is told to reply "Not covered in the video" if the excerpts do not contain the answer. Sources are not shown for refused answers.
- **Quiz generator:** MCQs with topic, difficulty, source timestamp. Each question is verified by a second model call that must pick the same answer using only the excerpt.
- **Learner model:** Bayesian Knowledge Tracing per topic. The next quiz question comes from the weakest topic, and already-asked questions are not repeated.
- **Dashboard:** topic mastery chart, weak topics, quiz accuracy.

## Architecture
1. `ingest_video.py` / `save_chunks.py`: transcript to timestamped chunks (`chunks.json`)
2. `build_index.py`: Gemini embeddings (`vectors.npy`)
3. `app.py` (Streamlit): retrieval (cosine similarity, top 3), grounded answer generation, quiz, BKT, dashboard
4. `make_quiz.py`: question generation and verification (`questions.json`)
5. `eval.py`, `tune.py`, `sim.py`: evaluation

## Grounding method
Query embedding vs chunk embeddings, top 3 chunks. If the best score is below 0.62, the question is declined. Otherwise the chunks are given to the LLM with an instruction to answer only from them and cite them.

## Learner model
Bayesian Knowledge Tracing per topic with P(init)=0.30, P(learn)=0.15, P(slip)=0.10, P(guess)=0.25. Mastery updates after every answer. The quiz picks the unasked question whose topic has the lowest mastery. Topics below 50% are shown as weak.

## Evaluation
Test set: 16 in-scope questions (from `questions.json`, each with a known source timestamp) and 8 off-topic questions.

| Metric | Result |
|---|---|
| Hit@1 (correct chunk ranked first) | 0.75 |
| Hit@3 (context recall) | 0.938 |
| MRR | 0.859 |
| Off-topic refusal rate (threshold 0.62) | 1.0 (8/8 off-topic questions declined) |

Simulated students (100 per profile, 3 sessions x 5 questions, 16 questions, 12 topics):

| Profile | Policy | Mastery gain | Repetition rate |
|---|---|---|---|
| Weak | Random | 0.139 | 33.6% |
| Weak | Adaptive | 0.150 | 0.0% |
| Average | Random | 0.093 | 33.6% |
| Average | Adaptive | 0.100 | 0.0% |
| Strong | Random | 0.047 | 33.6% |
| Strong | Adaptive | 0.050 | 0.0% |

## Limitations (please read)
- Evaluation is a custom script, not RAGAS. Faithfulness and answer relevancy were not measured with a framework.
- Test questions were generated from the same chunks they are evaluated against, so retrieval scores are likely optimistic.
- The refusal threshold (0.62) was tuned on our own test set, with a narrow gap between in-scope (min 0.64) and off-topic (max 0.59) scores. It may not generalize.
- Question verification uses the same model family, so it can share its mistakes.
- Simulated students follow our own assumptions (BKT parameters, 15% learning per question). The mastery gain difference between adaptive and random is small. The clear difference is repetition rate (0% vs 33.6%), which comes from the design.
- Only 16 questions and one single lecture were used. Slides, textbooks and figure extraction are not implemented.

## Run it
    pip install streamlit google-genai numpy youtube-transcript-api
    # create key.txt containing your own Gemini API key
    python -m streamlit run app.py

To use another video: run `python save_chunks.py`, `python build_index.py`, then `python make_quiz.py`.
