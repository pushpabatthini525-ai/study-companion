import json, random, statistics

questions = json.load(open("questions.json", encoding="utf-8"))
N = len(questions)
P0, T, S, G = 0.30, 0.15, 0.10, 0.25   # app.py lo unna BKT parameters
LEARN = 0.15                            # question chesina taruvata asalu knowledge perige rate

def norm(t):
    return t.lower().strip()

def bkt(p, correct):
    if correct:
        post = p * (1 - S) / (p * (1 - S) + (1 - p) * G)
    else:
        post = p * S / (p * S + (1 - p) * (1 - G))
    return post + (1 - post) * T

topics = sorted({norm(q["topic"]) for q in questions})

def simulate(policy, lo, hi, seed, sessions=3, per_session=5):
    rng = random.Random(seed)
    true = {t: rng.uniform(lo, hi) for t in topics}
    start_avg = statistics.mean(true.values())
    est, asked = {}, set()
    served = repeats = 0
    for _ in range(sessions):
        for _ in range(per_session):
            if policy == "adaptive":
                pool = [i for i in range(N) if i not in asked] or list(range(N))
                pick = min(pool, key=lambda i: (est.get(norm(questions[i]["topic"]), P0), rng.random()))
            else:
                pick = rng.randrange(N)
            served += 1
            if pick in asked:
                repeats += 1
            asked.add(pick)
            t = norm(questions[pick]["topic"])
            p_correct = true[t] * (1 - S) + (1 - true[t]) * G
            correct = rng.random() < p_correct
            est[t] = bkt(est.get(t, P0), correct)
            true[t] += LEARN * (1 - true[t])
    return statistics.mean(true.values()) - start_avg, repeats / served

profiles = {"weak": (0.05, 0.30), "average": (0.30, 0.60), "strong": (0.60, 0.85)}
results = {}
print(f"Questions: {N} | Topics: {len(topics)} | 3 sessions x 5 questions | 100 students per profile\n")
print(f"{'profile':9}{'policy':10}{'mastery gain':14}{'repetition rate'}")
for name, (lo, hi) in profiles.items():
    for policy in ["random", "adaptive"]:
        runs = [simulate(policy, lo, hi, seed) for seed in range(100)]
        gain = statistics.mean(r[0] for r in runs)
        rep = statistics.mean(r[1] for r in runs)
        results[f"{name}-{policy}"] = {"gain": round(gain, 3), "repetition": round(rep, 3)}
        print(f"{name:9}{policy:10}{gain:<14.3f}{rep:.1%}")

json.dump(results, open("sim_results.json", "w"), indent=2)
print("\nSaved to sim_results.json")