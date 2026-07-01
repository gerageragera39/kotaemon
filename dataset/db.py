import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
INPUT_FILE = "conversation_split.csv"
OUTPUT_FILE = "faq_output.csv"

SIMILARITY_THRESHOLD = 0.80  # tweak: 0.75–0.85
MODEL_NAME = "all-MiniLM-L6-v2"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["question"])
questions = df["question"].astype(str).tolist()

# =========================
# EMBEDDINGS
# =========================
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(questions, normalize_embeddings=True)

# =========================
# SIMILARITY MATRIX
# =========================
sim_matrix = cosine_similarity(embeddings)

K = 5  # wichtig: control parameter

neighbors = np.argsort(-sim_matrix, axis=1)[:, 1:K+1]

# =========================
# BUILD GROUPS (GRAPH CLUSTERING)
# =========================
parent = list(range(len(questions)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    pa, pb = find(a), find(b)
    if pa != pb:
        parent[pb] = pa

SIM_THRESHOLD = 0.7  # niedriger als vorher!

for i in range(len(questions)):
    for j in neighbors[i]:
        if sim_matrix[i][j] >= SIM_THRESHOLD:
            union(i, j)

groups = {}

for i in range(len(questions)):
    root = find(i)
    groups.setdefault(root, []).append(i)

faq = []

for cluster_id, idxs in groups.items():
    group_questions = [questions[i] for i in idxs]

    # bessere Repräsentation: längste Frage (simple but effective)
    representative = max(group_questions, key=len)

    faq.append({
        "cluster_id": cluster_id,
        "frequency": len(idxs),
        "faq_question": representative,
        "questions": group_questions
    })

faq_df = pd.DataFrame(faq)
faq_df = faq_df.sort_values("frequency", ascending=False)

faq_df.to_csv(OUTPUT_FILE, index=False)