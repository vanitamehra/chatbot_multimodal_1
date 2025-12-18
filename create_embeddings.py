# create_embeddings.py

import os
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------
# Folders
# ---------------------------
DOCS_FOLDER = os.path.join("backend", "data", "docs")  # folder with .txt files
MODEL_FOLDER = "models"  # folder to save FAISS index and metadata
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------------------------
# Load documents
# ---------------------------
docs = []

if not os.path.exists(DOCS_FOLDER):
    raise FileNotFoundError(f"❌ Docs folder not found: {DOCS_FOLDER}")

for filename in sorted(os.listdir(DOCS_FOLDER)):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)

if not docs:
    raise ValueError(f"❌ No text files found in {DOCS_FOLDER}")

print(f"[INFO] ✅ Loaded {len(docs)} documents.")

# ---------------------------
# Create embeddings
# ---------------------------
print("[INFO] Creating embeddings using 'all-mpnet-base-v2'...")
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

# Normalize for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings.astype("float32")

# ---------------------------
# Create FAISS index
# ---------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product ≈ cosine similarity
index.add(embeddings)
print(f"[INFO] ✅ FAISS index created with {index.ntotal} vectors (dim={dimension}).")

# ---------------------------
# Save index and metadata
# ---------------------------
faiss.write_index(index, os.path.join(MODEL_FOLDER, "index.faiss"))
with open(os.path.join(MODEL_FOLDER, "metadata.pkl"), "wb") as f:
    pickle.dump(docs, f)

print(f"[INFO] ✅ Saved FAISS index and metadata to '{MODEL_FOLDER}'")
