# backend/create_embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ---------------------------
# Folder containing your text files
# ---------------------------
DOCS_FOLDER = "D:/institute_project/backend/data/docs"  # change if different
MODEL_FOLDER = "D:/institute_project/backend/models"

# ---------------------------
# Check folder exists
# ---------------------------
if not os.path.exists(DOCS_FOLDER):
    raise FileNotFoundError(f"Folder not found: {DOCS_FOLDER}")

# ---------------------------
# Read all .txt files
# ---------------------------
docs = []
for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)

if not docs:
    raise ValueError(f"No text files found in {DOCS_FOLDER}")

print(f"[INFO] Loaded {len(docs)} documents from folder.")

# ---------------------------
# Create embeddings
# ---------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(docs).astype('float32')
print("[INFO] Created embeddings for all documents.")

# ---------------------------
# Create FAISS index
# ---------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"[INFO] FAISS index created with {index.ntotal} vectors.")

# ---------------------------
# Save index and metadata
# ---------------------------
os.makedirs(MODEL_FOLDER, exist_ok=True)
faiss.write_index(index, os.path.join(MODEL_FOLDER, "index.faiss"))
with open(os.path.join(MODEL_FOLDER, "metadata.pkl"), "wb") as f:
    pickle.dump(docs, f)

print(f"[INFO] Saved FAISS index and metadata to '{MODEL_FOLDER}'.")
