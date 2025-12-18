# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------
# Request model
# ---------------------------
class Query(BaseModel):
    question: str

# ---------------------------
# Paths
# ---------------------------
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "models")
INDEX_PATH = os.path.join(MODEL_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(MODEL_FOLDER, "metadata.pkl")

# ---------------------------
# Load FAISS and docs safely
# ---------------------------
try:
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        docs = pickle.load(f)
    if isinstance(docs, dict) and 'chunks' in docs:
        docs = docs['chunks']
    print(f"[INFO] Loaded {len(docs)} docs, FAISS index has {index.ntotal} vectors")
except Exception as e:
    print("[ERROR] Failed to load FAISS/index:", e)
    index = None
    docs = []

# ---------------------------
# Embedding model
# ---------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# FLAN-T5 generator
# ---------------------------
generator_model = None
tokenizer = None

def get_generator_model():
    global generator_model, tokenizer
    if generator_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "google/flan-t5-base"
        print("[INFO] Loading FLAN-T5 base...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[INFO] FLAN-T5 loaded")
    return generator_model, tokenizer

def generate_answer(user_question, retrieved_docs):
    model, tok = get_generator_model()
    context = "\n".join(retrieved_docs)
    prompt = f"""
You are a helpful assistant. Use ONLY the information below to answer.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {user_question}
Answer:
"""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tok.decode(outputs[0], skip_special_tokens=True)
    return answer

def fallback_answer(user_question):
    model, tok = get_generator_model()
    prompt = f"Answer the following question as a helpful assistant:\nQuestion: {user_question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tok.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat")
async def chat(query: Query):
    user_question = query.question.strip()
    if user_question == "":
        return {"answer": "Please type something!"}

    if index is None or len(docs) == 0:
        return {"answer": "Backend not ready. Index/docs missing."}

    # Compute embedding and retrieve top 3 docs
    q_emb = embedding_model.encode([user_question]).astype(np.float32)
    D, I = index.search(q_emb, 7)
    retrieved_docs = [docs[i] for i in I[0] if i < len(docs)]

    # Use retrieved docs if available, else fallback
    if len(retrieved_docs) == 0:
        answer = fallback_answer(user_question)
    else:
        answer = generate_answer(user_question, retrieved_docs)

    return {"answer": answer}

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
