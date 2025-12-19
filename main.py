import os
import pickle
import faiss
import numpy as np
import tempfile

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from TTS.api import TTS
import whisper

# ---------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------
FRONTEND_FOLDER = "frontend"
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_FOLDER, "js")), name="static")

@app.get("/{html_file}")
def serve_html(html_file: str):
    if html_file.endswith(".html"):
        fp = os.path.join(FRONTEND_FOLDER, html_file)
        if os.path.exists(fp):
            return FileResponse(fp)
    return FileResponse(os.path.join(FRONTEND_FOLDER, "chatbot.html"))

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_FOLDER, "chatbot.html"))

# ---------------------------------------------------------
# Load STT (Whisper)
# ---------------------------------------------------------
stt_model = whisper.load_model("base")

# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------
class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

# ---------------------------------------------------------
# Load FAISS Index + Metadata
# ---------------------------------------------------------
MODEL_FOLDER = "models"
INDEX_PATH = os.path.join(MODEL_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(MODEL_FOLDER, "metadata.pkl")

if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run create_embeddings.py first.")

index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    docs = pickle.load(f)

# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
tokenizer = embedding_model.tokenizer
MAX_TOKENS = 512
CHUNK_SIZE_WORDS = 200

# ---------------------------------------------------------
# Load RAG Model (Flan-T5)
# ---------------------------------------------------------
MODEL_NAME = "google/flan-t5-base"
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
flan_pipe = pipeline("text2text-generation", model=hf_model, tokenizer=hf_tokenizer)

# ---------------------------------------------------------
# Load TTS model
# ---------------------------------------------------------
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def speak_friendly(text):
    # Convert only emails or specific symbols
    text = text.replace("pcontact@proed.com", "p contact at pro ed dot com")
    text = text.replace("proed", "pro ed")
    replacements = {"@": " at ", "-": " dash ", "/": " slash "}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def chunk_text(text, size=CHUNK_SIZE_WORDS):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

def retrieve_docs(query_text, k=10):
    tokens = tokenizer.encode(query_text, truncation=True, max_length=MAX_TOKENS)
    truncated = tokenizer.decode(tokens)
    query_vector = embedding_model.encode(truncated, convert_to_numpy=True).astype("float32")
    query_vector /= np.linalg.norm(query_vector)
    D, I = index.search(np.array([query_vector]), k)
    return [docs[i] for i in I[0] if i < len(docs) and docs[i].strip()]


def handle_greetings(query_text: str):
    greetings = ["hello", "hi", "hey", "greetings"]
    if query_text.strip().lower() in greetings:
        return "Hello! How can I assist you with courses or admissions?"
    return None


def run_rag(query_text):

    greeting_response = handle_greetings(query_text)
    if greeting_response:
        return greeting_response
    
    retrieved = retrieve_docs(query_text)
    q_lower = query_text.lower()

    course_kw = ["course", "curriculum", "module", "syllabus"]
    fee_kw = ["fee", "enrollment", "tuition"]

    # -----------------------------
    # Fallback rules
    # -----------------------------
    if not retrieved:
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(g in q_lower for g in greetings):
            return "Hello! How can I assist you with courses or admissions?"
        if "weather" in q_lower or "rain" in q_lower:
            return "I can only provide information about courses or enrollment."
        if any(w in q_lower for w in course_kw):
            return "For details about courses, please contact the institute."
        if any(w in q_lower for w in fee_kw):
            return "For enrollment or fee details, please contact the institute directly."
        return "I can only answer questions related to courses."

    # -----------------------------
    # RAG processing
    # -----------------------------
    chunks = []
    for t in retrieved:
        chunks.extend(list(chunk_text(t)))
    context = "\n".join(chunks[:10])

    prompt = f"""
Answer ONLY using the context below. Do NOT make up answers.

Fallback:
1. Course questions → "For details about courses, please contact the institute."
2. Fee questions → "For enrollment or fee details, please contact the institute directly."
3. Others → "I can only answer questions related to courses."

Context:
{context}

Question: {query_text}
"""
    try:
        out = flan_pipe(prompt, max_new_tokens=256, do_sample=False)
        return out[0]["generated_text"].strip()
    except:
        return "Sorry, I could not generate an answer."

# ---------------------------------------------------------
# API: Text → Text
# ---------------------------------------------------------
@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    answer = run_rag(query.question)
    return Answer(answer=answer)

# ---------------------------------------------------------
# API: Audio → WAV (STT → RAG → TTS)
# ---------------------------------------------------------

# -----------------------------
# API: Audio → STT → RAG → TTS
# -----------------------------

@app.post("/chat_audio")
async def chat_audio(request: Request):
    audio_bytes = await request.body()

    # Save temp input WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        tmp_input.write(audio_bytes)
        input_path = tmp_input.name

    try:
        # STT: Audio → Text
        stt_result = stt_model.transcribe(input_path)
        user_text = stt_result["text"].strip()

        # Generate bot response
        greetings = ["hello", "hi", "hey", "greetings"]
        if user_text.lower() in greetings:
            bot_text = "Hello! How can I assist you with courses or admissions?"
        elif "weather" in user_text.lower() or "rain" in user_text.lower():
            bot_text = "I can only provide information about courses or enrollment."
        else:
            bot_text = run_rag(user_text)
            bot_text = speak_friendly(bot_text)

        # TTS: Generate audio
        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tts.tts_to_file(text=bot_text, file_path=tmp_output.name)
        tmp_output.close()

        audio_url = f"/play_audio/{os.path.basename(tmp_output.name)}"

        return {
            "user_text": user_text,
            "bot_text": bot_text,
            "audio_url": audio_url
        }

    finally:
        # Safe cleanup of input file
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.get("/play_audio/{filename}")
def play_audio(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}




# ---------------------------------------------------------
# Run Server
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
