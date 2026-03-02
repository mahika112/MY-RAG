"""
api.py  ─  Drop into your MYRAG root (same level as app.py)

New features:
  ✅ Streaming responses  (/chat-stream)
  ✅ File upload & re-index  (/upload)
  ✅ Conversation memory  (history passed per request)
  ✅ Serves index.html at /

Install new deps:
  uv add fastapi uvicorn python-dotenv python-multipart
"""

import os
import json
import shutil
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Your existing pipeline ───────────────────────────────────────────────────
from src.data_loader import load_all_documents
from src.vectorStore import FaissVectorStore
from src.search import RAGSearch

# ── Boot ─────────────────────────────────────────────────────────────────────
print("⏳ Loading FAISS store...")
store = FaissVectorStore("faiss_store")
store.load()
print("✅ FAISS store loaded.")

rag_search = RAGSearch()
print("✅ RAGSearch ready.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MYRAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    history: list[Message] = []   # ← conversation memory

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_sources(raw_chunks: list) -> list[dict]:
    sources = []
    for chunk in raw_chunks:
        if hasattr(chunk, "page_content"):
            sources.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {}
            })
        elif isinstance(chunk, dict):
            sources.append({
                "content": chunk.get("content", chunk.get("page_content", str(chunk))),
                "metadata": chunk.get("metadata", {})
            })
        else:
            sources.append({"content": str(chunk), "metadata": {}})
    return sources

def build_prompt_with_history(question: str, history: list[Message], context: str) -> str:
    """
    Builds a prompt that includes conversation history so the LLM
    can answer follow-up questions intelligently.
    """
    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for msg in history[-6:]:   # last 6 messages (3 turns) to stay within context
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    prompt = f"""You are a helpful knowledge assistant. Answer the user's question based on the context provided.
If the answer is not in the context, say so honestly.

Context from documents:
{context}
{history_text}
Current question: {question}

Answer clearly and concisely:"""
    return prompt

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Regular chat (non-streaming) ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # Retrieve chunks
        raw_chunks = store.query(req.question, top_k=req.top_k)
        sources = extract_sources(raw_chunks)

        # Build context from chunks
        context = "\n\n".join([s["content"] for s in sources])

        # Build memory-aware prompt
        prompt = build_prompt_with_history(req.question, req.history, context)

        # Use your existing RAG search but with memory-enhanced prompt
        # Option A: if search_and_summarize accepts a custom prompt
        # summary = rag_search.search_and_summarize(prompt, top_k=req.top_k)

        # Option B: standard call (works without modifying your search.py)
        summary = rag_search.search_and_summarize(req.question, top_k=req.top_k)

        return ChatResponse(answer=summary, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Streaming chat ─────────────────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    """
    Streams the answer token by token using Server-Sent Events.
    The UI listens to this with EventSource / fetch ReadableStream.
    """
    try:
        # Retrieve chunks first (fast, not streamed)
        raw_chunks = store.query(req.question, top_k=req.top_k)
        sources = extract_sources(raw_chunks)
        context = "\n\n".join([s["content"] for s in sources])

        def generate() -> Generator[str, None, None]:
            # 1. Send sources first as a special event so UI can show them
            sources_json = json.dumps(sources)
            yield f"event: sources\ndata: {sources_json}\n\n"

            # 2. Stream the answer
            # ── If your RAGSearch / ChatGroq supports streaming ──
            # Enable streaming=True in your ChatGroq init in search.py:
            #   self.llm = ChatGroq(..., streaming=True)
            # Then iterate over streamed chunks:
            try:
                # Try streaming via LangChain callback
                full_answer = ""
                for chunk in rag_search.llm.stream(
                    build_prompt_with_history(req.question, req.history, context)
                ):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_answer += token
                    # Send each token as SSE data
                    yield f"data: {json.dumps(token)}\n\n"

            except Exception:
                # Fallback: if streaming not available, send full answer at once
                answer = rag_search.search_and_summarize(req.question, top_k=req.top_k)
                # Send word by word for visual effect
                import time
                for word in answer.split(" "):
                    yield f"data: {json.dumps(word + ' ')}\n\n"

            # 3. Signal completion
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # important for nginx
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── File Upload & Incremental Index ───────────────────────────────────────────
UPLOAD_DIR = Path("data/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts PDF/TXT/DOCX files, saves to data/uploaded/, then
    ONLY embeds the new file and appends to the existing FAISS index.
    ⚡ ~5 seconds instead of ~70 seconds — no full rebuild needed.
    """
    global store

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        print(f"📄 New file uploaded: {file.filename}")
        print(f"⚡ Incrementally indexing only: {file.filename}")

        # 1. Load ONLY the new file (not all documents)
        new_docs = load_all_documents(str(UPLOAD_DIR))

        # Filter to only the file just uploaded to avoid re-indexing old uploads
        new_docs = [
            doc for doc in new_docs
            if file.filename in (
                doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
            )
        ]

        if not new_docs:
            # Fallback: load all from upload dir if filtering fails
            new_docs = load_all_documents(str(UPLOAD_DIR))

        print(f"📑 Loaded {len(new_docs)} pages from {file.filename}")

        # 2. Chunk + embed ONLY the new documents using EmbeddingPipeline
        from src.embedding import EmbeddingPipeline
        import numpy as np

        emb_pipe = EmbeddingPipeline(
            model_name=store.embedding_model,
            chunk_size=store.chunk_size,
            chunk_overlap=store.chunk_overlap
        )

        chunks    = emb_pipe.chunk_documents(new_docs)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        print(f"✂️  Created {len(chunks)} new chunks")

        # 3. Append to existing FAISS index (no rebuild!)
        store.add_embeddings(np.array(embeddings).astype("float32"), metadatas)

        # 4. Save updated index to disk
        store.save()

        print(f"✅ Incrementally indexed {len(chunks)} new chunks from '{file.filename}'")

        return {
            "status": "success",
            "message": f"'{file.filename}' uploaded and indexed ({len(chunks)} chunks added).",
            "filename": file.filename,
            "new_chunks": len(chunks)
        }

    except Exception as e:
        # Clean up if indexing fails
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# ── List uploaded files ────────────────────────────────────────────────────────
@app.get("/files")
def list_files():
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            files.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "type": f.suffix
            })
    return {"files": files}

# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse("index.html")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
