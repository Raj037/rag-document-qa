"""
=============================================================================
 main.py  -  The web server (FastAPI)
=============================================================================
 Exposes the RAG engine over HTTP so the React frontend can talk to it.

 Endpoints:
   GET  /            -> health check
   POST /upload      -> upload a PDF, index it
   POST /ask         -> ask a question, get an answer + sources

 Run with:  uvicorn main:app --reload
=============================================================================
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag


# -----------------------------------------------------------------------------
# Create the app and allow the frontend (a different port) to call it.
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# a place to temporarily save uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Health check - lets the frontend confirm the backend is alive.
# -----------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "RAG backend is running"}


# -----------------------------------------------------------------------------
# UPLOAD: receive a PDF -> read -> chunk -> embed -> store
# -----------------------------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # give this document a unique id
    doc_id = str(uuid.uuid4())

    # save the uploaded file to disk temporarily
    file_path = os.path.join(UPLOAD_DIR, doc_id + ".pdf")
    with open(file_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # run the indexing pipeline
    text = rag.read_pdf(file_path)
    chunks = rag.chunk_text(text)
    num_chunks = rag.index_document(doc_id, chunks)

    # clean up the temp file
    os.remove(file_path)

    return {
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "message": "Document indexed. You can now ask questions.",
    }


# -----------------------------------------------------------------------------
# ASK: receive a question -> retrieve -> prompt -> generate -> return answer
# -----------------------------------------------------------------------------
class Question(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
def ask(payload: Question):
    result = rag.ask(payload.question, payload.top_k)
    return result
