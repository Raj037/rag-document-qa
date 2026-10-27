# 📄 AI Document Q&A — a RAG app

Upload a PDF, ask questions about it in a chat UI, and get answers grounded in
the document. Built with **Retrieval-Augmented Generation (RAG)**.

## What it demonstrates (resume talking points)

- **Full-stack**: Python/FastAPI backend + React/Vite frontend
- **RAG pipeline**: chunking → embeddings → vector search → grounded generation
- **Vector database**: ChromaDB for semantic search
- **Local embeddings**: `sentence-transformers` (free, private — the full document never leaves the backend)
- **Swappable LLM**: runs free by default; set an OpenAI key to enable real answers

## Architecture

```
React UI  ──HTTP──>  FastAPI  ──>  RAG engine
 (upload PDF,                        - read PDF
  ask question)                      - chunk text
                                     - embed (local, free)
                                     - store/search in ChromaDB
                                     - build prompt
                                     - generate (OpenAI optional)
```

## Run it

### 1. Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: enable a real LLM
# cp .env.example .env  and put your key in it, then:
# export OPENAI_API_KEY=sk-...

uvicorn main:app --reload --port 8000
```
Backend runs at http://127.0.0.1:8000

### 2. Frontend (in a second terminal)
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually http://localhost:5173).

## Use it
1. Choose a PDF → **Upload & Index**
2. Type a question → **Ask**
3. See the answer + expandable **Sources** (the retrieved chunks)

Without an OpenAI key, the "answer" shows the retrieved context. With a key, a
real LLM writes the answer using only that context.
