"""
=============================================================================
 rag.py  -  The RAG engine (the brain of the app)
=============================================================================
 This is your complete_rag.py logic, grown up for a real app:
   - still: chunk -> embed -> store -> retrieve -> prompt -> generate
   - upgraded: a real vector database (ChromaDB) instead of a Python list
   - upgraded: reads real PDFs
   - upgraded: optional real LLM (OpenAI) with a free fallback

 Read it top to bottom. Each function matches one RAG step you already know.
=============================================================================
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader


# -----------------------------------------------------------------------------
# GLOBAL SETUP (runs once when the server starts)
# -----------------------------------------------------------------------------

# 1) The embedding model - same free local model you already know.
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) The vector database - ChromaDB. Think of it as our "chunk_vectors" list
#    from before, but a real database that stores vectors + their text and
#    does the similarity search for us.
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documents")

# 3) Optional OpenAI key. If present, we use a real LLM for the answer.
#    If absent, we fall back to showing the retrieved context (still free).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# -----------------------------------------------------------------------------
# STEP 1 - READ A PDF into plain text
# -----------------------------------------------------------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text = text + page_text + "\n"

    return text


# -----------------------------------------------------------------------------
# STEP 2 - CHUNK the text into overlapping pieces
#    Real documents need size-based chunks (not just lines). We take ~800
#    characters at a time, with 150 characters of overlap so a sentence that
#    straddles a boundary isn't lost.
# -----------------------------------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        # move forward, but step back by "overlap" so chunks share some text
        start = start + chunk_size - overlap

    return chunks


# -----------------------------------------------------------------------------
# STEP 3 - EMBED + STORE the chunks in the vector database
#    This is the INDEXING phase. Done once per uploaded document.
# -----------------------------------------------------------------------------
def index_document(doc_id, chunks):
    # embed every chunk into a vector (locally, free)
    vectors = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk).tolist()
        vectors.append(vector)

    # each stored item needs a unique id
    ids = []
    for i in range(len(chunks)):
        ids.append(doc_id + "_chunk_" + str(i))

    # store the vectors + their original text in ChromaDB
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
    )

    return len(chunks)


# -----------------------------------------------------------------------------
# STEP 4 + 5 - RETRIEVE the top-k most relevant chunks for a question
#    ChromaDB does the cosine-similarity search for us (the same math you
#    wrote by hand, just optimized).
# -----------------------------------------------------------------------------
def retrieve(question, top_k=3):
    question_vector = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
    )

    # results["documents"] is a list-of-lists; we asked one query so take [0]
    retrieved_chunks = results["documents"][0]
    return retrieved_chunks


# -----------------------------------------------------------------------------
# STEP 6 - BUILD THE PROMPT from the retrieved chunks + the question
# -----------------------------------------------------------------------------
def build_prompt(question, retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context = context + "- " + chunk + "\n\n"

    prompt = "Use ONLY the context below to answer the question. "
    prompt = prompt + "If the answer is not in the context, say you don't know.\n\n"
    prompt = prompt + "Context:\n" + context + "\n"
    prompt = prompt + "Question: " + question + "\n\nAnswer:"

    return prompt


# -----------------------------------------------------------------------------
# STEP 7 - GENERATE the final answer
#    If an OpenAI key is set, use a real LLM. Otherwise fall back to showing
#    the retrieved context so the app still works for free.
# -----------------------------------------------------------------------------
def generate_answer(prompt, retrieved_chunks):
    if OPENAI_API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # free fallback: no LLM, so we return the retrieved context directly
    fallback = "(No OpenAI key set - showing the retrieved context instead of an LLM answer.)\n\n"
    for chunk in retrieved_chunks:
        fallback = fallback + "- " + chunk + "\n\n"
    return fallback


# -----------------------------------------------------------------------------
# THE FULL "ASK" FLOW - ties steps 4-7 together
# -----------------------------------------------------------------------------
def ask(question, top_k=3):
    retrieved_chunks = retrieve(question, top_k)
    prompt = build_prompt(question, retrieved_chunks)
    answer = generate_answer(prompt, retrieved_chunks)

    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }
