import re
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS, SIMILARITY_THRESHOLD, INDEX_PATH, META_PATH
from utils import clean_pdf_text

# Global state (initialized in main.py lifespan)
text_splitter = None
embedder = None
faiss_index = None
metadata = []

def initialize_models(embed_model: str):
    """Load text splitter and embedder."""
    global text_splitter, embedder
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], length_function=len, is_separator_regex=False
    )
    embedder = SentenceTransformer(embed_model)

def ingest_pdf(pdf_path: str) -> int:
    """Ingest PDF → chunk → embed → index in FAISS."""
    global faiss_index, metadata
    if embedder is None or text_splitter is None:
        raise RuntimeError("Models not initialized. Call initialize_models() first.")
        
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages_text.append({"page": i + 1, "text": clean_pdf_text(text)})

    chunks = []
    for p in pages_text:
        for chunk in text_splitter.split_text(p["text"]):
            if chunk.strip() and len(chunk.split()) >= MIN_CHUNK_WORDS:
                chunks.append({
                    "page": p["page"],
                    "text": chunk,
                    "word_count": len(chunk.split())
                })

    if not chunks:
        raise ValueError("No readable text in PDF.")

    vectors = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True).astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, INDEX_PATH)
    metadata = chunks
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)
    
    return len(chunks)

def retrieve(query: str, is_summary: bool, k: int):
    """Context-aware retrieval with intent boosting."""
    global faiss_index, metadata, embedder
    if faiss_index is None or not metadata or embedder is None:
        return []
    
    fetch_k = 10 if is_summary else k * 2
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    distances, ids = faiss_index.search(query_vec, fetch_k)
    results, candidates = [], []
    
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1: continue
        txt = metadata[idx]["text"]
        if any(x in txt.lower() for x in ["references", "bibliography", "acknowledgments"]): continue
        if re.match(r'^\s*\[\d+\]', txt.strip()): continue
        if len(txt.split()) < MIN_CHUNK_WORDS: continue

        base_score = float(dist)
        page = metadata[idx]["page"]
        if page <= 3: base_score *= 1.15

        res = {"page": page, "text": txt, "score": base_score, "raw": float(dist)}
        candidates.append(res)
        if base_score >= SIMILARITY_THRESHOLD: results.append(res)

    if not results: results = sorted(candidates, key=lambda x: -x["score"])[:k]
    return sorted(results, key=lambda x: -x["score"])[:k]

def load_index():
    """Load existing FAISS index + metadata if available."""
    global faiss_index, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        return True
    return False