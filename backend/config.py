# Configuration constants
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-3-flash-preview" 
INDEX_PATH = "faiss_index.index"
META_PATH = "faiss_metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_HISTORY = 6
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHARS = 2500
MIN_CHUNK_WORDS = 20
SUMMARY_KEYWORDS = [
    "what is the document about", "summarize", "overview", "main topic", 
    "explain this paper", "abstract", "what does it cover", "tell me about"
]