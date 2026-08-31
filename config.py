from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
PDF_DIRECTORY = BASE_DIR / "data" / "pdfs"
PROCESSED_DIRECTORY = BASE_DIR / "data" / "processed"

# RAG configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS storage paths
FAISS_INDEX_PATH = (
    PROCESSED_DIRECTORY / "faiss.index"
)

FAISS_METADATA_PATH = (
    PROCESSED_DIRECTORY / "metadata.pkl"
)

# Llama base url & model
LLAMA_BASE_URL = "http://localhost:9000"
LLAMA_MODEL = "llama3.2:3b"
LLAMA_MAX_TOKEN = 256


def create_directories():
    """Create required directories if they don't exist."""
    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)