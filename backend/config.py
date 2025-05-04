# backend/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


# Paths
DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw"))
# Convert to absolute path
DATA_PATH = BASE_DIR / DATA_PATH

DB_FAISS_PATH = Path(os.getenv("DB_FAISS_PATH", "vectorstore/database_faiss"))
DB_FAISS_PATH = BASE_DIR / DB_FAISS_PATH

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval parameters
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))

# LLM settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "medllama2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.5))