# backend/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Paths
DATA_PATH = os.getenv("DATA_PATH", "data/raw")
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/database_faiss")

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