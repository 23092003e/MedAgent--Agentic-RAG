# backend/config.py
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config(BaseModel):
    """Configuration with validation"""
    
    # Base paths
    BASE_DIR: Path = Field(
        default=Path(__file__).resolve().parent.parent,
        description="Base directory of the project"
    )
    DATA_PATH: Path = Field(
        default=None,
        description="Path to raw document directory"
    )
    DB_FAISS_PATH: Path = Field(
        default=None,
        description="Path to FAISS vector store"
    )
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for embeddings"
    )
    
    # Chunking parameters
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="Size of text chunks for splitting documents"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks"
    )
    
    # Retrieval parameters
    RETRIEVAL_K: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of documents to retrieve"
    )
    
    # LLM settings
    LLM_MODEL_NAME: str = Field(
        default="medllama2",
        description="Name of the LLM model to use"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM sampling"
    )
    
    # Processing settings
    MAX_WORKERS: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Maximum number of worker threads"
    )
    
    @validator("DATA_PATH", pre=True)
    def validate_data_path(cls, v, values):
        if v is None:
            v = values["BASE_DIR"] / "data/raw"
        return Path(v)
        
    @validator("DB_FAISS_PATH", pre=True)
    def validate_db_path(cls, v, values):
        if v is None:
            v = values["BASE_DIR"] / "vectorstore/database_faiss"
        return Path(v)
        
    class Config:
        validate_assignment = True
        
# Create global config instance
config = Config(
    DATA_PATH=os.getenv("DATA_PATH"),
    DB_FAISS_PATH=os.getenv("DB_FAISS_PATH"),
    EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME"),
    CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", 1000)),
    CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", 200)),
    RETRIEVAL_K=int(os.getenv("RETRIEVAL_K", 3)),
    LLM_MODEL_NAME=os.getenv("LLM_MODEL_NAME", "medllama2"),
    LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", 0.5)),
    MAX_WORKERS=int(os.getenv("MAX_WORKERS", 4))
)

# Export all config variables
DATA_PATH = config.DATA_PATH
DB_FAISS_PATH = config.DB_FAISS_PATH
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
RETRIEVAL_K = config.RETRIEVAL_K
LLM_MODEL_NAME = config.LLM_MODEL_NAME
LLM_TEMPERATURE = config.LLM_TEMPERATURE
MAX_WORKERS = config.MAX_WORKERS