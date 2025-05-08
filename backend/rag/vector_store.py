# backend/ rag/ vector_store.py
import os
import logging
from typing import Optional
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from backend.rag.embeddings import get_embedding_model
from backend.rag.document_loader import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

def create_vector_store(
    data_path: str,
    db_faiss_path: str
) -> None:
    """
    Create and save FAISS vector store from documents

    Args:
        data_path (str): Directory with raw documents
        db_faiss_path (str): Path to save FAISS index
        
    Raises:
        VectorStoreError: If creation or saving fails
    """
    try:
        # Load documents
        documents = load_documents(data_path)
        if not documents:
            raise VectorStoreError("No documents were loaded")

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            raise VectorStoreError("Document splitting produced no chunks")

        # Create embeddings
        embedder = get_embedding_model()

        # Build and save FAISS
        logger.info(f"Creating FAISS index with {len(chunks)} chunks")
        db = FAISS.from_documents(chunks, embedder)
        
        os.makedirs(db_faiss_path, exist_ok=True)
        db.save_local(db_faiss_path)
        logger.info(f"Successfully saved vector store to {db_faiss_path}")
        
    except Exception as e:
        error_msg = f"Failed to create vector store: {str(e)}"
        logger.error(error_msg)
        raise VectorStoreError(error_msg) from e

@lru_cache(maxsize=1)
def load_vector_store(
    db_faiss_path: str
) -> Optional[FAISS]:
    """
    Load FAISS vector store from disk with caching
    
    Args:
        db_faiss_path (str): Path to FAISS index directory
    
    Returns:
        Optional[FAISS]: FAISS instance or None if loading fails
        
    Raises:
        VectorStoreError: If loading fails
    """
    try:
        if not os.path.exists(db_faiss_path):
            raise VectorStoreError(f"Vector store path does not exist: {db_faiss_path}")
            
        embedder = get_embedding_model()
        vectorstore = FAISS.load_local(
            db_faiss_path,
            embedder,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Successfully loaded vector store from {db_faiss_path}")
        return vectorstore
        
    except Exception as e:
        error_msg = f"Failed to load vector store: {str(e)}"
        logger.error(error_msg)
        raise VectorStoreError(error_msg) from e