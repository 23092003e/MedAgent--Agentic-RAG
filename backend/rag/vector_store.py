# backend/ rag/ vector_store.py
import os
from langchain_community.vectorstores import FAISS
from backend.rag.embeddings import get_embedding_model
from backend.rag.document_loader import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def create_vector_store(
    data_path: str,
    db_faiss_path: str
) -> None:
    """
    Create and save FAISS vector store from documents

    Args:
        data_path (str): Directory with raw documents
        db_faiss_path (str): Path to save FAISS index
    """
    # Load documents
    documents = load_documents(data_path)

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embedder = get_embedding_model()

    # Build and save FAISS
    db = FAISS.from_documents(chunks, embedder)
    os.makedirs(db_faiss_path, exist_ok=True)
    db.save_local(db_faiss_path)


def load_vector_store(
    db_faiss_path: str
) -> FAISS:
    """
    Load FAISS vector store from disk

    Args:
        db_faiss_path (str): Path to FAISS index directory

    Returns:
        FAISS instance
    """
    embedder = get_embedding_model()
    return FAISS.load_local(
        db_faiss_path,
        embedder,
        allow_dangerous_deserialization=True
    )