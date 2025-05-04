# backend/rag/embeddings.py

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    """
    Returns a simple HuggingFaceEmbeddings instance.

    Args:
        model_name (str): Name of the HuggingFace model, defaults to MiniLM.

    Returns:
        HuggingFaceEmbeddings: The initialized embedding model
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise
