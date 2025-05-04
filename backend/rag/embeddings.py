import logging
from ..config import EMBEDDING_MODEL_NAME
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance using the configured model

    Args:
        model_name (str): Name of HF embedding model
    Returns:
        HuggingFaceEmbeddings
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise