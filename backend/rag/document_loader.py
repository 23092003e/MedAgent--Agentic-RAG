import logging
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from ..config import DATA_PATH

logger = logging.getLogger(__name__)

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Load documents from data_path, including PDF and Word (.docx).
    """
    loaders = [
        ("*.pdf", PyPDFLoader),
        ("*.docx", UnstructuredWordDocumentLoader),
    ]
    documents = []

    for pattern, loader_cls in loaders:
        loader = DirectoryLoader(
            data_path,
            glob=pattern,
            loader_cls=loader_cls
        )
        try:
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from pattern: {pattern}")
        except Exception as e:
            logger.error(f"Failed loading {pattern}: {e}")

    return documents
