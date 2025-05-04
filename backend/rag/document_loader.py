# backend/document_loader.py
import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(data_path: str) -> List[Document]:
    """
    Load PDF and DOC files from a specified directory

    Args:
        data_path (str): Path to the directory containing PDF and DOC files

    Returns:
        List[Document]: Loaded PDF and DOC documents
    """
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return []

    pdf_loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    doc_loader = DirectoryLoader(
        data_path,
        glob='*.doc',
        loader_cls=UnstructuredWordDocumentLoader
    )

    documents = []
    try:
        pdf_docs = pdf_loader.load()
        doc_docs = doc_loader.load()
        documents.extend(pdf_docs)
        documents.extend(doc_docs)
        if not documents:
            logger.warning(f"No PDF or DOC documents found in directory: {data_path}")
        else:
            logger.info(f"Loaded {len(documents)} documents (PDF and DOC) from {data_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []
