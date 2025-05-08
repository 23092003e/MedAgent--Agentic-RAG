import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from ..config import DATA_PATH, MAX_WORKERS

logger = logging.getLogger(__name__)

class DocumentLoadError(Exception):
    """Custom exception for document loading errors"""
    pass

def load_single_document(file_path: str, loader_cls) -> Optional[List[Document]]:
    """
    Load a single document with error handling
    
    Args:
        file_path: Path to the document
        loader_cls: Document loader class to use
        
    Returns:
        Optional[List[Document]]: List of loaded documents or None if loading fails
    """
    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        logger.info(f"Successfully loaded document: {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {str(e)}")
        return None

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Load documents from data_path in parallel, including PDF and Word (.docx).
    
    Args:
        data_path: Directory containing documents
        
    Returns:
        List[Document]: List of loaded documents
        
    Raises:
        DocumentLoadError: If no documents could be loaded
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise DocumentLoadError(f"Data path does not exist: {data_path}")

    supported_formats: List[Tuple[str, type]] = [
        ("*.pdf", PyPDFLoader),
        ("*.docx", UnstructuredWordDocumentLoader),
    ]
    
    all_documents = []
    
    try:
        # Find all matching files
        files_to_process = []
        for pattern, loader_cls in supported_formats:
            files_to_process.extend([
                (str(f), loader_cls) 
                for f in data_path.glob(pattern)
            ])
            
        if not files_to_process:
            raise DocumentLoadError(f"No supported documents found in {data_path}")
            
        logger.info(f"Found {len(files_to_process)} documents to process")
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(load_single_document, file_path, loader_cls): file_path
                for file_path, loader_cls in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    if docs:
                        all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {str(e)}")
                    
        if not all_documents:
            raise DocumentLoadError("Failed to load any documents successfully")
            
        logger.info(f"Successfully loaded {len(all_documents)} documents")
        return all_documents
        
    except Exception as e:
        error_msg = f"Error during document loading: {str(e)}"
        logger.error(error_msg)
        raise DocumentLoadError(error_msg) from e
