import logging
import concurrent.futures
from langchain.document_loaders.base import DocumentLoadError
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from ..config import DATA_PATH, MAX_WORKERS
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Cache for loaded documents
_document_cache: Dict[str, List[Document]] = {}

def get_file_hash(file_path: str) -> str:
    """Get hash of file contents for caching"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_single_document(file_path: str, loader_cls) -> Optional[List[Document]]:
    """Load a single document with caching and optimized processing"""
    try:
        # Check cache first
        file_hash = get_file_hash(file_path)
        if file_hash in _document_cache:
            logger.info(f"Using cached document: {file_path}")
            return _document_cache[file_hash]

        loader = loader_cls(file_path)
        docs = loader.load()
        
        if docs:
            # Add detailed metadata
            file_name = Path(file_path).name
            for i, doc in enumerate(docs):
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                    
                # Basic metadata
                doc.metadata.update({
                    'source': file_name,
                    'file_path': file_path,
                    'date_loaded': datetime.now().isoformat(),
                    'hash': file_hash
                })
                
                # For PDF documents, page numbers should already be included
                # For other documents, we'll add section numbers
                if 'page' not in doc.metadata:
                    doc.metadata['section'] = i + 1
                
                # Add content length and hash for reference
                doc.metadata['content_length'] = len(doc.page_content)
                doc.metadata['content_hash'] = hashlib.md5(
                    doc.page_content.encode()
                ).hexdigest()
            
            # Cache the processed documents
            _document_cache[file_hash] = docs
            
            # Log document details
            doc_count = len(docs)
            total_chars = sum(len(doc.page_content) for doc in docs)
            logger.info(f"Loaded document: {file_path}")
            logger.info(f"Pages/Sections: {doc_count}, Total characters: {total_chars}")
            
            # Log sample content for verification
            if doc_count > 0:
                logger.debug(f"Sample content from first page: {docs[0].page_content[:200]}...")
                
        return docs
        
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {str(e)}")
        return None

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """Load documents with parallel processing and caching"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise DocumentLoadError(f"Data path does not exist: {data_path}")

    logger.info(f"Starting document loading from: {data_path}")
    
    supported_formats: List[Tuple[str, type]] = [
        ("*.pdf", PyPDFLoader),
        ("*.docx", UnstructuredWordDocumentLoader),
    ]
    
    all_documents = []
    
    try:
        # Find all matching files
        files_to_process = []
        for pattern, loader_cls in supported_formats:
            matched_files = list(data_path.glob(pattern))
            logger.info(f"Found {len(matched_files)} {pattern} files")
            files_to_process.extend([
                (str(f), loader_cls) 
                for f in matched_files
            ])
            
        if not files_to_process:
            raise DocumentLoadError(f"No supported documents found in {data_path}")
            
        logger.info(f"Found {len(files_to_process)} total documents to process")
        
        # Process files in parallel with optimized thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(MAX_WORKERS, len(files_to_process))
        ) as executor:
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
                        logger.info(f"Successfully processed: {file_path}")
                    else:
                        logger.warning(f"No documents extracted from: {file_path}")
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {str(e)}")
                    
        if not all_documents:
            raise DocumentLoadError("Failed to load any documents successfully")
            
        # Log final statistics
        total_docs = len(all_documents)
        total_chars = sum(len(doc.page_content) for doc in all_documents)
        logger.info(f"Successfully loaded {total_docs} documents")
        logger.info(f"Total characters across all documents: {total_chars}")
        
        return all_documents
        
    except Exception as e:
        error_msg = f"Error during document loading: {str(e)}"
        logger.error(error_msg)
        raise DocumentLoadError(error_msg) from e
