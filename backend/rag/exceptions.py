"""Custom exceptions for MedAgent"""

from .exceptions import MedAgentError, VectorStoreError
from .logging_config import setup_logging

class MedAgentError(Exception):
    """Base exception for MedAgent"""
    pass

class VectorStoreError(MedAgentError):
    """Vector store related errors"""
    pass

class DocumentError(MedAgentError):
    """Document processing related errors"""
    pass

class ModelError(MedAgentError):
    """ML model related errors"""
    pass

class ConfigError(MedAgentError):
    """Configuration related errors"""
    pass

class ConnectionError(MedAgentError):
    """Connection related errors (e.g. Ollama server)"""
    pass

class ResourceError(MedAgentError):
    """Resource management related errors"""
    pass