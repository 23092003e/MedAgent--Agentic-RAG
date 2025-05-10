from .exceptions import MedAgentError, VectorStoreError
from .logging_config import setup_logging

class MedAgentError(Exception):
    """Base exception for MedAgent"""
    pass

class VectorStoreError(MedAgentError):
    """Vector store related errors"""
    pass

class DocumentError(MedAgentError):
    """Document processing errors"""
    pass

class ReflectionError(MedAgentError):
    """Self-reflection related errors"""
    pass

class ConfigError(MedAgentError):
    """Configuration related errors"""
    pass