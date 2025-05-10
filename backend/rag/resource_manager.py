"""Resource management utilities for MedAgent"""

import logging
from typing import Any, Optional
from contextlib import contextmanager
from threading import Lock
from .exceptions import ResourceError

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages shared resources and ensures proper cleanup"""
    
    def __init__(self):
        self._resources = {}
        self._locks = {}
        self._global_lock = Lock()
        
    def register(self, name: str, resource: Any) -> None:
        """Register a resource with the manager"""
        with self._global_lock:
            if name in self._resources:
                raise ResourceError(f"Resource {name} already registered")
            self._resources[name] = resource
            self._locks[name] = Lock()
            logger.info(f"Registered resource: {name}")
            
    def unregister(self, name: str) -> None:
        """Unregister a resource"""
        with self._global_lock:
            if name not in self._resources:
                raise ResourceError(f"Resource {name} not found")
            del self._resources[name]
            del self._locks[name]
            logger.info(f"Unregistered resource: {name}")
            
    def get(self, name: str) -> Optional[Any]:
        """Get a registered resource"""
        with self._global_lock:
            return self._resources.get(name)
            
    @contextmanager
    def acquire(self, name: str):
        """Acquire a resource lock"""
        if name not in self._locks:
            raise ResourceError(f"Resource {name} not found")
            
        lock = self._locks[name]
        try:
            lock.acquire()
            yield self._resources[name]
        finally:
            lock.release()
            
    def cleanup(self) -> None:
        """Clean up all resources"""
        with self._global_lock:
            for name, resource in self._resources.items():
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up resource {name}: {e}")
            self._resources.clear()
            self._locks.clear()
            logger.info("All resources cleaned up")

# Global resource manager instance
resource_manager = ResourceManager() 