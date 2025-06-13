"""
Redis Cache Service for caching processing results and session data.

This service provides Redis-based caching for PDF processing results, field mappings,
and session data to improve performance and reduce redundant AI API calls with
comprehensive error handling and TTL management.
"""

import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
except ImportError:
    redis = None
    RedisError = Exception
    ConnectionError = Exception
    TimeoutError = Exception

from app.models.schemas import ProcessingResult, FieldMapping, ConfidenceScore
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheKey:
    """Cache key patterns for different data types."""
    SESSION_PREFIX = "session"
    PROCESSING_RESULT = "processing_result"
    FIELD_MAPPINGS = "field_mappings"
    EXTRACTION_RESULT = "extraction_result"
    FORM_TEMPLATE = "form_template"
    PDF_HASH = "pdf_hash"
    
    @staticmethod
    def session_key(session_id: str) -> str:
        """Generate session-based cache key."""
        return f"{CacheKey.SESSION_PREFIX}:{session_id}"
    
    @staticmethod
    def processing_key(session_id: str) -> str:
        """Generate processing result cache key."""
        return f"{CacheKey.SESSION_PREFIX}:{session_id}:{CacheKey.PROCESSING_RESULT}"
    
    @staticmethod
    def field_mappings_key(session_id: str) -> str:
        """Generate field mappings cache key."""
        return f"{CacheKey.SESSION_PREFIX}:{session_id}:{CacheKey.FIELD_MAPPINGS}"
    
    @staticmethod
    def extraction_key(session_id: str, file_hash: str) -> str:
        """Generate extraction result cache key."""
        return f"{CacheKey.SESSION_PREFIX}:{session_id}:{CacheKey.EXTRACTION_RESULT}:{file_hash}"
    
    @staticmethod
    def pdf_hash_key(file_path: str) -> str:
        """Generate PDF hash cache key."""
        return f"{CacheKey.PDF_HASH}:{hashlib.md5(file_path.encode()).hexdigest()}"


class CacheService:
    """
    Advanced Redis cache service for PA processing workflows.
    
    Provides high-performance caching for processing results, extracted data,
    field mappings, and session information with automatic TTL management,
    error handling, and cache invalidation strategies.
    """
    
    def __init__(self):
        """Initialize the cache service with Redis connection."""
        self.redis_client = None
        self.connected = False
        self.connection_retries = 3
        self.connection_timeout = 5
        
        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            "session_data": 24 * 60 * 60,        # 24 hours
            "processing_results": 12 * 60 * 60,   # 12 hours
            "extraction_results": 6 * 60 * 60,    # 6 hours
            "field_mappings": 4 * 60 * 60,        # 4 hours
            "form_templates": 7 * 24 * 60 * 60,   # 7 days
            "pdf_hashes": 30 * 24 * 60 * 60       # 30 days
        }
        
        # Initialize connection
        if redis:
            self.connect_redis()
        else:
            logger.warning("Redis not available - caching disabled")
    
    def connect_redis(self) -> bool:
        """
        Establish connection to Redis server with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not redis:
            logger.error("Redis package not installed - caching unavailable")
            return False
        
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            
            # Parse Redis URL and create connection
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=self.connection_timeout,
                socket_connect_timeout=self.connection_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            
            logger.info(f"Successfully connected to Redis at {redis_url}")
            return True
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            self.redis_client = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self.connected = False
            self.redis_client = None
            return False
    
    def cache_result(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
        cache_type: str = "session_data"
    ) -> bool:
        """
        Cache processing result or data with automatic serialization.
        
        Args:
            key: Cache key identifier
            data: Data to cache (will be JSON serialized)
            ttl: Time to live in seconds (optional)
            cache_type: Type of cache for TTL selection
            
        Returns:
            True if caching successful, False otherwise
        """
        if not self.connected or not self.redis_client:
            logger.warning("Redis not connected - skipping cache operation")
            return False
        
        try:
            # Determine TTL
            if ttl is None:
                ttl = self.ttl_settings.get(cache_type, self.ttl_settings["session_data"])
            
            # Serialize data
            if isinstance(data, dict):
                serialized_data = json.dumps(data, default=self._json_serializer)
            elif hasattr(data, 'dict'):
                # Pydantic model
                serialized_data = json.dumps(data.dict(), default=self._json_serializer)
            else:
                # Try direct JSON serialization
                serialized_data = json.dumps(data, default=self._json_serializer)
            
            # Cache with TTL
            success = self.redis_client.setex(key, ttl, serialized_data)
            
            if success:
                logger.debug(f"Cached data with key: {key} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"Failed to cache data with key: {key}")
                return False
                
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Failed to cache result for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error caching result: {e}")
            return False
    
    def get_cached_result(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Retrieve cached result with automatic deserialization.
        
        Args:
            key: Cache key identifier
            default: Default value if key not found
            
        Returns:
            Cached data or default value
        """
        if not self.connected or not self.redis_client:
            logger.debug("Redis not connected - returning default value")
            return default
        
        try:
            # Retrieve from cache
            cached_data = self.redis_client.get(key)
            
            if cached_data is None:
                logger.debug(f"Cache miss for key: {key}")
                return default
            
            # Deserialize data
            try:
                deserialized_data = json.loads(cached_data)
                logger.debug(f"Cache hit for key: {key}")
                return deserialized_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to deserialize cached data for key {key}: {e}")
                # Remove corrupted cache entry
                self.redis_client.delete(key)
                return default
                
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to retrieve cached result for key {key}: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected error retrieving cached result: {e}")
            return default
    
    def invalidate_cache(
        self,
        pattern: Optional[str] = None,
        session_id: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> int:
        """
        Invalidate cached data by pattern, session, or specific keys.
        
        Args:
            pattern: Redis key pattern to match (e.g., "session:*")
            session_id: Session ID to invalidate all related cache
            keys: Specific keys to invalidate
            
        Returns:
            Number of keys deleted
        """
        if not self.connected or not self.redis_client:
            logger.warning("Redis not connected - cannot invalidate cache")
            return 0
        
        try:
            keys_to_delete = []
            
            # Build keys list based on parameters
            if session_id:
                # Invalidate all session-related cache
                session_pattern = f"{CacheKey.session_key(session_id)}*"
                keys_to_delete.extend(self.redis_client.keys(session_pattern))
                
            elif pattern:
                # Use provided pattern
                keys_to_delete.extend(self.redis_client.keys(pattern))
                
            elif keys:
                # Use specific keys
                keys_to_delete.extend(keys)
            
            # Delete keys
            if keys_to_delete:
                deleted_count = self.redis_client.delete(*keys_to_delete)
                logger.info(f"Invalidated {deleted_count} cache entries")
                return deleted_count
            else:
                logger.debug("No cache keys found to invalidate")
                return 0
                
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error invalidating cache: {e}")
            return 0
    
    def cache_processing_result(
        self,
        session_id: str,
        processing_result: ProcessingResult
    ) -> bool:
        """
        Cache complete processing result for a session.
        
        Args:
            session_id: Session identifier
            processing_result: Processing result to cache
            
        Returns:
            True if caching successful
        """
        cache_key = CacheKey.processing_key(session_id)
        return self.cache_result(
            cache_key,
            processing_result,
            cache_type="processing_results"
        )
    
    def get_cached_processing_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached processing result for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Cached processing result or None
        """
        cache_key = CacheKey.processing_key(session_id)
        return self.get_cached_result(cache_key)
    
    def cache_field_mappings(
        self,
        session_id: str,
        field_mappings: Dict[str, FieldMapping]
    ) -> bool:
        """
        Cache field mappings for a session.
        
        Args:
            session_id: Session identifier
            field_mappings: Field mappings to cache
            
        Returns:
            True if caching successful
        """
        cache_key = CacheKey.field_mappings_key(session_id)
        
        # Convert FieldMapping objects to dictionaries
        serializable_mappings = {}
        for field_name, mapping in field_mappings.items():
            if hasattr(mapping, 'dict'):
                serializable_mappings[field_name] = mapping.dict()
            else:
                serializable_mappings[field_name] = mapping
        
        return self.cache_result(
            cache_key,
            serializable_mappings,
            cache_type="field_mappings"
        )
    
    def get_cached_field_mappings(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached field mappings for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Cached field mappings or None
        """
        cache_key = CacheKey.field_mappings_key(session_id)
        return self.get_cached_result(cache_key)
    
    def cache_extraction_result(
        self,
        session_id: str,
        file_path: str,
        extraction_data: Dict[str, Any]
    ) -> bool:
        """
        Cache PDF extraction result with file hash.
        
        Args:
            session_id: Session identifier
            file_path: Path to the PDF file
            extraction_data: Extracted data to cache
            
        Returns:
            True if caching successful
        """
        try:
            # Generate file hash for cache key
            file_hash = self._generate_file_hash(file_path)
            cache_key = CacheKey.extraction_key(session_id, file_hash)
            
            # Include metadata
            cache_data = {
                "extraction_data": extraction_data,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "cached_at": datetime.now(timezone.utc).isoformat()
            }
            
            return self.cache_result(
                cache_key,
                cache_data,
                cache_type="extraction_results"
            )
            
        except Exception as e:
            logger.error(f"Failed to cache extraction result: {e}")
            return False
    
    def get_cached_extraction_result(
        self,
        session_id: str,
        file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached extraction result by file hash.
        
        Args:
            session_id: Session identifier
            file_path: Path to the PDF file
            
        Returns:
            Cached extraction data or None
        """
        try:
            file_hash = self._generate_file_hash(file_path)
            cache_key = CacheKey.extraction_key(session_id, file_hash)
            
            cached_data = self.get_cached_result(cache_key)
            if cached_data:
                return cached_data.get("extraction_data")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached extraction result: {e}")
            return None
    
    def is_connected(self) -> bool:
        """
        Check if Redis connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.connected or not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except (RedisError, ConnectionError, TimeoutError):
            self.connected = False
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.connected or not self.redis_client:
            return {
                "connected": False,
                "error": "Redis not connected"
            }
        
        try:
            info = self.redis_client.info()
            
            # Get key counts by pattern
            session_keys = len(self.redis_client.keys(f"{CacheKey.SESSION_PREFIX}:*"))
            processing_keys = len(self.redis_client.keys(f"*:{CacheKey.PROCESSING_RESULT}"))
            mapping_keys = len(self.redis_client.keys(f"*:{CacheKey.FIELD_MAPPINGS}"))
            
            return {
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "session_keys": session_keys,
                "processing_result_keys": processing_keys,
                "field_mapping_keys": mapping_keys,
                "uptime_seconds": info.get("uptime_in_seconds"),
                "ttl_settings": self.ttl_settings
            }
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries manually.
        
        Returns:
            Number of expired keys cleaned up
        """
        if not self.connected or not self.redis_client:
            return 0
        
        try:
            # This is typically handled automatically by Redis TTL
            # But we can identify and remove any corrupted entries
            all_keys = self.redis_client.keys("*")
            expired_count = 0
            
            for key in all_keys:
                try:
                    # Check if key exists and is valid
                    if self.redis_client.ttl(key) == -2:  # Key doesn't exist
                        expired_count += 1
                except Exception:
                    # Remove any problematic keys
                    self.redis_client.delete(key)
                    expired_count += 1
            
            logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash for file content."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return hashlib.md5(str(file_path).encode()).hexdigest()
            
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate file hash: {e}")
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


# Global cache service instance
cache_service = CacheService()


def get_cache_service() -> CacheService:
    """
    Get the global cache service instance.
    
    Returns:
        CacheService instance for dependency injection
    """
    return cache_service