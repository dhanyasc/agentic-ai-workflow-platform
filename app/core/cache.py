import json
import hashlib
from typing import Any

import redis.asyncio as redis

from app.core.config import get_settings


class RedisCache:
    """Async Redis caching layer for embeddings and LLM responses."""

    def __init__(self):
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self._client = redis.from_url(
            settings.redis_url, decode_responses=True
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> redis.Redis:
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    @staticmethod
    def _hash_key(prefix: str, content: str) -> str:
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{prefix}:{digest}"

    async def get_cached(self, prefix: str, key: str) -> Any | None:
        cache_key = self._hash_key(prefix, key)
        raw = await self.client.get(cache_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_cached(
        self, prefix: str, key: str, value: Any, ttl: int | None = None
    ) -> None:
        settings = get_settings()
        cache_key = self._hash_key(prefix, key)
        await self.client.set(
            cache_key,
            json.dumps(value),
            ex=ttl or settings.cache_ttl,
        )

    async def invalidate(self, prefix: str, key: str) -> None:
        cache_key = self._hash_key(prefix, key)
        await self.client.delete(cache_key)


cache = RedisCache()
