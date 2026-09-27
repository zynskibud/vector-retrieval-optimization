"""Embedding cache backends (CONTRACT section 14.2): none, lru (in process), redis (shared).

Every backend has get / put / stats. stats() returns hits, misses, evictions, entries, bytes.
bytes counts the stored vector values only (entries x 1,536 B for lru); redis reports its
used_memory, which includes key and allocator overhead.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Protocol

import numpy as np

DIM = 384
VECTOR_BYTES = DIM * 4


class EmbeddingCache(Protocol):
    def get(self, key: str) -> np.ndarray | None: ...
    def put(self, key: str, vector: np.ndarray) -> None: ...
    def stats(self) -> dict: ...


def normalize(text: str) -> str:
    """Lowercase, collapse every run of whitespace to one space, strip."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def key(text: str, model_name: str, model_version: str) -> str:
    return hashlib.sha256(f"{normalize(text)}|{model_name}|{model_version}".encode()).hexdigest()


class NoCache:
    """Every get misses; put stores nothing."""

    def __init__(self):
        self.misses = 0

    def get(self, key: str):
        self.misses += 1
        return None

    def put(self, key: str, vector) -> None:
        pass

    def stats(self) -> dict:
        return {"hits": 0, "misses": self.misses, "evictions": 0, "entries": 0, "bytes": 0}


class LRUCache:
    """In-process LRU on an OrderedDict: the first item is the least recently used."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.data: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key: str):
        v = self.data.get(key)
        if v is None:
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: str, vector) -> None:
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = np.asarray(vector, dtype=np.float32)
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "entries": len(self.data), "bytes": len(self.data) * VECTOR_BYTES}


class RedisCache:
    """Values are the raw float32 bytes (1,536 B). The capacity is the server's maxmemory
    (512 MB, allkeys-lru, docker-compose.yml), not an entry count. evictions is the
    server's evicted_keys since this object was made (INFO stats)."""

    def __init__(self, host: str = "redis", port: int = 6379, ttl: int | None = None, flush: bool = True):
        import redis

        self.r = redis.Redis(host=host, port=port)
        self.ttl = ttl
        if flush:
            self.r.flushdb()
        self.hits = self.misses = 0
        self._evicted0 = self._evicted()

    def _evicted(self) -> int:
        return int(self.r.info("stats")["evicted_keys"])

    def get(self, key: str):
        raw = self.r.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return np.frombuffer(raw, dtype=np.float32)

    def put(self, key: str, vector) -> None:
        self.r.set(key, np.asarray(vector, dtype=np.float32).tobytes(), ex=self.ttl)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "evictions": self._evicted() - self._evicted0,
                "entries": int(self.r.dbsize()), "bytes": int(self.r.info("memory")["used_memory"])}


def make(backend: str, capacity: int = 0, host: str = "redis", ttl: int | None = None):
    if backend == "none":
        return NoCache()
    if backend == "lru":
        return LRUCache(capacity)
    if backend == "redis":
        return RedisCache(host=host, ttl=ttl)
    raise ValueError(f"unknown backend {backend!r}; known: none, lru, redis")
