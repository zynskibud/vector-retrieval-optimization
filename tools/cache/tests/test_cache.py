import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tools.cache import workload
from tools.cache.bench import replay
from tools.cache.cache import LRUCache, NoCache, RedisCache, key, normalize

DEV = Path("data/processed/dev")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
needs_dev = pytest.mark.skipif(not (DEV / "metadata.parquet").exists(), reason="dev data missing")


def fake_embed(text: str) -> np.ndarray:
    """A deterministic unit vector per text; hit rates do not depend on the model."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_key_normalization_and_version():
    assert normalize("  Hello \n\t WORLD  ") == "hello world"
    a = key("Hello   World", MODEL, "v1")
    assert a == key("  hello world\n", MODEL, "v1")
    assert a != key("Hello World", MODEL, "v2")
    assert a != key("Hello World", "other-model", "v1")
    c = LRUCache(10)
    c.put(a, fake_embed("x"))
    assert c.get(key("HELLO world", MODEL, "v1")) is not None
    assert c.get(key("hello world", MODEL, "v1-2")) is None


def test_lru_evicts_least_recently_used():
    c = LRUCache(2)
    c.put("a", np.zeros(384, np.float32))
    c.put("b", np.ones(384, np.float32))
    assert c.get("a") is not None      # a is now more recent than b
    c.put("c", np.ones(384, np.float32))
    assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None
    s = c.stats()
    assert s == {"hits": 3, "misses": 1, "evictions": 1, "entries": 2, "bytes": 2 * 1536}


def test_nocache_always_misses():
    c = NoCache()
    c.put("a", np.zeros(384, np.float32))
    assert c.get("a") is None and c.stats()["misses"] == 1


def _redis():
    host = os.environ.get("VRO_REDIS_HOST", "redis")
    try:
        import redis

        redis.Redis(host=host, socket_connect_timeout=1).ping()
    except Exception as e:  # no Redis on this network (the bench service has none)
        pytest.skip(f"redis at {host} not reachable: {e}")
    return RedisCache(host=host)


def test_redis_round_trip_bit_exact():
    c = _redis()
    v = fake_embed("round trip")
    v[0] = np.float32(1e-38)  # a subnormal-adjacent value, to catch any float conversion
    c.put("k", v)
    got = c.get("k")
    assert got.dtype == np.float32 and got.tobytes() == v.tobytes()
    assert c.get("missing") is None
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["entries"] == 1 and s["bytes"] > 0


def test_redis_ttl_sets_expiry():
    c = _redis()
    c.ttl = 60
    c.put("t", fake_embed("t"))
    assert 0 < c.r.ttl("t") <= 60


@needs_dev
def test_workload_files(tmp_path):
    out = workload.build(DEV, "zipf", requests=2000, out_dir=tmp_path)
    t = pq.read_table(out)
    assert t.column_names == ["request_id", "pool_id", "text"] and t.num_rows == 2000
    ids = t.column("pool_id").to_numpy()
    assert ids.min() >= 0 and ids.max() < 5000
    queries = pq.read_table(DEV / "query_meta.parquet", columns=["text"]).column("text").to_pylist()
    rows = [(p, s) for p, s in zip(ids, t.column("text").to_pylist()) if p < 1000]
    assert rows and all(queries[p] == s for p, s in rows)
    # same seed, same stream
    again = pq.read_table(workload.build(DEV, "zipf", requests=2000, out_dir=tmp_path / "b"))
    assert again.column("pool_id").to_pylist() == ids.tolist()


@needs_dev
def test_zipf_hit_rate_above_uniform(tmp_path):
    rates = {}
    for name in ("zipf", "uniform"):
        texts = pq.read_table(workload.build(DEV, name, requests=2000, out_dir=tmp_path)).column("text").to_pylist()
        res = replay(texts, LRUCache(500), fake_embed, None, MODEL, "v1")
        rates[name] = res["hit"].mean()
    assert rates["zipf"] > rates["uniform"], rates


def test_invalidation_gives_100_misses():
    texts = [f"text number {i}" for i in range(100)]
    res = replay(texts + texts, LRUCache(1000), fake_embed, None, MODEL, "v1", invalidate_at=100)
    assert not res["hit"][100:200].any()
    same = replay(texts + texts, LRUCache(1000), fake_embed, None, MODEL, "v1")
    assert same["hit"][100:200].all()  # without the version change they all hit
