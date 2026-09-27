import copy

from tools.bench.schema import validate


def make_doc() -> dict:
    """3 queries, k = 2, n = 10."""
    return {
        "contract_version": 1, "language": "python", "index": "flat", "data_dir": "x",
        "n": 10, "dim": 4, "q": 3, "k": 2, "threads": 1, "seed": 42,
        "build_params": {}, "build": {"train_s": 0.0, "add_s": 0.0, "total_s": 0.0, "peak_rss_mb": 1.5, "index_bytes": 0},
        "searches": [{
            "search_params": {}, "ids": [[0, 1], [2, 3], [4, -1]],
            "scores": [[0.9, 0.8], [0.7, 0.6], [0.5, None]],
            "latency_ms": [0.1, 0.2, 0.3], "total_s": 0.001, "qps": 3000.0, "distance_computations": 10, "extra": {},
        }],
        "machine": {"os": "darwin", "arch": "arm64", "cpu": "Apple M4", "cores": 10}, "extra": {},
    }


def test_valid():
    assert validate(make_doc()) == []


def test_missing_key():
    doc = make_doc()
    del doc["seed"]
    assert validate(doc) == ["top level: missing key 'seed'"]


def test_ids_shape_and_type():
    doc = make_doc()
    doc["searches"][0]["ids"][1] = [2]
    assert validate(doc) == ["searches[0].ids[1]: expected 2 values, got 1"]
    doc = make_doc()
    doc["searches"][0]["ids"][0][0] = 1.5
    assert "searches[0].ids[0][0]" in validate(doc)[0]


def test_ids_out_of_range_and_bool():
    doc = make_doc()
    doc["searches"][0]["ids"][0][0] = 10
    assert len(validate(doc)) == 1
    doc["searches"][0]["ids"][0][0] = True
    assert len(validate(doc)) == 1


def test_latency_length_and_empty_searches():
    doc = make_doc()
    doc["searches"][0]["latency_ms"] = [0.1]
    assert validate(doc) == ["searches[0].latency_ms: expected 3 values, got 1"]
    doc = make_doc()
    doc["searches"] = []
    assert validate(doc) == ["searches: expected at least one search run, got an empty list"]


def test_extra_keys_and_params_types():
    doc = make_doc()
    doc["foo"] = 1
    doc["build_params"] = []
    errs = validate(doc)
    assert any("unknown keys ['foo']" in e for e in errs)
    assert "top level.build_params: expected dict, got list" in errs


def test_machine_and_distance_computations():
    doc = make_doc()
    del doc["machine"]["cores"]
    del doc["searches"][0]["distance_computations"]
    assert len(validate(copy.deepcopy(doc))) == 2


def make_load_doc() -> dict:
    """make_doc with one load run (CONTRACT section 12.1): 5 latencies from 2 clients."""
    doc = make_doc()
    run = doc["searches"][0]
    run["latency_ms"] = [0.1, 0.2, 0.3, 0.2, 0.1]
    run["extra"] = {"errors": 0, "cpu_pct": 150.0, "clients": 2, "duration_s": 1, "queries_done": 5}
    return doc


def test_load_run_valid():
    assert validate(make_load_doc()) == []


def test_load_run_missing_keys():
    doc = make_load_doc()
    del doc["searches"][0]["extra"]["cpu_pct"]
    del doc["searches"][0]["extra"]["errors"]
    assert validate(doc) == ["searches[0].extra: missing key 'errors'", "searches[0].extra: missing key 'cpu_pct'"]


def test_load_run_latency_must_match_queries_done():
    doc = make_load_doc()
    doc["searches"][0]["extra"]["queries_done"] = 6
    assert "queries_done" in validate(doc)[0]


def test_non_load_run_still_needs_q_latencies():
    doc = make_doc()
    doc["searches"][0]["latency_ms"] = [0.1, 0.2, 0.3, 0.4]
    assert validate(doc) == ["searches[0].latency_ms: expected 3 values, got 4"]


def test_phase():
    doc = make_doc()
    doc["searches"][0]["search_params"]["phase"] = "after_inserts"
    assert validate(doc) == []
    doc["searches"][0]["search_params"]["phase"] = "during"
    assert "phase" in validate(doc)[0]


def make_cache_doc() -> dict:
    doc = make_doc()
    doc["language"], doc["index"] = "cache", "hnsw"
    run = doc["searches"][0]
    run["search_params"] = {"backend": "lru", "capacity": 2, "workload": "zipf", "model_version": "v1"}
    run["latency_ms"] = [0.1, 0.2, 0.3, 0.4]
    run["distance_computations"] = None
    run["extra"] = {"requests": 4, "hit_rate": 0.25, "embed_p50_ms": 5.0, "search_p50_ms": 0.1, "e2e_p50_ms": 1.0,
                    "e2e_p99_ms": 6.0, "entries": 2, "cache_bytes": 3072, "evictions": 1, "request_pool_ids": [0, 7, 1]}
    return doc


def test_cache_doc():
    assert validate(make_cache_doc()) == []
    doc = make_cache_doc()
    doc["searches"][0]["latency_ms"].pop()
    assert validate(doc) == ["searches[0].latency_ms: 3 values, but extra.requests = 4"]
    doc = make_cache_doc()
    del doc["searches"][0]["extra"]["hit_rate"]
    assert validate(doc) == ["searches[0].extra: missing key 'hit_rate'"]
    doc = make_cache_doc()
    doc["searches"][0]["search_params"]["invalidate_at"] = 2
    assert validate(doc) == ["searches[0].extra: missing key 'hit_rate_after_invalidate' (search_params has invalidate_at)"]
