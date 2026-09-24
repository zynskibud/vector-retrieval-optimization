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
            "latency_ms": [0.1, 0.2, 0.3], "total_s": 0.001, "qps": 3000.0, "distance_computations": 10,
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
