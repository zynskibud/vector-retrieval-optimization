import json

from indexes.python import bench

DEV = "data/processed/dev"
TOP = {"contract_version", "language", "index", "data_dir", "n", "dim", "q", "k", "threads", "seed",
       "build_params", "build", "searches", "machine", "extra"}


def test_flat_json(tmp_path):
    out = tmp_path / "r.json"
    rc = bench.main(["--index", "flat", "--data", DEV, "--out", str(out), "--limit", "20000", "--warmup", "10"])
    assert rc == 0
    r = json.loads(out.read_text())
    assert set(r) == TOP
    assert (r["n"], r["dim"], r["q"], r["k"], r["language"]) == (20000, 384, 1000, 10, "python")
    assert set(r["build"]) == {"train_s", "add_s", "total_s", "peak_rss_mb", "index_bytes"}
    assert set(r["machine"]) == {"os", "arch", "cpu", "cores"}
    (s,) = r["searches"]
    assert set(s) == {"search_params", "ids", "scores", "latency_ms", "total_s", "qps", "distance_computations", "extra"}
    assert len(s["ids"]) == len(s["scores"]) == len(s["latency_ms"]) == 1000
    assert all(len(row) == 10 and all(isinstance(i, int) for i in row) for row in s["ids"])
    assert all(isinstance(x, float) for row in s["scores"] for x in row)
    assert s["distance_computations"] == 20000


def test_usage_errors(tmp_path, capsys):
    out = str(tmp_path / "r.json")
    assert bench.main(["--index", "nope", "--data", DEV, "--out", out]) == 2
    assert bench.main(["--index", "hnsw", "--data", DEV, "--out", out, "--build", "zzz=1"]) == 2
    assert bench.main(["--index", "diskann", "--data", DEV, "--out", out, "--limit", "100"]) == 1
    assert "not implemented" in capsys.readouterr().err
