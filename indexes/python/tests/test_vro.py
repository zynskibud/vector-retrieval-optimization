"""Phase 7 save and load (CONTRACT 15): the .vro file for flat, ivf, hnsw on 20,000 dev rows.

One build per index is shared (module fixture). A check that deletes rows works on a loaded copy,
so the shared build never changes.
"""

import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import changes, flat, hnsw, ivf, vro
from indexes.python.npy import read_npy

DEV = "data/processed/dev"
N = 20000
K = 10
BUILD = {"flat": {}, "ivf": {"nlist": 256, "train_size": N, "iters": 20}, "hnsw": {"m": 16, "ef_construct": 100}}
PARAMS = {"flat": {"filter": "none"}, "ivf": {"nprobe": 8, "filter": "none"}, "hnsw": {"ef": 64, "filter": "none"}}
MODS = {"flat": flat, "ivf": ivf, "hnsw": hnsw}


def run(mod, index, queries, params):
    out = [mod.search(index, q, K, params) for q in queries]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


@pytest.fixture(scope="module")
def data():
    for mod in MODS.values():
        mod.DATA_DIR = DEV
    return read_npy(f"{DEV}/vectors.npy", N), read_npy(f"{DEV}/queries.npy")


@pytest.fixture(scope="module")
def built(data):
    vectors, _ = data
    return {name: mod.build(vectors, BUILD[name], 1, 42) for name, mod in MODS.items()}


def roundtrip(name, index, tmp_path, tag=""):
    path = tmp_path / f"{name}{tag}.vro"
    size = MODS[name].save(index, path, BUILD[name], 42)
    assert size == path.stat().st_size
    return MODS[name].load(path, BUILD[name], dim=384), path


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_roundtrip(name, data, built, tmp_path):
    _, queries = data
    mod, index = MODS[name], built[name]
    loaded, _ = roundtrip(name, index, tmp_path)
    a_ids, a_s = run(mod, index, queries, PARAMS[name])
    b_ids, b_s = run(mod, loaded, queries, PARAMS[name])
    assert (a_ids == b_ids).all()
    assert (a_s == b_s).all()


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_roundtrip_after_delete(name, data, built, tmp_path):
    """Delete del30 on a loaded copy, save, load again: same IDs and scores, no deleted ID returned.
    Also with filter=top10 on the loaded index."""
    _, queries = data
    mod = MODS[name]
    copy, _ = roundtrip(name, built[name], tmp_path, "-copy")
    dead = changes.delete_mask(DEV, "del30", N)
    copy = mod.delete(copy, dead)
    loaded, _ = roundtrip(name, copy, tmp_path, "-del30")
    assert (loaded["deleted"] == dead).all()
    for params in (PARAMS[name], {**PARAMS[name], "filter": "top10"}):
        a_ids, a_s = run(mod, copy, queries, params)
        b_ids, b_s = run(mod, loaded, queries, params)
        assert (a_ids == b_ids).all() and (a_s == b_s).all(), params
        got = a_ids[a_ids >= 0]
        assert not dead[got].any()


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_roundtrip_after_compact(name, data, built, tmp_path):
    """del30, compact (rebuild), save, load: the file holds all N rows, and the loaded index returns
    the same IDs and scores as the compacted one and no deleted ID."""
    _, queries = data
    mod = MODS[name]
    copy, _ = roundtrip(name, built[name], tmp_path, "-c")
    dead = changes.delete_mask(DEV, "del30", N)
    compacted = mod.compact(mod.delete(copy, dead), "rebuild")
    loaded, path = roundtrip(name, compacted, tmp_path, "-compacted")
    assert loaded["header"]["n"] == N
    assert (loaded["deleted"] == dead).all()
    a_ids, a_s = run(mod, compacted, queries, PARAMS[name])
    b_ids, b_s = run(mod, loaded, queries, PARAMS[name])
    assert (a_ids == b_ids).all() and (a_s == b_s).all()
    assert not dead[b_ids[b_ids >= 0]].any()


def test_hnsw_supports_update_after_load(data, built, tmp_path):
    """A loaded hnsw index accepts an update (Phase 5) and a save of the result loads again."""
    _, queries = data
    loaded, _ = roundtrip("hnsw", built["hnsw"], tmp_path, "-u")
    ids, vecs = changes.update_set(DEV, "upd10", N)
    loaded = hnsw.update(loaded, ids[:200], vecs[:200])
    again, _ = roundtrip("hnsw", loaded, tmp_path, "-u2")
    a, _ = run(hnsw, loaded, queries[:200], PARAMS["hnsw"])
    b, _ = run(hnsw, again, queries[:200], PARAMS["hnsw"])
    assert (a == b).all()
    top1 = np.array([hnsw.search(again, v, 1, PARAMS["hnsw"])[0][0] for v in vecs[:100]])
    assert (top1 == ids[:100]).all()


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_header(name, built, tmp_path):
    path = tmp_path / f"{name}.vro"
    MODS[name].save(built[name], path, BUILD[name], 42)
    raw = path.read_bytes()
    assert raw[:8] == b"VROIDX01"
    (hlen,) = struct.unpack("<I", raw[8:12])
    h = json.loads(raw[12 : 12 + hlen].decode("ascii"))
    assert (h["index"], h["n"], h["dim"], h["seed"], h["contract_version"], h["language"]) == \
        (name, N, 384, 42, 1, "python")
    assert h["build_params"] == BUILD[name]
    want = {"flat": ["vectors", "tombstones"],
            "ivf": ["vectors", "tombstones", "centers", "list_ids", "list_offsets"],
            "hnsw": ["vectors", "tombstones", "levels", "entry", "layer0_slots", "layer0_counts",
                     "upper_slots", "upper_counts", "upper_offsets"]}[name]
    assert [s["name"] for s in h["sections"]] == want
    end = 12 + hlen
    for s in h["sections"]:
        assert s["offset"] % 64 == 0 and s["offset"] >= end
        end = s["offset"] + s["bytes"]
    assert end == len(raw)
    sec = {s["name"]: s for s in h["sections"]}
    assert sec["vectors"]["shape"] == [N, 384] and sec["tombstones"]["shape"] == [(N + 7) // 8]
    if name == "hnsw":
        g = built["hnsw"]["graph"]
        L = int(g.levels.sum())
        assert sec["layer0_slots"]["shape"] == [N, 32] and sec["upper_slots"]["shape"] == [L, 16]
        off = np.frombuffer(raw, "<i4", N + 1, sec["upper_offsets"]["offset"])
        assert off[0] == 0 and off[-1] == L
        slots = np.frombuffer(raw, "<i4", N * 32, sec["layer0_slots"]["offset"]).reshape(N, 32)
        counts = np.frombuffer(raw, "<i4", N, sec["layer0_counts"]["offset"])
        assert ((slots == -1) == (np.arange(32)[None, :] >= counts[:, None])).all()


def test_refusals(built, tmp_path):
    path = tmp_path / "flat.vro"
    flat.save(built["flat"], path, {}, 42)
    raw = bytearray(path.read_bytes())
    (hlen,) = struct.unpack("<I", raw[8:12])
    text = raw[12 : 12 + hlen].decode("ascii")
    assert '"dim": 384' in text
    bad = tmp_path / "dim.vro"
    bad.write_bytes(bytes(raw[:12]) + text.replace('"dim": 384', '"dim": 385').encode() + bytes(raw[12 + hlen :]))
    with pytest.raises(vro.FormatError):
        flat.load(bad)
    with pytest.raises(vro.FormatError):
        flat.load(path, dim=383)
    magic = tmp_path / "magic.vro"
    magic.write_bytes(b"VROIDX02" + bytes(raw[8:]))
    with pytest.raises(vro.FormatError):
        flat.load(magic)
    with pytest.raises(vro.FormatError):
        ivf.load(path)  # wrong index
    ipath = tmp_path / "ivf.vro"
    ivf.save(built["ivf"], ipath, BUILD["ivf"], 42)
    with pytest.raises(vro.FormatError):
        ivf.load(ipath, {"nlist": 128})


def bench(tmp_path, name, tag, *extra):
    out = tmp_path / f"{name}-{tag}.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", name, "--data", DEV, "--out", str(out),
           "--limit", "5000", "--threads", "1", "--warmup", "10", *extra]
    rc = subprocess.run(cmd).returncode
    return rc, (json.loads(out.read_text()) if rc == 0 else None)


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_bench_save_load(name, tmp_path):
    from tools.bench import schema

    path = tmp_path / f"{name}.vro"
    build = ["--build", "nlist=64"] if name == "ivf" else []
    rc, a = bench(tmp_path, name, "w", *build, "--save", str(path))
    assert rc == 0
    rc, b = bench(tmp_path, name, "r", "--load", str(path))
    assert rc == 0
    assert schema.validate(b) == []
    assert a["searches"][0]["ids"] == b["searches"][0]["ids"]
    assert a["searches"][0]["scores"] == b["searches"][0]["scores"]
    assert a["extra"]["file_bytes"] == path.stat().st_size and a["extra"]["save_s"] >= 0
    assert b["build"]["train_s"] == 0 and b["build"]["add_s"] == 0
    assert b["extra"]["loaded_from"] == str(path) and b["extra"]["load_s"] >= 0
    assert b["build_params"] == a["build_params"]
    if name == "ivf":
        rc, _ = bench(tmp_path, name, "x", "--load", str(path), "--build", "nlist=128")
        assert rc == 2
