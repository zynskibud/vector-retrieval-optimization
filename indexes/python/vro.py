"""The .vro index file (CONTRACT 15.1): write and read, shared by flat, ivf, hnsw.

Layout, little-endian:
    magic       8 bytes  b"VROIDX01"
    header_len  uint32
    header      ASCII JSON, header_len bytes (index, n, dim, build_params, seed,
                contract_version, language, sections: [{name, dtype, shape, offset, bytes}])
    padding     zero bytes to the next multiple of 64
    sections    raw arrays in table order, each at its offset (a multiple of 64), zero padding between

dtypes: "f32" (float32), "int32", "u8". The reader uses the section table and never assumes offsets.
"""

import json
import struct

import numpy as np

MAGIC = b"VROIDX01"
ALIGN = 64
DTYPES = {"f32": np.dtype("<f4"), "int32": np.dtype("<i4"), "u8": np.dtype("u1")}


class FormatError(ValueError):
    """The file is not a .vro file, or it does not match what the caller expects."""


def _align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def pack_tombstones(dead, n: int) -> np.ndarray:
    """bool (n,) or None -> u8 bit set, ceil(n/8) bytes; bit i (LSB first in a byte) = row i deleted."""
    if dead is None:
        return np.zeros((n + 7) // 8, dtype=np.uint8)
    return np.packbits(np.asarray(dead, dtype=bool)[:n], bitorder="little")


def unpack_tombstones(bits: np.ndarray, n: int):
    """u8 bit set -> bool (n,), or None when no bit is set."""
    dead = np.unpackbits(bits, count=n, bitorder="little").astype(bool)
    return dead if dead.any() else None


def write(path, index: str, n: int, dim: int, build_params: dict, seed: int, sections: list) -> int:
    """Write the file. sections: [(name, dtype_name, array)] in file order. Returns the file size."""
    arrays = []
    for name, dt, arr in sections:
        a = np.ascontiguousarray(arr, dtype=DTYPES[dt])
        arrays.append((name, dt, a))
    header = {"index": index, "n": int(n), "dim": int(dim), "build_params": build_params, "seed": int(seed),
              "contract_version": 1, "language": "python", "sections": []}
    # The offsets depend on the header length, which depends on the offsets: iterate to a fixed point.
    start = 0
    while True:
        table, off = [], _align(8 + 4 + start)
        for name, dt, a in arrays:
            table.append({"name": name, "dtype": dt, "shape": list(a.shape), "offset": off, "bytes": int(a.nbytes)})
            off = _align(off + a.nbytes)
        header["sections"] = table
        text = json.dumps(header, ensure_ascii=True).encode("ascii")
        if len(text) == start:
            break
        start = len(text)
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(text)) + text)
        for entry, (_, _, a) in zip(table, arrays):
            f.write(b"\0" * (entry["offset"] - f.tell()))
            f.write(a.tobytes())
        size = f.tell()
    return size


class Reader:
    """Parsed header plus typed section reads (np.fromfile at the section's offset)."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:8] != MAGIC:
                raise FormatError(f"{self.path}: not a .vro file (bad magic {head[:8]!r})")
            (hlen,) = struct.unpack("<I", head[8:12])
            text = f.read(hlen)
            f.seek(0, 2)
            self.file_bytes = f.tell()
        if len(text) != hlen:
            raise FormatError(f"{self.path}: header is truncated")
        try:
            self.header = json.loads(text.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise FormatError(f"{self.path}: bad header JSON: {e}") from None
        for key in ("index", "n", "dim", "build_params", "seed", "sections"):
            if key not in self.header:
                raise FormatError(f"{self.path}: header has no {key!r}")
        self.sections = {s["name"]: s for s in self.header["sections"]}
        for s in self.header["sections"]:
            if s["dtype"] not in DTYPES:
                raise FormatError(f"{self.path}: section {s['name']}: unknown dtype {s['dtype']!r}")
            count = int(np.prod(s["shape"], dtype=np.int64))
            if s["offset"] % ALIGN or count * DTYPES[s["dtype"]].itemsize != s["bytes"] \
                    or s["offset"] + s["bytes"] > self.file_bytes:
                raise FormatError(f"{self.path}: section {s['name']}: bad offset, shape, or size")

    def check(self, index: str, dim=None, build_params=None) -> None:
        """Refuse a file whose index, dim, or given build_params differ (CONTRACT 15.1)."""
        h = self.header
        if h["index"] != index:
            raise FormatError(f"{self.path}: file holds index {h['index']!r}, expected {index!r}")
        if dim is not None and h["dim"] != dim:
            raise FormatError(f"{self.path}: file has dim {h['dim']}, expected {dim}")
        for key, value in (build_params or {}).items():
            if value is None:
                continue
            if key not in h["build_params"] or h["build_params"][key] != value:
                raise FormatError(f"{self.path}: build param {key}={h['build_params'].get(key)!r} in file, "
                                  f"expected {value!r}")
        vec = self.sections.get("vectors")
        if vec is None or vec["shape"] != [h["n"], h["dim"]]:
            raise FormatError(f"{self.path}: vectors shape does not match n and dim in the header")

    def array(self, name: str, dtype: str, shape=None) -> np.ndarray:
        """Read one section as a new, writable native-endian array."""
        s = self.sections.get(name)
        if s is None:
            raise FormatError(f"{self.path}: no section {name!r}")
        if s["dtype"] != dtype:
            raise FormatError(f"{self.path}: section {name}: dtype {s['dtype']}, expected {dtype}")
        if shape is not None and list(shape) != s["shape"]:
            raise FormatError(f"{self.path}: section {name}: shape {s['shape']}, expected {list(shape)}")
        count = int(np.prod(s["shape"], dtype=np.int64))
        a = np.fromfile(self.path, dtype=DTYPES[dtype], count=count, offset=s["offset"])
        return a.astype(DTYPES[dtype].newbyteorder("="), copy=False).reshape(s["shape"])


def read(path, index: str, dim=None, build_params=None) -> Reader:
    r = Reader(path)
    r.check(index, dim, build_params)
    return r
