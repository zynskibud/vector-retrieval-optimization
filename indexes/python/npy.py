"""Hand-written reader for NumPy .npy files, format version 1.0 (CONTRACT 1.1)."""

import ast
import struct
from pathlib import Path

import numpy as np

MAGIC = b"\x93NUMPY"
DTYPES = {"<f4": np.float32, "<i8": np.int64}


def parse_header(raw: bytes, path: Path) -> tuple[np.dtype, tuple[int, ...], int]:
    """Return (dtype, shape, data_offset) from the first bytes of a .npy file."""
    if raw[:6] != MAGIC:
        raise ValueError(f"{path}: not a .npy file (bad magic)")
    if raw[6] != 1 or raw[7] != 0:
        raise ValueError(f"{path}: unsupported .npy version {raw[6]}.{raw[7]}, want 1.0")
    (hlen,) = struct.unpack("<H", raw[8:10])
    text = raw[10 : 10 + hlen].decode("ascii")
    header = ast.literal_eval(text.strip())
    descr = header.get("descr")
    if descr not in DTYPES:
        raise ValueError(f"{path}: unsupported descr {descr!r}, want one of {list(DTYPES)}")
    if header.get("fortran_order") is not False:
        raise ValueError(f"{path}: fortran_order must be False")
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not all(isinstance(s, int) and s >= 0 for s in shape):
        raise ValueError(f"{path}: bad shape {shape!r}")
    return np.dtype(DTYPES[descr]), shape, 10 + hlen


def read_npy(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Read a .npy file into a contiguous C-order array. `limit` keeps only the first rows."""
    path = Path(path)
    with open(path, "rb") as f:
        dtype, shape, offset = parse_header(f.read(65536), path)
        rows = shape[0] if shape else 1
        if limit is not None:
            rows = min(rows, limit)
        row_items = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        count = rows * row_items
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError(f"{path}: file is truncated ({data.size} of {count} items)")
    return np.ascontiguousarray(data.reshape((rows, *shape[1:]) if shape else ()))
