"""DiskANN / Vamana graph index with on-disk records (CONTRACT 6.7), after Subramanya et al. 2019.

Recall floor (CONTRACT 9, dev set): l=100, recall@10 >= 0.90 (metric=ip and metric=l2).

Build (sequential; `threads` is used only for the independent PQ codebooks)
-------------------------------------------------------------------------------
1. Entry point = the row with the highest dot product with the mean of all rows (medoid).
2. Init: each node gets r random out-edges, next_below(N) from one SplitMix64 seeded `seed`,
   in row order, skipping self and repeats.
3. Two passes over the rows in row order, alpha = 1.0 then `alpha`. Row i: greedy search from
   the entry point with list size l_build -> V = the expanded (visited) nodes;
   robust_prune(i, V, alpha, r); for each out-neighbor j of i, add j -> i and, if j now has
   more than r edges, robust_prune(j, N_out(j), alpha, r).
   The graph uses the full-precision corpus and the dot product. robust_prune needs a
   distance, so it uses d(a, b) = ||a - b||^2 = 2 - 2 a.b (all vectors are unit length).
4. PQ (6.4, m = pq_m, seed + j, no normalization, assignment by `metric`), trained on all
   rows; every row is encoded.
5. File <OUT_PATH>.diskann: one record per node = dim float32, then r int32
   out-edges (-1 = empty), zero-padded to a multiple of 4096 bytes. The corpus array is then
   dropped: the index holds only PQ codes, codebooks, and the entry point.

train_s = PQ codebook training. add_s = medoid + graph + encoding + file write.

Search (beam search)
--------------------
Candidate list of size max(l, k), ordered by PQ table score (higher first, lower ID first on
ties). Each step expands the `beam` best unexpanded candidates: read their records from the
file, score their unseen out-neighbors with the PQ table. Stop when every candidate in the
list is expanded. Then re-score the top `rerank` candidates with the full vectors from their
records (already read, because every listed node was expanded) by the dot product q . x in
both metrics (6.7: the metric changes only the PQ codes and tables). rerank = 0 returns the
PQ scores.

io=mmap: np.memmap over the file. io=nocache: os.pread of each record from a descriptor with
fcntl(F_NOCACHE, 1) on macOS, or O_DIRECT with a 4096-aligned buffer on Linux.

distance_computations = PQ table scores (1 each) + rerank dot products.
search_extra: disk_reads = records read, disk_bytes_read = records read x record size.
index_bytes = PQ codes (N x pq_m) + codebooks (pq_m x 256 x dim/pq_m x 4) + entry point (4).
"""

import heapq
import mmap
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import pq, splitmix

BUILD_PARAMS: dict = {"r": 64, "l_build": 100, "alpha": 1.2, "pq_m": 48, "metric": "ip"}
SEARCH_PARAMS: dict = {"l": 100, "beam": 4, "rerank": 100, "io": "mmap"}

OUT_PATH = None  # bench sets this to --out before build; the file is <out>.diskann
BLOCK = 4096
WRITE_CHUNK = 4096  # records per write call


def medoid(vectors: np.ndarray) -> int:
    mean = vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
    s = vectors @ mean
    return int(np.flatnonzero(s == s.max())[0])


def init_graph(n: int, r: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random out-edges. Returns (edges int32 (N, r + 1), counts int32 (N,)); the extra slot
    holds a back-edge before the prune that brings the node back to r."""
    rng = splitmix.new(seed)
    deg = min(r, n - 1)
    edges = np.full((n, r + 1), -1, dtype=np.int32)
    for i in range(n):
        out: list[int] = []
        seen = {i}
        while len(out) < deg:
            j = splitmix.next_below(rng, n)
            if j not in seen:
                seen.add(j)
                out.append(j)
        edges[i, :deg] = out
    return edges, np.full(n, deg, dtype=np.int32)


class _Builder:
    """Vamana build state: fixed-slot adjacency (N, r + 1) with per-node counts."""

    def __init__(self, vectors, edges, counts, r):
        self.v = vectors
        self.e = edges
        self.d = counts
        self.r = r
        self.mark = np.zeros(len(vectors), dtype=np.int64)
        self.gen = 0

    def greedy(self, q, entry, l):
        """Greedy search with list size l. Returns the expanded nodes (the visited set V).

        Same result as the paper's loop "expand the best unexpanded node of the list of size
        l until none is left": the heap stops when the best open candidate is worse than the
        l-th best seen score."""
        self.gen += 1
        gen, mark, v, e, d = self.gen, self.mark, self.v, self.e, self.d
        push, pop, replace = heapq.heappush, heapq.heappop, heapq.heapreplace
        s0 = float(v[entry] @ q)
        mark[entry] = gen
        cand = [(-s0, entry)]  # max-heap on score
        res = [(s0, -entry)]  # min-heap: worst first (lowest score, then highest id)
        expanded = []
        while cand:
            ns, c = pop(cand)
            if len(res) >= l and -ns < res[0][0]:
                break
            expanded.append(c)
            nb = e[c, : d[c]]
            nb = nb[mark[nb] != gen]
            if not nb.size:
                continue
            mark[nb] = gen
            s = v[nb] @ q
            if len(res) >= l:  # only scores above the current worst can enter the list
                keep = s > res[0][0]
                nb, s = nb[keep], s[keep]
            for x, sx in zip(nb.tolist(), s.tolist()):
                if len(res) < l:
                    push(cand, (-sx, x))
                    push(res, (sx, -x))
                elif sx > res[0][0]:
                    push(cand, (-sx, x))
                    replace(res, (sx, -x))
        return expanded

    def prune(self, p, cands, alpha):
        """RobustPrune(p, cands, alpha, r): writes p's new out-edges."""
        ids = np.unique(np.asarray(cands, dtype=np.int64))  # sorted, so ties go to the lower ID
        ids = ids[ids != p]
        if not ids.size:
            return
        v = self.v
        x = v[ids]
        sp = x @ v[p]
        order = np.argsort(-sp, kind="stable")  # nearest first, lower ID first on ties
        x, ids, sp = x[order], ids[order], sp[order]
        dp = 2.0 - 2.0 * sp  # d(p, c)
        # keep[a, b]: b survives selection of a, i.e. alpha * d(a, b) > d(p, b).
        keep = alpha * (2.0 - 2.0 * (x @ x.T)) > dp[None, :]
        alive = np.ones(len(ids), dtype=bool)
        out = []
        pos = 0
        while len(out) < self.r:
            out.append(pos)
            alive &= keep[pos]
            alive[pos] = False
            pos = int(alive.argmax())
            if not alive[pos]:
                break
        out_ids = ids[out]
        self.e[p, : len(out_ids)] = out_ids
        self.e[p, len(out_ids) :] = -1
        self.d[p] = len(out_ids)

    def pass_(self, entry, l, alpha):
        e, d, r, v = self.e, self.d, self.r, self.v
        for i in range(len(v)):
            visited = self.greedy(v[i], entry, l)
            self.prune(i, visited + e[i, : d[i]].tolist(), alpha)
            for j in e[i, : d[i]].tolist():
                dj = d[j]
                if i in e[j, :dj]:
                    continue
                e[j, dj] = i
                d[j] = dj + 1
                if dj + 1 > r:
                    self.prune(j, e[j, : dj + 1], alpha)


def _disk_path(n: int) -> Path:
    if OUT_PATH is not None:
        return Path(str(OUT_PATH) + ".diskann")  # CONTRACT 6.7: <out>.diskann, as the other languages do
    path = Path(tempfile.gettempdir()) / f"diskann-{os.getpid()}-{n}.diskann"
    print(f"diskann: OUT_PATH is not set, writing {path}", file=sys.stderr)
    return path


def _no_cache(fd: int) -> None:
    import fcntl

    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)


def write_file(path: Path, vectors: np.ndarray, edges: np.ndarray, rec: int) -> int:
    n, dim = vectors.shape
    r = edges.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        _no_cache(f.fileno())  # do not leave warm pages from this process (6.7.1 rule 1)
        for start in range(0, n, WRITE_CHUNK):
            stop = min(n, start + WRITE_CHUNK)
            buf = np.zeros((stop - start, rec), dtype=np.uint8)
            buf[:, : dim * 4] = np.ascontiguousarray(vectors[start:stop], dtype="<f4").view(np.uint8)
            buf[:, dim * 4 : dim * 4 + r * 4] = np.ascontiguousarray(edges[start:stop], dtype="<i4").view(np.uint8)
            f.write(buf.tobytes())
    return path.stat().st_size


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    n, dim = vectors.shape
    r, l_build = int(params["r"]), int(params["l_build"])
    alpha, pq_m, metric = float(params["alpha"]), int(params["pq_m"]), params["metric"]
    if n < 2 or r < 1 or l_build < 1 or alpha < 1.0:
        raise ValueError("diskann: need N >= 2, r >= 1, l_build >= 1, alpha >= 1")
    pq._check({"m": pq_m, "nbits": 8, "metric": metric}, dim)
    sub = dim // pq_m

    # train_s: PQ codebooks on all rows (independent per sub-vector, so threads do not change them).
    t0 = time.perf_counter()
    codebooks = np.empty((pq_m, 256, sub), dtype=np.float32)

    def train_one(j: int) -> None:
        sub_train = np.ascontiguousarray(vectors[:, j * sub : (j + 1) * sub], dtype=np.float32)
        codebooks[j] = pq.train_codebook(sub_train, 20, seed + j, metric)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        list(pool.map(train_one, range(pq_m)))
    train_s = time.perf_counter() - t0

    # add_s: graph, encoding, file.
    t0 = time.perf_counter()
    entry = medoid(vectors)
    b = _Builder(vectors, *init_graph(n, r, seed), r)
    b.pass_(entry, l_build, 1.0)
    b.pass_(entry, l_build, alpha)
    edges = np.ascontiguousarray(b.e[:, :r])
    b = None
    codes = pq.encode(vectors, codebooks, metric)
    rec = -(-(dim * 4 + r * 4) // BLOCK) * BLOCK
    path = _disk_path(n)
    disk_bytes = write_file(path, vectors, edges, rec)
    add_s = time.perf_counter() - t0
    deg = (edges >= 0).sum(axis=1)
    del vectors  # the index keeps no reference to the corpus array

    return {
        "path": str(path),
        "n": n,
        "dim": dim,
        "r": r,
        "rec": rec,
        "entry": entry,
        "codebooks": codebooks,
        "codebook_sq": np.einsum("jcs,jcs->jc", codebooks, codebooks),
        "codes": codes,
        "offsets": np.arange(pq_m, dtype=np.intp) * 256,
        "metric": metric,
        "readers": {},
        "train_s": train_s,
        "add_s": add_s,
        "extra": {
            "disk_bytes": disk_bytes,
            "record_bytes": rec,
            "entry_point": entry,
            "mean_out_degree": float(deg.mean()),
            "min_out_degree": int(deg.min()),
            "max_out_degree": int(deg.max()),
            "build_threads": 1,
        },
    }


class _MmapReader:
    def __init__(self, index):
        n, rec, dim, r = index["n"], index["rec"], index["dim"], index["r"]
        mm = np.memmap(index["path"], dtype="<f4", mode="r", shape=(n, rec // 4))
        self.vec = mm[:, :dim]
        self.edges = mm[:, dim : dim + r].view("<i4")

    def read(self, i):
        return self.vec[i], self.edges[i].tolist()


class _NoCacheReader:
    def __init__(self, index):
        self.dim, self.r, self.rec = index["dim"], index["r"], index["rec"]
        flags = os.O_RDONLY
        self.direct = hasattr(os, "O_DIRECT") and sys.platform.startswith("linux")
        if self.direct:
            flags |= os.O_DIRECT
            self.buf = mmap.mmap(-1, self.rec)  # page-aligned buffer for O_DIRECT
        self.fd = os.open(index["path"], flags)
        if not self.direct:
            _no_cache(self.fd)

    def read(self, i):
        off = i * self.rec
        if self.direct:
            os.preadv(self.fd, [self.buf], off)
            data = bytes(self.buf)
        else:
            data = os.pread(self.fd, self.rec, off)
        vec = np.frombuffer(data, dtype="<f4", count=self.dim)
        edges = np.frombuffer(data, dtype="<i4", count=self.r, offset=self.dim * 4)
        return vec, edges.tolist()


def evict(path: str) -> None:
    """Drop the file's pages from the OS page cache, so io=nocache reads reach the SSD after an
    io=mmap run warmed them (6.7.1 rule 1). F_NOCACHE alone does not bypass pages that are
    already cached. macOS: msync(MS_INVALIDATE) over a shared read-only map (no page is read).
    Linux: posix_fadvise(DONTNEED); O_DIRECT bypasses the cache anyway."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            return
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return
        libc.msync(addr, size, 2)  # MS_INVALIDATE on macOS
        libc.munmap(addr, size)
    finally:
        os.close(fd)


def _reader(index, io):
    if io == "nocache" and index.get("last_io") == "mmap":
        index["readers"].pop("mmap", None)  # close the map; a later mmap run reopens it
        evict(index["path"])
    index["last_io"] = io
    rd = index["readers"].get(io)
    if rd is None:
        if io == "mmap":
            rd = _MmapReader(index)
        elif io == "nocache":
            rd = _NoCacheReader(index)
        else:
            raise ValueError(f"diskann: io={io!r} is not supported, want mmap or nocache")
        index["readers"][io] = rd
    return rd


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    l = max(int(params["l"]), k)
    beam = max(1, int(params["beam"]))
    rerank = int(params["rerank"])
    rd = _reader(index, params["io"])
    codes, offsets = index["codes"], index["offsets"]
    t = pq.table(index, query).ravel()

    entry = index["entry"]
    seen = {entry}
    s0 = float(np.take(t, codes[entry] + offsets).sum(dtype=np.float32))
    cand = [(-s0, entry)]  # sorted: best score first, lower ID first on ties
    expanded: set[int] = set()
    vecs: dict[int, np.ndarray] = {}
    ndist = 1
    reads = 0
    while True:
        batch = []
        for _, c in cand:
            if c not in expanded:
                batch.append(c)
                if len(batch) == beam:
                    break
        if not batch:
            break
        new = []
        for c in batch:
            expanded.add(c)
            vec, nb = rd.read(c)
            reads += 1
            vecs[c] = vec
            for e in nb:
                if e >= 0 and e not in seen:
                    seen.add(e)
                    new.append(e)
        if new:
            s = np.take(t, codes[new] + offsets).sum(axis=1, dtype=np.float32)
            ndist += len(new)
            cand.extend(zip((-s).tolist(), new))
            cand.sort()
            del cand[l:]

    out_ids = np.full(k, -1, dtype=np.int64)
    out_scores = np.full(k, -np.inf, dtype=np.float32)
    if rerank > 0:
        top = [c for _, c in cand[:rerank]]
        for c in top:  # every listed node is expanded, so this reads nothing in practice
            if c not in vecs:
                vecs[c], _ = rd.read(c)
                reads += 1
        x = np.stack([vecs[c] for c in top])
        exact = x @ query  # rerank is the dot product in both metrics (6.7)
        ndist += len(top)
        ranked = sorted(zip((-exact).tolist(), top))[:k]
    else:
        ranked = cand[:k]
    out_ids[: len(ranked)] = [c for _, c in ranked]
    out_scores[: len(ranked)] = [-s for s, _ in ranked]
    index["distance_computations"] = ndist
    index["search_extra"] = {"disk_reads": reads, "disk_bytes_read": reads * index["rec"]}
    return out_ids, out_scores


def index_bytes(index: dict) -> int:
    return int(index["codes"].size + index["codebooks"].size * 4 + 4)
