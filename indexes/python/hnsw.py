"""HNSW graph index (CONTRACT 6.6), after Malkov and Yashunin 2018.

Recall floor (CONTRACT 9, dev set): ef=64, recall@10 >= 0.95.

Build
-----
- Levels: one SplitMix64 PRNG seeded `seed`. For rows 0..N-1 in order, before any insertion,
  u = next_f64() and level = floor(-ln(u) / ln(m)). The entry point is the node with the
  highest level, ties to the lowest row.
- Insertion is sequential, strictly in row order (Algorithm 1), for every `threads` value.
  The Python build ignores threads > 1: the per-node work is small NumPy calls, and a
  parallel build would need per-node locks without a speed gain under the GIL.
- Per layer: search-layer (Algorithm 2) with ef_construct, heuristic neighbor selection of
  m neighbors on every layer (Algorithm 4, extendCandidates=false,
  keepPrunedConnections=false), bidirectional edges, and a heuristic shrink of a neighbor
  list that exceeds its cap (2m on layer 0, m above).
- Repair pass after the build, inside add_s (CONTRACT 6.6). Step A: each node x with no
  layer-0 in-edge (row order, entry point skipped) gets u -> x, u = its nearest out-neighbor
  (or the nearest search-layer result from the entry point if x has no out-edges). Step B:
  each node that a directed BFS from the entry point on layer 0 does not reach gets u -> x,
  u = the nearest search-layer result with a free slot, else the nearest. Repair edges are
  protected: a shrink never prunes them. Repeat while step B finds a node, max 3 passes.
  `extra` reports unreachable_before_repair, repair_added (step A), and
  repair_added_unreachable (step B).

Storage (no per-node Python lists in the final structure)
---------------------------------------------------------
During the build, adjacency lives in per-node Python lists, because appends and element
reads on NumPy arrays cost more Python overhead per call. At the end of the build, `freeze()`
copies it into the fixed-slot int32 arrays below and drops the lists. Search reads only these:
- Layer 0: `nbr0` int32 (N, 2m), `cnt0` int32 (N,).
- Layers >= 1: one block of rows in `up` int32 (sum of levels, m) with `up_cnt` int32
  (sum of levels,). Node i on layer l >= 1 uses row `up_off[i] + l - 1`.
- `levels` int32 (N,), `up_off` int32 (N,).

index_bytes = 4 * (N * 2m + L * m)      # edge slots, layer 0 and upper layers
            + 4 * (N + L)               # per-node, per-layer edge counts
            + 4 * N + 4 * N             # levels and up_off (level bookkeeping)
with L = sum over nodes of level(i) (the number of upper-layer node entries).

Concurrency (CONTRACT 12)
-------------------------
- `build(..., n_build=B)` draws levels for all N rows (same PRNG stream, row order) and builds
  the graph on rows 0..B-1 only. `freeze()` allocates slots for all N rows, so `insert()` can
  add rows B..N-1 later without moving any array. `n` is the published row count.
- `insert(index, ids, vectors)` adds rows in row order into the frozen arrays. A write to a
  node's list takes that node's lock from a stripe of 1024 threading.Lock objects. The slots
  are written first and the count last. A shrink rewrites the slots and then the count; slots
  past the new count keep old (valid) ids, so a reader with an older count still reads node
  ids. The entry point and top layer change together, as one tuple, under a global lock.
  The new row's own lists are complete before any edge to it is added.
- Search takes no locks. It reads a count, then a slot slice, with `.tolist()`. A torn value
  cannot happen: an int32 element is copied as one aligned 4-byte load, and the whole slice
  copy runs while the thread holds the GIL, so no other Python thread writes in between.
- Per-search state (visited bytes, distance counter, expanded counter) is thread-local.
- What the GIL serializes: every Python-level step (heap pushes and pops, list work, the
  visited-byte loop, `.tolist()`) runs in one thread at a time. NumPy releases the GIL in the
  gather `v[nb]` and the dot product, so only those parts of different searches overlap.
  With 1-thread BLAS and ~16-32 rows per product, they are a small part of a search, so
  more threads add little QPS.
- After inserts, `repair(index)` copies the arrays back to lists, runs the repair pass on
  all published rows, and freezes again. Call it with no searches running.

Updates and deletes (CONTRACT 13.3)
-----------------------------------
- delete(index, mask): a tombstone bool array in index["deleted"]. Search uses the filtered
  search-layer with allow = not deleted (and the filter, if any): a tombstoned node is still
  scored, enters the candidate heap, and is expanded, but never enters the result heap
  (hnswlib markDelete). Greedy descent on the upper layers ignores tombstones.
- update(index, ids, vectors): on a copy of the corpus array (made on the first update), the
  batch runs in two steps. Step 1: overwrite the vectors, drop every updated node's out-edges on
  every layer, and remove the updated nodes from all other lists (one scan over all lists).
  Step 2: re-insert the updated nodes in row order with their existing level (the insert
  procedure of the build). Then the repair pass. This is the per-node "delete + insert" of the
  contract, done as one batch of deletes followed by one batch of inserts, so the list scan
  runs once and not K times. If the entry point is updated, the re-inserts start from a
  temporary entry point (the live node with the highest level, ties to the lowest row) until
  the entry point itself is re-inserted; the entry point and all levels stay unchanged. In
  that case, layers above the temporary entry point's level keep no edges (only updated nodes
  live there), which greedy descent passes through.
- compact(index, "rebuild"): build a new graph from the live rows (vectors copied in row
  order) with the same m, ef_construct, and seed; levels are redrawn for the live rows in row
  order. index["id_map"] maps node IDs to row IDs; search returns row IDs.
- compact(index, "repair"): in place. For each live node whose list on a layer holds a deleted
  node, the new candidates are its live neighbors plus the live neighbors of each deleted
  neighbor (the deleted node's list on that layer); the heuristic (Algorithm 4) selects up to
  the cap (2m on layer 0, m above). Then the deleted nodes' own lists are cleared. If the entry
  point was deleted, the new entry point is the live node with the highest level (ties to the
  lowest row). Then the repair pass (steps A and B) over live nodes only. The deleted nodes keep
  their slots but have no edges and no in-edges, so no search reaches them; the tombstone
  array is dropped. index_bytes is unchanged apart from the dropped bit set.
- index_bytes adds the tombstone bit set (N/8) and, after a rebuild, the int64 ID map.

Similarity is the dot product; a higher score is better. `distance_computations` counts every
dot product in one search, including each row of a vectorized batch.
"""

import heapq
import math
import threading
import time

import numpy as np

from . import changes, filters, splitmix, vro

BUILD_PARAMS: dict = {"m": 16, "ef_construct": 100}
SEARCH_PARAMS: dict = {"ef": 64, "filter": "none"}
# bench adds phase="after_inserts" to the last run of an insert load run (CONTRACT 12.2).
# search() ignores it; it is a label only, so it is not in SEARCH_PARAMS (defaults unchanged).
PHASES = ("none", "after_inserts")
N_STRIPES = 1024
DATA_DIR = None  # set by bench; filter_<name>.npy lives here (CONTRACT 11)


def draw_levels(n: int, m: int, seed: int) -> np.ndarray:
    rng = splitmix.new(seed)
    ml = 1.0 / math.log(m)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        u = splitmix.next_f64(rng)
        out[i] = int(math.floor(-math.log(u) * ml)) if u > 0.0 else 64
    return out


class _Graph:
    """Graph state. During the build, adjacency is held in Python lists (fast appends and
    element reads). `freeze()` copies it into the fixed-slot int32 arrays that search uses."""

    def __init__(self, vectors, levels, m, n_build=None):
        # vectors and levels cover all N rows (the capacity); rows >= n are not yet inserted.
        n = len(vectors) if n_build is None else n_build
        self.v = vectors
        self.n = n
        self.m = m
        self.levels = levels
        # adj[layer][node] -> list of neighbor ids (build time only).
        top = int(levels[:n].max()) if n else 0
        self.adj = [[None] * n for _ in range(top + 1)]
        for i, lv in enumerate(levels[:n].tolist()):
            for layer in range(lv + 1):
                self.adj[layer][i] = []
        self._tls = threading.local()
        self.locks = [threading.Lock() for _ in range(N_STRIPES)]
        self.frozen = False

    # Per-thread search state: concurrent searches must not share these.
    @property
    def visited(self):
        t = self._tls
        vis = getattr(t, "visited", None)
        if vis is None:
            vis = t.visited = bytearray(len(self.v))
        return vis

    @property
    def dist(self):
        return getattr(self._tls, "dist", 0)

    @dist.setter
    def dist(self, value):
        self._tls.dist = value

    @property
    def expanded(self):
        return getattr(self._tls, "expanded", 0)

    @expanded.setter
    def expanded(self, value):
        self._tls.expanded = value

    def freeze(self):
        """Copy the lists into fixed-slot arrays sized for all len(self.v) rows (capacity)."""
        n, m = len(self.v), self.m
        self.nbr0 = np.full((n, 2 * m), -1, dtype=np.int32)
        self.cnt0 = np.zeros(n, dtype=np.int32)
        self.up_off = np.zeros(n, dtype=np.int32)
        if n:
            self.up_off[1:] = np.cumsum(self.levels[:-1], dtype=np.int64)
        total = int(self.levels.sum())
        self.up = np.full((max(total, 1), m), -1, dtype=np.int32)
        self.up_cnt = np.zeros(max(total, 1), dtype=np.int32)
        for i in range(self.n):
            nb = self.adj[0][i]
            self.nbr0[i, : len(nb)] = nb
            self.cnt0[i] = len(nb)
            for layer in range(1, int(self.levels[i]) + 1):
                r = self.up_off[i] + layer - 1
                nb = self.adj[layer][i]
                self.up[r, : len(nb)] = nb
                self.up_cnt[r] = len(nb)
        self.adj = None
        self.frozen = True

    def thaw(self):
        """Copy the frozen arrays back into build-time lists for rows 0..n-1."""
        n = self.n
        top = int(self.levels[:n].max()) if n else 0
        self.adj = [[None] * n for _ in range(top + 1)]
        for i in range(n):
            self.adj[0][i] = self.nbr0[i, : self.cnt0[i]].tolist()
            for layer in range(1, int(self.levels[i]) + 1):
                r = self.up_off[i] + layer - 1
                self.adj[layer][i] = self.up[r, : self.up_cnt[r]].tolist()
        self.frozen = False

    def _write(self, node, layer, ids):
        """Frozen insert: write node's list on layer, slots first, count last. Caller holds the lock."""
        if layer == 0:
            self.nbr0[node, : len(ids)] = ids
            self.cnt0[node] = len(ids)
        else:
            r = self.up_off[node] + layer - 1
            self.up[r, : len(ids)] = ids
            self.up_cnt[r] = len(ids)

    def insert_frozen(self, i, ep, top, ef_c):
        """Insert row i into the frozen arrays while searches run (CONTRACT 12.2)."""
        v = self.v
        q = v[i]
        lvl = int(self.levels[i])
        ep_s = float(v[ep] @ q)
        for layer in range(top, lvl, -1):
            ep, ep_s = self.greedy(q, ep, ep_s, layer)
        eps = [(ep_s, ep)]
        for layer in range(min(lvl, top), -1, -1):
            w = self.search_layer(q, eps, ef_c, layer)
            limit = 2 * self.m if layer == 0 else self.m
            chosen = self.select(w, self.m)
            with self.locks[i % N_STRIPES]:
                self._write(i, layer, chosen)
            for e in chosen:
                with self.locks[e % N_STRIPES]:
                    nb = self.neighbors(e, layer)
                    if i in nb:
                        continue
                    if len(nb) < limit:
                        self._write(e, layer, nb + [i])
                    else:
                        cands = nb + [i]
                        s = (v[cands] @ v[e]).tolist()
                        self._write(e, layer, self.select(list(zip(s, cands)), limit))
            eps = w

    def neighbors(self, node, layer):
        """Neighbor ids of node on layer, as a Python list."""
        if not self.frozen:
            return self.adj[layer][node]
        if layer == 0:
            return self.nbr0[node, : self.cnt0[node]].tolist()
        r = self.up_off[node] + layer - 1
        return self.up[r, : self.up_cnt[r]].tolist()

    def greedy(self, q, ep, ep_s, layer):
        """Greedy descent on one layer (search-layer with ef = 1)."""
        v = self.v
        changed = True
        while changed:
            changed = False
            nb = self.neighbors(ep, layer)
            if not nb:
                break
            s = v[nb] @ q
            self.dist += len(nb)
            j = int(np.argmax(s))
            if s[j] > ep_s:
                ep, ep_s = nb[j], float(s[j])
                changed = True
        return ep, ep_s

    def search_layer(self, q, eps, ef, layer):
        """Algorithm 2. eps: list of (score, id). Returns a list of (score, id), size <= ef."""
        visited = self.visited
        touched = []
        v = self.v
        push, pop = heapq.heappush, heapq.heappop
        cand = []  # max-heap on score: (-score, id)
        res = []  # min-heap: (score, -id); the worst (lowest score, then highest id) pops first
        for s, i in eps:
            if not visited[i]:
                visited[i] = 1
                touched.append(i)
            cand.append((-s, i))
            res.append((s, -i))
        heapq.heapify(cand)
        heapq.heapify(res)
        while len(res) > ef:
            pop(res)
        expanded = 0
        while cand:
            ns, c = pop(cand)
            if len(res) >= ef and -ns < res[0][0]:
                break
            expanded += 1
            nb = [e for e in self.neighbors(c, layer) if not visited[e]]
            if not nb:
                continue
            for e in nb:
                visited[e] = 1
            touched.extend(nb)
            s = (v[nb] @ q).tolist()
            self.dist += len(nb)
            for e, se in zip(nb, s):
                if len(res) < ef:
                    push(cand, (-se, e))
                    push(res, (se, -e))
                elif se > res[0][0]:
                    push(cand, (-se, e))
                    heapq.heapreplace(res, (se, -e))
        for i in touched:
            visited[i] = 0
        self.expanded = expanded
        return [(s, -ni) for s, ni in res]

    def search_layer_filtered(self, q, eps, ef, layer, allow):
        """Search-layer with a filter (CONTRACT 11.3). allow: bool (N,).

        A node enters the result heap only if allow[node]. Every scored node that is better than
        the worst result (or any node while the result heap is not full) enters the candidate heap
        and is expanded later, so the walk can cross failing regions. The stop rule is unchanged.
        Sets self.expanded to the number of nodes popped and expanded.
        """
        visited = self.visited
        touched = []
        v = self.v
        push, pop = heapq.heappush, heapq.heappop
        cand = []
        res = []
        for s, i in eps:
            if not visited[i]:
                visited[i] = 1
                touched.append(i)
            cand.append((-s, i))
            if allow[i]:
                res.append((s, -i))
        heapq.heapify(cand)
        heapq.heapify(res)
        while len(res) > ef:
            pop(res)
        expanded = 0
        while cand:
            ns, c = pop(cand)
            if len(res) >= ef and -ns < res[0][0]:
                break
            expanded += 1
            nb = [e for e in self.neighbors(c, layer) if not visited[e]]
            if not nb:
                continue
            for e in nb:
                visited[e] = 1
            touched.extend(nb)
            s = (v[nb] @ q).tolist()
            self.dist += len(nb)
            for e, se in zip(nb, s):
                full = len(res) >= ef
                if full and se <= res[0][0]:
                    continue
                push(cand, (-se, e))
                if allow[e]:
                    if full:
                        heapq.heapreplace(res, (se, -e))
                    else:
                        push(res, (se, -e))
        for i in touched:
            visited[i] = 0
        self.expanded = expanded
        return [(s, -ni) for s, ni in res]

    def select(self, scored, limit, keep=None):
        """Algorithm 4 heuristic. scored: list of (score_to_base, id). Returns ids, best first.

        `keep` (repair pass): a set of protected ids. They are selected first (best first) and
        never pruned; the other candidates must also be closer to the base than to them."""
        scored = sorted(scored, key=lambda t: (-t[0], t[1]))
        if keep:
            scored = [(float("inf"), i) for _, i in scored if i in keep] + [
                t for t in scored if t[1] not in keep
            ]
        if len(scored) <= 1:
            return [i for _, i in scored]
        ids = [i for _, i in scored]
        vs = self.v[ids]
        gram = (vs @ vs.T).tolist()
        out = []
        sel = []
        for p, (s, i) in enumerate(scored):
            if len(out) >= limit:
                break
            # Keep e only if it is closer to the base than to every selected neighbor.
            row = gram[p]
            for r in sel:
                if row[r] >= s:
                    break
            else:
                out.append(i)
                sel.append(p)
        return out

    def insert(self, i, ep, top, ef_c):
        v = self.v
        q = v[i]
        lvl = int(self.levels[i])
        ep_s = float(v[ep] @ q)
        for layer in range(top, lvl, -1):
            ep, ep_s = self.greedy(q, ep, ep_s, layer)
        eps = [(ep_s, ep)]
        for layer in range(min(lvl, top), -1, -1):
            w = self.search_layer(q, eps, ef_c, layer)
            limit = 2 * self.m if layer == 0 else self.m
            chosen = self.select(w, self.m)  # the new node selects m on every layer
            adj = self.adj[layer]
            adj[i] = chosen
            for e in chosen:
                nb = adj[e]
                if len(nb) < limit:
                    nb.append(i)
                else:
                    cands = nb + [i]
                    s = (v[cands] @ v[e]).tolist()
                    adj[e] = self.select(list(zip(s, cands)), limit)
            eps = w

    def in_degree0(self):
        deg = np.zeros(self.n, dtype=np.int64)
        for nb in self.adj[0]:
            for e in nb:
                deg[e] += 1
        return deg

    def reachable0(self, entry):
        n = self.n
        seen = bytearray(n)
        if n == 0:
            return 0
        seen[entry] = 1
        stack = [entry]
        count = 1
        adj = self.adj[0]
        while stack:
            c = stack.pop()
            for e in adj[c]:
                if not seen[e]:
                    seen[e] = 1
                    count += 1
                    stack.append(e)
        return count

    def _nearest_from_entry(self, x, entry, top, ef_c):
        """search-layer on layer 0 from the entry point (greedy descent above), without x.
        Returns ids nearest first."""
        v = self.v
        q = v[x]
        ep, ep_s = entry, float(v[entry] @ q)
        for layer in range(top, 0, -1):
            ep, ep_s = self.greedy(q, ep, ep_s, layer)
        w = [t for t in self.search_layer(q, [(ep_s, ep)], ef_c, 0) if t[1] != x]
        w.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in w]

    def _add_protected(self, u, x, protected):
        """Add u -> x on layer 0 and protect it. If u exceeds its cap, shrink it with the
        heuristic; x and every edge the repair added earlier are never pruned."""
        adj, v, cap = self.adj[0], self.v, 2 * self.m
        if x in adj[u]:
            protected.setdefault(u, set()).add(x)
            return False
        protected.setdefault(u, set()).add(x)
        nb = adj[u]
        if len(nb) < cap:
            nb.append(x)
        else:
            cs = nb + [x]
            sc = (v[cs] @ v[u]).tolist()
            adj[u] = self.select(list(zip(sc, cs)), cap, keep=protected[u])
        return True

    def _unreachable0(self, entry, dead=None):
        n = self.n
        seen = bytearray(n)
        seen[entry] = 1
        stack = [entry]
        adj = self.adj[0]
        while stack:
            c = stack.pop()
            for e in adj[c]:
                if not seen[e]:
                    seen[e] = 1
                    stack.append(e)
        if dead is None:
            return [i for i in range(n) if not seen[i]]
        return [i for i in range(n) if not seen[i] and not dead[i]]

    def repair(self, entry, top, ef_c, dead=None):
        """CONTRACT 6.6 repair pass. Returns (added_in_step_a, added_in_step_b).
        dead: optional bool (n,); these nodes are skipped (compact repair mode)."""
        v, adj, cap = self.v, self.adj[0], 2 * self.m
        protected: dict[int, set] = {}
        added_a = added_b = 0
        if self.n == 0:
            return 0, 0
        for _ in range(3):
            # Step A: nodes with zero layer-0 in-degree, in row order, entry point skipped.
            deg = self.in_degree0()
            for x in np.flatnonzero(deg == 0).tolist():
                if x == entry or (dead is not None and dead[x]):
                    continue
                if adj[x]:
                    cands = adj[x]
                    s = (v[cands] @ v[x]).tolist()
                    u = max(zip(s, cands), key=lambda t: (t[0], -t[1]))[1]
                else:
                    near = self._nearest_from_entry(x, entry, top, ef_c)
                    if not near:
                        continue
                    u = near[0]
                added_a += self._add_protected(u, x, protected)
            # Step B: nodes not reached by a directed BFS from the entry point.
            unreached = self._unreachable0(entry, dead)
            for x in unreached:
                near = self._nearest_from_entry(x, entry, top, ef_c)
                if not near:
                    continue
                u = next((c for c in near if len(adj[c]) < cap), near[0])
                added_b += self._add_protected(u, x, protected)
            if not unreached:
                break
        return added_a, added_b


def _pick_entry(levels: np.ndarray, live: np.ndarray) -> tuple[int, int]:
    """The live node with the highest level, ties to the lowest row."""
    cand = np.flatnonzero(live)
    lv = levels[cand]
    best = int(cand[np.flatnonzero(lv == lv.max())[0]])
    return best, int(levels[best])


def _update_nodes(g: _Graph, ids: list, ep: int, top: int, ef_c: int) -> None:
    """Thawed graph: step 1 and step 2 of update (module docstring)."""
    pend = set(ids)
    for adj in g.adj:
        for node, nb in enumerate(adj):
            if nb is None:
                continue
            if node in pend:
                adj[node] = []
            elif any(e in pend for e in nb):
                adj[node] = [e for e in nb if e not in pend]
    if ep in pend:
        live = np.ones(g.n, dtype=bool)
        live[list(pend)] = False
        start, s_top = _pick_entry(g.levels[: g.n], live)
    else:
        start, s_top = ep, top
    for i in sorted(pend):
        g.insert(i, start, s_top, ef_c)
        if i == ep:
            start, s_top = ep, top


def _refill(g: _Graph, dead: np.ndarray) -> None:
    """Thawed graph: repair-mode compaction of the lists (module docstring)."""
    v = g.v
    for layer, adj in enumerate(g.adj):
        cap = 2 * g.m if layer == 0 else g.m
        for x, nb in enumerate(adj):
            if nb is None or dead[x] or not any(dead[e] for e in nb):
                continue
            cands = [e for e in nb if not dead[e]]
            seen = set(cands)
            seen.add(x)
            for d in nb:
                if dead[d]:
                    for e in adj[d]:
                        if not dead[e] and e not in seen:
                            seen.add(e)
                            cands.append(e)
            if cands:
                s = (v[cands] @ v[x]).tolist()
                adj[x] = g.select(list(zip(s, cands)), cap)
            else:
                adj[x] = []
    for adj in g.adj:
        for x in np.flatnonzero(dead).tolist():
            if adj[x] is not None:
                adj[x] = []


def build(vectors: np.ndarray, params: dict, threads: int, seed: int, n_build: int | None = None) -> dict:
    """Build on rows 0..n_build-1 (default: all). Levels and slots cover all len(vectors) rows."""
    m = int(params["m"])
    ef_c = int(params["ef_construct"])
    if m < 2 or ef_c < 1:
        raise ValueError("hnsw: need m >= 2 and ef_construct >= 1")
    t0 = time.perf_counter()
    levels = draw_levels(len(vectors), m, seed)
    n = len(vectors) if n_build is None else int(n_build)
    g = _Graph(vectors, levels, m, n)
    ep, top = 0, int(levels[0]) if n else 0
    for i in range(1, n):
        g.insert(i, ep, top, ef_c)
        if levels[i] > top:
            ep, top = i, int(levels[i])
    unreachable = n - g.reachable0(ep)
    repair_added, repair_added_unreachable = g.repair(ep, top, ef_c)
    g.freeze()
    add_s = time.perf_counter() - t0
    layer_nodes = [int((levels[:n] >= l).sum()) for l in range(top + 1)]
    return {
        "graph": g,
        "entry": ep,
        "top": top,
        "ept": (ep, top),  # read by search as one object, so entry and top always match
        "ef_construct": ef_c,
        "global_lock": threading.Lock(),
        "m": m,
        "seed": seed,
        "n_orig": len(vectors),
        "train_s": 0.0,
        "add_s": add_s,
        "extra": {
            "top_layer": top,
            "entry_point": ep,
            "nodes_per_layer": layer_nodes,
            "unreachable_before_repair": unreachable,
            "repair_added": repair_added,
            "repair_added_unreachable": repair_added_unreachable,
            "build_threads": 1,
        },
    }


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    g = index["graph"]
    ef = max(int(params["ef"]), k)
    g.dist = 0
    out_ids = np.full(k, -1, dtype=np.int64)
    out_scores = np.full(k, -np.inf, dtype=np.float32)
    if len(g.v) == 0:
        index["distance_computations"] = 0
        return out_ids, out_scores
    ep, top = index["ept"]
    ep_s = float(g.v[ep] @ query)
    g.dist = 1
    for layer in range(top, 0, -1):
        ep, ep_s = g.greedy(query, ep, ep_s, layer)
    # The upper-layer descent above ignores the filter and tombstones; only layer 0 applies them.
    mask = _allow(index, str(params.get("filter", "none")))
    if mask is None:
        w = g.search_layer(query, [(ep_s, ep)], ef, 0)
        index["search_extra"] = {"filter_rows": len(g.v), "visited": g.expanded}
    else:
        w = g.search_layer_filtered(query, [(ep_s, ep)], ef, 0, mask)
        index["search_extra"] = {"filter_rows": filters.count(mask), "visited": g.expanded}
    w.sort(key=lambda t: (-t[0], t[1]))
    w = w[:k]
    out_ids[: len(w)] = [i for _, i in w]
    id_map = index.get("id_map")
    if id_map is not None and len(w):
        out_ids[: len(w)] = id_map[out_ids[: len(w)]]
    out_scores[: len(w)] = [s for s, _ in w]
    index["distance_computations"] = g.dist
    return out_ids, out_scores


def _allow(index: dict, name: str):
    """bool (n,) of nodes allowed into the result heap, or None (all). Cached per filter."""
    dead = index.get("deleted")
    id_map = index.get("id_map")
    if name == "none" and dead is None:
        return None
    cache = index.setdefault("allow", {})
    if name not in cache:
        n = len(index["graph"].v)
        allow = np.ones(n, dtype=bool)
        if name != "none":
            fm = filters.mask(DATA_DIR, name, index.get("n_orig", n))
            allow &= fm if id_map is None else fm[id_map]
        if dead is not None:
            allow &= ~dead
        cache[name] = allow
    return cache[name]


def index_bytes(index: dict) -> int:
    g = index["graph"]
    n = len(g.v)
    total_up = int(g.levels.sum())
    slots = n * 2 * g.m + total_up * g.m
    total = 4 * slots + 4 * (n + total_up) + 4 * n + 4 * n
    if index.get("deleted") is not None:
        total += changes.tombstone_bytes(len(index["deleted"]))
    if index.get("id_map") is not None:
        total += int(index["id_map"].nbytes)
    return total


def clone(index: dict) -> dict:
    """Deep copy of a frozen index (graph arrays, vectors, tombstones). For tests."""
    g = index["graph"]
    h = _Graph.__new__(_Graph)
    h.v, h.n, h.m, h.levels = g.v.copy(), g.n, g.m, g.levels.copy()
    for a in ("nbr0", "cnt0", "up_off", "up", "up_cnt"):
        setattr(h, a, getattr(g, a).copy())
    h.adj, h.frozen = None, True
    h._tls = threading.local()
    h.locks = [threading.Lock() for _ in range(N_STRIPES)]
    out = {k: v for k, v in index.items() if k not in ("graph", "allow", "global_lock")}
    for k in ("deleted", "id_map"):
        if out.get(k) is not None:
            out[k] = out[k].copy()
    out["graph"], out["global_lock"], out["extra"] = h, threading.Lock(), dict(index.get("extra", {}))
    return out


def delete(index: dict, mask: np.ndarray) -> dict:
    """Tombstone the rows in mask (bool, one per node)."""
    index["deleted"] = np.asarray(mask, dtype=bool).copy()
    index.pop("allow", None)
    return index


def update(index: dict, ids, vectors) -> dict:
    """Replace the vectors of nodes `ids` (module docstring). No searches may run."""
    g = index["graph"]
    if index.get("id_map") is not None:
        raise ValueError("hnsw.update: index is compacted; update before compaction")
    ids = [int(i) for i in ids]
    if not index.get("owned"):
        g.v = g.v.copy()
        index["owned"] = True
    g.v[ids] = vectors
    g.thaw()
    ep, top = index["ept"]
    _update_nodes(g, ids, ep, top, index["ef_construct"])
    added = g.repair(ep, top, index["ef_construct"])
    g.freeze()
    index["extra"]["update_repair_added"] = int(added[0])
    index["extra"]["update_repair_added_unreachable"] = int(added[1])
    index.pop("allow", None)
    return index


def compact(index: dict, mode: str = "rebuild") -> dict:
    """mode "rebuild" returns a new index; mode "repair" changes this one (module docstring)."""
    g = index["graph"]
    dead = index.get("deleted")
    if mode == "rebuild":
        live = np.arange(g.n, dtype=np.int64) if dead is None else np.flatnonzero(~dead[: g.n]).astype(np.int64)
        new = build(np.ascontiguousarray(g.v[live]), {"m": index["m"], "ef_construct": index["ef_construct"]},
                    1, index["seed"])
        old_map = index.get("id_map")
        new["id_map"] = live if old_map is None else old_map[live]
        new["n_orig"] = index.get("n_orig", len(g.v))
        new["owned"] = True
        return new
    if mode != "repair":
        raise ValueError(f"unknown compact mode {mode!r}")
    if dead is None:
        return index
    dead = dead[: g.n]
    g.thaw()
    _refill(g, dead)
    ep, top = index["ept"]
    if dead[ep]:
        ep, top = _pick_entry(g.levels[: g.n], ~dead)
    added = g.repair(ep, top, index["ef_construct"], dead=dead)
    g.freeze()
    index["ept"] = (ep, top)
    index["entry"], index["top"] = ep, top
    index["extra"]["compact_repair_added"] = int(added[0])
    index["extra"]["compact_repair_added_unreachable"] = int(added[1])
    index.pop("deleted", None)
    index.pop("allow", None)
    return index


def insert(index: dict, ids, vectors=None) -> None:
    """Add rows `ids` (the next row indices, in order) while searches may run (CONTRACT 12.2).

    The vectors are already in the capacity array given to build(); `vectors`, if given, must
    equal those rows. Only one thread may call insert() at a time.
    """
    g = index["graph"]
    ids = [int(i) for i in ids]
    if vectors is not None and not np.array_equal(np.asarray(vectors), g.v[ids]):
        raise ValueError("hnsw.insert: vectors differ from the rows given to build")
    ef_c = index["ef_construct"]
    for i in ids:
        if i != g.n or i >= len(g.v):
            raise ValueError(f"hnsw.insert: next row must be {g.n}, got {i}")
        ep, top = index["ept"]
        g.insert_frozen(i, ep, top, ef_c)
        lvl = int(g.levels[i])
        if lvl > top:
            with index["global_lock"]:
                index["ept"] = (i, lvl)
                index["entry"], index["top"] = i, lvl
        g.n = i + 1  # publish the row count last


def repair(index: dict) -> tuple[int, int]:
    """Run the repair pass on all published rows. No searches or inserts may run."""
    g = index["graph"]
    g.thaw()
    ep, top = index["ept"]
    added = g.repair(ep, top, index["ef_construct"])
    g.freeze()
    return added


def _masked(slots: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Copy of slots with every slot at or past the node's count set to -1 (CONTRACT 15.1)."""
    out = slots.copy()
    out[np.arange(slots.shape[1])[None, :] >= counts[:, None]] = -1
    return out


def save(index: dict, path, build_params: dict | None = None, seed: int | None = None) -> int:
    """Write the .vro file (CONTRACT 15.1). The in-memory layout maps to the file as:
    nbr0 -> layer0_slots, cnt0 -> layer0_counts, up[:L] -> upper_slots, up_cnt[:L] -> upper_counts,
    up_off + [L] -> upper_offsets (N + 1), levels (int32) -> levels (u8), ept[0] -> entry.
    Slots past a node's count are written as -1.

    A rebuilt (compacted) index has an ID map from node to row. The file always holds all N original
    rows in row order (CONTRACT 15.1), so save expands it: node j becomes row id_map[j] (edges are
    mapped the same way), and a dropped row gets a zero vector, level 0, no edges, and its tombstone
    bit. id_map ascends, so the upper-layer blocks keep their order."""
    g = index["graph"]
    n = len(g.v)
    if g.n != n or not g.frozen:
        raise ValueError("hnsw.save: the index has rows that are not inserted yet")
    if n and int(g.levels.max()) > 255:
        raise ValueError("hnsw.save: a level does not fit in u8")
    m = g.m
    L = int(g.levels.sum())
    v, levels, dead, ep = g.v, g.levels, index.get("deleted"), int(index["ept"][0])
    slots0, cnt0 = _masked(g.nbr0, g.cnt0), g.cnt0
    up_slots, up_cnt = _masked(g.up[:L], g.up_cnt[:L]), g.up_cnt[:L]
    id_map = index.get("id_map")
    if id_map is not None:
        if (np.diff(id_map) <= 0).any():
            raise ValueError("hnsw.save: the ID map must ascend")
        n_full = int(index["n_orig"])
        lut = np.append(id_map, -1).astype(np.int32)  # lut[-1] == -1 keeps empty slots empty

        def remap(slots):
            return lut[slots]

        v = np.zeros((n_full, g.v.shape[1]), dtype=np.float32)
        v[id_map] = g.v
        levels = np.zeros(n_full, dtype=np.int32)
        levels[id_map] = g.levels
        full_dead = np.ones(n_full, dtype=bool)
        full_dead[id_map] = False if dead is None else dead
        dead = full_dead
        s0 = np.full((n_full, 2 * m), -1, dtype=np.int32)
        s0[id_map] = remap(slots0)
        c0 = np.zeros(n_full, dtype=np.int32)
        c0[id_map] = cnt0
        slots0, cnt0 = s0, c0
        up_slots = remap(up_slots)
        ep = int(id_map[ep])
        n = n_full
    up_off = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(levels, out=up_off[1:])
    bp = {"m": m, "ef_construct": index["ef_construct"]} if build_params is None else build_params
    return vro.write(path, "hnsw", n, v.shape[1], bp, index.get("seed", 0) if seed is None else seed, [
        ("vectors", "f32", v),
        ("tombstones", "u8", vro.pack_tombstones(dead, n)),
        ("levels", "u8", levels),
        ("entry", "int32", np.array([ep], dtype=np.int32)),
        ("layer0_slots", "int32", slots0),
        ("layer0_counts", "int32", cnt0),
        ("upper_slots", "int32", up_slots),
        ("upper_counts", "int32", up_cnt),
        ("upper_offsets", "int32", up_off),
    ])


def load(path, params: dict | None = None, dim: int | None = None) -> dict:
    """Read a .vro file written by any language. Refuses a wrong index, dim, or build_params.
    The loaded index is complete: no rebuild and no repair."""
    r = vro.read(path, "hnsw", dim, params)
    h = r.header
    n, d = h["n"], h["dim"]
    m = int(h["build_params"]["m"])
    ef_c = int(h["build_params"]["ef_construct"])
    levels = r.array("levels", "u8", (n,)).astype(np.int32)
    L = int(levels.sum())
    up_off = r.array("upper_offsets", "int32", (n + 1,))
    if n and (up_off[0] != 0 or up_off[n] != L or (np.diff(up_off) != levels).any()):
        raise vro.FormatError(f"{path}: upper_offsets do not match the levels")
    g = _Graph.__new__(_Graph)
    g.v, g.n, g.m, g.levels = r.array("vectors", "f32", (n, d)), n, m, levels
    g.nbr0 = r.array("layer0_slots", "int32", (n, 2 * m))
    g.cnt0 = r.array("layer0_counts", "int32", (n,))
    g.up_off = np.ascontiguousarray(up_off[:n])
    g.up = np.full((max(L, 1), m), -1, dtype=np.int32)
    g.up_cnt = np.zeros(max(L, 1), dtype=np.int32)
    g.up[:L] = r.array("upper_slots", "int32", (L, m))
    g.up_cnt[:L] = r.array("upper_counts", "int32", (L,))
    if (g.cnt0 < 0).any() or (g.cnt0 > 2 * m).any() or (g.up_cnt < 0).any() or (g.up_cnt > m).any():
        raise vro.FormatError(f"{path}: an edge count is out of range")
    g.adj, g.frozen = None, True
    g._tls = threading.local()
    g.locks = [threading.Lock() for _ in range(N_STRIPES)]
    ep = int(r.array("entry", "int32", (1,))[0]) if n else 0
    top = int(levels[ep]) if n else 0
    index = {
        "graph": g,
        "entry": ep,
        "top": top,
        "ept": (ep, top),
        "ef_construct": ef_c,
        "global_lock": threading.Lock(),
        "m": m,
        "seed": int(h["seed"]),
        "n_orig": n,
        "train_s": 0.0,
        "add_s": 0.0,
        "extra": {"top_layer": top, "entry_point": ep,
                  "nodes_per_layer": [int((levels >= l).sum()) for l in range(top + 1)]},
        "header": h,
    }
    dead = vro.unpack_tombstones(r.array("tombstones", "u8", ((n + 7) // 8,)), n)
    if dead is not None:
        index["deleted"] = dead
    return index
