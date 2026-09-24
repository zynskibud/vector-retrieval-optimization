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

Similarity is the dot product; a higher score is better. `distance_computations` counts every
dot product in one search, including each row of a vectorized batch.
"""

import heapq
import math
import time

import numpy as np

from . import splitmix

BUILD_PARAMS: dict = {"m": 16, "ef_construct": 100}
SEARCH_PARAMS: dict = {"ef": 64}


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

    def __init__(self, vectors, levels, m):
        n = len(vectors)
        self.v = vectors
        self.m = m
        self.levels = levels
        # adj[layer][node] -> list of neighbor ids (build time only).
        top = int(levels.max()) if n else 0
        self.adj = [[None] * n for _ in range(top + 1)]
        for i, lv in enumerate(levels.tolist()):
            for layer in range(lv + 1):
                self.adj[layer][i] = []
        self.visited = bytearray(n)
        self.dist = 0
        self.frozen = False

    def freeze(self):
        n, m = len(self.v), self.m
        self.nbr0 = np.full((n, 2 * m), -1, dtype=np.int32)
        self.cnt0 = np.zeros(n, dtype=np.int32)
        self.up_off = np.zeros(n, dtype=np.int32)
        if n:
            self.up_off[1:] = np.cumsum(self.levels[:-1], dtype=np.int64)
        total = int(self.levels.sum())
        self.up = np.full((max(total, 1), m), -1, dtype=np.int32)
        self.up_cnt = np.zeros(max(total, 1), dtype=np.int32)
        for i in range(n):
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
        while cand:
            ns, c = pop(cand)
            if len(res) >= ef and -ns < res[0][0]:
                break
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
        deg = np.zeros(len(self.v), dtype=np.int64)
        for nb in self.adj[0]:
            for e in nb:
                deg[e] += 1
        return deg

    def reachable0(self, entry):
        n = len(self.v)
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

    def _unreachable0(self, entry):
        n = len(self.v)
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
        return [i for i in range(n) if not seen[i]]

    def repair(self, entry, top, ef_c):
        """CONTRACT 6.6 repair pass. Returns (added_in_step_a, added_in_step_b)."""
        v, adj, cap = self.v, self.adj[0], 2 * self.m
        protected: dict[int, set] = {}
        added_a = added_b = 0
        if len(v) == 0:
            return 0, 0
        for _ in range(3):
            # Step A: nodes with zero layer-0 in-degree, in row order, entry point skipped.
            deg = self.in_degree0()
            for x in np.flatnonzero(deg == 0).tolist():
                if x == entry:
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
            unreached = self._unreachable0(entry)
            for x in unreached:
                near = self._nearest_from_entry(x, entry, top, ef_c)
                if not near:
                    continue
                u = next((c for c in near if len(adj[c]) < cap), near[0])
                added_b += self._add_protected(u, x, protected)
            if not unreached:
                break
        return added_a, added_b


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    m = int(params["m"])
    ef_c = int(params["ef_construct"])
    if m < 2 or ef_c < 1:
        raise ValueError("hnsw: need m >= 2 and ef_construct >= 1")
    t0 = time.perf_counter()
    n = len(vectors)
    levels = draw_levels(n, m, seed)
    g = _Graph(vectors, levels, m)
    ep, top = 0, int(levels[0]) if n else 0
    for i in range(1, n):
        g.insert(i, ep, top, ef_c)
        if levels[i] > top:
            ep, top = i, int(levels[i])
    unreachable = n - g.reachable0(ep)
    repair_added, repair_added_unreachable = g.repair(ep, top, ef_c)
    g.freeze()
    add_s = time.perf_counter() - t0
    layer_nodes = [int((levels >= l).sum()) for l in range(top + 1)]
    return {
        "graph": g,
        "entry": ep,
        "top": top,
        "m": m,
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
    ep = index["entry"]
    ep_s = float(g.v[ep] @ query)
    g.dist = 1
    for layer in range(index["top"], 0, -1):
        ep, ep_s = g.greedy(query, ep, ep_s, layer)
    w = g.search_layer(query, [(ep_s, ep)], ef, 0)
    w.sort(key=lambda t: (-t[0], t[1]))
    w = w[:k]
    out_ids[: len(w)] = [i for _, i in w]
    out_scores[: len(w)] = [s for s, _ in w]
    index["distance_computations"] = g.dist
    return out_ids, out_scores


def index_bytes(index: dict) -> int:
    g = index["graph"]
    n = len(g.v)
    total_up = int(g.levels.sum())
    slots = n * 2 * g.m + total_up * g.m
    return 4 * slots + 4 * (n + total_up) + 4 * n + 4 * n
