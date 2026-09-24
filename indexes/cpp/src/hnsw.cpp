#include "hnsw.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "distance.hpp"
#include "splitmix.hpp"

namespace vro::hnsw {

namespace {

constexpr std::size_t kStripes = 65536;
constexpr std::size_t kChunk = 64;  // rows claimed per worker step  // mutex stripe over node IDs (parallel build)

struct Cand {
    float score;
    std::int32_t id;
};

// "a is better than b": higher score, ties to the lower ID.
inline bool better(const Cand& a, const Cand& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}
struct BestOnTop {  // max-heap: top is the best candidate
    bool operator()(const Cand& a, const Cand& b) const { return better(b, a); }
};
struct WorstOnTop {  // min-heap: top is the worst result
    bool operator()(const Cand& a, const Cand& b) const { return better(a, b); }
};

// Visited set: node is visited when mark[node] == gen.
struct Visited {
    std::vector<std::uint32_t> mark;
    std::uint32_t gen = 0;
    void reset(std::size_t n) {
        if (mark.size() != n) {
            mark.assign(n, 0);
            gen = 0;
        }
        if (++gen == 0) {
            std::fill(mark.begin(), mark.end(), 0);
            gen = 1;
        }
    }
    bool test_and_set(std::int32_t id) {
        auto& v = mark[static_cast<std::size_t>(id)];
        if (v == gen) return true;
        v = gen;
        return false;
    }
};

// Graph access during build and search. With locks == nullptr (search, or a
// one-thread build) lists are read in place. Otherwise the list is copied
// under the node's stripe lock.
struct Access {
    const Index* ix;
    std::mutex* locks;  // nullptr or kStripes mutexes
    std::vector<std::int32_t> buf;

    std::mutex& lock_of(std::int32_t node) const {
        return locks[static_cast<std::size_t>(node) & (kStripes - 1)];
    }
    const std::int32_t* list(int l, std::int32_t node, std::int32_t& n) {
        if (!locks) return ix->neighbors(l, node, n);
        std::lock_guard<std::mutex> g(lock_of(node));
        const std::int32_t* p = ix->neighbors(l, node, n);
        buf.assign(p, p + n);
        return buf.data();
    }
};

std::size_t slot_row(const Index& ix, int l, std::int32_t node) {
    return l == 0 ? static_cast<std::size_t>(node) : static_cast<std::size_t>(ix.ord1[node]);
}
std::size_t cap(const Index& ix, int l) { return l == 0 ? ix.m0 : ix.m; }

// Greedy descent with ef = 1 on layer l from cur.
Cand greedy(Access& a, const float* q, Cand cur, int l, std::int64_t& dc) {
    const Matrix& v = *a.ix->vectors;
    bool moved = true;
    while (moved) {
        moved = false;
        std::int32_t n;
        const std::int32_t* nb = a.list(l, cur.id, n);
        for (std::int32_t j = 0; j < n; ++j) {
            Cand c{dot(q, v.row(static_cast<std::size_t>(nb[j])), v.dim), nb[j]};
            ++dc;
            if (better(c, cur)) {
                cur = c;
                moved = true;
            }
        }
    }
    return cur;
}

// Algorithm 2. Returns up to ef results, best first.
std::vector<Cand> search_layer(Access& a, const float* q, const std::vector<Cand>& eps,
                               std::size_t ef, int l, Visited& vis, std::int64_t& dc) {
    const Matrix& v = *a.ix->vectors;
    vis.reset(v.rows);
    std::priority_queue<Cand, std::vector<Cand>, BestOnTop> cand;
    std::priority_queue<Cand, std::vector<Cand>, WorstOnTop> res;
    for (const Cand& e : eps) {
        if (vis.test_and_set(e.id)) continue;
        cand.push(e);
        res.push(e);
        if (res.size() > ef) res.pop();
    }
    while (!cand.empty()) {
        Cand c = cand.top();
        if (res.size() >= ef && better(res.top(), c)) break;
        cand.pop();
        std::int32_t n;
        const std::int32_t* nb = a.list(l, c.id, n);
        for (std::int32_t j = 0; j < n; ++j) {
            std::int32_t e = nb[j];
            if (vis.test_and_set(e)) continue;
            Cand ce{dot(q, v.row(static_cast<std::size_t>(e)), v.dim), e};
            ++dc;
            if (res.size() < ef || better(ce, res.top())) {
                cand.push(ce);
                res.push(ce);
                if (res.size() > ef) res.pop();
            }
        }
    }
    std::vector<Cand> out(res.size());
    for (std::size_t i = out.size(); i-- > 0;) {
        out[i] = res.top();
        res.pop();
    }
    return out;
}

// Algorithm 4 with extendCandidates = false, keepPrunedConnections = false.
// cands sorted best first; scores are similarity to the base point.
std::vector<std::int32_t> select_heuristic(const Index& ix, const std::vector<Cand>& cands,
                                           std::size_t limit) {
    const Matrix& v = *ix.vectors;
    std::vector<std::int32_t> out;
    out.reserve(limit);
    for (const Cand& e : cands) {
        if (out.size() >= limit) break;
        const float* xe = v.row(static_cast<std::size_t>(e.id));
        bool keep = true;
        for (std::int32_t r : out) {
            if (dot(xe, v.row(static_cast<std::size_t>(r)), v.dim) > e.score) {
                keep = false;
                break;
            }
        }
        if (keep) out.push_back(e.id);
    }
    return out;
}

// Scores of ids against node, sorted best first, duplicates and node removed.
std::vector<Cand> scored(const Index& ix, std::int32_t node, std::vector<std::int32_t> ids) {
    const Matrix& v = *ix.vectors;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const float* x = v.row(static_cast<std::size_t>(node));
    std::vector<Cand> c;
    c.reserve(ids.size());
    for (std::int32_t id : ids)
        if (id != node) c.push_back({dot(x, v.row(static_cast<std::size_t>(id)), v.dim), id});
    std::sort(c.begin(), c.end(), better);
    return c;
}

// Caller holds node's lock (if any).
void write_list(Index& ix, int l, std::int32_t node, const std::vector<std::int32_t>& nb) {
    std::size_t r = slot_row(ix, l, node);
    std::copy(nb.begin(), nb.end(),
              ix.links[l].begin() + static_cast<std::ptrdiff_t>(r * cap(ix, l)));
    ix.counts[l][r] = static_cast<std::int32_t>(nb.size());
}

// Writes node's own list on layer l, merged with edges that other threads
// added meanwhile. Shrinks to the cap with the heuristic if needed.
void set_own_list(Index& ix, Access& a, int l, std::int32_t node, std::vector<std::int32_t> sel) {
    std::unique_lock<std::mutex> g;
    if (a.locks) g = std::unique_lock<std::mutex>(a.lock_of(node));
    std::int32_t n;
    const std::int32_t* cur = ix.neighbors(l, node, n);
    if (n == 0) {
        write_list(ix, l, node, sel);
        return;
    }
    sel.insert(sel.end(), cur, cur + n);
    std::vector<Cand> c = scored(ix, node, sel);
    if (c.size() <= cap(ix, l)) {
        std::vector<std::int32_t> ids;
        for (const Cand& x : c) ids.push_back(x.id);
        write_list(ix, l, node, ids);
    } else {
        write_list(ix, l, node, select_heuristic(ix, c, cap(ix, l)));
    }
}

// Adds edge from -> to on layer l; shrinks with the heuristic if over the cap.
void add_edge(Index& ix, Access& a, int l, std::int32_t from, std::int32_t to) {
    std::unique_lock<std::mutex> g;
    if (a.locks) g = std::unique_lock<std::mutex>(a.lock_of(from));
    std::size_t r = slot_row(ix, l, from);
    std::size_t slots = cap(ix, l);
    std::int32_t& cnt = ix.counts[l][r];
    std::int32_t* base = ix.links[l].data() + r * slots;
    for (std::int32_t j = 0; j < cnt; ++j)
        if (base[j] == to) return;
    if (static_cast<std::size_t>(cnt) < slots) {
        base[cnt++] = to;
        return;
    }
    std::vector<std::int32_t> ids(base, base + cnt);
    ids.push_back(to);
    write_list(ix, l, from, select_heuristic(ix, scored(ix, from, ids), slots));
}

struct Shared {
    std::mutex ep_mu;  // guards entry and top
};

void insert(Index& ix, Access& a, Shared& sh, std::int32_t i, Visited& vis) {
    const Matrix& v = *ix.vectors;
    const float* q = v.row(static_cast<std::size_t>(i));
    int li = ix.level[static_cast<std::size_t>(i)];

    std::unique_lock<std::mutex> ep_lock;
    if (a.locks) ep_lock = std::unique_lock<std::mutex>(sh.ep_mu);
    std::int32_t entry = ix.entry;
    int top = ix.top;
    // A node that raises the top layer keeps the lock for its whole insert.
    if (ep_lock.owns_lock() && li <= top) ep_lock.unlock();
    if (entry < 0) {
        ix.entry = i;
        ix.top = li;
        return;
    }

    std::int64_t dc = 0;
    Cand cur{dot(q, v.row(static_cast<std::size_t>(entry)), v.dim), entry};
    for (int l = top; l > li; --l) cur = greedy(a, q, cur, l, dc);
    std::vector<Cand> eps{cur};
    for (int l = std::min(li, top); l >= 0; --l) {
        std::vector<Cand> w = search_layer(a, q, eps, ix.ef_construct, l, vis, dc);
        // In a parallel build, other threads can link to i before i reaches
        // this layer, so the search can find i itself.
        w.erase(std::remove_if(w.begin(), w.end(), [i](const Cand& c) { return c.id == i; }),
                w.end());
        if (w.empty()) w.push_back(cur);
        std::vector<std::int32_t> nb = select_heuristic(ix, w, ix.m);  // m on every layer
        set_own_list(ix, a, l, i, nb);
        for (std::int32_t e : nb) add_edge(ix, a, l, e, i);
        eps = std::move(w);
    }
    if (li > top) {
        ix.top = li;
        ix.entry = i;
    }
}

std::vector<std::int32_t> in_degree0(const Index& ix) {
    const std::size_t n = ix.vectors->rows;
    std::vector<std::int32_t> indeg(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        std::int32_t c;
        const std::int32_t* nb = ix.neighbors(0, static_cast<std::int32_t>(i), c);
        for (std::int32_t j = 0; j < c; ++j) ++indeg[static_cast<std::size_t>(nb[j])];
    }
    std::vector<std::int32_t> zero;
    for (std::size_t i = 0; i < n; ++i)
        if (indeg[i] == 0 && static_cast<std::int32_t>(i) != ix.entry)
            zero.push_back(static_cast<std::int32_t>(i));
    return zero;
}

std::vector<char> reachable0(const Index& ix) {
    const std::size_t n = ix.vectors->rows;
    std::vector<char> seen(n, 0);
    std::vector<std::int32_t> queue{ix.entry};
    seen[static_cast<std::size_t>(ix.entry)] = 1;
    for (std::size_t h = 0; h < queue.size(); ++h) {
        std::int32_t c;
        const std::int32_t* nb = ix.neighbors(0, queue[h], c);
        for (std::int32_t j = 0; j < c; ++j)
            if (!seen[static_cast<std::size_t>(nb[j])]) {
                seen[static_cast<std::size_t>(nb[j])] = 1;
                queue.push_back(nb[j]);
            }
    }
    return seen;
}

// Repair edges added so far, per source node. Never pruned by the repair.
using Protected = std::map<std::int32_t, std::vector<std::int32_t>>;

// Adds u -> v on layer 0. If u's list is over the cap, v and every edge that
// the repair added earlier are protected; the heuristic picks the rest.
bool repair_edge(Index& ix, Protected& prot, std::int32_t u, std::int32_t vtx) {
    std::int32_t cu;
    const std::int32_t* nu = ix.neighbors(0, u, cu);
    std::vector<std::int32_t> ids(nu, nu + cu);
    if (std::find(ids.begin(), ids.end(), vtx) != ids.end()) return false;
    std::vector<std::int32_t>& keep = prot[u];
    keep.push_back(vtx);
    if (ids.size() < ix.m0) {
        ids.push_back(vtx);
    } else {
        std::vector<std::int32_t> rest;
        for (std::int32_t x : ids)
            if (std::find(keep.begin(), keep.end(), x) == keep.end()) rest.push_back(x);
        std::size_t room = ix.m0 > keep.size() ? ix.m0 - keep.size() : 0;
        ids = select_heuristic(ix, scored(ix, u, rest), room);
        ids.insert(ids.end(), keep.begin(), keep.end());
    }
    write_list(ix, 0, u, ids);
    ++ix.repair_added;
    return true;
}

// search-layer on layer 0 from the entry point with ef_construct. It visits
// only nodes reachable from the entry point.
std::vector<Cand> search_from_entry(const Index& ix, std::int32_t vtx, Visited& vis) {
    const Matrix& v = *ix.vectors;
    Access a{&ix, nullptr, {}};
    const float* q = v.row(static_cast<std::size_t>(vtx));
    std::int64_t dc = 0;
    Cand ep{dot(q, v.row(static_cast<std::size_t>(ix.entry)), v.dim), ix.entry};
    return search_layer(a, q, {ep}, ix.ef_construct, 0, vis, dc);
}

// Repair pass (CONTRACT 6.6, Parallel build item 1). Sequential, row order.
void repair(Index& ix) {
    Visited vis;
    Protected prot;
    for (int pass = 0; pass < 3; ++pass) {
        ++ix.repair_passes;
        std::vector<std::int32_t> zero = in_degree0(ix);  // step A
        if (pass == 0) ix.zero_in_before_repair = zero.size();
        for (std::int32_t vtx : zero) {
            std::int32_t c;
            const std::int32_t* nb = ix.neighbors(0, vtx, c);
            std::int32_t u = -1;
            if (c > 0) {
                u = scored(ix, vtx, std::vector<std::int32_t>(nb, nb + c)).front().id;
            } else {
                for (const Cand& x : search_from_entry(ix, vtx, vis))
                    if (x.id != vtx) {
                        u = x.id;
                        break;
                    }
            }
            if (u >= 0) repair_edge(ix, prot, u, vtx);
        }
        std::vector<char> seen = reachable0(ix);  // step B
        bool found = false;
        for (std::size_t i = 0; i < seen.size(); ++i) {
            if (seen[i]) continue;
            found = true;
            std::int32_t vtx = static_cast<std::int32_t>(i);
            std::int32_t u = -1, u_free = -1;
            for (const Cand& x : search_from_entry(ix, vtx, vis)) {
                if (x.id == vtx) continue;
                if (u < 0) u = x.id;
                if (static_cast<std::size_t>(ix.counts[0][static_cast<std::size_t>(x.id)]) < ix.m0) {
                    u_free = x.id;
                    break;
                }
            }
            if (u_free >= 0) u = u_free;
            if (u >= 0 && repair_edge(ix, prot, u, vtx)) ++ix.repair_step_b;
        }
        if (!found) break;
    }
}

}  // namespace

std::vector<std::uint8_t> draw_levels(std::size_t n, std::size_t m, std::uint64_t seed) {
    SplitMix64 rng(seed);
    const double ml = 1.0 / std::log(static_cast<double>(m));
    std::vector<std::uint8_t> lv(n);
    for (std::size_t i = 0; i < n; ++i) {
        double u = rng.next_f64();
        double l = u > 0.0 ? std::floor(-std::log(u) * ml) : 255.0;
        lv[i] = static_cast<std::uint8_t>(std::min(l, 255.0));
    }
    return lv;
}

std::size_t unreachable_layer0(const Index& ix) {
    const std::size_t n = ix.vectors->rows;
    if (ix.entry < 0) return n;
    std::vector<char> seen(n, 0);
    std::vector<std::int32_t> queue{ix.entry};
    seen[static_cast<std::size_t>(ix.entry)] = 1;
    for (std::size_t h = 0; h < queue.size(); ++h) {
        std::int32_t c;
        const std::int32_t* nb = ix.neighbors(0, queue[h], c);
        for (std::int32_t j = 0; j < c; ++j)
            if (!seen[static_cast<std::size_t>(nb[j])]) {
                seen[static_cast<std::size_t>(nb[j])] = 1;
                queue.push_back(nb[j]);
            }
    }
    return n - queue.size();
}

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx) {
    long long m = params.get_int("m");
    long long efc = params.get_int("ef_construct");
    if (m < 2) throw ParamError("hnsw: m must be >= 2");
    if (efc < 1) throw ParamError("hnsw: ef_construct must be >= 1");
    if (vectors.rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("hnsw: too many rows for int32 IDs");

    Index ix;
    ix.vectors = &vectors;
    ix.m = static_cast<std::size_t>(m);
    ix.m0 = 2 * ix.m;
    ix.ef_construct = static_cast<std::size_t>(efc);
    ix.threads = std::max(1, ctx.threads);
    const std::size_t n = vectors.rows;

    auto t0 = std::chrono::steady_clock::now();
    ix.level = draw_levels(n, ix.m, ctx.seed);
    int max_level = 0;
    ix.ord1.assign(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        max_level = std::max(max_level, static_cast<int>(ix.level[i]));
        if (ix.level[i] >= 1) ix.ord1[i] = static_cast<std::int32_t>(ix.n_upper++);
    }
    ix.links.resize(static_cast<std::size_t>(max_level) + 1);
    ix.counts.resize(static_cast<std::size_t>(max_level) + 1);
    ix.links[0].assign(n * ix.m0, -1);
    ix.counts[0].assign(n, 0);
    for (int l = 1; l <= max_level; ++l) {
        ix.links[l].assign(ix.n_upper * ix.m, -1);
        ix.counts[l].assign(ix.n_upper, 0);
    }

    Shared sh;
    if (ix.threads == 1 || n < 2) {
        Access a{&ix, nullptr, {}};
        Visited vis;
        for (std::size_t i = 0; i < n; ++i) insert(ix, a, sh, static_cast<std::int32_t>(i), vis);
    } else {
        std::unique_ptr<std::mutex[]> locks(new std::mutex[kStripes]);
        {
            Access a{&ix, locks.get(), {}};
            Visited vis;
            insert(ix, a, sh, 0, vis);  // row 0 sets the first entry point
        }
        std::atomic<std::size_t> next{1};
        std::vector<std::thread> pool;
        for (int t = 0; t < ix.threads; ++t)
            pool.emplace_back([&] {
                Access a{&ix, locks.get(), {}};
                Visited vis;
                // Chunks of 64 consecutive rows keep near-duplicates in order.
                for (std::size_t c0 = next.fetch_add(kChunk); c0 < n; c0 = next.fetch_add(kChunk))
                    for (std::size_t i = c0; i < std::min(n, c0 + kChunk); ++i)
                        insert(ix, a, sh, static_cast<std::int32_t>(i), vis);
            });
        for (auto& th : pool) th.join();
    }
    if (n > 0) {
        ix.unreachable_before_repair = unreachable_layer0(ix);
        repair(ix);
    }
    ix.times.add_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ix;
}

SearchResult search(const Index& ix, const float* query, std::size_t k, const Params& params) {
    SearchResult r;
    r.ids.assign(k, -1);
    r.scores.assign(k, -std::numeric_limits<float>::infinity());
    r.distance_computations = 0;
    if (ix.entry < 0 || k == 0) return r;
    long long ef_p = params.get_int("ef");
    if (ef_p < 1) throw ParamError("hnsw: ef must be >= 1");
    std::size_t ef = std::max(static_cast<std::size_t>(ef_p), k);

    const Matrix& v = *ix.vectors;
    Access a{&ix, nullptr, {}};
    std::int64_t dc = 0;
    Cand cur{dot(query, v.row(static_cast<std::size_t>(ix.entry)), v.dim), ix.entry};
    ++dc;
    for (int l = ix.top; l >= 1; --l) cur = greedy(a, query, cur, l, dc);
    thread_local Visited vis;
    std::vector<Cand> w = search_layer(a, query, {cur}, ef, 0, vis, dc);
    for (std::size_t i = 0; i < k && i < w.size(); ++i) {
        r.ids[i] = w[i].id;
        r.scores[i] = w[i].score;
    }
    r.distance_computations = dc;
    return r;
}

std::size_t index_bytes(const Index& ix) {
    // edges x 4 bytes + per node: level (1 byte) and ord1 (4 bytes) + one 4-byte count per list
    std::size_t edges = 0, counts = 0;
    for (const auto& c : ix.counts) {
        counts += c.size() * sizeof(std::int32_t);
        for (std::int32_t x : c) edges += static_cast<std::size_t>(x);
    }
    return edges * sizeof(std::int32_t) +
           ix.level.size() * (sizeof(std::uint8_t) + sizeof(std::int32_t)) + counts;
}

std::map<std::string, double> extra(const Index& ix) {
    std::map<std::string, double> e;
    e["top_layer"] = ix.top;
    e["entry_point"] = ix.entry;
    e["unreachable_before_repair"] = static_cast<double>(ix.unreachable_before_repair);
    e["zero_in_degree_before_repair"] = static_cast<double>(ix.zero_in_before_repair);
    e["repair_added"] = static_cast<double>(ix.repair_added);
    e["repair_added_unreachable"] = static_cast<double>(ix.repair_step_b);
    e["repair_passes"] = static_cast<double>(ix.repair_passes);
    e["build_threads"] = ix.threads;
    std::vector<std::size_t> per(ix.links.size(), 0);
    for (std::uint8_t lv : ix.level)
        for (std::size_t l = 0; l <= lv && l < per.size(); ++l) ++per[l];
    for (std::size_t l = 0; l < per.size(); ++l)
        e["nodes_layer_" + std::to_string(l)] = static_cast<double>(per[l]);
    return e;
}

}  // namespace vro::hnsw
