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

// Graph access during build and search. The list is copied into buf: under
// the node's stripe lock if locks != nullptr (parallel build, insert), else
// with atomic loads and no lock (search, one-thread build). Without the lock,
// a concurrent insert can make the copy shorter or longer than the final
// list, but every slot value is whole (CONTRACT 12.2); -1 slots are skipped.
struct Access {
    const Index* ix;
    std::mutex* locks;  // nullptr or kStripes mutexes
    std::vector<std::int32_t> buf;

    std::mutex& lock_of(std::int32_t node) const {
        return locks[static_cast<std::size_t>(node) & (kStripes - 1)];
    }
    void copy(int l, std::int32_t node) {
        std::size_t r = l == 0 ? static_cast<std::size_t>(node)
                               : static_cast<std::size_t>(ix->ord1[static_cast<std::size_t>(node)]);
        std::size_t slots = l == 0 ? ix->m0 : ix->m;
        std::int32_t n = ix->counts[static_cast<std::size_t>(l)][r].load(std::memory_order_acquire);
        if (static_cast<std::size_t>(n) > slots) n = static_cast<std::int32_t>(slots);
        const std::atomic<std::int32_t>* p = ix->links[static_cast<std::size_t>(l)].data() + r * slots;
        buf.clear();
        for (std::int32_t j = 0; j < n; ++j) {
            std::int32_t x = p[j].load(std::memory_order_acquire);
            if (x >= 0) buf.push_back(x);
        }
    }
    const std::int32_t* list(int l, std::int32_t node, std::int32_t& n) {
        if (locks) {
            std::lock_guard<std::mutex> g(lock_of(node));
            copy(l, node);
        } else {
            copy(l, node);
        }
        n = static_cast<std::int32_t>(buf.size());
        return buf.data();
    }
};

std::size_t slot_row(const Index& ix, int l, std::int32_t node) {
    return l == 0 ? static_cast<std::size_t>(node) : static_cast<std::size_t>(ix.ord1[node]);
}
std::size_t cap(const Index& ix, int l) { return l == 0 ? ix.m0 : ix.m; }

// Greedy descent with ef = 1 on layer l from cur. Never moves to node skip.
Cand greedy(Access& a, const float* q, Cand cur, int l, std::int64_t& dc,
            std::int32_t skip = -1) {
    const Matrix& v = *a.ix->vectors;
    bool moved = true;
    while (moved) {
        moved = false;
        std::int32_t n;
        const std::int32_t* nb = a.list(l, cur.id, n);
        for (std::int32_t j = 0; j < n; ++j) {
            if (nb[j] == skip) continue;
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

// Algorithm 2. Returns up to ef results, best first. Node skip (if >= 0) is
// treated as visited: it is never scored or expanded.
std::vector<Cand> search_layer(Access& a, const float* q, const std::vector<Cand>& eps,
                               std::size_t ef, int l, Visited& vis, std::int64_t& dc,
                               std::int32_t skip = -1) {
    const Matrix& v = *a.ix->vectors;
    vis.reset(v.rows);
    if (skip >= 0) vis.test_and_set(skip);
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

// Filtered search-layer on layer 0 (CONTRACT 11.3). Same as search_layer, with
// one change: a node enters the result heap only if pass[node] is 1. Every
// scored node that is better than the worst result (or scored while the result
// heap is not full) enters the candidate heap and is expanded, also if it fails.
// pass == nullptr: no filter (identical to search_layer from one entry point).
// expanded = nodes popped from the candidate heap and expanded.
// Tombstones (CONTRACT 13.3) use the same rule: a tombstoned node is scored and
// expanded but never enters the result heap. pass is indexed by row ID; after
// a compact (rebuild), idmap maps node -> row ID.
std::vector<Cand> search_layer0_filtered(Access& a, const float* q, Cand ep, std::size_t ef,
                                         const std::uint8_t* pass, const Tombstones* dead,
                                         const std::int32_t* idmap, Visited& vis,
                                         std::int64_t& dc, std::int64_t& expanded) {
    const Matrix& v = *a.ix->vectors;
    vis.reset(v.rows);
    auto admit = [&](std::int32_t node) {
        if (dead && dead->test(static_cast<std::size_t>(node))) return false;
        if (!pass) return true;
        return pass[static_cast<std::size_t>(idmap ? idmap[node] : node)] != 0;
    };
    std::priority_queue<Cand, std::vector<Cand>, BestOnTop> cand;
    std::priority_queue<Cand, std::vector<Cand>, WorstOnTop> res;
    vis.test_and_set(ep.id);
    cand.push(ep);
    if (admit(ep.id)) res.push(ep);
    while (!cand.empty()) {
        Cand c = cand.top();
        if (res.size() >= ef && better(res.top(), c)) break;
        cand.pop();
        ++expanded;
        std::int32_t n;
        const std::int32_t* nb = a.list(0, c.id, n);
        for (std::int32_t j = 0; j < n; ++j) {
            std::int32_t e = nb[j];
            if (vis.test_and_set(e)) continue;
            Cand ce{dot(q, v.row(static_cast<std::size_t>(e)), v.dim), e};
            ++dc;
            if (res.size() < ef || better(ce, res.top())) {
                cand.push(ce);
                if (admit(e)) {
                    res.push(ce);
                    if (res.size() > ef) res.pop();
                }
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
    // Slots first, then the count, both with release order (CONTRACT 12.2).
    std::size_t r = slot_row(ix, l, node);
    std::atomic<std::int32_t>* base = ix.links[static_cast<std::size_t>(l)].data() + r * cap(ix, l);
    for (std::size_t j = 0; j < nb.size(); ++j) base[j].store(nb[j], std::memory_order_release);
    ix.counts[static_cast<std::size_t>(l)][r].store(static_cast<std::int32_t>(nb.size()),
                                                   std::memory_order_release);
}

// Writes node's own list on layer l, merged with edges that other threads
// added meanwhile. Shrinks to the cap with the heuristic if needed.
void set_own_list(Index& ix, Access& a, int l, std::int32_t node, std::vector<std::int32_t> sel) {
    std::unique_lock<std::mutex> g;
    if (a.locks) g = std::unique_lock<std::mutex>(a.lock_of(node));
    std::int32_t n;
    const auto* cur = ix.neighbors(l, node, n);
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
    std::atomic<std::int32_t>& cnt_a = ix.counts[static_cast<std::size_t>(l)][r];
    std::int32_t cnt = cnt_a.load(std::memory_order_relaxed);
    std::atomic<std::int32_t>* base = ix.links[static_cast<std::size_t>(l)].data() + r * slots;
    for (std::int32_t j = 0; j < cnt; ++j)
        if (base[j].load(std::memory_order_relaxed) == to) return;
    if (static_cast<std::size_t>(cnt) < slots) {
        base[cnt].store(to, std::memory_order_release);
        cnt_a.store(cnt + 1, std::memory_order_release);
        return;
    }
    std::vector<std::int32_t> ids(base, base + cnt);
    ids.push_back(to);
    write_list(ix, l, from, select_heuristic(ix, scored(ix, from, ids), slots));
}

void insert_one(Index& ix, Access& a, std::int32_t i, Visited& vis) {
    const Matrix& v = *ix.vectors;
    const float* q = v.row(static_cast<std::size_t>(i));
    int li = ix.level[static_cast<std::size_t>(i)];

    std::unique_lock<std::mutex> ep_lock;
    if (a.locks) ep_lock = std::unique_lock<std::mutex>(*ix.ep_mu);
    std::int32_t entry = ix.entry.load();
    int top = ix.top.load();
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
    if (li > top) {  // still under ep_mu (kept above)
        ix.top = li;
        ix.entry = i;  // published after i's lists are linked
    }
}

std::vector<std::int32_t> in_degree0(const Index& ix) {
    const std::size_t n = ix.n_live.load();
    std::vector<std::int32_t> indeg(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        std::int32_t c;
        const auto* nb = ix.neighbors(0, static_cast<std::int32_t>(i), c);
        for (std::int32_t j = 0; j < c; ++j) ++indeg[static_cast<std::size_t>(nb[j])];
    }
    std::vector<std::int32_t> zero;
    for (std::size_t i = 0; i < n; ++i)
        if (indeg[i] == 0 && static_cast<std::int32_t>(i) != ix.entry)
            zero.push_back(static_cast<std::int32_t>(i));
    return zero;
}

std::vector<char> reachable0(const Index& ix) {
    const std::size_t n = ix.n_live.load();
    std::vector<char> seen(n, 0);
    std::vector<std::int32_t> queue{ix.entry};
    seen[static_cast<std::size_t>(ix.entry)] = 1;
    for (std::size_t h = 0; h < queue.size(); ++h) {
        std::int32_t c;
        const auto* nb = ix.neighbors(0, queue[h], c);
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
    const auto* nu = ix.neighbors(0, u, cu);
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
            if (ix.dead.test(static_cast<std::size_t>(vtx))) continue;  // repair mode: no edges to dead nodes
            std::int32_t c;
            const auto* nb = ix.neighbors(0, vtx, c);
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
            if (seen[i] || ix.dead.test(i)) continue;
            found = true;
            std::int32_t vtx = static_cast<std::int32_t>(i);
            std::int32_t u = -1, u_free = -1;
            for (const Cand& x : search_from_entry(ix, vtx, vis)) {
                if (x.id == vtx) continue;
                if (u < 0) u = x.id;
                if (static_cast<std::size_t>(ix.counts[0][static_cast<std::size_t>(x.id)].load()) < ix.m0) {
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
    const std::size_t n = ix.n_live.load();
    if (ix.entry < 0) return n;
    std::vector<char> seen(n, 0);
    std::vector<std::int32_t> queue{ix.entry};
    seen[static_cast<std::size_t>(ix.entry)] = 1;
    for (std::size_t h = 0; h < queue.size(); ++h) {
        std::int32_t c;
        const auto* nb = ix.neighbors(0, queue[h], c);
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
    ix.data_dir = ctx.data_dir;
    ix.m = static_cast<std::size_t>(m);
    ix.m0 = 2 * ix.m;
    ix.ef_construct = static_cast<std::size_t>(efc);
    ix.threads = std::max(1, ctx.threads);
    ix.seed = ctx.seed;
    ix.n_orig = vectors.rows;
    const std::size_t n_all = vectors.rows;  // storage and levels for every row
    const std::size_t n = ctx.build_rows > 0 ? std::min(ctx.build_rows, n_all) : n_all;

    auto t0 = std::chrono::steady_clock::now();
    ix.level = draw_levels(n_all, ix.m, ctx.seed);  // one stream, row order, all rows
    int max_level = 0;
    ix.ord1.assign(n_all, -1);
    for (std::size_t i = 0; i < n_all; ++i) {
        max_level = std::max(max_level, static_cast<int>(ix.level[i]));
        if (ix.level[i] >= 1) ix.ord1[i] = static_cast<std::int32_t>(ix.n_upper++);
    }
    ix.links.resize(static_cast<std::size_t>(max_level) + 1);
    ix.counts.resize(static_cast<std::size_t>(max_level) + 1);
    ix.links[0].assign(n_all * ix.m0, -1);
    ix.counts[0].assign(n_all, 0);
    for (int l = 1; l <= max_level; ++l) {
        ix.links[l].assign(ix.n_upper * ix.m, -1);
        ix.counts[l].assign(ix.n_upper, 0);
    }

    ix.locks.reset(new std::mutex[kStripes]);  // kept for insert()
    ix.ep_mu = std::make_unique<std::mutex>();
    std::mutex* locks = ix.locks.get();
    ix.n_live = n;
    if (ix.threads == 1 || n < 2) {
        Access a{&ix, nullptr, {}};
        Visited vis;
        for (std::size_t i = 0; i < n; ++i) insert_one(ix, a, static_cast<std::int32_t>(i), vis);
    } else {
        {
            Access a{&ix, locks, {}};
            Visited vis;
            insert_one(ix, a, 0, vis);  // row 0 sets the first entry point
        }
        std::atomic<std::size_t> next{1};
        std::vector<std::thread> pool;
        for (int t = 0; t < ix.threads; ++t)
            pool.emplace_back([&] {
                Access a{&ix, locks, {}};
                Visited vis;
                // Chunks of 64 consecutive rows keep near-duplicates in order.
                for (std::size_t c0 = next.fetch_add(kChunk); c0 < n; c0 = next.fetch_add(kChunk))
                    for (std::size_t i = c0; i < std::min(n, c0 + kChunk); ++i)
                        insert_one(ix, a, static_cast<std::int32_t>(i), vis);
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
    const std::int32_t entry = ix.entry.load();  // top = level[entry], read no second value
    if (entry < 0 || k == 0) return r;
    long long ef_p = params.get_int("ef");
    if (ef_p < 1) throw ParamError("hnsw: ef must be >= 1");
    std::size_t ef = std::max(static_cast<std::size_t>(ef_p), k);

    const Matrix& v = *ix.vectors;
    thread_local Access a{nullptr, nullptr, {}};  // per-thread scratch, no shared lock
    a.ix = &ix;
    std::int64_t dc = 0;
    Cand cur{dot(query, v.row(static_cast<std::size_t>(entry)), v.dim), entry};
    ++dc;
    for (int l = ix.level[static_cast<std::size_t>(entry)]; l >= 1; --l) cur = greedy(a, query, cur, l, dc);
    thread_local Visited vis;
    const FilterMask* f = get_filter(ix.data_dir, params, ix.n_orig);
    const std::int32_t* idmap = ix.ids.empty() ? nullptr : ix.ids.data();
    std::int64_t expanded = 0;
    std::vector<Cand> w = search_layer0_filtered(a, query, cur, ef, f ? f->pass.data() : nullptr,
                                                 ix.dead.any() ? &ix.dead : nullptr, idmap, vis,
                                                 dc, expanded);
    if (f) r.counters["filter_rows"] = static_cast<double>(f->rows);  // omitted for none (11.2)
    r.counters["visited"] = static_cast<double>(expanded);
    for (std::size_t i = 0; i < k && i < w.size(); ++i) {
        r.ids[i] = idmap ? idmap[w[i].id] : w[i].id;
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
        for (std::size_t i = 0; i < c.size(); ++i) edges += static_cast<std::size_t>(c[i].load());
    }
    // + the tombstone bit set and, after a compact (rebuild), the node -> ID map.
    return edges * sizeof(std::int32_t) +
           ix.level.size() * (sizeof(std::uint8_t) + sizeof(std::int32_t)) + counts +
           ix.dead.bytes() + ix.ids.size() * sizeof(std::int32_t);
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
    for (std::size_t i = 0; i < ix.n_live.load(); ++i)
        for (std::size_t l = 0, lv = ix.level[i]; l <= lv && l < per.size(); ++l) ++per[l];
    for (std::size_t l = 0; l < per.size(); ++l)
        e["nodes_layer_" + std::to_string(l)] = static_cast<double>(per[l]);
    return e;
}

void insert(Index& ix, const std::vector<std::int64_t>& ids, const Matrix& rows) {
    if (rows.rows != ids.size() || rows.dim != ix.vectors->dim)
        throw std::runtime_error("hnsw insert: ids and rows differ in shape");
    Matrix& v = *ix.vectors;
    Access a{&ix, ix.locks.get(), {}};
    thread_local Visited vis;
    for (std::size_t j = 0; j < ids.size(); ++j) {
        std::int64_t id = ids[j];
        if (id < static_cast<std::int64_t>(ix.n_live.load()) || static_cast<std::size_t>(id) >= v.rows)
            throw std::runtime_error("hnsw insert: row " + std::to_string(id) + " is not free");
        // The vector is written before any edge to the node exists.
        std::copy(rows.row(j), rows.row(j) + v.dim, v.row(static_cast<std::size_t>(id)));
        insert_one(ix, a, static_cast<std::int32_t>(id), vis);
        std::size_t live = ix.n_live.load();
        if (static_cast<std::size_t>(id) + 1 > live) ix.n_live.store(static_cast<std::size_t>(id) + 1);
    }
}

void repair_after_inserts(Index& ix) {
    if (ix.entry.load() >= 0) repair(ix);
}

// ---------- Changes (CONTRACT 13.3) ----------

namespace {

// Live node with the highest level, lowest ID on a tie; not skip. -1 if none.
std::int32_t top_live_node(const Index& ix, std::int32_t skip) {
    std::int32_t best = -1;
    const std::size_t n = ix.n_live.load();
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<std::int32_t>(i) == skip || ix.dead.test(i)) continue;
        if (best < 0 || ix.level[i] > ix.level[static_cast<std::size_t>(best)])
            best = static_cast<std::int32_t>(i);
    }
    return best;
}

// Rewrites the list of node on layer l without the entry x (if present).
void remove_from_list(Index& ix, int l, std::int32_t node, std::int32_t x) {
    std::int32_t c;
    const auto* nb = ix.neighbors(l, node, c);
    std::vector<std::int32_t> keep;
    keep.reserve(static_cast<std::size_t>(c));
    for (std::int32_t j = 0; j < c; ++j)
        if (nb[j] != x) keep.push_back(nb[j]);
    if (keep.size() != static_cast<std::size_t>(c)) write_list(ix, l, node, keep);
}

// Insert procedure (Algorithm 1) for a node that is already in the graph with
// no out-edges, at its existing level. The search skips node i.
void relink(Index& ix, Access& a, std::int32_t i, Visited& vis) {
    const Matrix& v = *ix.vectors;
    const float* q = v.row(static_cast<std::size_t>(i));
    const int li = ix.level[static_cast<std::size_t>(i)];
    std::int32_t start = ix.entry.load();
    if (start == i) start = top_live_node(ix, i);
    if (start < 0) return;  // the only node
    const int top = ix.level[static_cast<std::size_t>(start)];
    std::int64_t dc = 0;
    Cand cur{dot(q, v.row(static_cast<std::size_t>(start)), v.dim), start};
    for (int l = top; l > li; --l) cur = greedy(a, q, cur, l, dc, i);
    std::vector<Cand> eps{cur};
    for (int l = std::min(li, top); l >= 0; --l) {
        std::vector<Cand> w = search_layer(a, q, eps, ix.ef_construct, l, vis, dc, i);
        if (w.empty()) w.push_back(cur);
        std::vector<std::int32_t> nb = select_heuristic(ix, w, ix.m);
        set_own_list(ix, a, l, i, nb);
        for (std::int32_t e : nb) add_edge(ix, a, l, e, i);
        eps = std::move(w);
    }
}

// In-place repair (mode repair); see hnsw.hpp.
void repair_in_place(Index& ix) {
    const std::size_t n = ix.n_live.load();
    for (int l = 0; l < static_cast<int>(ix.links.size()); ++l) {
        for (std::size_t u = 0; u < n; ++u) {
            if (ix.level[u] < l || ix.dead.test(u)) continue;
            const auto node = static_cast<std::int32_t>(u);
            std::int32_t c;
            const auto* nb = ix.neighbors(l, node, c);
            bool has_dead = false;
            for (std::int32_t j = 0; j < c && !has_dead; ++j)
                has_dead = ix.dead.test(static_cast<std::size_t>(nb[j]));
            if (!has_dead) continue;
            std::vector<std::int32_t> cand;
            for (std::int32_t j = 0; j < c; ++j) {
                const std::int32_t x = nb[j];
                if (!ix.dead.test(static_cast<std::size_t>(x))) {
                    cand.push_back(x);
                    continue;
                }
                std::int32_t cx;
                const auto* nx = ix.neighbors(l, x, cx);
                for (std::int32_t t = 0; t < cx; ++t)
                    if (!ix.dead.test(static_cast<std::size_t>(nx[t]))) cand.push_back(nx[t]);
            }
            std::vector<Cand> sc = scored(ix, node, cand);
            std::vector<std::int32_t> out;
            if (sc.size() <= cap(ix, l)) {
                for (const Cand& x : sc) out.push_back(x.id);
            } else {
                out = select_heuristic(ix, sc, cap(ix, l));
            }
            write_list(ix, l, node, out);
        }
    }
    // Tombstoned nodes lose their out-edges (all lists above no longer hold them).
    for (std::size_t u = 0; u < n; ++u) {
        if (!ix.dead.test(u)) continue;
        for (int l = 0; l <= ix.level[u] && l < static_cast<int>(ix.links.size()); ++l)
            write_list(ix, l, static_cast<std::int32_t>(u), {});
    }
    if (ix.entry.load() >= 0 && ix.dead.test(static_cast<std::size_t>(ix.entry.load()))) {
        std::int32_t e = top_live_node(ix, -1);
        ix.entry = e;
        ix.top = e >= 0 ? static_cast<int>(ix.level[static_cast<std::size_t>(e)]) : -1;
    }
    if (ix.entry.load() >= 0) repair(ix);
}

}  // namespace

void delete_rows(Index& ix, const std::vector<std::uint8_t>& mask) {
    if (!ix.ids.empty()) throw std::runtime_error("hnsw: delete after compact is not supported");
    if (mask.size() != ix.vectors->rows) throw std::runtime_error("hnsw: delete mask size != rows");
    ix.dead.set(mask);
}

void update_rows(Index& ix, const std::vector<std::int64_t>& ids, const Matrix& rows) {
    if (!ix.ids.empty()) throw std::runtime_error("hnsw: update after compact is not supported");
    Matrix& v = *ix.vectors;
    if (rows.rows != ids.size() || rows.dim != v.dim)
        throw std::runtime_error("hnsw: update ids and rows differ in shape");
    Access a{&ix, nullptr, {}};  // one thread: no locks
    Visited vis;
    for (std::size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] < 0 || static_cast<std::size_t>(ids[j]) >= ix.n_live.load())
            throw std::runtime_error("hnsw: update ID not in the graph");
        const auto i = static_cast<std::int32_t>(ids[j]);
        std::copy(rows.row(j), rows.row(j) + v.dim, v.row(static_cast<std::size_t>(i)));
        for (int l = 0; l <= ix.level[static_cast<std::size_t>(i)]; ++l) {
            std::int32_t c;
            const auto* nb = ix.neighbors(l, i, c);
            std::vector<std::int32_t> old(nb, nb + c);
            for (std::int32_t u : old) remove_from_list(ix, l, u, i);
            write_list(ix, l, i, {});
        }
        relink(ix, a, i, vis);
    }
    repair(ix);
}

void compact(Index& ix, CompactMode mode) {
    if (!ix.ids.empty()) return;  // already rebuilt
    if (mode == CompactMode::kRepair) {
        if (ix.dead.any()) repair_in_place(ix);
        return;
    }
    // Rebuild from the live rows, in row order.
    const Matrix& v = *ix.vectors;
    const std::size_t n = ix.n_live.load();
    std::vector<std::int32_t> live;
    std::unique_ptr<Matrix> own;
    if (ix.dead.any()) {
        own = std::make_unique<Matrix>();
        own->dim = v.dim;
        own->rows = n - std::min(n, ix.dead.count);
        own->data.reserve(own->rows * v.dim);
        for (std::size_t i = 0; i < n; ++i) {
            if (ix.dead.test(i)) continue;
            own->data.insert(own->data.end(), v.row(i), v.row(i) + v.dim);
            live.push_back(static_cast<std::int32_t>(i));
        }
        own->rows = live.size();
    }
    Params p;
    p.values["m"] = std::to_string(ix.m);
    p.values["ef_construct"] = std::to_string(ix.ef_construct);
    BuildContext ctx{ix.threads, ix.seed, "", ix.data_dir};
    ctx.build_rows = own ? 0 : n;
    Index fresh = build(own ? *own : *ix.vectors, p, ctx);
    fresh.n_orig = ix.n_orig;
    fresh.own = std::move(own);  // the heap Matrix does not move: fresh.vectors stays valid
    fresh.ids = std::move(live);
    fresh.times = ix.times;      // build times of the original build
    ix = std::move(fresh);
}

}  // namespace vro::hnsw

namespace vro::hnsw {

// File layout (CONTRACT 15.1) versus memory: layer 0 is the same (row ID x
// 2m slots). Upper layers: memory keeps one slot array per layer, indexed by
// ord1 (rank among nodes with level >= 1); the file keeps one block of m
// slots per (row, layer) pair, row-major: row i's blocks are
// upper_offsets[i] .. upper_offsets[i+1], for layers 1..level(i).
// Empty slots (at or past the count) are written as -1.
// After compact (rebuild), node k holds row ids[k]; the file describes all
// n_orig rows: node IDs map back to row IDs, and a dropped row gets its
// tombstone bit, a zero vector, level 0, and no edges.
std::uint64_t save(const Index& ix, const std::string& path, const vrofile::Meta& meta) {
    const Matrix& v = *ix.vectors;
    const std::size_t nodes = ix.level.size();
    if (ix.n_live.load() != nodes)
        throw std::runtime_error("hnsw: save while rows are not yet inserted is not supported");
    const bool mapped = !ix.ids.empty();
    const std::size_t n = mapped ? ix.n_orig : nodes;
    const std::size_t m = ix.m, m0 = ix.m0;
    auto row_of = [&](std::int32_t node) { return mapped ? ix.ids[static_cast<std::size_t>(node)] : node; };

    std::vector<std::uint8_t> level(n, 0);
    for (std::size_t k = 0; k < nodes; ++k) level[static_cast<std::size_t>(row_of(static_cast<std::int32_t>(k)))] = ix.level[k];
    std::vector<float> vec_full;  // only when mapped
    Tombstones dead_full;
    if (mapped) {
        vec_full.assign(n * v.dim, 0.0f);
        std::vector<std::uint8_t> mask(n, 1);
        for (std::size_t k = 0; k < nodes; ++k) {
            const auto r = static_cast<std::size_t>(ix.ids[k]);
            std::copy(v.row(k), v.row(k) + v.dim, vec_full.data() + r * v.dim);
            mask[r] = ix.dead.test(k) ? 1 : 0;
        }
        dead_full.set(mask);
    }
    std::vector<std::int32_t> l0(n * m0, -1), c0(n, 0);
    std::vector<std::int32_t> uoff(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) uoff[i + 1] = uoff[i] + level[i];
    const std::size_t L = static_cast<std::size_t>(uoff[n]);
    std::vector<std::int32_t> us(L * m, -1), uc(L, 0);
    for (std::size_t k = 0; k < nodes; ++k) {
        const auto i = static_cast<std::size_t>(row_of(static_cast<std::int32_t>(k)));
        std::int32_t cnt = 0;
        const std::atomic<std::int32_t>* nb = ix.neighbors(0, static_cast<std::int32_t>(k), cnt);
        c0[i] = cnt;
        for (std::int32_t j = 0; j < cnt; ++j) l0[i * m0 + static_cast<std::size_t>(j)] = row_of(nb[j].load());
        for (int l = 1; l <= ix.level[k]; ++l) {
            const std::size_t b = static_cast<std::size_t>(uoff[i]) + static_cast<std::size_t>(l - 1);
            nb = ix.neighbors(l, static_cast<std::int32_t>(k), cnt);
            uc[b] = cnt;
            for (std::int32_t j = 0; j < cnt; ++j) us[b * m + static_cast<std::size_t>(j)] = row_of(nb[j].load());
        }
    }
    const std::int32_t entry = ix.entry.load() >= 0 ? row_of(ix.entry.load()) : -1;
    std::vector<std::uint8_t> bits = vrofile::tombstones_to_bytes(mapped ? dead_full : ix.dead, n);

    vrofile::Writer w;
    w.add("vectors", "f32", {n, v.dim}, mapped ? vec_full.data() : v.data.data());
    w.add("tombstones", "u8", {bits.size()}, bits.data());
    w.add("levels", "u8", {n}, level.data());
    w.add("entry", "int32", {1}, &entry);
    w.add("layer0_slots", "int32", {n, m0}, l0.data());
    w.add("layer0_counts", "int32", {n}, c0.data());
    w.add("upper_slots", "int32", {L, m}, us.data());
    w.add("upper_counts", "int32", {L}, uc.data());
    w.add("upper_offsets", "int32", {n + 1}, uoff.data());
    vrofile::Meta mt = meta;
    mt.index = "hnsw";
    mt.n = n;
    mt.dim = v.dim;
    return w.write(path, mt);
}

Index load(const std::string& path, const Params& params, const BuildContext& ctx) {
    vrofile::Reader r(path);
    r.expect("hnsw", ctx.expect_dim, params);
    const std::size_t n = r.n(), dim = r.dim();
    const Params bp = vrofile::params_from_header(r.header().at("build_params"));
    const long long m_ll = bp.get_int("m"), efc = bp.get_int("ef_construct");
    if (m_ll < 2 || efc < 1) throw vrofile::FormatError(path + ": bad m or ef_construct");
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw vrofile::FormatError(path + ": too many rows for int32 IDs");

    Index ix;
    ix.own = std::make_unique<Matrix>();
    ix.own->rows = n;
    ix.own->dim = dim;
    ix.own->data = r.read_f32("vectors", {n, dim});
    ix.vectors = ix.own.get();
    ix.data_dir = ctx.data_dir;
    ix.m = static_cast<std::size_t>(m_ll);
    ix.m0 = 2 * ix.m;
    ix.ef_construct = static_cast<std::size_t>(efc);
    ix.threads = std::max(1, ctx.threads);
    ix.seed = r.seed();  // compact (rebuild) redraws levels with the build seed
    ix.n_orig = n;
    ix.level = r.read_u8("levels", {n});
    const std::vector<std::int32_t> entry = r.read_i32("entry", {1});
    const std::vector<std::int32_t> uoff = r.read_i32("upper_offsets", {n + 1});
    if (uoff[0] != 0) throw vrofile::FormatError(path + ": upper_offsets[0] != 0");
    for (std::size_t i = 0; i < n; ++i)
        if (uoff[i + 1] - uoff[i] != ix.level[i])
            throw vrofile::FormatError(path + ": upper_offsets do not match levels");
    const std::size_t L = static_cast<std::size_t>(uoff[n]);
    const std::size_t m = ix.m, m0 = ix.m0;
    const std::vector<std::int32_t> l0 = r.read_i32("layer0_slots", {n, m0});
    const std::vector<std::int32_t> c0 = r.read_i32("layer0_counts", {n});
    const std::vector<std::int32_t> us = r.read_i32("upper_slots", {L, m});
    const std::vector<std::int32_t> uc = r.read_i32("upper_counts", {L});

    int max_level = 0;
    ix.ord1.assign(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        max_level = std::max(max_level, static_cast<int>(ix.level[i]));
        if (ix.level[i] >= 1) ix.ord1[i] = static_cast<std::int32_t>(ix.n_upper++);
    }
    ix.links.resize(static_cast<std::size_t>(max_level) + 1);
    ix.counts.resize(static_cast<std::size_t>(max_level) + 1);
    ix.links[0].assign(n * m0, -1);
    ix.counts[0].assign(n, 0);
    for (int l = 1; l <= max_level; ++l) {
        ix.links[static_cast<std::size_t>(l)].assign(ix.n_upper * m, -1);
        ix.counts[static_cast<std::size_t>(l)].assign(ix.n_upper, 0);
    }
    auto check_list = [&](std::int32_t cnt, std::size_t cap, const std::int32_t* s, int l) {
        if (cnt < 0 || static_cast<std::size_t>(cnt) > cap)
            throw vrofile::FormatError(path + ": neighbor count out of range");
        for (std::int32_t j = 0; j < cnt; ++j)
            if (s[j] < 0 || static_cast<std::size_t>(s[j]) >= n || ix.level[static_cast<std::size_t>(s[j])] < l)
                throw vrofile::FormatError(path + ": neighbor ID out of range");
    };
    for (std::size_t i = 0; i < n; ++i) {
        check_list(c0[i], m0, &l0[i * m0], 0);
        for (std::int32_t j = 0; j < c0[i]; ++j)
            ix.links[0][i * m0 + static_cast<std::size_t>(j)].store(l0[i * m0 + static_cast<std::size_t>(j)],
                                                                     std::memory_order_relaxed);
        ix.counts[0][i].store(c0[i], std::memory_order_relaxed);
        for (int l = 1; l <= ix.level[i]; ++l) {
            const std::size_t b = static_cast<std::size_t>(uoff[i]) + static_cast<std::size_t>(l - 1);
            const std::size_t o = static_cast<std::size_t>(ix.ord1[i]);
            check_list(uc[b], m, &us[b * m], l);
            for (std::int32_t j = 0; j < uc[b]; ++j)
                ix.links[static_cast<std::size_t>(l)][o * m + static_cast<std::size_t>(j)].store(
                    us[b * m + static_cast<std::size_t>(j)], std::memory_order_relaxed);
            ix.counts[static_cast<std::size_t>(l)][o].store(uc[b], std::memory_order_relaxed);
        }
    }
    if (n > 0 && (entry[0] < 0 || static_cast<std::size_t>(entry[0]) >= n))
        throw vrofile::FormatError(path + ": entry out of range");
    ix.entry = n > 0 ? entry[0] : -1;
    ix.top = n > 0 ? static_cast<int>(ix.level[static_cast<std::size_t>(entry[0])]) : -1;
    ix.n_live = n;
    ix.locks.reset(new std::mutex[kStripes]);
    ix.ep_mu = std::make_unique<std::mutex>();
    ix.dead = vrofile::tombstones_from_bytes(r.read_u8("tombstones", {(n + 7) / 8}), n);
    std::atomic_thread_fence(std::memory_order_release);
    return ix;
}

}  // namespace vro::hnsw
