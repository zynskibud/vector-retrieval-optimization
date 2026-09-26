// hnsw: hierarchical navigable small world graph (CONTRACT 6.6).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.95 at ef=64 on the dev set.
// threads == 1: rows are inserted strictly in row order (deterministic).
// threads > 1: rows are inserted in parallel with a stripe of 65536 mutexes
// over node IDs. A repair pass runs after every build (CONTRACT 6.6).
// Concurrency (CONTRACT 12.2): search takes no locks. Slot arrays and counts
// are std::atomic<int32_t>; insert() stores the slots, then the count, with
// release order; search loads them with acquire order. insert() may run on one
// thread while other threads search.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common.hpp"

namespace vro::hnsw {

// std::atomic that can be moved (Index is returned by value). A move is not
// atomic; it happens only while no other thread uses the index.
template <typename T>
struct MovableAtomic : std::atomic<T> {
    MovableAtomic(T v = T()) : std::atomic<T>(v) {}
    MovableAtomic(MovableAtomic&& o) noexcept : std::atomic<T>(o.load()) {}
    MovableAtomic& operator=(MovableAtomic&& o) noexcept {
        this->store(o.load());
        return *this;
    }
    using std::atomic<T>::operator=;
};

// Fixed-size array of std::atomic<int32_t> (one layer's slots or counts).
struct AtomicArray {
    std::unique_ptr<std::atomic<std::int32_t>[]> p;
    std::size_t n = 0;
    void assign(std::size_t size, std::int32_t value) {
        p.reset(new std::atomic<std::int32_t>[size]);
        n = size;
        for (std::size_t i = 0; i < size; ++i) p[i].store(value, std::memory_order_relaxed);
    }
    std::size_t size() const { return n; }
    std::atomic<std::int32_t>& operator[](std::size_t i) { return p[i]; }
    const std::atomic<std::int32_t>& operator[](std::size_t i) const { return p[i]; }
    std::atomic<std::int32_t>* data() { return p.get(); }
    const std::atomic<std::int32_t>* data() const { return p.get(); }
};

struct Index {
    // Holds all rows (after --limit), also the rows that insert() adds later.
    // Rows at or above n_live are not in the graph yet.
    Matrix* vectors = nullptr;
    std::string data_dir;             // for filter_<name>.npy (CONTRACT 11)
    std::size_t m = 16;               // slots per node on layers >= 1
    std::size_t m0 = 32;              // slots per node on layer 0 (2m)
    std::size_t ef_construct = 100;
    std::vector<std::uint8_t> level;  // level of each node
    std::vector<std::int32_t> ord1;   // rank among nodes with level >= 1, else -1
    std::size_t n_upper = 0;          // number of nodes with level >= 1
    // links[l]: flat slot array. Layer 0 is indexed by node ID (m0 slots each);
    // layers >= 1 are indexed by ord1 (m slots each).
    std::vector<AtomicArray> links;
    std::vector<AtomicArray> counts;  // same indexing as links, one count per node
    MovableAtomic<std::int32_t> entry{-1};
    MovableAtomic<int> top{-1};  // always level[entry]; search reads entry only
    MovableAtomic<std::size_t> n_live{0};  // rows 0..n_live-1 are in the graph
    std::unique_ptr<std::mutex[]> locks;   // stripe of per-node locks (build and insert)
    std::unique_ptr<std::mutex> ep_mu;     // guards entry and top during build and insert
    std::size_t unreachable_before_repair = 0;  // directed layer-0 BFS misses before repair
    std::size_t zero_in_before_repair = 0;      // layer-0 nodes with in-degree 0 before repair
    std::size_t repair_added = 0;               // edges added by the repair pass
    std::size_t repair_step_b = 0;              // of repair_added: edges to BFS-unreachable nodes
    std::size_t repair_passes = 0;
    int threads = 1;
    BuildTimes times;

    // Neighbors of node on layer l: pointer to the first slot, count in n.
    // Not for use during concurrent inserts (search copies lists instead).
    const std::atomic<std::int32_t>* neighbors(int l, std::int32_t node, std::int32_t& n) const {
        std::size_t r = l == 0 ? static_cast<std::size_t>(node) : static_cast<std::size_t>(ord1[node]);
        std::size_t slots = l == 0 ? m0 : m;
        n = counts[l][r].load(std::memory_order_acquire);
        return links[l].data() + r * slots;
    }
};

// Builds on rows 0..ctx.build_rows-1 (0 = all rows). Levels and slot storage
// cover all vectors.rows rows, so insert() can add the others later.
Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
// Adds rows ids[j] (vector rows.row(j)) to the graph. Copies each vector into
// index.vectors, then links it. One inserter thread at a time; searches may
// run at the same time. ids must be >= n_live and < vectors.rows.
void insert(Index& index, const std::vector<std::int64_t>& ids, const Matrix& rows);
// Runs the repair pass once (CONTRACT 12.2). No searches or inserts may run.
void repair_after_inserts(Index& index);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

// Level of each row for n rows (CONTRACT 6.6), exposed for tests.
// Nodes that a directed BFS on layer 0 from the entry point does not reach.
std::size_t unreachable_layer0(const Index& index);

std::vector<std::uint8_t> draw_levels(std::size_t n, std::size_t m, std::uint64_t seed);

}  // namespace vro::hnsw
