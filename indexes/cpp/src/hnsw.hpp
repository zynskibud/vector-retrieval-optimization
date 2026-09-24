// hnsw: hierarchical navigable small world graph (CONTRACT 6.6).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.95 at ef=64 on the dev set.
// threads == 1: rows are inserted strictly in row order (deterministic).
// threads > 1: rows are inserted in parallel with a stripe of 65536 mutexes
// over node IDs. A repair pass runs after every build (CONTRACT 6.6).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace vro::hnsw {

struct Index {
    const Matrix* vectors = nullptr;
    std::size_t m = 16;               // slots per node on layers >= 1
    std::size_t m0 = 32;              // slots per node on layer 0 (2m)
    std::size_t ef_construct = 100;
    std::vector<std::uint8_t> level;  // level of each node
    std::vector<std::int32_t> ord1;   // rank among nodes with level >= 1, else -1
    std::size_t n_upper = 0;          // number of nodes with level >= 1
    // links[l]: flat slot array. Layer 0 is indexed by node ID (m0 slots each);
    // layers >= 1 are indexed by ord1 (m slots each).
    std::vector<std::vector<std::int32_t>> links;
    std::vector<std::vector<std::int32_t>> counts;  // same indexing as links, one count per node
    std::int32_t entry = -1;
    int top = -1;
    std::size_t unreachable_before_repair = 0;  // directed layer-0 BFS misses before repair
    std::size_t zero_in_before_repair = 0;      // layer-0 nodes with in-degree 0 before repair
    std::size_t repair_added = 0;               // edges added by the repair pass
    std::size_t repair_step_b = 0;              // of repair_added: edges to BFS-unreachable nodes
    std::size_t repair_passes = 0;
    int threads = 1;
    BuildTimes times;

    // Neighbors of node on layer l: pointer to the first slot, count in n.
    const std::int32_t* neighbors(int l, std::int32_t node, std::int32_t& n) const {
        std::size_t r = l == 0 ? static_cast<std::size_t>(node) : static_cast<std::size_t>(ord1[node]);
        std::size_t slots = l == 0 ? m0 : m;
        n = counts[l][r];
        return links[l].data() + r * slots;
    }
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

// Level of each row for n rows (CONTRACT 6.6), exposed for tests.
// Nodes that a directed BFS on layer 0 from the entry point does not reach.
std::size_t unreachable_layer0(const Index& index);

std::vector<std::uint8_t> draw_levels(std::size_t n, std::size_t m, std::uint64_t seed);

}  // namespace vro::hnsw
