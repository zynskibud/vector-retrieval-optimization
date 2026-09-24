// ivf: inverted file with k-means clusters (CONTRACT 6.3).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.75 at nprobe=8 on the dev set.
//
// Build: train = shared k-means (dot product, normalized centers).
// Add = assign each corpus row to its best center (std::thread over rows).
// Lists use CSR layout: list_ids holds the row IDs grouped by list, and the
// IDs of list c are list_ids[offsets[c] .. offsets[c+1]).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace vro::ivf {

struct Index {
    const Matrix* vectors = nullptr;   // the corpus; search scans full vectors
    std::size_t nlist = 0;
    std::size_t dim = 0;
    std::vector<float> centers;         // nlist x dim, row-major
    std::vector<std::int32_t> list_ids; // N row IDs, grouped by list
    std::vector<std::int32_t> offsets;  // nlist + 1 entries; offsets[nlist] == N
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::ivf
