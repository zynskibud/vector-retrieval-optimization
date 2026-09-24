// ivf_pq: inverted file with PQ-coded residuals (CONTRACT 6.5).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.45 at default params on the dev set, for metric=ip and metric=l2.
//
// Build: coarse centers = shared k-means (dot product, normalized) on the first
// train_size rows. Residual r = x - c(x). Codebooks = one k-means per sub-vector
// on the training residuals (seed + j, no normalization, assignment under metric).
// Add: assign every row, encode its residual, store CSR lists: list_ids and codes
// grouped by list, offsets[c] .. offsets[c+1] for list c.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "kmeans.hpp"

namespace vro::ivf_pq {

constexpr std::size_t kCentroids = 256;

struct Index {
    const Matrix* vectors = nullptr;     // full vectors, for rerank only
    std::size_t n = 0;
    std::size_t dim = 0;
    std::size_t nlist = 0;
    std::size_t m = 0;                   // sub-vectors per vector
    std::size_t dsub = 0;                // dim / m
    Metric metric = Metric::kDot;        // PQ metric; coarse centers always use dot
    std::vector<float> centers;          // nlist x dim
    std::vector<float> codebooks;        // (m, 256, dsub)
    std::vector<std::int32_t> list_ids;  // N row IDs, grouped by list
    std::vector<std::uint8_t> codes;     // N x m, in list order (row p matches list_ids[p])
    std::vector<std::int32_t> offsets;   // nlist + 1
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::ivf_pq
