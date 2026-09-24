// pq: product quantization with a full scan (CONTRACT 6.4).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.50 at default params on the dev set, for metric=ip and metric=l2.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "kmeans.hpp"

namespace vro::pq {

struct Index {
    const Matrix* vectors = nullptr;  // full vectors, for rerank only
    std::size_t n = 0;
    std::size_t dim = 0;
    std::size_t m = 0;       // sub-vectors per vector
    std::size_t dsub = 0;    // dim / m
    Metric metric = Metric::kDot;
    std::vector<float> codebooks;     // (m, 256, dsub), contiguous
    std::vector<std::uint8_t> codes;  // (n, m), contiguous
    BuildTimes times;
};

constexpr std::size_t kCentroids = 256;

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::pq
