// ivf_pq: inverted file with PQ-coded residuals (CONTRACT 6.5).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.45 at default params on the dev set, for metric=ip and metric=l2.
// Stub. build() throws NotImplemented until the Wave 2 agent writes it.
#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "common.hpp"

namespace vro::ivf_pq {

struct Index {
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::ivf_pq
