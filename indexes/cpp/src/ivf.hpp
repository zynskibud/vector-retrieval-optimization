// ivf: inverted file with k-means clusters (CONTRACT 6.3).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.80 at nprobe=8 on the dev set.
// Stub. build() throws NotImplemented until the Wave 2 agent writes it.
#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "common.hpp"

namespace vro::ivf {

struct Index {
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::ivf
