// flat: exact search by brute force (CONTRACT 6.1). No parameters.
// Recall floor: recall@10 = 1.0 on the dev set (CONTRACT 9, item 3).
#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "common.hpp"

namespace vro::flat {

struct Index {
    const Matrix* vectors = nullptr;  // the corpus array is the index
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::flat
