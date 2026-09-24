#include "flat.hpp"

#include "distance.hpp"

namespace vro::flat {

Index build(Matrix& vectors, const Params& /*params*/, const BuildContext& /*ctx*/) {
    Index index;
    index.vectors = &vectors;
    return index;  // no train, no add: both times stay 0
}

SearchResult search(const Index& index, const float* query, std::size_t k,
                    const Params& /*params*/) {
    const Matrix& v = *index.vectors;
    TopK top(k);
    for (std::size_t i = 0; i < v.rows; ++i) {
        float s = dot(query, v.row(i), v.dim);
        if (s >= top.threshold()) top.push(static_cast<std::int64_t>(i), s);
    }
    SearchResult r;
    top.result(r.ids, r.scores);
    r.distance_computations = static_cast<std::int64_t>(v.rows);
    return r;
}

std::size_t index_bytes(const Index& /*index*/) { return 0; }

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::flat
