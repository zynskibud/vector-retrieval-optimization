#include "flat.hpp"

#include "distance.hpp"

namespace vro::flat {

Index build(Matrix& vectors, const Params& /*params*/, const BuildContext& ctx) {
    Index index;
    index.vectors = &vectors;
    index.data_dir = ctx.data_dir;
    return index;  // no train, no add: both times stay 0
}

SearchResult search(const Index& index, const float* query, std::size_t k,
                    const Params& params) {
    const Matrix& v = *index.vectors;
    const FilterMask* f = get_filter(index.data_dir, params, v.rows);
    TopK top(k);
    SearchResult r;
    if (!f) {
        for (std::size_t i = 0; i < v.rows; ++i) {
            float s = dot(query, v.row(i), v.dim);
            if (s >= top.threshold()) top.push(static_cast<std::int64_t>(i), s);
        }
        r.distance_computations = static_cast<std::int64_t>(v.rows);
    } else {
        // Iterate all rows and skip the rows that fail the mask (CONTRACT 11.3).
        const std::uint8_t* pass = f->pass.data();
        for (std::size_t i = 0; i < v.rows; ++i) {
            if (!pass[i]) continue;
            float s = dot(query, v.row(i), v.dim);
            if (s >= top.threshold()) top.push(static_cast<std::int64_t>(i), s);
        }
        r.distance_computations = static_cast<std::int64_t>(f->rows);
        r.counters["filter_rows"] = static_cast<double>(f->rows);
    }
    top.result(r.ids, r.scores);
    return r;
}

std::size_t index_bytes(const Index& /*index*/) { return 0; }

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::flat
