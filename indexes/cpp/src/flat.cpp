#include "flat.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "distance.hpp"

namespace vro::flat {

Index build(Matrix& vectors, const Params& /*params*/, const BuildContext& ctx) {
    Index index;
    index.vectors = &vectors;
    index.data_dir = ctx.data_dir;
    index.n_rows = vectors.rows;
    return index;  // no train, no add: both times stay 0
}

SearchResult search(const Index& index, const float* query, std::size_t k,
                    const Params& params) {
    const Matrix& v = index.compacted ? index.own : *index.vectors;
    const FilterMask* f = get_filter(index.data_dir, params, index.n_rows);
    TopK top(k);
    SearchResult r;
    if (!f && !index.compacted && !index.dead.any()) {
        for (std::size_t i = 0; i < v.rows; ++i) {
            float s = dot(query, v.row(i), v.dim);
            if (s >= top.threshold()) top.push(static_cast<std::int64_t>(i), s);
        }
        r.distance_computations = static_cast<std::int64_t>(v.rows);
    } else {
        // Iterate all stored rows; skip tombstoned rows and rows that fail the
        // mask (CONTRACT 11.3, 13.3). The mask and the tombstones use row IDs.
        const std::uint8_t* pass = f ? f->pass.data() : nullptr;
        std::int64_t scored = 0;
        for (std::size_t i = 0; i < v.rows; ++i) {
            const std::size_t id = index.compacted ? static_cast<std::size_t>(index.ids[i]) : i;
            if (pass && !pass[id]) continue;
            if (index.dead.test(id)) continue;
            ++scored;
            float s = dot(query, v.row(i), v.dim);
            if (s >= top.threshold()) top.push(static_cast<std::int64_t>(id), s);
        }
        r.distance_computations = scored;
        if (f) r.counters["filter_rows"] = static_cast<double>(f->rows);
    }
    top.result(r.ids, r.scores);
    return r;
}

std::size_t index_bytes(const Index& index) {
    if (!index.changed) return 0;
    const Matrix& v = index.compacted ? index.own : *index.vectors;
    return v.rows * v.dim * sizeof(float) + index.ids.size() * sizeof(std::int32_t) +
           index.dead.bytes();
}

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

void delete_rows(Index& index, const std::vector<std::uint8_t>& mask) {
    if (index.compacted) throw std::runtime_error("flat: delete after compact is not supported");
    if (mask.size() != index.n_rows) throw std::runtime_error("flat: delete mask size != rows");
    index.dead.set(mask);
    index.changed = true;
}

void update_rows(Index& index, const std::vector<std::int64_t>& ids, const Matrix& rows) {
    if (index.compacted) throw std::runtime_error("flat: update after compact is not supported");
    Matrix& v = *index.vectors;
    if (rows.rows != ids.size() || rows.dim != v.dim)
        throw std::runtime_error("flat: update ids and rows differ in shape");
    for (std::size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] < 0 || static_cast<std::size_t>(ids[j]) >= v.rows)
            throw std::runtime_error("flat: update ID out of range");
        std::copy(rows.row(j), rows.row(j) + v.dim, v.row(static_cast<std::size_t>(ids[j])));
    }
    index.changed = true;
}

void compact(Index& index, CompactMode /*mode*/) {
    // Nothing to drop (an update only): the arrays stay as they are.
    if (index.compacted || !index.dead.any()) return;
    const Matrix& v = *index.vectors;
    if (v.rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("flat: corpus too large for int32 IDs");
    index.own.dim = v.dim;
    index.own.rows = v.rows - index.dead.count;
    index.own.data.resize(index.own.rows * v.dim);
    index.ids.clear();
    index.ids.reserve(index.own.rows);
    std::size_t j = 0;
    for (std::size_t i = 0; i < v.rows; ++i) {
        if (index.dead.test(i)) continue;
        std::copy(v.row(i), v.row(i) + v.dim, index.own.row(j++));
        index.ids.push_back(static_cast<std::int32_t>(i));
    }
    index.dead.clear();
    index.compacted = true;
    index.changed = true;
}

}  // namespace vro::flat
