#include "ivf.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "distance.hpp"
#include "kmeans.hpp"

namespace vro::ivf {

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx) {
    const long long nlist_ll = params.has("nlist") ? params.get_int("nlist") : 1024;
    const long long iters = params.has("iters") ? params.get_int("iters") : 20;
    const long long train_ll = params.has("train_size") ? params.get_int("train_size") : 0;
    const std::size_t n = vectors.rows, dim = vectors.dim;
    if (nlist_ll < 1) throw ParamError("ivf: nlist must be >= 1");
    if (iters < 1) throw ParamError("ivf: iters must be >= 1");
    if (train_ll < 0) throw ParamError("ivf: train_size must be >= 0");
    const auto nlist = static_cast<std::size_t>(nlist_ll);
    if (nlist > n) throw ParamError("ivf: nlist must be <= number of corpus rows");
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw ParamError("ivf: corpus too large for int32 IDs");
    std::size_t train_size = static_cast<std::size_t>(train_ll);
    if (train_size > n) train_size = n;
    if (train_size != 0 && train_size < nlist) train_size = nlist;

    Index index;
    index.vectors = &vectors;
    index.data_dir = ctx.data_dir;
    index.nlist = nlist;
    index.dim = dim;

    // Train: k-means on the first train_size rows.
    auto t0 = std::chrono::steady_clock::now();
    KMeansOptions opt;
    opt.k = nlist;
    opt.iters = static_cast<int>(iters);
    opt.train_size = train_size;
    opt.seed = ctx.seed;
    opt.threads = std::max(1, ctx.threads);
    opt.metric = Metric::kDot;
    opt.normalize = true;
    index.centers = kmeans(vectors.data.data(), n, dim, opt);
    index.times.train_s = seconds_since(t0);

    // Add: label every row, then counting sort into CSR lists (row order kept).
    t0 = std::chrono::steady_clock::now();
    std::vector<std::int32_t> labels(n);
    const std::size_t nthreads = std::min<std::size_t>(std::max(1, ctx.threads), std::max<std::size_t>(1, n));
    const std::size_t chunk = (n + nthreads - 1) / nthreads;
    auto work = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i)
            labels[i] = static_cast<std::int32_t>(
                nearest_center(vectors.row(i), index.centers.data(), nlist, dim, Metric::kDot));
    };
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < nthreads; ++t) {
        std::size_t lo = std::min(n, t * chunk), hi = std::min(n, lo + chunk);
        pool.emplace_back(work, lo, hi);
    }
    work(0, std::min(n, chunk));
    for (auto& th : pool) th.join();

    index.offsets.assign(nlist + 1, 0);
    for (std::size_t i = 0; i < n; ++i) ++index.offsets[static_cast<std::size_t>(labels[i]) + 1];
    for (std::size_t c = 0; c < nlist; ++c) index.offsets[c + 1] += index.offsets[c];
    index.list_ids.resize(n);
    std::vector<std::int32_t> cursor(index.offsets.begin(), index.offsets.end() - 1);
    for (std::size_t i = 0; i < n; ++i)
        index.list_ids[static_cast<std::size_t>(cursor[static_cast<std::size_t>(labels[i])]++)] =
            static_cast<std::int32_t>(i);
    index.times.add_s = seconds_since(t0);
    return index;
}

SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params) {
    const long long nprobe_ll = params.has("nprobe") ? params.get_int("nprobe") : 8;
    if (nprobe_ll < 1) throw ParamError("ivf: nprobe must be >= 1");
    const std::size_t nprobe = std::min(static_cast<std::size_t>(nprobe_ll), index.nlist);
    const std::size_t dim = index.dim;

    // Score all centers; choose the nprobe best (ties to the lower center index).
    std::vector<std::pair<float, std::int32_t>> cs(index.nlist);
    for (std::size_t c = 0; c < index.nlist; ++c)
        cs[c] = {dot(query, index.centers.data() + c * dim, dim), static_cast<std::int32_t>(c)};
    auto better = [](const std::pair<float, std::int32_t>& a,
                     const std::pair<float, std::int32_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    std::nth_element(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(nprobe - 1), cs.end(),
                     better);

    const Matrix& v = *index.vectors;
    const FilterMask* f = get_filter(index.data_dir, params, v.rows);
    const std::uint8_t* pass = f ? f->pass.data() : nullptr;
    TopK top(k);
    std::int64_t scanned = 0;  // rows scored (only passing rows when filtered)
    for (std::size_t p = 0; p < nprobe; ++p) {
        const auto c = static_cast<std::size_t>(cs[p].second);
        const std::int32_t lo = index.offsets[c], hi = index.offsets[c + 1];
        for (std::int32_t j = lo; j < hi; ++j) {
            const std::int32_t id = index.list_ids[static_cast<std::size_t>(j)];
            if (pass && !pass[static_cast<std::size_t>(id)]) continue;
            if (index.dead.test(static_cast<std::size_t>(id))) continue;
            ++scanned;
            float s = dot(query, v.row(static_cast<std::size_t>(id)), dim);
            if (s >= top.threshold()) top.push(id, s);
        }
    }
    SearchResult r;
    top.result(r.ids, r.scores);
    r.distance_computations = static_cast<std::int64_t>(index.nlist) + scanned;
    if (f) r.counters["filter_rows"] = static_cast<double>(f->rows);  // omitted for none (11.2)
    return r;
}

std::size_t index_bytes(const Index& index) {
    return index.centers.size() * sizeof(float) + index.list_ids.size() * sizeof(std::int32_t) +
           index.offsets.size() * sizeof(std::int32_t) + index.dead.bytes();
}

std::map<std::string, double> extra(const Index& index) {
    std::size_t largest = 0, empty = 0;
    for (std::size_t c = 0; c < index.nlist; ++c) {
        auto sz = static_cast<std::size_t>(index.offsets[c + 1] - index.offsets[c]);
        largest = std::max(largest, sz);
        if (sz == 0) ++empty;
    }
    return {{"id_bytes", 4.0},  // list IDs are int32 (extra holds numbers only)
            {"largest_list", static_cast<double>(largest)},
            {"empty_lists", static_cast<double>(empty)}};
}

namespace {

// Rebuilds the CSR lists from labels (one per row; -1 = leave the row out).
// Counting sort: IDs keep row order inside each list.
void rebuild_lists(Index& index, const std::vector<std::int32_t>& labels) {
    std::vector<std::int32_t> offsets(index.nlist + 1, 0);
    std::size_t total = 0;
    for (std::int32_t c : labels)
        if (c >= 0) {
            ++offsets[static_cast<std::size_t>(c) + 1];
            ++total;
        }
    for (std::size_t c = 0; c < index.nlist; ++c) offsets[c + 1] += offsets[c];
    std::vector<std::int32_t> ids(total);
    std::vector<std::int32_t> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (labels[i] >= 0)
            ids[static_cast<std::size_t>(cursor[static_cast<std::size_t>(labels[i])]++)] =
                static_cast<std::int32_t>(i);
    index.offsets = std::move(offsets);
    index.list_ids = std::move(ids);
}

// Current label of every row (-1 = in no list), from the CSR lists.
std::vector<std::int32_t> current_labels(const Index& index) {
    std::vector<std::int32_t> labels(index.vectors->rows, -1);
    for (std::size_t c = 0; c < index.nlist; ++c)
        for (std::int32_t j = index.offsets[c]; j < index.offsets[c + 1]; ++j)
            labels[static_cast<std::size_t>(index.list_ids[static_cast<std::size_t>(j)])] =
                static_cast<std::int32_t>(c);
    return labels;
}

}  // namespace

void delete_rows(Index& index, const std::vector<std::uint8_t>& mask) {
    if (mask.size() != index.vectors->rows) throw std::runtime_error("ivf: delete mask size != rows");
    index.dead.set(mask);
}

void update_rows(Index& index, const std::vector<std::int64_t>& ids, const Matrix& rows) {
    Matrix& v = *index.vectors;
    if (rows.rows != ids.size() || rows.dim != v.dim)
        throw std::runtime_error("ivf: update ids and rows differ in shape");
    std::vector<std::int32_t> labels = current_labels(index);
    for (std::size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] < 0 || static_cast<std::size_t>(ids[j]) >= v.rows)
            throw std::runtime_error("ivf: update ID out of range");
        const auto id = static_cast<std::size_t>(ids[j]);
        std::copy(rows.row(j), rows.row(j) + v.dim, v.row(id));
        if (labels[id] < 0) continue;  // compacted away: stays out
        labels[id] = static_cast<std::int32_t>(
            nearest_center(v.row(id), index.centers.data(), index.nlist, index.dim, Metric::kDot));
    }
    rebuild_lists(index, labels);
}

void compact(Index& index, CompactMode /*mode*/) {
    if (!index.dead.any()) return;  // nothing to drop (an update only)
    std::vector<std::int32_t> labels = current_labels(index);
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (index.dead.test(i)) labels[i] = -1;
    rebuild_lists(index, labels);
    index.dead.clear();
}

}  // namespace vro::ivf

namespace vro::ivf {

// The lists are written as they are (CONTRACT 15.1). After a compact,
// list_ids is shorter than N: a dropped row is in no list and gets its
// tombstone bit. The vectors section keeps the corpus rows.
std::uint64_t save(const Index& index, const std::string& path, const vrofile::Meta& meta) {
    const Matrix& v = *index.vectors;
    const std::size_t n = v.rows;
    Tombstones dead_full;
    if (index.list_ids.size() != n) {
        std::vector<std::uint8_t> mask(n, 1);
        for (std::int32_t id : index.list_ids) mask[static_cast<std::size_t>(id)] = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (index.dead.test(i)) mask[i] = 1;
        dead_full.set(mask);
    }
    const Tombstones& t = index.list_ids.size() != n ? dead_full : index.dead;
    std::vector<std::uint8_t> bits = vrofile::tombstones_to_bytes(t, n);
    vrofile::Writer w;
    w.add("vectors", "f32", {n, v.dim}, v.data.data());
    w.add("tombstones", "u8", {bits.size()}, bits.data());
    w.add("centers", "f32", {index.nlist, index.dim}, index.centers.data());
    w.add("list_ids", "int32", {index.list_ids.size()}, index.list_ids.data());
    w.add("list_offsets", "int32", {index.offsets.size()}, index.offsets.data());
    vrofile::Meta m = meta;
    m.index = "ivf";
    m.n = n;
    m.dim = v.dim;
    return w.write(path, m);
}

Index load(const std::string& path, const Params& params, const BuildContext& ctx) {
    vrofile::Reader r(path);
    r.expect("ivf", ctx.expect_dim, params);
    const std::size_t n = r.n(), dim = r.dim();
    const vrofile::Section& cs = r.section("centers");
    if (cs.shape.size() != 2 || cs.shape[1] != dim) throw vrofile::FormatError(path + ": bad centers shape");
    const std::size_t nlist = cs.shape[0];
    if (!r.header().at("build_params").contains("nlist") ||
        r.header().at("build_params").at("nlist").get<std::size_t>() != nlist)
        throw vrofile::FormatError(path + ": centers rows differ from build_params.nlist");
    Index index;
    index.loaded = std::make_unique<Matrix>();
    index.loaded->rows = n;
    index.loaded->dim = dim;
    index.loaded->data = r.read_f32("vectors", {n, dim});
    index.vectors = index.loaded.get();
    index.data_dir = ctx.data_dir;
    index.nlist = nlist;
    index.dim = dim;
    index.centers = r.read_f32("centers", {nlist, dim});
    // list_ids may be shorter than N (rows dropped by a compaction are in no
    // list); its length must equal list_offsets[nlist].
    const vrofile::Section& ls = r.section("list_ids");
    if (ls.shape.size() != 1 || ls.shape[0] > n) throw vrofile::FormatError(path + ": bad list_ids shape");
    index.list_ids = r.read_i32("list_ids", {ls.shape[0]});
    index.offsets = r.read_i32("list_offsets", {nlist + 1});
    if (index.offsets[0] != 0 || static_cast<std::size_t>(index.offsets[nlist]) != index.list_ids.size())
        throw vrofile::FormatError(path + ": list_offsets do not cover list_ids");
    for (std::size_t c = 0; c < nlist; ++c)
        if (index.offsets[c + 1] < index.offsets[c]) throw vrofile::FormatError(path + ": list_offsets decrease");
    for (std::int32_t id : index.list_ids)
        if (id < 0 || static_cast<std::size_t>(id) >= n) throw vrofile::FormatError(path + ": list ID out of range");
    index.dead = vrofile::tombstones_from_bytes(r.read_u8("tombstones", {(n + 7) / 8}), n);
    return index;
}

}  // namespace vro::ivf
