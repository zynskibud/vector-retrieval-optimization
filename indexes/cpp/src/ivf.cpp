#include "ivf.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
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
    TopK top(k);
    std::int64_t scanned = 0;
    for (std::size_t p = 0; p < nprobe; ++p) {
        const auto c = static_cast<std::size_t>(cs[p].second);
        const std::int32_t lo = index.offsets[c], hi = index.offsets[c + 1];
        scanned += hi - lo;
        for (std::int32_t j = lo; j < hi; ++j) {
            const std::int32_t id = index.list_ids[static_cast<std::size_t>(j)];
            float s = dot(query, v.row(static_cast<std::size_t>(id)), dim);
            if (s >= top.threshold()) top.push(id, s);
        }
    }
    SearchResult r;
    top.result(r.ids, r.scores);
    r.distance_computations = static_cast<std::int64_t>(index.nlist) + scanned;
    return r;
}

std::size_t index_bytes(const Index& index) {
    return index.centers.size() * sizeof(float) + index.list_ids.size() * sizeof(std::int32_t) +
           index.offsets.size() * sizeof(std::int32_t);
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

}  // namespace vro::ivf
