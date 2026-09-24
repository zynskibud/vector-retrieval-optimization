#include "ivf_pq.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>

#include "distance.hpp"

namespace vro::ivf_pq {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Runs fn(lo, hi) over [0, n) split into contiguous chunks, one per thread.
template <typename Fn>
void parallel_rows(std::size_t n, int threads, Fn fn) {
    const std::size_t nt = std::min<std::size_t>(static_cast<std::size_t>(std::max(1, threads)),
                                                 std::max<std::size_t>(1, n));
    const std::size_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < nt; ++t) {
        std::size_t lo = std::min(n, t * chunk), hi = std::min(n, lo + chunk);
        if (lo < hi) pool.emplace_back(fn, lo, hi);
    }
    fn(std::size_t{0}, std::min(n, chunk));
    for (auto& th : pool) th.join();
}

long long get_or(const Params& p, const char* key, long long def) {
    return p.has(key) ? p.get_int(key) : def;
}

}  // namespace

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx) {
    const long long nlist_ll = get_or(params, "nlist", 1024);
    const long long iters = get_or(params, "iters", 20);
    const long long m_ll = get_or(params, "m", 48);
    const long long nbits = get_or(params, "nbits", 8);
    const long long train_ll = get_or(params, "train_size", 100000);
    const std::string metric = params.has("metric") ? params.get_string("metric") : "ip";
    const std::size_t n = vectors.rows, dim = vectors.dim;
    if (nlist_ll < 1) throw ParamError("ivf_pq: nlist must be >= 1");
    if (iters < 1) throw ParamError("ivf_pq: iters must be >= 1");
    if (nbits != 8) throw ParamError("ivf_pq: nbits must be 8");
    if (metric != "ip" && metric != "l2") throw ParamError("ivf_pq: metric must be ip or l2");
    if (m_ll < 1 || dim % static_cast<std::size_t>(m_ll) != 0)
        throw ParamError("ivf_pq: m must divide dim " + std::to_string(dim));
    if (train_ll < 1) throw ParamError("ivf_pq: train_size must be >= 1");
    const auto nlist = static_cast<std::size_t>(nlist_ll);
    if (nlist > n) throw ParamError("ivf_pq: nlist must be <= number of corpus rows");
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw ParamError("ivf_pq: corpus too large for int32 IDs");
    std::size_t train_n = std::min(n, static_cast<std::size_t>(train_ll));
    if (train_n < nlist) train_n = nlist;  // 6.2: never below k
    if (train_n < kCentroids && n >= kCentroids) train_n = std::min(n, kCentroids);
    if (train_n < kCentroids) throw ParamError("ivf_pq: need at least 256 training rows");

    Index idx;
    idx.vectors = &vectors;
    idx.n = n;
    idx.dim = dim;
    idx.nlist = nlist;
    idx.m = static_cast<std::size_t>(m_ll);
    idx.dsub = dim / idx.m;
    idx.metric = metric == "l2" ? Metric::kL2 : Metric::kDot;
    const std::size_t m = idx.m, dsub = idx.dsub;
    const int threads = std::max(1, ctx.threads);

    // ---- Train ----
    auto t0 = std::chrono::steady_clock::now();
    KMeansOptions copt;
    copt.k = nlist;
    copt.iters = static_cast<int>(iters);
    copt.train_size = train_n;
    copt.seed = ctx.seed;
    copt.threads = threads;
    copt.metric = Metric::kDot;
    copt.normalize = true;
    idx.centers = kmeans(vectors.data.data(), n, dim, copt);

    // Residuals of the training rows, laid out per sub-vector: sub[j] is (train_n, dsub).
    std::vector<float> resid(train_n * dim);
    parallel_rows(train_n, threads, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const float* x = vectors.row(i);
            std::size_t c = nearest_center(x, idx.centers.data(), nlist, dim, Metric::kDot);
            const float* cc = idx.centers.data() + c * dim;
            float* r = resid.data() + i * dim;
            for (std::size_t d = 0; d < dim; ++d) r[d] = x[d] - cc[d];
        }
    });
    idx.codebooks.resize(m * kCentroids * dsub);
    std::vector<float> sub(train_n * dsub);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < train_n; ++i)
            std::copy_n(resid.data() + i * dim + j * dsub, dsub, sub.data() + i * dsub);
        KMeansOptions opt;
        opt.k = kCentroids;
        opt.iters = static_cast<int>(iters);
        opt.train_size = train_n;
        opt.seed = ctx.seed + j;
        opt.threads = threads;
        opt.metric = idx.metric;
        opt.normalize = false;
        std::vector<float> cb = kmeans(sub.data(), train_n, dsub, opt);
        std::copy(cb.begin(), cb.end(), idx.codebooks.begin() + static_cast<std::ptrdiff_t>(j * kCentroids * dsub));
    }
    resid.clear();
    resid.shrink_to_fit();
    idx.times.train_s = seconds_since(t0);

    // ---- Add: assign, encode residual (row order), then counting sort into CSR ----
    t0 = std::chrono::steady_clock::now();
    std::vector<std::int32_t> labels(n);
    std::vector<std::uint8_t> row_codes(n * m);
    parallel_rows(n, threads, [&](std::size_t lo, std::size_t hi) {
        std::vector<float> r(dim);
        for (std::size_t i = lo; i < hi; ++i) {
            const float* x = vectors.row(i);
            std::size_t c = nearest_center(x, idx.centers.data(), nlist, dim, Metric::kDot);
            labels[i] = static_cast<std::int32_t>(c);
            const float* cc = idx.centers.data() + c * dim;
            for (std::size_t d = 0; d < dim; ++d) r[d] = x[d] - cc[d];
            for (std::size_t j = 0; j < m; ++j)
                row_codes[i * m + j] = static_cast<std::uint8_t>(nearest_center(
                    r.data() + j * dsub, idx.codebooks.data() + j * kCentroids * dsub, kCentroids,
                    dsub, idx.metric));
        }
    });
    idx.offsets.assign(nlist + 1, 0);
    for (std::size_t i = 0; i < n; ++i) ++idx.offsets[static_cast<std::size_t>(labels[i]) + 1];
    for (std::size_t c = 0; c < nlist; ++c) idx.offsets[c + 1] += idx.offsets[c];
    idx.list_ids.resize(n);
    idx.codes.resize(n * m);
    std::vector<std::int32_t> cursor(idx.offsets.begin(), idx.offsets.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const auto p = static_cast<std::size_t>(cursor[static_cast<std::size_t>(labels[i])]++);
        idx.list_ids[p] = static_cast<std::int32_t>(i);
        std::copy_n(row_codes.data() + i * m, m, idx.codes.data() + p * m);
    }
    idx.times.add_s = seconds_since(t0);
    return idx;
}

SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params) {
    const long long nprobe_ll = params.has("nprobe") ? params.get_int("nprobe") : 8;
    const long long rerank_ll = params.has("rerank") ? params.get_int("rerank") : 0;
    if (nprobe_ll < 1) throw ParamError("ivf_pq: nprobe must be >= 1");
    if (rerank_ll < 0) throw ParamError("ivf_pq: rerank must be >= 0");
    const std::size_t nprobe = std::min(static_cast<std::size_t>(nprobe_ll), index.nlist);
    const auto rerank = static_cast<std::size_t>(rerank_ll);
    const std::size_t dim = index.dim, m = index.m, dsub = index.dsub;
    const bool l2 = index.metric == Metric::kL2;

    // Coarse step: nprobe best centers by dot product (ties to the lower index).
    std::vector<std::pair<float, std::int32_t>> cs(index.nlist);
    for (std::size_t c = 0; c < index.nlist; ++c)
        cs[c] = {dot(query, index.centers.data() + c * dim, dim), static_cast<std::int32_t>(c)};
    auto better = [](const std::pair<float, std::int32_t>& a,
                     const std::pair<float, std::int32_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    std::nth_element(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(nprobe - 1), cs.end(),
                     better);

    std::vector<float> table(m * kCentroids);
    auto fill_table = [&](const float* q) {
        for (std::size_t j = 0; j < m; ++j) {
            const float* qj = q + j * dsub;
            const float* cb = index.codebooks.data() + j * kCentroids * dsub;
            for (std::size_t c = 0; c < kCentroids; ++c)
                table[j * kCentroids + c] =
                    l2 ? -l2_sq(qj, cb + c * dsub, dsub) : dot(qj, cb + c * dsub, dsub);
        }
    };
    std::vector<float> qres;
    if (l2) qres.resize(dim);
    else fill_table(query);  // ip: one table for every list

    TopK top(rerank > 0 ? rerank : k);
    std::int64_t scanned = 0;
    const float* t = table.data();
    for (std::size_t p = 0; p < nprobe; ++p) {
        const auto c = static_cast<std::size_t>(cs[p].second);
        float base = 0.0f;
        if (l2) {
            const float* cc = index.centers.data() + c * dim;
            for (std::size_t d = 0; d < dim; ++d) qres[d] = query[d] - cc[d];
            fill_table(qres.data());
        } else {
            base = cs[p].first;
        }
        const std::int32_t lo = index.offsets[c], hi = index.offsets[c + 1];
        scanned += hi - lo;
        for (std::int32_t e = lo; e < hi; ++e) {
            const std::uint8_t* code = index.codes.data() + static_cast<std::size_t>(e) * m;
            float s = base;
            for (std::size_t j = 0; j < m; ++j) s += t[j * kCentroids + code[j]];
            if (s >= top.threshold())
                top.push(index.list_ids[static_cast<std::size_t>(e)], s);
        }
    }

    SearchResult r;
    r.distance_computations = static_cast<std::int64_t>(index.nlist) + scanned;
    if (rerank == 0) {
        top.result(r.ids, r.scores);
        return r;
    }
    std::vector<std::int64_t> cand;
    std::vector<float> approx;
    top.result(cand, approx);
    TopK exact(k);
    const Matrix& v = *index.vectors;
    for (std::int64_t id : cand) {
        if (id < 0) continue;
        const float* x = v.row(static_cast<std::size_t>(id));
        exact.push(id, l2 ? -l2_sq(query, x, dim) : dot(query, x, dim));
        ++r.distance_computations;
    }
    exact.result(r.ids, r.scores);
    return r;
}

std::size_t index_bytes(const Index& index) {
    return index.centers.size() * sizeof(float) + index.list_ids.size() * sizeof(std::int32_t) +
           index.offsets.size() * sizeof(std::int32_t) + index.codebooks.size() * sizeof(float) +
           index.codes.size();
}

std::map<std::string, double> extra(const Index& index) {
    std::size_t largest = 0, empty = 0;
    for (std::size_t c = 0; c < index.nlist; ++c) {
        auto sz = static_cast<std::size_t>(index.offsets[c + 1] - index.offsets[c]);
        largest = std::max(largest, sz);
        if (sz == 0) ++empty;
    }
    return {{"id_bytes", 4.0},
            {"largest_list", static_cast<double>(largest)},
            {"empty_lists", static_cast<double>(empty)}};
}

}  // namespace vro::ivf_pq
