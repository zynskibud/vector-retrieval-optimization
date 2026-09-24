#include "pq.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "distance.hpp"

namespace vro::pq {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx) {
    const long long m_in = params.get_int("m");
    const long long nbits = params.get_int("nbits");
    const std::string& metric = params.get_string("metric");
    const long long train_size = params.get_int("train_size");
    const long long iters = params.get_int("iters");
    if (nbits != 8) throw std::invalid_argument("pq: nbits must be 8, got " + std::to_string(nbits));
    if (metric != "ip" && metric != "l2")
        throw std::invalid_argument("pq: metric must be ip or l2, got " + metric);
    if (m_in <= 0 || vectors.dim % static_cast<std::size_t>(m_in) != 0)
        throw std::invalid_argument("pq: m must divide dim " + std::to_string(vectors.dim) +
                                    ", got " + std::to_string(m_in));
    if (train_size <= 0) throw std::invalid_argument("pq: train_size must be > 0");
    if (iters <= 0) throw std::invalid_argument("pq: iters must be > 0");

    Index idx;
    idx.vectors = &vectors;
    idx.n = vectors.rows;
    idx.dim = vectors.dim;
    idx.m = static_cast<std::size_t>(m_in);
    idx.dsub = idx.dim / idx.m;
    idx.metric = metric == "l2" ? Metric::kL2 : Metric::kDot;
    const std::size_t m = idx.m, dsub = idx.dsub, n = idx.n;
    const int threads = std::max(1, ctx.threads);

    // Train: one k-means per sub-vector position on the first train_n rows.
    auto t0 = std::chrono::steady_clock::now();
    const std::size_t train_n = std::min(n, static_cast<std::size_t>(train_size));
    idx.codebooks.resize(m * kCentroids * dsub);
    std::vector<float> sub(train_n * dsub);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < train_n; ++i)
            std::copy_n(vectors.row(i) + j * dsub, dsub, sub.data() + i * dsub);
        KMeansOptions opt;
        opt.k = kCentroids;
        opt.iters = static_cast<int>(iters);
        opt.train_size = train_n;
        opt.seed = ctx.seed + j;
        opt.threads = threads;
        opt.metric = idx.metric;
        opt.normalize = false;
        std::vector<float> centers = kmeans(sub.data(), train_n, dsub, opt);
        std::copy(centers.begin(), centers.end(), idx.codebooks.begin() + j * kCentroids * dsub);
    }
    idx.times.train_s = seconds_since(t0);

    // Add: encode every row, rows split across threads.
    t0 = std::chrono::steady_clock::now();
    idx.codes.resize(n * m);
    auto encode = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const float* x = vectors.row(i);
            for (std::size_t j = 0; j < m; ++j)
                idx.codes[i * m + j] = static_cast<std::uint8_t>(
                    nearest_center(x + j * dsub, idx.codebooks.data() + j * kCentroids * dsub,
                                   kCentroids, dsub, idx.metric));
        }
    };
    const std::size_t t_count = std::min<std::size_t>(static_cast<std::size_t>(threads), std::max<std::size_t>(n, 1));
    std::vector<std::thread> pool;
    const std::size_t chunk = (n + t_count - 1) / t_count;
    for (std::size_t t = 0; t < t_count; ++t) {
        std::size_t lo = t * chunk, hi = std::min(n, lo + chunk);
        if (lo < hi) pool.emplace_back(encode, lo, hi);
    }
    for (auto& th : pool) th.join();
    idx.times.add_s = seconds_since(t0);
    return idx;
}

SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params) {
    const long long rerank_in = params.get_int("rerank");
    if (rerank_in < 0) throw std::invalid_argument("pq: rerank must be >= 0");
    const std::size_t rerank = static_cast<std::size_t>(rerank_in);
    const std::size_t m = index.m, dsub = index.dsub, n = index.n;
    const bool l2 = index.metric == Metric::kL2;

    // Table T (m, 256): higher is better in both modes.
    std::vector<float> table(m * kCentroids);
    for (std::size_t j = 0; j < m; ++j) {
        const float* qj = query + j * dsub;
        const float* cb = index.codebooks.data() + j * kCentroids * dsub;
        for (std::size_t c = 0; c < kCentroids; ++c)
            table[j * kCentroids + c] = l2 ? -l2_sq(qj, cb + c * dsub, dsub) : dot(qj, cb + c * dsub, dsub);
    }

    TopK top(rerank > 0 ? rerank : k);
    const float* t = table.data();
    const std::uint8_t* codes = index.codes.data();
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* code = codes + i * m;
        float s = 0.0f;
        for (std::size_t j = 0; j < m; ++j) s += t[j * kCentroids + code[j]];
        if (s >= top.threshold()) top.push(static_cast<std::int64_t>(i), s);
    }

    SearchResult r;
    r.distance_computations = static_cast<std::int64_t>(n);
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
        float s = l2 ? -l2_sq(query, x, index.dim) : dot(query, x, index.dim);
        exact.push(id, s);
        ++r.distance_computations;
    }
    exact.result(r.ids, r.scores);
    return r;
}

std::size_t index_bytes(const Index& index) {
    return index.codebooks.size() * sizeof(float) + index.codes.size();
}

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::pq
