#include "kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "distance.hpp"
#include "splitmix.hpp"

namespace vro {

std::size_t default_train_size(std::size_t n, std::size_t k) {
    return std::max(std::min(n, 256 * k), k);
}

std::size_t nearest_center(const float* point, const float* centers, std::size_t k,
                           std::size_t dim, Metric metric, float* score_out) {
    std::size_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < k; ++c) {
        const float* center = centers + c * dim;
        float s = metric == Metric::kDot ? dot(point, center, dim) : -l2_sq(point, center, dim);
        if (s > best_score) {
            best_score = s;
            best = c;
        }
    }
    if (score_out) *score_out = best_score;
    return best;
}

namespace {

// Step 2: k distinct training rows, drawn with next_below; a repeat draws again.
std::vector<float> init_centers(const float* points, std::size_t train_n, std::size_t dim,
                                std::size_t k, std::uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<bool> taken(train_n, false);
    std::vector<float> centers(k * dim);
    for (std::size_t c = 0; c < k; ++c) {
        std::size_t row;
        do {
            row = static_cast<std::size_t>(rng.next_below(train_n));
        } while (taken[row]);
        taken[row] = true;
        std::copy(points + row * dim, points + (row + 1) * dim, centers.begin() + c * dim);
    }
    return centers;
}

// Assigns every training row to its best center, in parallel over row ranges.
// Returns true if any assignment changed.
bool assign_all(const float* points, std::size_t train_n, std::size_t dim,
                const std::vector<float>& centers, std::size_t k, Metric metric, int threads,
                std::vector<std::int64_t>& assign, std::vector<float>& score) {
    std::size_t t_count = static_cast<std::size_t>(std::max(1, threads));
    t_count = std::min(t_count, std::max<std::size_t>(1, train_n));
    std::vector<char> changed(t_count, 0);
    auto work = [&](std::size_t t) {
        std::size_t begin = train_n * t / t_count;
        std::size_t end = train_n * (t + 1) / t_count;
        for (std::size_t i = begin; i < end; ++i) {
            float s = 0.0f;
            auto c = static_cast<std::int64_t>(
                nearest_center(points + i * dim, centers.data(), k, dim, metric, &s));
            if (c != assign[i]) changed[t] = 1;
            assign[i] = c;
            score[i] = s;
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < t_count; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();
    return std::any_of(changed.begin(), changed.end(), [](char c) { return c != 0; });
}

void normalize_row(float* v, std::size_t dim) {
    float norm = std::sqrt(dot(v, v, dim));
    if (norm > 0.0f)
        for (std::size_t j = 0; j < dim; ++j) v[j] /= norm;
}

// Step (c): for each empty cluster in index order, move the worst-fit point
// (lowest score to its own center) into it. Only points whose cluster has at
// least 2 members qualify, so no cluster becomes empty by donating.
void fix_empty(std::size_t k, std::vector<std::int64_t>& labels, const std::vector<float>& score) {
    std::vector<std::size_t> counts(k, 0);
    for (auto l : labels) ++counts[static_cast<std::size_t>(l)];
    std::vector<bool> moved(labels.size(), false);
    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] != 0) continue;
        std::size_t worst = labels.size();
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (moved[i] || counts[static_cast<std::size_t>(labels[i])] < 2) continue;
            if (worst == labels.size() || score[i] < score[worst]) worst = i;
        }
        if (worst == labels.size()) break;  // cannot happen while train_n >= k
        --counts[static_cast<std::size_t>(labels[worst])];
        ++counts[c];
        labels[worst] = static_cast<std::int64_t>(c);
        moved[worst] = true;  // its old score no longer applies to its new cluster
    }
}

// Step (d): each center = mean of its points, then L2-normalized unless disabled.
void update_centers(const float* points, std::size_t dim, std::size_t k, bool normalize,
                    const std::vector<std::int64_t>& labels, std::vector<float>& centers) {
    std::vector<double> sums(k * dim, 0.0);
    std::vector<std::size_t> counts(k, 0);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        auto c = static_cast<std::size_t>(labels[i]);
        ++counts[c];
        const float* p = points + i * dim;
        for (std::size_t j = 0; j < dim; ++j) sums[c * dim + j] += p[j];
    }
    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        float* center = centers.data() + c * dim;
        for (std::size_t j = 0; j < dim; ++j)
            center[j] = static_cast<float>(sums[c * dim + j] / static_cast<double>(counts[c]));
        if (normalize) normalize_row(center, dim);
    }
}

}  // namespace

std::vector<float> kmeans(const float* points, std::size_t n, std::size_t dim,
                          const KMeansOptions& opt) {
    std::size_t k = opt.k;
    std::size_t train_n = opt.train_size ? std::min(opt.train_size, n) : default_train_size(n, k);
    train_n = std::min(std::max(train_n, k), n);
    if (k == 0 || train_n < k) throw std::invalid_argument("kmeans: need at least k training rows");

    std::vector<float> centers = init_centers(points, train_n, dim, k, opt.seed);
    std::vector<std::int64_t> assign(train_n, -1);
    std::vector<float> score(train_n, 0.0f);
    // Per iteration: (a) assign, (b) stop if no label changed since the
    // previous (a), (c) fix empty clusters, (d) recompute centers.
    for (int it = 0; it < opt.iters; ++it) {
        bool changed = assign_all(points, train_n, dim, centers, k, opt.metric, opt.threads,
                                  assign, score);
        if (!changed) break;
        std::vector<std::int64_t> labels = assign;  // (b) compares against unfixed labels
        fix_empty(k, labels, score);
        update_centers(points, dim, k, opt.normalize, labels, centers);
    }
    return centers;
}

}  // namespace vro
