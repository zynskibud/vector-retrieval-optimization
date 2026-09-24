// Shared k-means (CONTRACT 6.2), used by ivf, pq, ivf_pq, and diskann.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vro {

enum class Metric { kDot, kL2 };

struct KMeansOptions {
    std::size_t k = 0;
    int iters = 20;
    std::size_t train_size = 0;  // 0 = default: min(n, 256 * k), never below k
    std::uint64_t seed = 42;
    int threads = 1;
    Metric metric = Metric::kDot;  // kL2 for PQ codebooks with metric=l2 (6.4.1)
    bool normalize = true;         // false for PQ codebooks (6.4)
};

// Default training-set size of 6.2: min(n, 256 * k), but never below k.
std::size_t default_train_size(std::size_t n, std::size_t k);

// points: n rows of dim floats, row-major. Uses the first train_size rows.
// Returns the centers, k rows of dim floats, row-major.
std::vector<float> kmeans(const float* points, std::size_t n, std::size_t dim,
                          const KMeansOptions& opt);

// Index of the best center for one point, and its score (dot product, or
// negative squared L2 distance for kL2). Ties go to the lower center index.
std::size_t nearest_center(const float* point, const float* centers, std::size_t k,
                           std::size_t dim, Metric metric, float* score_out = nullptr);

}  // namespace vro
