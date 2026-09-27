#include "distance.hpp"

#include <algorithm>
#include <limits>

namespace vro {

float dot(const float* a, const float* b, std::size_t n) {
    // 16 lanes, then a left-to-right sum of the lanes, then the tail: the same
    // order of float additions as the Rust reference (indexes/rust/src/distance.rs),
    // so both languages get bit-identical scores and walk the graph the same way.
    constexpr std::size_t W = 16;
    const std::size_t body = n - n % W;
    float acc[W] = {};
    for (std::size_t i = 0; i < body; i += W)
        for (std::size_t l = 0; l < W; ++l) acc[l] += a[i + l] * b[i + l];
    float tail = 0.0f;
    for (std::size_t i = body; i < n; ++i) tail += a[i] * b[i];
    float sum = 0.0f;
    for (std::size_t l = 0; l < W; ++l) sum += acc[l];
    return sum + tail;
}

float l2_sq(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

namespace {

// Strict order "a is better than b": higher score, then lower id.
// Used as the heap's less-than, so the heap top is the worst item, and as
// the sort order, so a sort puts the best item first.
struct Better {
    template <typename T>
    bool operator()(const T& a, const T& b) const {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    }
};

}  // namespace

TopK::TopK(std::size_t k) : k_(k) { heap_.reserve(k + 1); }

float TopK::threshold() const {
    if (heap_.size() < k_) return -std::numeric_limits<float>::infinity();
    return heap_.front().score;
}

void TopK::push(std::int64_t id, float score) {
    if (k_ == 0) return;
    Better better;
    Item item{score, id};
    if (heap_.size() < k_) {
        heap_.push_back(item);
        std::push_heap(heap_.begin(), heap_.end(), better);
        return;
    }
    // Replace the worst item only if the new item is better.
    if (!better(item, heap_.front())) return;
    std::pop_heap(heap_.begin(), heap_.end(), better);
    heap_.back() = item;
    std::push_heap(heap_.begin(), heap_.end(), better);
}

void TopK::result(std::vector<std::int64_t>& ids, std::vector<float>& scores) const {
    std::vector<Item> sorted = heap_;
    std::sort(sorted.begin(), sorted.end(), Better{});  // best first
    ids.assign(k_, -1);
    scores.assign(k_, -std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        ids[i] = sorted[i].id;
        scores[i] = sorted[i].score;
    }
}

}  // namespace vro
