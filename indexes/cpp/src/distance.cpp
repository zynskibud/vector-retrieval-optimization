#include "distance.hpp"

#include <algorithm>
#include <limits>

namespace vro {

float dot(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float l2_sq(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
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
