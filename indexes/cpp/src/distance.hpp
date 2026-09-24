// Dot product, squared L2 distance, and top-k selection.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vro {

float dot(const float* a, const float* b, std::size_t n);
float l2_sq(const float* a, const float* b, std::size_t n);

// Keeps the k best (id, score) pairs. Higher score is better. On equal
// scores, the lower id wins. Uses a bounded min-heap: O(log k) per push.
class TopK {
public:
    explicit TopK(std::size_t k);

    void push(std::int64_t id, float score);
    // Lowest score in the heap, or -inf while the heap holds fewer than k items.
    float threshold() const;
    std::size_t size() const { return heap_.size(); }

    // Writes exactly k entries, best first. Pads with id -1 and score -inf.
    void result(std::vector<std::int64_t>& ids, std::vector<float>& scores) const;

private:
    struct Item {
        float score;
        std::int64_t id;
    };
    std::size_t k_;
    std::vector<Item> heap_;
};

}  // namespace vro
