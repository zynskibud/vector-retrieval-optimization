#include "splitmix.hpp"

namespace vro {

std::uint64_t SplitMix64::next_u64() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double SplitMix64::next_f64() {
    return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
}

std::uint64_t SplitMix64::next_below(std::uint64_t n) {
    return next_u64() % n;
}

}  // namespace vro
