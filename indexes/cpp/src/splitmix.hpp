// SplitMix64 PRNG (CONTRACT 5). Same sequence in all four languages.
#pragma once

#include <cstdint>

namespace vro {

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next_u64();
    double next_f64();                          // in [0, 1)
    std::uint64_t next_below(std::uint64_t n);  // next_u64() % n

private:
    std::uint64_t state_;
};

}  // namespace vro
