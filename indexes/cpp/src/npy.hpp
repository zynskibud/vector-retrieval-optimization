// Hand-written .npy reader, NumPy format version 1.0 (CONTRACT 1.1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

namespace vro::npy {

struct Int64Array {
    std::vector<std::int64_t> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

// Reads a 2-D '<f4' array. If max_rows > 0, reads only the first max_rows rows.
// Throws std::runtime_error with a clear message on any format problem.
Matrix read_f32(const std::string& path, std::size_t max_rows = 0);

// Reads a 2-D '<i8' array.
Int64Array read_i64(const std::string& path);

}  // namespace vro::npy
