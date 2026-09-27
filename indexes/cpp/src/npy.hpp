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

// Reads a 1-D '<i8' array (for example update_upd10_ids.npy, CONTRACT 13.1).
std::vector<std::int64_t> read_i64_1d(const std::string& path);

// Reads a 1-D '|b1' (bool) array: one byte per entry, 0 or 1.
std::vector<std::uint8_t> read_bool(const std::string& path);

}  // namespace vro::npy
