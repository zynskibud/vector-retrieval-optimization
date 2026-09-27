// Types shared by bench and every index module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vro {

// Row-major float32 matrix: rows x dim, one contiguous buffer.
struct Matrix {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t dim = 0;

    const float* row(std::size_t i) const { return data.data() + i * dim; }
    float* row(std::size_t i) { return data.data() + i * dim; }
};

// Thrown by an index module that has no implementation yet. bench exits 1.
struct NotImplemented : std::runtime_error {
    explicit NotImplemented(const std::string& index)
        : std::runtime_error(index + ": not implemented") {}
};

// Thrown for a bad parameter value. bench exits 2.
struct ParamError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parameter values as strings, with typed getters. bench fills in every
// default before an index sees it, so a missing key is a program error.
class Params {
public:
    std::map<std::string, std::string> values;

    bool has(const std::string& key) const { return values.count(key) != 0; }
    const std::string& get_string(const std::string& key) const;
    long long get_int(const std::string& key) const;
    double get_double(const std::string& key) const;
};

// Everything that build() needs besides vectors and params.
struct BuildContext {
    BuildContext() = default;
    BuildContext(int t, std::uint64_t s, std::string out, std::string data = "")
        : threads(t), seed(s), out_path(std::move(out)), data_dir(std::move(data)) {}
    int threads = 1;
    std::uint64_t seed = 42;
    std::string out_path;  // the output JSON path; diskann writes <out_path>.diskann
    std::string data_dir;  // the --data directory; filtered search reads filter_<name>.npy here
    std::size_t build_rows = 0;  // hnsw: build on the first build_rows rows only (0 = all); CONTRACT 12.2
};

// Metadata filter (CONTRACT 11). pass[i] is 1 if row i passes; pass has one
// byte per corpus row. rows = number of passing rows.
struct FilterMask {
    std::vector<std::uint8_t> pass;
    std::size_t rows = 0;
};

// Returns the mask for params "filter", or nullptr for "none" or a missing key.
// Loads <data_dir>/filter_<name>.npy on first use and caches it (thread-safe).
// The mask is cut to the first n rows (bench --limit). Bad name: ParamError.
const FilterMask* get_filter(const std::string& data_dir, const Params& params, std::size_t n);

// Tombstones (CONTRACT 13.3): one bit per corpus row; bit i set = row i is
// deleted. Empty (no bits) until the first delete, so bytes() is 0 on runs
// without changes. index_bytes counts bytes() (N/8, rounded up to 8 bytes).
struct Tombstones {
    std::vector<std::uint64_t> bits;
    std::size_t count = 0;  // number of set bits

    bool any() const { return count != 0; }
    bool test(std::size_t i) const {
        return count != 0 && ((bits[i >> 6] >> (i & 63)) & 1u) != 0;
    }
    // Sets the bits of every row with mask[i] != 0. mask has one byte per row.
    void set(const std::vector<std::uint8_t>& mask) {
        if (bits.size() * 64 < mask.size()) bits.resize((mask.size() + 63) / 64, 0);
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            std::uint64_t b = std::uint64_t{1} << (i & 63);
            if ((bits[i >> 6] & b) == 0) {
                bits[i >> 6] |= b;
                ++count;
            }
        }
    }
    void clear() {
        bits.clear();
        bits.shrink_to_fit();
        count = 0;
    }
    std::size_t bytes() const { return bits.size() * sizeof(std::uint64_t); }
};

// --compact-mode (CONTRACT 13.3).
enum class CompactMode { kRebuild, kRepair };

struct BuildTimes {
    double train_s = 0.0;
    double add_s = 0.0;
};

// One query's result. ids and scores have k entries, best first.
// Pad: id -1 and score -inf.
struct SearchResult {
    std::vector<std::int64_t> ids;
    std::vector<float> scores;
    std::int64_t distance_computations = -1;  // dot products in this query; -1 = not counted
    // Extra per-query counters, for example "disk_reads". bench writes the mean
    // per query to searches[i].extra.
    std::map<std::string, double> counters;
};

}  // namespace vro
