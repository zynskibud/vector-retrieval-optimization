// Types shared by bench and every index module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
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
    int threads = 1;
    std::uint64_t seed = 42;
    std::string out_path;  // the output JSON path; diskann writes <out_path>.diskann
};

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
