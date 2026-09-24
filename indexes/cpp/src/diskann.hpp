// diskann: Vamana graph on disk with PQ codes in RAM (CONTRACT 6.7).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.90 at l=100 on the dev set, for metric=ip and metric=l2.
//
// Build: medoid entry point, random init graph, two Vamana passes (alpha 1.0,
// then alpha) with full-precision vectors, then PQ codes (pq::build with
// train_size = N), then the file <out_path>.diskann. After the file is
// written, build() clears the corpus Matrix: search reads full vectors and
// edges only from the file.
// threads == 1: rows are processed strictly in row order (deterministic).
// threads > 1: rows are processed in parallel in chunks of 64 consecutive
// rows, with a stripe of 65536 mutexes over node IDs. A thread holds at most
// one node lock at a time, and every write keeps the list at <= r distinct
// edges without self-loops.
// Distance in robust-prune: 2 - 2 * dot (squared Euclidean distance of unit
// vectors), computed from the dot product.
// Search is safe from several threads at once when all use the same io mode.
// Switching io mode (mmap <-> nocache) must not overlap with other searches.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "kmeans.hpp"

namespace vro::diskann {

struct FileIo;  // mmap / uncached file access, defined in diskann.cpp

struct Index {
    std::size_t n = 0;
    std::size_t dim = 0;
    std::size_t r = 64;             // edge slots per record
    std::size_t record_bytes = 0;   // dim*4 + r*4, rounded up to a multiple of 4096
    std::int32_t entry = -1;        // medoid
    Metric metric = Metric::kDot;   // PQ codebooks and table only
    std::size_t m = 0;              // PQ sub-vectors
    std::size_t dsub = 0;           // dim / m
    std::vector<float> codebooks;     // (m, 256, dsub)
    std::vector<std::uint8_t> codes;  // (n, m)
    std::string path;               // <out_path>.diskann
    std::size_t disk_bytes = 0;     // file size
    double mean_out_degree = 0.0;
    int threads = 1;
    BuildTimes times;
    std::shared_ptr<FileIo> io;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

}  // namespace vro::diskann
