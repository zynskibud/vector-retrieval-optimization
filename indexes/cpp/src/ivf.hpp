// ivf: inverted file with k-means clusters (CONTRACT 6.3).
// Recall floor (CONTRACT 9, item 4): recall@10 >= 0.75 at nprobe=8 on the dev set.
//
// Build: train = shared k-means (dot product, normalized centers).
// Add = assign each corpus row to its best center (std::thread over rows).
// Lists use CSR layout: list_ids holds the row IDs grouped by list, and the
// IDs of list c are list_ids[offsets[c] .. offsets[c+1]).
//
// Changes (CONTRACT 13.3): delete_rows sets tombstone bits; search skips
// tombstoned IDs in the scanned lists. update_rows overwrites the corpus rows,
// assigns each updated row to its new best center, and rebuilds the CSR lists
// once for the batch (a counting sort, O(N)), so each ID moves to its new list.
// compact rebuilds the CSR lists without the tombstoned IDs and drops the
// tombstones (both modes). index_bytes counts the tombstone bit set.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "vro.hpp"

namespace vro::ivf {

struct Index {
    Matrix* vectors = nullptr;         // the corpus; search scans full vectors
    std::unique_ptr<Matrix> loaded;    // after load(): the rows from the file (vectors points here)
    std::string data_dir;               // for filter_<name>.npy (CONTRACT 11)
    std::size_t nlist = 0;
    std::size_t dim = 0;
    std::vector<float> centers;         // nlist x dim, row-major
    std::vector<std::int32_t> list_ids; // N row IDs, grouped by list
    std::vector<std::int32_t> offsets;  // nlist + 1 entries; offsets[nlist] == N
    Tombstones dead;                    // CONTRACT 13.3
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

// CONTRACT 13.3. mask: one byte per corpus row, 1 = delete.
void delete_rows(Index& index, const std::vector<std::uint8_t>& mask);
// Overwrites row ids[j] with rows.row(j) and moves the ID to its new list.
void update_rows(Index& index, const std::vector<std::int64_t>& ids, const Matrix& rows);
// Drops tombstoned IDs from the lists. Both modes do the same for ivf.
void compact(Index& index, CompactMode mode);

// CONTRACT 15.1: writes the index to a .vro file. meta supplies build_params
// and seed; save fills in index, n, and dim. Returns the file size in bytes.
std::uint64_t save(const Index& index, const std::string& path, const vrofile::Meta& meta);
// Loads a .vro file. Refuses (vrofile::FormatError) a file whose index, dim
// (ctx.expect_dim), or build_params (every key of params) differ. The index
// owns its vectors. No rebuild and no repair.
Index load(const std::string& path, const Params& params, const BuildContext& ctx);

}  // namespace vro::ivf
