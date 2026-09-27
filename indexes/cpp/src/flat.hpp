// flat: exact search by brute force (CONTRACT 6.1). No parameters.
// Recall floor: recall@10 = 1.0 on the dev set (CONTRACT 9, item 3).
//
// Changes (CONTRACT 13.3): delete_rows sets tombstone bits and search skips
// those rows. update_rows overwrites the corpus rows. compact copies the live
// rows into an array that the index owns (own) with an int32 row -> ID map
// (ids), then drops the tombstones.
// index_bytes: 0 while the corpus array is the index (CONTRACT 4). After a
// change, the index holds its own rows (a table with deleted rows in it), so
// index_bytes = stored rows x dim x 4 + ID map + tombstone bits. This makes
// index_bytes_after < index_bytes_before_compact measure what compaction frees.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace vro::flat {

struct Index {
    Matrix* vectors = nullptr;        // the corpus array is the index
    std::string data_dir;             // for filter_<name>.npy (CONTRACT 11)
    std::size_t n_rows = 0;           // corpus rows (IDs are 0..n_rows-1)
    Tombstones dead;                  // CONTRACT 13.3
    bool changed = false;             // a delete or update was applied
    bool compacted = false;           // rows live in own; ids maps row -> ID
    Matrix own;
    std::vector<std::int32_t> ids;
    BuildTimes times;
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx);
SearchResult search(const Index& index, const float* query, std::size_t k, const Params& params);
std::size_t index_bytes(const Index& index);
std::map<std::string, double> extra(const Index& index);

// CONTRACT 13.3. mask: one byte per corpus row, 1 = delete.
void delete_rows(Index& index, const std::vector<std::uint8_t>& mask);
// Overwrites row ids[j] with rows.row(j).
void update_rows(Index& index, const std::vector<std::int64_t>& ids, const Matrix& rows);
// Drops tombstoned rows. Both modes do the same for flat.
void compact(Index& index, CompactMode mode);

}  // namespace vro::flat
