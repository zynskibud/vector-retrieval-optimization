// The .vro index file (CONTRACT 15.1): one container format that all four
// languages read and write.
//
//   magic      8 bytes   "VROIDX01"
//   header_len uint32, little-endian
//   header     ASCII JSON: index, n, dim, build_params, seed, contract_version,
//              language, sections [{name, dtype, shape, offset, bytes}, ...]
//   padding    zero bytes to the next multiple of 64
//   sections   raw little-endian arrays, each at a 64-byte-aligned offset
//
// dtypes: "f32" (float32), "int32", "u8". The reader reads the section table;
// it never assumes an offset.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common.hpp"

namespace vro::vrofile {

// Thrown when a file does not match what the caller expects (wrong magic,
// index, dim, or build_params). bench exits 2 for a build_params conflict
// that the command line causes, and 1 otherwise.
struct FormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// What the caller knows about the index that is not in the index struct.
struct Meta {
    std::string index;             // "flat" | "ivf" | "hnsw"
    std::size_t n = 0;             // corpus rows
    std::size_t dim = 0;
    nlohmann::ordered_json build_params = nlohmann::ordered_json::object();  // typed, as in the output JSON
    std::uint64_t seed = 42;
};

struct Section {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

// Collects sections, then writes the file in one pass. The data pointers must
// stay valid until write() returns (no copy is made).
class Writer {
public:
    void add(const std::string& name, const std::string& dtype,
             std::vector<std::uint64_t> shape, const void* data);
    // Writes the file. Returns the file size in bytes.
    std::uint64_t write(const std::string& path, const Meta& meta) const;

private:
    struct Item {
        Section s;
        const void* data;
    };
    std::vector<Item> items_;
};

// A parsed file: the header and the section table. Section data is read on
// request from the open path.
class Reader {
public:
    // Parses the magic, header, and section table. Throws FormatError on a
    // wrong magic, a bad header, or a section that is not aligned, not inside
    // the file, or has a size that differs from shape x dtype size.
    explicit Reader(const std::string& path);

    const nlohmann::ordered_json& header() const { return header_; }
    const std::vector<Section>& sections() const { return sections_; }
    const Section& section(const std::string& name) const;  // FormatError if missing
    std::string index() const { return header_.at("index").get<std::string>(); }
    std::size_t n() const { return header_.at("n").get<std::size_t>(); }
    std::size_t dim() const { return header_.at("dim").get<std::size_t>(); }
    std::uint64_t seed() const { return header_.at("seed").get<std::uint64_t>(); }
    std::uint64_t file_bytes() const { return file_bytes_; }

    // Checks the header against what the caller expects. expect_dim 0 = any.
    // params: every key must be in the header's build_params with an equal
    // value. Throws FormatError on a mismatch.
    void expect(const std::string& index, std::size_t expect_dim, const Params& params) const;

    // Typed readers. shape: expected shape; the section must match it exactly.
    std::vector<float> read_f32(const std::string& name, const std::vector<std::uint64_t>& shape) const;
    std::vector<std::int32_t> read_i32(const std::string& name, const std::vector<std::uint64_t>& shape) const;
    std::vector<std::uint8_t> read_u8(const std::string& name, const std::vector<std::uint64_t>& shape) const;
    // Reads a f32 section into dst (bytes of the section), without a copy.
    void read_into(const std::string& name, const std::string& dtype,
                   const std::vector<std::uint64_t>& shape, void* dst) const;

private:
    std::string path_;
    nlohmann::ordered_json header_;
    std::vector<Section> sections_;
    std::uint64_t file_bytes_ = 0;
};

// Params from a header's build_params (values as strings, as bench keeps them).
Params params_from_header(const nlohmann::ordered_json& build_params);

// Tombstones <-> the file's u8 bit set (bit i = row i, LSB first in a byte,
// ceil(n/8) bytes).
std::vector<std::uint8_t> tombstones_to_bytes(const Tombstones& t, std::size_t n);
Tombstones tombstones_from_bytes(const std::vector<std::uint8_t>& b, std::size_t n);

}  // namespace vro::vrofile
