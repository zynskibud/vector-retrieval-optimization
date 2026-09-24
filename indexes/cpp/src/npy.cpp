#include "npy.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vro::npy {
namespace {

struct Header {
    std::string descr;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error(path + ": " + msg);
}

// Returns the text after "'key':" in the header dict, with leading spaces removed.
std::string value_after(const std::string& dict, const std::string& key, const std::string& path) {
    std::string pattern = "'" + key + "':";
    auto pos = dict.find(pattern);
    if (pos == std::string::npos) fail(path, "header has no '" + key + "'");
    pos += pattern.size();
    while (pos < dict.size() && dict[pos] == ' ') ++pos;
    return dict.substr(pos);
}

Header parse_header(const std::string& dict, const std::string& path) {
    Header h;
    std::string d = value_after(dict, "descr", path);
    if (d.empty() || d[0] != '\'') fail(path, "bad 'descr'");
    h.descr = d.substr(1, d.find('\'', 1) - 1);

    std::string f = value_after(dict, "fortran_order", path);
    if (f.rfind("False", 0) != 0) fail(path, "fortran_order must be False");

    std::string s = value_after(dict, "shape", path);
    if (s.empty() || s[0] != '(') fail(path, "bad 'shape'");
    std::string inner = s.substr(1, s.find(')') - 1);
    std::vector<std::size_t> dims;
    std::size_t i = 0;
    while (i < inner.size()) {
        while (i < inner.size() && (inner[i] == ' ' || inner[i] == ',')) ++i;
        if (i >= inner.size()) break;
        std::size_t used = 0;
        dims.push_back(std::stoull(inner.substr(i), &used));
        i += used;
    }
    if (dims.size() != 2) fail(path, "expected a 2-D array");
    h.rows = dims[0];
    h.cols = dims[1];
    return h;
}

// Opens the file, checks magic and version, parses the header. Leaves the
// stream at the data offset (10 + HLEN).
Header open_and_parse(std::ifstream& in, const std::string& path) {
    in.open(path, std::ios::binary);
    if (!in) fail(path, "cannot open file");
    unsigned char pre[10];
    if (!in.read(reinterpret_cast<char*>(pre), 10)) fail(path, "file too short");
    if (std::memcmp(pre, "\x93NUMPY", 6) != 0) fail(path, "not a .npy file (bad magic)");
    if (pre[6] != 1 || pre[7] != 0) fail(path, "only .npy version 1.0 is supported");
    std::size_t hlen = static_cast<std::size_t>(pre[8]) | (static_cast<std::size_t>(pre[9]) << 8);
    std::string dict(hlen, '\0');
    if (!in.read(dict.data(), static_cast<std::streamsize>(hlen))) fail(path, "header truncated");
    return parse_header(dict, path);
}

template <typename T>
void read_data(std::ifstream& in, std::vector<T>& out, std::size_t count, const std::string& path) {
    out.resize(count);
    auto bytes = static_cast<std::streamsize>(count * sizeof(T));
    if (!in.read(reinterpret_cast<char*>(out.data()), bytes)) fail(path, "data truncated");
}

}  // namespace

Matrix read_f32(const std::string& path, std::size_t max_rows) {
    std::ifstream in;
    Header h = open_and_parse(in, path);
    if (h.descr != "<f4") fail(path, "descr is '" + h.descr + "', expected '<f4'");
    Matrix m;
    m.rows = (max_rows > 0 && max_rows < h.rows) ? max_rows : h.rows;
    m.dim = h.cols;
    read_data(in, m.data, m.rows * m.dim, path);
    return m;
}

Int64Array read_i64(const std::string& path) {
    std::ifstream in;
    Header h = open_and_parse(in, path);
    if (h.descr != "<i8") fail(path, "descr is '" + h.descr + "', expected '<i8'");
    Int64Array a;
    a.rows = h.rows;
    a.cols = h.cols;
    read_data(in, a.data, a.rows * a.cols, path);
    return a;
}

}  // namespace vro::npy
