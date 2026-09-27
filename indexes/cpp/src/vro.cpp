#include "vro.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the .vro writer and reader assume a little-endian host"
#endif

namespace vro::vrofile {

namespace {

using json = nlohmann::ordered_json;

constexpr char kMagic[8] = {'V', 'R', 'O', 'I', 'D', 'X', '0', '1'};
constexpr std::uint64_t kAlign = 64;

std::uint64_t align_up(std::uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

std::uint64_t dtype_size(const std::string& dtype) {
    if (dtype == "f32" || dtype == "int32") return 4;
    if (dtype == "u8") return 1;
    throw FormatError("unknown dtype: " + dtype);
}

std::uint64_t product(const std::vector<std::uint64_t>& shape) {
    std::uint64_t p = 1;
    for (auto d : shape) p *= d;
    return p;
}

std::string shape_str(const std::vector<std::uint64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

// Header text with the given section offsets, starting at data_start.
std::string header_text(const Meta& meta, std::vector<Section>& secs, std::uint64_t data_start) {
    std::uint64_t off = data_start;
    json table = json::array();
    for (auto& s : secs) {
        s.offset = off;
        off = align_up(off + s.bytes);
        table.push_back({{"name", s.name},
                         {"dtype", s.dtype},
                         {"shape", s.shape},
                         {"offset", s.offset},
                         {"bytes", s.bytes}});
    }
    json h;
    h["index"] = meta.index;
    h["n"] = meta.n;
    h["dim"] = meta.dim;
    h["build_params"] = meta.build_params;
    h["seed"] = meta.seed;
    h["contract_version"] = 1;
    h["language"] = "cpp";
    h["sections"] = std::move(table);
    return h.dump(-1, ' ', /*ensure_ascii=*/true);
}

}  // namespace

void Writer::add(const std::string& name, const std::string& dtype,
                 std::vector<std::uint64_t> shape, const void* data) {
    Section s;
    s.name = name;
    s.dtype = dtype;
    s.bytes = product(shape) * dtype_size(dtype);
    s.shape = std::move(shape);
    items_.push_back({std::move(s), data});
}

std::uint64_t Writer::write(const std::string& path, const Meta& meta) const {
    std::vector<Section> secs;
    for (const auto& it : items_) secs.push_back(it.s);
    // The offsets are in the header, and the header length sets the first
    // offset: iterate until the first offset is stable.
    std::uint64_t start = 0;
    std::string text = header_text(meta, secs, start);
    for (int iter = 0; iter < 16; ++iter) {
        std::uint64_t need = align_up(sizeof(kMagic) + 4 + text.size());
        if (need == start) break;
        start = need;
        text = header_text(meta, secs, start);
    }
    if (align_up(sizeof(kMagic) + 4 + text.size()) != start)
        throw std::runtime_error("vro: header offsets did not converge");
    if (text.size() > 0xffffffffULL) throw std::runtime_error("vro: header too long");

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + path);
    const char zeros[kAlign] = {};
    out.write(kMagic, sizeof(kMagic));
    const auto len = static_cast<std::uint32_t>(text.size());
    out.write(reinterpret_cast<const char*>(&len), 4);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    std::uint64_t pos = sizeof(kMagic) + 4 + text.size();
    for (std::size_t i = 0; i < secs.size(); ++i) {
        out.write(zeros, static_cast<std::streamsize>(secs[i].offset - pos));
        const char* p = static_cast<const char*>(items_[i].data);
        std::uint64_t left = secs[i].bytes;
        constexpr std::uint64_t kChunk = std::uint64_t{1} << 30;
        while (left > 0) {
            std::uint64_t c = std::min(left, kChunk);
            out.write(p, static_cast<std::streamsize>(c));
            p += c;
            left -= c;
        }
        pos = secs[i].offset + secs[i].bytes;
    }
    // Pad the end to 64 too, so every section size is covered by the file.
    out.write(zeros, static_cast<std::streamsize>(align_up(pos) - pos));
    pos = align_up(pos);
    out.flush();
    if (!out) throw std::runtime_error("write failed: " + path);
    return pos;
}

Reader::Reader(const std::string& path) : path_(path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    file_bytes_ = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0);
    char magic[8] = {};
    std::uint32_t len = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&len), 4);
    if (!in || std::memcmp(magic, kMagic, 8) != 0)
        throw FormatError(path + ": not a .vro file (wrong magic)");
    if (12 + static_cast<std::uint64_t>(len) > file_bytes_)
        throw FormatError(path + ": header runs past the end of the file");
    std::string text(len, '\0');
    in.read(text.data(), len);
    if (!in) throw FormatError(path + ": short header");
    try {
        header_ = json::parse(text);
        for (const char* key : {"index", "n", "dim", "build_params", "seed", "sections"})
            if (!header_.contains(key)) throw FormatError(std::string("header has no key ") + key);
        const std::uint64_t data_start = align_up(12 + static_cast<std::uint64_t>(len));
        for (const auto& e : header_.at("sections")) {
            Section s;
            s.name = e.at("name").get<std::string>();
            s.dtype = e.at("dtype").get<std::string>();
            s.shape = e.at("shape").get<std::vector<std::uint64_t>>();
            s.offset = e.at("offset").get<std::uint64_t>();
            s.bytes = e.at("bytes").get<std::uint64_t>();
            if (s.offset % kAlign != 0) throw FormatError("section " + s.name + " is not 64-byte aligned");
            if (s.offset < data_start) throw FormatError("section " + s.name + " overlaps the header");
            if (s.offset + s.bytes > file_bytes_) throw FormatError("section " + s.name + " runs past the end of the file");
            if (s.bytes != product(s.shape) * dtype_size(s.dtype))
                throw FormatError("section " + s.name + ": bytes differ from shape x dtype size");
            sections_.push_back(std::move(s));
        }
    } catch (const FormatError& e) {
        throw FormatError(path + ": " + e.what());
    } catch (const json::exception& e) {
        throw FormatError(path + ": bad header: " + e.what());
    }
}

const Section& Reader::section(const std::string& name) const {
    for (const auto& s : sections_)
        if (s.name == name) return s;
    throw FormatError(path_ + ": no section " + name);
}

void Reader::expect(const std::string& index, std::size_t expect_dim, const Params& params) const {
    if (this->index() != index)
        throw FormatError(path_ + ": file holds index " + this->index() + ", expected " + index);
    if (expect_dim != 0 && dim() != expect_dim)
        throw FormatError(path_ + ": file has dim " + std::to_string(dim()) + ", expected " +
                          std::to_string(expect_dim));
    const json& bp = header_.at("build_params");
    for (const auto& [key, value] : params.values) {
        if (!bp.contains(key))
            throw FormatError(path_ + ": build_params has no " + key);
        const json& v = bp.at(key);
        bool same = false;
        try {
            if (v.is_string()) same = v.get<std::string>() == value;
            else if (v.is_number_integer()) same = v.get<long long>() == params.get_int(key);
            else if (v.is_number_float()) same = v.get<double>() == params.get_double(key);
        } catch (const ParamError&) {
            same = false;
        }
        if (!same)
            throw FormatError(path_ + ": build_params." + key + " is " + v.dump() + ", expected " + value);
    }
}

void Reader::read_into(const std::string& name, const std::string& dtype,
                       const std::vector<std::uint64_t>& shape, void* dst) const {
    const Section& s = section(name);
    if (s.dtype != dtype) throw FormatError(path_ + ": section " + name + " has dtype " + s.dtype + ", expected " + dtype);
    if (s.shape != shape)
        throw FormatError(path_ + ": section " + name + " has shape " + shape_str(s.shape) +
                          ", expected " + shape_str(shape));
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path_);
    in.seekg(static_cast<std::streamoff>(s.offset));
    char* p = static_cast<char*>(dst);
    std::uint64_t left = s.bytes;
    constexpr std::uint64_t kChunk = std::uint64_t{1} << 30;
    while (left > 0) {
        std::uint64_t c = std::min(left, kChunk);
        in.read(p, static_cast<std::streamsize>(c));
        p += c;
        left -= c;
    }
    if (!in) throw FormatError(path_ + ": short read in section " + name);
}

std::vector<float> Reader::read_f32(const std::string& name, const std::vector<std::uint64_t>& shape) const {
    std::vector<float> v(product(shape));
    read_into(name, "f32", shape, v.data());
    return v;
}

std::vector<std::int32_t> Reader::read_i32(const std::string& name, const std::vector<std::uint64_t>& shape) const {
    std::vector<std::int32_t> v(product(shape));
    read_into(name, "int32", shape, v.data());
    return v;
}

std::vector<std::uint8_t> Reader::read_u8(const std::string& name, const std::vector<std::uint64_t>& shape) const {
    std::vector<std::uint8_t> v(product(shape));
    read_into(name, "u8", shape, v.data());
    return v;
}

Params params_from_header(const nlohmann::ordered_json& build_params) {
    Params p;
    for (const auto& [key, v] : build_params.items()) {
        if (v.is_string()) p.values[key] = v.get<std::string>();
        else if (v.is_number_integer()) p.values[key] = std::to_string(v.get<long long>());
        else if (v.is_number_float()) p.values[key] = v.dump();
        else throw FormatError("build_params." + key + " is not a string or a number");
    }
    return p;
}

std::vector<std::uint8_t> tombstones_to_bytes(const Tombstones& t, std::size_t n) {
    std::vector<std::uint8_t> b((n + 7) / 8, 0);
    if (!t.any()) return b;
    for (std::size_t i = 0; i < n; ++i)
        if (t.test(i)) b[i >> 3] = static_cast<std::uint8_t>(b[i >> 3] | (1u << (i & 7)));
    return b;
}

Tombstones tombstones_from_bytes(const std::vector<std::uint8_t>& b, std::size_t n) {
    std::vector<std::uint8_t> mask(n, 0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mask[i] = static_cast<std::uint8_t>((b[i >> 3] >> (i & 7)) & 1u);
        count += mask[i];
    }
    Tombstones t;
    if (count > 0) t.set(mask);  // no bits at all when nothing is deleted
    return t;
}

}  // namespace vro::vrofile
