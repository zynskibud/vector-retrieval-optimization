// Tests for the .vro index file (CONTRACT 15.4) on flat, ivf, hnsw.
// Usage: test_vro <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
// Uses the first 20000 dev rows, all 1000 queries, and the dev del30 set.
#include <sys/wait.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "flat.hpp"
#include "hnsw.hpp"
#include "ivf.hpp"
#include "npy.hpp"
#include "vro.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

constexpr std::size_t kRows = 20000, kK = 10;
std::string g_root, g_bench;

vro::Params params(std::initializer_list<std::pair<const std::string, std::string>> kv) {
    vro::Params p;
    p.values = kv;
    return p;
}

using SearchFn = std::function<vro::SearchResult(const float*)>;

// Number of queries whose IDs or scores differ (bitwise) between a and b.
std::size_t differ(const SearchFn& a, const SearchFn& b, const vro::Matrix& q) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::SearchResult ra = a(q.row(i)), rb = b(q.row(i));
        if (ra.ids != rb.ids ||
            std::memcmp(ra.scores.data(), rb.scores.data(), ra.scores.size() * sizeof(float)) != 0 ||
            ra.scores.size() != rb.scores.size())
            ++bad;
    }
    return bad;
}

int run_bench(const std::string& args) {
    std::string cmd = "\"" + g_bench + "\" --data \"" + g_root + "/data/processed/dev\" " + args;
    int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

// Save, load, compare at `search` with and without del30 and with filter=top10.
template <typename Idx, typename Build, typename Search, typename Save, typename Load, typename Del,
          typename Compact>
void round_trip(const std::string& name, const vro::Matrix& base, const vro::Matrix& q,
                const std::vector<std::uint8_t>& dead, const vro::Params& bp,
                const nlohmann::ordered_json& bp_json, const vro::Params& sp, Build build,
                Search search, Save save, Load load, Del del, Compact compact) {
    const std::string dev = g_root + "/data/processed/dev";
    const std::string path = "/tmp/vro-cpp-test-" + name + ".vro";
    vro::BuildContext ctx{4, 42, "", dev};
    ctx.expect_dim = base.dim;
    vro::vrofile::Meta meta;
    meta.build_params = bp_json;
    meta.seed = 42;
    vro::Params sp_filter = sp;
    sp_filter.values["filter"] = "top10";

    vro::Matrix v = base;
    Idx ix = build(v, ctx);
    save(ix, path, meta);
    {
        Idx lx = load(path, bp, ctx);
        SearchFn a = [&](const float* x) { return search(ix, x, sp); };
        SearchFn b = [&](const float* x) { return search(lx, x, sp); };
        std::size_t d0 = differ(a, b, q);
        SearchFn af = [&](const float* x) { return search(ix, x, sp_filter); };
        SearchFn bf = [&](const float* x) { return search(lx, x, sp_filter); };
        std::size_t d1 = differ(af, bf, q);
        std::cerr << name << ": queries that differ after load: " << d0 << ", with filter=top10: " << d1 << "\n";
        CHECK(d0 == 0);
        CHECK(d1 == 0);
        // A change on the loaded index works like on the built one.
        del(lx, dead);
        del(ix, dead);
        std::size_t d2 = differ(a, b, q);
        std::cerr << name << ": after del30 on both: " << d2 << "\n";
        CHECK(d2 == 0);
    }
    // del30 then save, load: identical, filter too.
    save(ix, path, meta);
    Idx lx = load(path, bp, ctx);
    SearchFn a = [&](const float* x) { return search(ix, x, sp); };
    SearchFn b = [&](const float* x) { return search(lx, x, sp); };
    SearchFn af = [&](const float* x) { return search(ix, x, sp_filter); };
    SearchFn bf = [&](const float* x) { return search(lx, x, sp_filter); };
    std::size_t d3 = differ(a, b, q), d4 = differ(af, bf, q);
    std::cerr << name << ": del30 saved and loaded: " << d3 << ", with filter=top10: " << d4 << "\n";
    CHECK(d3 == 0);
    CHECK(d4 == 0);

    // The header: parses, 64-byte-aligned offsets, tombstones match del30.
    vro::vrofile::Reader r(path);
    CHECK(r.index() == name);
    CHECK(r.n() == kRows);
    CHECK(r.dim() == base.dim);
    for (const auto& s : r.sections()) CHECK(s.offset % 64 == 0);
    std::vector<std::uint8_t> bits = r.read_u8("tombstones", {(kRows + 7) / 8});
    std::size_t mism = 0;
    for (std::size_t i = 0; i < kRows; ++i)
        if (((bits[i >> 3] >> (i & 7)) & 1u) != dead[i]) ++mism;
    CHECK(mism == 0);

    // Refusals: wrong expected dim, wrong index, other build_params.
    vro::BuildContext ctx2 = ctx;
    ctx2.expect_dim = base.dim + 1;
    bool refused = false;
    try { load(path, bp, ctx2); } catch (const vro::vrofile::FormatError&) { refused = true; }
    CHECK(refused);

    // del30, compact (rebuild), save, load: the file describes all N rows
    // (dropped rows tombstoned); the loaded index searches identically.
    vro::Matrix vc = base;
    Idx cx = build(vc, ctx);
    del(cx, dead);
    compact(cx, vro::CompactMode::kRebuild);
    save(cx, path, meta);
    Idx cl = load(path, bp, ctx);
    SearchFn ca = [&](const float* x) { return search(cx, x, sp); };
    SearchFn cb = [&](const float* x) { return search(cl, x, sp); };
    SearchFn caf = [&](const float* x) { return search(cx, x, sp_filter); };
    SearchFn cbf = [&](const float* x) { return search(cl, x, sp_filter); };
    std::size_t d5 = differ(ca, cb, q), d6 = differ(caf, cbf, q);
    std::cerr << name << ": compacted, saved, loaded: " << d5 << ", with filter=top10: " << d6 << "\n";
    CHECK(d5 == 0);
    CHECK(d6 == 0);
    vro::vrofile::Reader rc(path);
    CHECK(rc.n() == kRows);
    bits = rc.read_u8("tombstones", {(kRows + 7) / 8});
    mism = 0;
    for (std::size_t i = 0; i < kRows; ++i)
        if (((bits[i >> 3] >> (i & 7)) & 1u) != dead[i]) ++mism;
    CHECK(mism == 0);
}

// Rewrites the header of src with dim = dim + 1 (same length: 384 -> 385).
void write_dim_changed(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::size_t pos = data.find("\"dim\":384");
    if (pos == std::string::npos) throw std::runtime_error("no dim in header");
    data.replace(pos, 9, "\"dim\":385");
    std::ofstream out(dst, std::ios::binary);
    out << data;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_vro <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        const std::string dev = g_root + "/data/processed/dev";
        vro::Matrix v = vro::npy::read_f32(dev + "/vectors.npy", kRows);
        vro::Matrix q = vro::npy::read_f32(dev + "/queries.npy");
        std::vector<std::uint8_t> dead = vro::npy::read_bool(dev + "/delete_del30.npy");
        dead.resize(kRows);

        round_trip<vro::flat::Index>(
            "flat", v, q, dead, vro::Params{}, nlohmann::ordered_json::object(), vro::Params{},
            [](vro::Matrix& m, const vro::BuildContext& c) { return vro::flat::build(m, {}, c); },
            [](const vro::flat::Index& i, const float* x, const vro::Params& p) { return vro::flat::search(i, x, kK, p); },
            [](const vro::flat::Index& i, const std::string& p, const vro::vrofile::Meta& m) { vro::flat::save(i, p, m); },
            [](const std::string& p, const vro::Params& bp, const vro::BuildContext& c) { return vro::flat::load(p, bp, c); },
            [](vro::flat::Index& i, const std::vector<std::uint8_t>& d) { vro::flat::delete_rows(i, d); },
            [](vro::flat::Index& i, vro::CompactMode m) { vro::flat::compact(i, m); });

        const vro::Params ivf_bp = params({{"nlist", "256"}, {"iters", "20"}, {"train_size", "20000"}});
        round_trip<vro::ivf::Index>(
            "ivf", v, q, dead, ivf_bp,
            nlohmann::ordered_json{{"nlist", 256}, {"train_size", 20000}, {"iters", 20}},
            params({{"nprobe", "8"}}),
            [&](vro::Matrix& m, const vro::BuildContext& c) { return vro::ivf::build(m, ivf_bp, c); },
            [](const vro::ivf::Index& i, const float* x, const vro::Params& p) { return vro::ivf::search(i, x, kK, p); },
            [](const vro::ivf::Index& i, const std::string& p, const vro::vrofile::Meta& m) { vro::ivf::save(i, p, m); },
            [](const std::string& p, const vro::Params& bp, const vro::BuildContext& c) { return vro::ivf::load(p, bp, c); },
            [](vro::ivf::Index& i, const std::vector<std::uint8_t>& d) { vro::ivf::delete_rows(i, d); },
            [](vro::ivf::Index& i, vro::CompactMode m) { vro::ivf::compact(i, m); });

        const vro::Params hnsw_bp = params({{"m", "16"}, {"ef_construct", "100"}});
        round_trip<vro::hnsw::Index>(
            "hnsw", v, q, dead, hnsw_bp, nlohmann::ordered_json{{"m", 16}, {"ef_construct", 100}},
            params({{"ef", "64"}}),
            [&](vro::Matrix& m, const vro::BuildContext& c) { return vro::hnsw::build(m, hnsw_bp, c); },
            [](const vro::hnsw::Index& i, const float* x, const vro::Params& p) { return vro::hnsw::search(i, x, kK, p); },
            [](const vro::hnsw::Index& i, const std::string& p, const vro::vrofile::Meta& m) { vro::hnsw::save(i, p, m); },
            [](const std::string& p, const vro::Params& bp, const vro::BuildContext& c) { return vro::hnsw::load(p, bp, c); },
            [](vro::hnsw::Index& i, const std::vector<std::uint8_t>& d) { vro::hnsw::delete_rows(i, d); },
            [](vro::hnsw::Index& i, vro::CompactMode m) { vro::hnsw::compact(i, m); });

        // A file with dim changed in the header is refused (the vectors shape no
        // longer matches, and a caller that expects 384 refuses it first).
        write_dim_changed("/tmp/vro-cpp-test-hnsw.vro", "/tmp/vro-cpp-test-dim.vro");
        {
            vro::BuildContext c{1, 42, "", dev};
            c.expect_dim = 384;
            bool refused = false;
            try { vro::hnsw::load("/tmp/vro-cpp-test-dim.vro", hnsw_bp, c); }
            catch (const vro::vrofile::FormatError& e) { refused = true; std::cerr << "dim refused: " << e.what() << "\n"; }
            CHECK(refused);
            c.expect_dim = 0;
            refused = false;
            try { vro::hnsw::load("/tmp/vro-cpp-test-dim.vro", hnsw_bp, c); }
            catch (const vro::vrofile::FormatError&) { refused = true; }
            CHECK(refused);
            refused = false;  // wrong index
            try { vro::flat::load("/tmp/vro-cpp-test-hnsw.vro", {}, c); }
            catch (const vro::vrofile::FormatError&) { refused = true; }
            CHECK(refused);
            refused = false;  // other build_params
            try { vro::hnsw::load("/tmp/vro-cpp-test-hnsw.vro", params({{"m", "8"}}), c); }
            catch (const vro::vrofile::FormatError&) { refused = true; }
            CHECK(refused);
            refused = false;  // wrong magic
            { std::ofstream o("/tmp/vro-cpp-test-bad.vro", std::ios::binary); o << "NOTAVRO!xxxxxxxx"; }
            try { vro::vrofile::Reader bad("/tmp/vro-cpp-test-bad.vro"); }
            catch (const vro::vrofile::FormatError&) { refused = true; }
            CHECK(refused);
        }

        // bench: a build run with --save, then a separate run with --load.
        const std::string lim = " --limit " + std::to_string(kRows);
        for (const std::string idx : {"flat", "ivf", "hnsw"}) {
            const std::string f = "/tmp/vro-cpp-test-bench-" + idx + ".vro";
            const std::string o1 = "/tmp/vro-cpp-test-bench-" + idx + "-save.json";
            const std::string o2 = "/tmp/vro-cpp-test-bench-" + idx + "-load.json";
            const std::string b = idx == "ivf" ? " --build nlist=256" : "";
            CHECK(run_bench("--index " + idx + lim + b + " --out " + o1 + " --save " + f + " 2>/dev/null") == 0);
            CHECK(run_bench("--index " + idx + " --out " + o2 + " --load " + f + " 2>/dev/null") == 0);
            nlohmann::json j1 = read_json(o1), j2 = read_json(o2);
            CHECK(j1["searches"][0]["ids"] == j2["searches"][0]["ids"]);
            CHECK(j1["searches"][0]["scores"] == j2["searches"][0]["scores"]);
            CHECK(j1["build_params"] == j2["build_params"]);
            CHECK(j2["build"]["train_s"] == 0.0 && j2["build"]["add_s"] == 0.0);
            CHECK(j1["extra"].contains("save_s") && j1["extra"].contains("file_bytes"));
            CHECK(j2["extra"].contains("load_s") && j2["extra"]["loaded_from"] == f);
            std::cerr << "bench " << idx << ": file_bytes " << j1["extra"]["file_bytes"]
                      << ", ids identical " << (j1["searches"][0]["ids"] == j2["searches"][0]["ids"]) << "\n";
            // del30 with --save, then --load --delete del30 is not needed: the
            // tombstones are in the file. A load with a conflicting --build exits 2.
            if (idx == "hnsw")
                CHECK(run_bench("--index hnsw --out /tmp/x.json --load " + f + " --build m=8 2>/dev/null") == 2);
            if (idx == "ivf")
                CHECK(run_bench("--index ivf --out /tmp/x.json --load " + f + " --build nlist=256 2>/dev/null") == 0);
        }
        // bench: --delete del30 --save, then --load gives the same IDs.
        {
            const std::string f = "/tmp/vro-cpp-test-bench-hnsw-del.vro";
            CHECK(run_bench("--index hnsw" + lim + " --delete del30 --out /tmp/vro-d1.json --save " + f + " 2>/dev/null") == 0);
            CHECK(run_bench("--index hnsw --out /tmp/vro-d2.json --load " + f + " 2>/dev/null") == 0);
            CHECK(read_json("/tmp/vro-d1.json")["searches"][0]["ids"] == read_json("/tmp/vro-d2.json")["searches"][0]["ids"]);
        }
        // bench: --delete del30 --compact --save, then --load, per index.
        for (const std::string idx : {"flat", "ivf", "hnsw"}) {
            const std::string f = "/tmp/vro-cpp-test-bench-" + idx + "-cmp.vro";
            const std::string b = idx == "ivf" ? " --build nlist=256" : "";
            CHECK(run_bench("--index " + idx + lim + b + " --delete del30 --compact --out /tmp/vro-c1.json --save " + f + " 2>/dev/null") == 0);
            CHECK(run_bench("--index " + idx + " --out /tmp/vro-c2.json --load " + f + " 2>/dev/null") == 0);
            const bool same = read_json("/tmp/vro-c1.json")["searches"][0]["ids"] == read_json("/tmp/vro-c2.json")["searches"][0]["ids"];
            std::cerr << "bench " << idx << " compacted: ids identical " << same << "\n";
            CHECK(same);
            std::remove(f.c_str());
        }
        CHECK(run_bench("--index pq --out /tmp/x.json --save /tmp/x.vro 2>/dev/null") == 2);
        CHECK(run_bench("--index flat --out /tmp/x.json --load /tmp/vro-cpp-test-bench-hnsw.vro 2>/dev/null") == 2);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    for (const char* f : {"flat", "ivf", "hnsw", "dim", "bad", "bench-flat", "bench-ivf", "bench-hnsw", "bench-hnsw-del"})
        std::remove((std::string("/tmp/vro-cpp-test-") + f + ".vro").c_str());
    if (g_failures) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cerr << "vro: PASS\n";
    return 0;
}
