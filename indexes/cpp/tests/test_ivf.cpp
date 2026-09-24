// Tests for the ivf index (CONTRACT 9, items 4 and 7).
// Usage: test_ivf <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
#include <sys/wait.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ivf.hpp"
#include "kmeans.hpp"
#include "npy.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

std::string g_root, g_bench;
std::string dev(const std::string& f) { return g_root + "/data/processed/dev/" + f; }

vro::Params build_params(std::size_t n) {
    vro::Params p;
    p.values["nlist"] = "1024";
    p.values["iters"] = "20";
    p.values["train_size"] = std::to_string(vro::default_train_size(n, 1024));
    return p;
}

double recall_at_10(const vro::ivf::Index& index, const vro::Matrix& queries,
                    const vro::npy::Int64Array& gt, int nprobe, bool check_counts) {
    vro::Params sp;
    sp.values["nprobe"] = std::to_string(nprobe);
    const std::size_t k = 10, n = index.vectors->rows;
    std::size_t hits = 0;
    for (std::size_t qi = 0; qi < queries.rows; ++qi) {
        vro::SearchResult r = vro::ivf::search(index, queries.row(qi), k, sp);
        CHECK(r.ids.size() == k && r.scores.size() == k);
        for (std::size_t a = 0; a < k; ++a) {
            CHECK(r.ids[a] == -1 || (r.ids[a] >= 0 && static_cast<std::size_t>(r.ids[a]) < n));
            for (std::size_t b = 0; b < k; ++b)
                if (r.ids[a] == gt.data[qi * gt.cols + b]) ++hits;
        }
        if (check_counts) {
            CHECK(r.distance_computations >= static_cast<std::int64_t>(index.nlist));
            CHECK(r.distance_computations <= static_cast<std::int64_t>(index.nlist + n));
        }
    }
    return static_cast<double>(hits) / static_cast<double>(queries.rows * k);
}

void test_index() {
    vro::Matrix vectors = vro::npy::read_f32(dev("vectors.npy"));
    vro::Matrix queries = vro::npy::read_f32(dev("queries.npy"));
    vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
    const std::size_t n = vectors.rows;
    vro::Params bp = build_params(n);

    vro::BuildContext ctx4{4, 42, ""};
    vro::ivf::Index idx = vro::ivf::build(vectors, bp, ctx4);

    // CSR structure: offsets end at N, every row appears exactly once.
    CHECK(idx.offsets.size() == 1025);
    CHECK(idx.offsets.front() == 0);
    CHECK(static_cast<std::size_t>(idx.offsets.back()) == n);
    for (std::size_t c = 0; c < 1024; ++c) CHECK(idx.offsets[c] <= idx.offsets[c + 1]);
    std::vector<int> seen(n, 0);
    for (auto id : idx.list_ids) {
        CHECK(id >= 0 && static_cast<std::size_t>(id) < n);
        if (id >= 0 && static_cast<std::size_t>(id) < n) ++seen[static_cast<std::size_t>(id)];
    }
    std::size_t once = 0;
    for (int s : seen) once += (s == 1);
    CHECK(once == n);
    CHECK(vro::ivf::index_bytes(idx) == 1024 * vectors.dim * 4 + n * 4 + 1025 * 4);

    double r8 = recall_at_10(idx, queries, gt, 8, true);
    double r64 = recall_at_10(idx, queries, gt, 64, true);
    std::cerr << "ivf recall@10 nprobe=8: " << r8 << "  nprobe=64: " << r64 << "\n";
    CHECK(r8 >= 0.75);
    CHECK(r64 >= r8);

    // 1 thread and 4 threads give identical lists (first 20000 rows, nlist 256, for speed).
    vro::Matrix sub = vro::npy::read_f32(dev("vectors.npy"), 20000);
    vro::Params sp;
    sp.values["nlist"] = "256";
    sp.values["iters"] = "20";
    sp.values["train_size"] = std::to_string(vro::default_train_size(sub.rows, 256));
    vro::ivf::Index a = vro::ivf::build(sub, sp, vro::BuildContext{1, 42, ""});
    vro::ivf::Index b = vro::ivf::build(sub, sp, vro::BuildContext{4, 42, ""});
    CHECK(a.offsets == b.offsets);
    CHECK(a.list_ids == b.list_ids);
    CHECK(a.centers == b.centers);
}

void test_bench_json() {
    std::string out = "/tmp/vro-cpp-test-ivf.json";
    std::string cmd = "\"" + g_bench + "\" --index ivf --data \"" + g_root +
                      "/data/processed/dev\" --out " + out +
                      " --limit 20000 --search nprobe=4 --search nprobe=16";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in, nullptr, false);
    CHECK(!doc.is_discarded());
    if (doc.is_discarded()) return;
    CHECK(doc["index"] == "ivf");
    CHECK(doc["n"] == 20000);
    CHECK(doc["searches"].is_array() && doc["searches"].size() == 2);
    for (const auto& s : doc["searches"]) {
        const auto& ids = s["ids"];
        CHECK(ids.is_array() && ids.size() == 1000);
        for (const auto& row : ids) CHECK(row.is_array() && row.size() == 10);
        CHECK(s["distance_computations"].is_number());
    }
    CHECK(doc["searches"][0]["search_params"]["nprobe"] == 4);
    CHECK(doc["searches"][1]["search_params"]["nprobe"] == 16);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_ivf <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        test_index();
        test_bench_json();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "ivf: all checks passed\n";
    return g_failures ? 1 : 0;
}
