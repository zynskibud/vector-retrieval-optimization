// Tests for the ivf_pq index (CONTRACT 9, items 4, 5, 7).
// Usage: test_ivf_pq <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
#include <sys/wait.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ivf_pq.hpp"
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

int hw_threads() { return static_cast<int>(std::max(1u, std::thread::hardware_concurrency())); }

vro::Params build_params(const std::string& metric) {
    vro::Params p;
    p.values = {{"nlist", "1024"}, {"iters", "20"}, {"m", "48"},
                {"nbits", "8"},    {"metric", metric}, {"train_size", "100000"}};
    return p;
}

void check_csr(const vro::ivf_pq::Index& idx) {
    const std::size_t n = idx.n;
    CHECK(idx.offsets.size() == idx.nlist + 1);
    CHECK(idx.offsets.front() == 0);
    CHECK(static_cast<std::size_t>(idx.offsets.back()) == n);
    for (std::size_t c = 0; c < idx.nlist; ++c) CHECK(idx.offsets[c] <= idx.offsets[c + 1]);
    CHECK(idx.list_ids.size() == n);
    CHECK(idx.codes.size() == n * idx.m);
    std::vector<int> seen(n, 0);
    for (auto id : idx.list_ids)
        if (id >= 0 && static_cast<std::size_t>(id) < n) ++seen[static_cast<std::size_t>(id)];
    CHECK(std::count(seen.begin(), seen.end(), 1) == static_cast<std::ptrdiff_t>(n));
}

double recall_at_10(const vro::ivf_pq::Index& index, const vro::Matrix& queries,
                    const vro::npy::Int64Array& gt, int nprobe, int rerank) {
    vro::Params sp;
    sp.values["nprobe"] = std::to_string(nprobe);
    sp.values["rerank"] = std::to_string(rerank);
    const std::size_t k = 10;
    std::vector<vro::SearchResult> results(queries.rows);
    const auto nt = static_cast<std::size_t>(hw_threads());
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t i = t; i < queries.rows; i += nt)
                results[i] = vro::ivf_pq::search(index, queries.row(i), k, sp);
        });
    for (auto& th : pool) th.join();
    const bool l2 = index.metric == vro::Metric::kL2;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < queries.rows; ++i) {
        const vro::SearchResult& r = results[i];
        CHECK(r.ids.size() == k && r.scores.size() == k);
        CHECK(r.distance_computations >= static_cast<std::int64_t>(index.nlist));
        CHECK(r.distance_computations <=
              static_cast<std::int64_t>(index.nlist + index.n + static_cast<std::size_t>(rerank)));
        for (std::size_t j = 1; j < k; ++j) CHECK(r.scores[j - 1] >= r.scores[j]);
        if (l2)
            for (float s : r.scores) CHECK(s <= 0.0f);
        std::set<std::int64_t> truth(gt.data.begin() + static_cast<std::ptrdiff_t>(i * gt.cols),
                                     gt.data.begin() + static_cast<std::ptrdiff_t>(i * gt.cols + k));
        for (auto id : r.ids) hits += truth.count(id);
    }
    return static_cast<double>(hits) / static_cast<double>(queries.rows * k);
}

void test_recall(vro::Matrix& vectors, const vro::Matrix& queries,
                 const vro::npy::Int64Array& gt) {
    for (const std::string metric : {"ip", "l2"}) {
        vro::ivf_pq::Index idx =
            vro::ivf_pq::build(vectors, build_params(metric), vro::BuildContext{hw_threads(), 42, ""});
        check_csr(idx);
        const std::size_t n = vectors.rows;
        CHECK(vro::ivf_pq::index_bytes(idx) ==
              1024 * 384 * 4 + n * 4 + 1025 * 4 + 48 * 256 * 8 * 4 + n * 48);
        double r8 = recall_at_10(idx, queries, gt, 8, 0);
        double r8r = recall_at_10(idx, queries, gt, 8, 100);
        double r32 = recall_at_10(idx, queries, gt, 32, 0);
        std::cerr << "ivf_pq metric=" << metric << " train_s=" << idx.times.train_s
                  << " add_s=" << idx.times.add_s << " recall@10 nprobe=8: " << r8
                  << " nprobe=8,rerank=100: " << r8r << " nprobe=32: " << r32 << "\n";
        CHECK(r8 >= 0.45);
        CHECK(r8r >= r8);
        CHECK(r32 >= r8);
    }
}

void test_determinism(const vro::Matrix& vectors) {
    vro::Matrix small;
    small.dim = vectors.dim;
    small.rows = 20000;
    small.data.assign(vectors.data.begin(),
                      vectors.data.begin() + static_cast<std::ptrdiff_t>(small.rows * small.dim));
    for (const std::string metric : {"ip", "l2"}) {
        vro::Params p = build_params(metric);
        p.values["nlist"] = "64";
        p.values["iters"] = "5";
        p.values["train_size"] = "20000";
        vro::ivf_pq::Index a = vro::ivf_pq::build(small, p, vro::BuildContext{1, 7, ""});
        vro::ivf_pq::Index b = vro::ivf_pq::build(small, p, vro::BuildContext{4, 7, ""});
        check_csr(a);
        CHECK(a.centers == b.centers);
        CHECK(a.codebooks == b.codebooks);
        CHECK(a.offsets == b.offsets);
        CHECK(a.list_ids == b.list_ids);
        CHECK(a.codes == b.codes);
    }
}

void test_bench_json() {
    std::string out = "/tmp/vro-cpp-test-ivf_pq.json";
    std::string cmd = "\"" + g_bench + "\" --index ivf_pq --data \"" + g_root +
                      "/data/processed/dev\" --out " + out +
                      " --limit 20000 --build metric=l2 --search nprobe=8,rerank=0"
                      " --search nprobe=8,rerank=100";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in, nullptr, false);
    CHECK(!doc.is_discarded());
    if (doc.is_discarded()) return;
    CHECK(doc["index"] == "ivf_pq");
    CHECK(doc["n"] == 20000);
    CHECK(doc["build_params"]["metric"] == "l2");
    CHECK(doc["build"]["index_bytes"] ==
          1024 * 384 * 4 + 20000 * 4 + 1025 * 4 + 48 * 256 * 8 * 4 + 20000 * 48);
    CHECK(doc["extra"]["id_bytes"] == 4.0);
    CHECK(doc["searches"].is_array() && doc["searches"].size() == 2);
    for (const auto& s : doc["searches"]) {
        CHECK(s["ids"].size() == 1000);
        for (const auto& row : s["ids"]) CHECK(row.size() == 10);
        CHECK(s["scores"].size() == 1000);
        CHECK(s["latency_ms"].size() == 1000);
        CHECK(s["distance_computations"].is_number());
    }
    CHECK(doc["searches"][1]["search_params"]["rerank"] == 100);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_ivf_pq <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        vro::Matrix vectors = vro::npy::read_f32(dev("vectors.npy"));
        vro::Matrix queries = vro::npy::read_f32(dev("queries.npy"));
        vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
        test_determinism(vectors);
        test_recall(vectors, queries, gt);
        test_bench_json();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "ivf_pq: PASS\n";
    return g_failures ? 1 : 0;
}
