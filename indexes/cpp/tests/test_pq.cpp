// Tests for the pq index (CONTRACT 9, items 4, 5, 7).
// Usage: test_pq <repo_root> <bench_path>
#include <sys/wait.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "npy.hpp"
#include "pq.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

std::string g_root;
std::string g_bench;

std::string dev(const std::string& file) { return g_root + "/data/processed/dev/" + file; }

vro::Params build_params(const std::string& metric) {
    vro::Params p;
    p.values = {{"m", "48"}, {"nbits", "8"}, {"metric", metric}, {"train_size", "100000"},
                {"iters", "20"}};
    return p;
}

vro::Params search_params(int rerank) {
    vro::Params p;
    p.values["rerank"] = std::to_string(rerank);
    return p;
}

int hw_threads() { return static_cast<int>(std::max(1u, std::thread::hardware_concurrency())); }

double recall_at_10(const vro::pq::Index& index, const vro::Matrix& queries,
                    const vro::npy::Int64Array& gt, int rerank, bool check_l2_scores) {
    const std::size_t k = 10;
    std::size_t hits = 0;
    vro::Params sp = search_params(rerank);
    // Search in parallel (the test only checks results), then check serially.
    std::vector<vro::SearchResult> results(queries.rows);
    const std::size_t nt = static_cast<std::size_t>(hw_threads());
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t i = t; i < queries.rows; i += nt)
                results[i] = vro::pq::search(index, queries.row(i), k, sp);
        });
    for (auto& th : pool) th.join();
    for (std::size_t i = 0; i < queries.rows; ++i) {
        const vro::SearchResult& r = results[i];
        CHECK(r.ids.size() == k && r.scores.size() == k);
        CHECK(r.distance_computations ==
              static_cast<std::int64_t>(index.n + static_cast<std::size_t>(rerank)));
        for (std::size_t j = 1; j < k; ++j) CHECK(r.scores[j - 1] >= r.scores[j]);
        if (check_l2_scores)
            for (float s : r.scores) CHECK(s <= 0.0f);
        std::set<std::int64_t> truth(gt.data.begin() + i * gt.cols,
                                     gt.data.begin() + i * gt.cols + k);
        for (auto id : r.ids) hits += truth.count(id);
    }
    return static_cast<double>(hits) / static_cast<double>(queries.rows * k);
}

void test_recall(vro::Matrix& vectors, const vro::Matrix& queries,
                 const vro::npy::Int64Array& gt) {
    for (const std::string metric : {"ip", "l2"}) {
        vro::BuildContext ctx{hw_threads(), 42, ""};
        vro::pq::Index index = vro::pq::build(vectors, build_params(metric), ctx);
        CHECK(index.codes.size() == vectors.rows * 48);
        CHECK(vro::pq::index_bytes(index) == 48 * 256 * 8 * 4 + vectors.rows * 48);
        bool l2 = metric == "l2";
        double r0 = recall_at_10(index, queries, gt, 0, l2);
        double r100 = recall_at_10(index, queries, gt, 100, l2);
        std::cerr << "pq metric=" << metric << " train_s=" << index.times.train_s
                  << " add_s=" << index.times.add_s << " recall@10 rerank=0: " << r0
                  << " rerank=100: " << r100 << "\n";
        CHECK(r0 >= 0.50);
        CHECK(r100 >= 0.50);
        CHECK(r100 >= r0);
    }
}

void test_determinism(vro::Matrix& vectors) {
    vro::Matrix small;
    small.dim = vectors.dim;
    small.rows = 20000;
    small.data.assign(vectors.data.begin(), vectors.data.begin() + small.rows * small.dim);
    vro::Params p = build_params("l2");
    p.values["train_size"] = "20000";
    p.values["iters"] = "5";
    vro::pq::Index a = vro::pq::build(small, p, vro::BuildContext{1, 7, ""});
    vro::pq::Index b = vro::pq::build(small, p, vro::BuildContext{4, 7, ""});
    vro::pq::Index c = vro::pq::build(small, p, vro::BuildContext{4, 7, ""});
    CHECK(a.codes == b.codes);          // 1 vs 4 threads
    CHECK(b.codes == c.codes);          // same seed, two builds
    CHECK(a.codebooks == b.codebooks);
}

void test_bad_params(vro::Matrix& vectors) {
    for (const auto& kv : {std::pair<std::string, std::string>{"nbits", "4"},
                           {"metric", "cosine"}, {"m", "5"}}) {
        vro::Params p = build_params("ip");
        p.values[kv.first] = kv.second;
        bool threw = false;
        try {
            vro::pq::build(vectors, p, vro::BuildContext{});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

void test_bench_json() {
    std::string out = "/tmp/vro-cpp-test-pq.json";
    std::string cmd = "\"" + g_bench + "\" --index pq --data \"" + g_root +
                      "/data/processed/dev\" --out " + out +
                      " --limit 20000 --build metric=l2 --search rerank=0 --search rerank=100";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in);
    CHECK(doc["index"] == "pq");
    CHECK(doc["n"] == 20000);
    CHECK(doc["build_params"]["metric"] == "l2");
    CHECK(doc["build"]["index_bytes"] == 48 * 256 * 8 * 4 + 20000 * 48);
    CHECK(doc["searches"].size() == 2);
    for (const auto& s : doc["searches"]) {
        CHECK(s["ids"].size() == 1000);
        for (const auto& row : s["ids"]) CHECK(row.size() == 10);
        CHECK(s["scores"].size() == 1000);
        CHECK(s["latency_ms"].size() == 1000);
    }
    CHECK(doc["searches"][0]["distance_computations"] == 20000.0);
    CHECK(doc["searches"][1]["distance_computations"] == 20100.0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_pq <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        vro::Matrix vectors = vro::npy::read_f32(dev("vectors.npy"));
        vro::Matrix queries = vro::npy::read_f32(dev("queries.npy"));
        vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
        test_bad_params(vectors);
        test_determinism(vectors);
        test_recall(vectors, queries, gt);
        test_bench_json();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "pq: PASS\n";
    return g_failures ? 1 : 0;
}
