// Tests for the hnsw index (CONTRACT 6.6, 9).
// Usage: test_hnsw <repo_root> <bench_path>
// Builds on the full dev set (100k rows, all cores), then threads=1 and all-cores
// builds on the first 20000 rows with brute-force truth computed here.
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "distance.hpp"
#include "hnsw.hpp"
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

vro::Params params_of(std::initializer_list<std::pair<const std::string, std::string>> kv) {
    vro::Params p;
    p.values = kv;
    return p;
}

void test_levels() {
    auto a = vro::hnsw::draw_levels(100000, 16, 42);
    auto b = vro::hnsw::draw_levels(100000, 16, 42);
    CHECK(a == b);
    std::size_t up = 0;
    for (auto l : a) up += l >= 1;
    double frac = static_cast<double>(up) / static_cast<double>(a.size());
    std::cerr << "levels: fraction with level >= 1 = " << frac << "\n";
    CHECK(frac >= 0.04 && frac <= 0.09);
}

void test_graph(const vro::hnsw::Index& ix) {
    const std::size_t n = ix.vectors->rows;
    // Slot counts within limits, edge targets valid and present on the layer.
    for (std::size_t l = 0; l < ix.counts.size(); ++l) {
        std::size_t lim = l == 0 ? ix.m0 : ix.m;
        for (std::size_t i = 0; i < n; ++i) {
            if (ix.level[i] < l) continue;
            std::int32_t c;
            const std::int32_t* nb = ix.neighbors(static_cast<int>(l), static_cast<std::int32_t>(i), c);
            CHECK(c >= 0 && static_cast<std::size_t>(c) <= lim);
            for (std::int32_t j = 0; j < c; ++j) {
                if (nb[j] < 0 || static_cast<std::size_t>(nb[j]) >= n || ix.level[nb[j]] < l ||
                    nb[j] == static_cast<std::int32_t>(i)) {
                    CHECK(false && "bad edge");
                    return;
                }
            }
        }
    }
    // Entry point: highest level, ties to lowest row.
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (ix.level[i] > ix.level[best]) best = i;
    CHECK(ix.entry == static_cast<std::int32_t>(best));
    CHECK(ix.top == ix.level[best]);
    // Layer-0 directed BFS from the entry point reaches every node (after repair).
    std::size_t unreach = vro::hnsw::unreachable_layer0(ix);
    std::cerr << "threads=" << ix.threads << ": unreachable before repair "
              << ix.unreachable_before_repair << ", zero in-degree before repair "
              << ix.zero_in_before_repair << ", repair_added " << ix.repair_added
              << " (to unreachable nodes " << ix.repair_step_b << "), passes " << ix.repair_passes << ", unreachable after " << unreach << "\n";
    CHECK(unreach == 0);
}

double recall(const vro::hnsw::Index& ix, const vro::Matrix& q, const vro::npy::Int64Array& gt,
              int ef, double* mean_dc) {
    const std::size_t k = 10, n = ix.vectors->rows;
    vro::Params p = params_of({{"ef", std::to_string(ef)}});
    std::size_t hits = 0;
    double dc = 0;
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::SearchResult r = vro::hnsw::search(ix, q.row(i), k, p);
        CHECK(r.ids.size() == k && r.scores.size() == k);
        CHECK(r.distance_computations > 0 && r.distance_computations < static_cast<std::int64_t>(n));
        dc += static_cast<double>(r.distance_computations);
        for (std::size_t j = 1; j < k; ++j) CHECK(r.scores[j - 1] >= r.scores[j]);
        std::set<std::int64_t> truth(gt.data.begin() + i * gt.cols, gt.data.begin() + i * gt.cols + k);
        for (auto id : r.ids) hits += truth.count(id);
    }
    *mean_dc = dc / static_cast<double>(q.rows);
    return static_cast<double>(hits) / static_cast<double>(q.rows * k);
}

void test_threads(const vro::Matrix& queries) {
    vro::Matrix v = vro::npy::read_f32(dev("vectors.npy"), 20000);
    const std::size_t k = 10;
    vro::npy::Int64Array truth;
    truth.rows = queries.rows;
    truth.cols = k;
    truth.data.resize(queries.rows * k);
    for (std::size_t i = 0; i < queries.rows; ++i) {
        vro::TopK top(k);
        for (std::size_t j = 0; j < v.rows; ++j)
            top.push(static_cast<std::int64_t>(j), vro::dot(queries.row(i), v.row(j), v.dim));
        std::vector<std::int64_t> ids;
        std::vector<float> sc;
        top.result(ids, sc);
        std::copy(ids.begin(), ids.end(), truth.data.begin() + static_cast<std::ptrdiff_t>(i * k));
    }
    vro::Params bp = params_of({{"m", "16"}, {"ef_construct", "100"}});
    double rec[2];
    int th[2] = {1, static_cast<int>(std::max(2u, std::thread::hardware_concurrency()))};
    for (int t = 0; t < 2; ++t) {
        vro::BuildContext ctx;
        ctx.threads = th[t];
        vro::hnsw::Index ix = vro::hnsw::build(v, bp, ctx);
        std::cerr << "20k build threads=" << th[t] << ": add_s " << ix.times.add_s << "\n";
        test_graph(ix);
        double dc;
        rec[t] = recall(ix, queries, truth, 64, &dc);
        std::cerr << "20k threads=" << th[t] << " recall@10 ef=64 " << rec[t] << "\n";
    }
    CHECK(std::abs(rec[0] - rec[1]) <= 0.01);
}

void test_json() {
    std::string out = "/tmp/vro-cpp-test-hnsw.json";
    std::string cmd = "\"" + g_bench + "\" --index hnsw --data \"" + g_root +
                      "/data/processed/dev\" --out " + out +
                      " --limit 20000 --search ef=16 --search ef=64";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in);
    CHECK(doc["index"] == "hnsw");
    CHECK(doc["n"] == 20000);
    CHECK(doc["build_params"]["m"] == 16);
    CHECK(doc["build_params"]["ef_construct"] == 100);
    CHECK(doc["searches"].size() == 2);
    for (const auto& s : doc["searches"]) {
        CHECK(s["ids"].size() == 1000);
        for (const auto& row : s["ids"]) CHECK(row.size() == 10);
        CHECK(s["distance_computations"].is_number());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_hnsw <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        test_levels();
        vro::Matrix vectors = vro::npy::read_f32(dev("vectors.npy"));
        vro::Matrix queries = vro::npy::read_f32(dev("queries.npy"));
        vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
        vro::Params bp = params_of({{"m", "16"}, {"ef_construct", "100"}});
        vro::BuildContext ctx;
        ctx.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        auto t0 = std::chrono::steady_clock::now();
        vro::hnsw::Index ix = vro::hnsw::build(vectors, bp, ctx);
        std::cerr << "build " << vectors.rows << " rows: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s, top layer " << ix.top << "\n";
        CHECK(ix.level == vro::hnsw::draw_levels(vectors.rows, 16, 42));
        test_graph(ix);
        double dc16, dc64, dc128;
        double r16 = recall(ix, queries, gt, 16, &dc16);
        double r64 = recall(ix, queries, gt, 64, &dc64);
        double r128 = recall(ix, queries, gt, 128, &dc128);
        std::cerr << "recall@10 ef=16 " << r16 << " (dc " << dc16 << "), ef=64 " << r64 << " (dc "
                  << dc64 << "), ef=128 " << r128 << " (dc " << dc128 << ")\n";
        CHECK(r64 >= 0.95);
        CHECK(r128 >= r64 && r64 >= r16);
        test_threads(queries);
        test_json();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "hnsw: PASS\n";
    return g_failures ? 1 : 0;
}
