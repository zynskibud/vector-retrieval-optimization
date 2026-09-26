// Tests for load runs (CONTRACT 12.5) on the hnsw bench.
// Usage: test_load <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
// Runs bench on the first 20000 dev rows; the truth is computed here.
#include <sys/wait.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "distance.hpp"
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

constexpr std::size_t kRows = 20000, kK = 10;
std::string g_root, g_bench;

std::vector<std::vector<std::int64_t>> truth(const vro::Matrix& v, const vro::Matrix& q) {
    std::vector<std::vector<std::int64_t>> out(q.rows);
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::TopK top(kK);
        for (std::size_t j = 0; j < v.rows; ++j)
            top.push(static_cast<std::int64_t>(j), vro::dot(q.row(i), v.row(j), v.dim));
        std::vector<float> sc;
        top.result(out[i], sc);
    }
    return out;
}

double recall(const nlohmann::json& ids, const std::vector<std::vector<std::int64_t>>& t) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        std::set<std::int64_t> s(t[i].begin(), t[i].end());
        for (const auto& id : ids[i]) hits += s.count(id.get<std::int64_t>());
    }
    return static_cast<double>(hits) / static_cast<double>(t.size() * kK);
}

nlohmann::json bench(const std::string& name, const std::string& flags) {
    std::string out = "/tmp/vro-cpp-test-load-" + name + ".json";
    std::string cmd = "\"" + g_bench + "\" --index hnsw --data \"" + g_root +
                      "/data/processed/dev\" --out " + out + " --limit " + std::to_string(kRows) +
                      " --search ef=64 " + flags;
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    return nlohmann::json::parse(in);
}

void check_load_keys(const nlohmann::json& s, int clients, double duration) {
    const auto& e = s["extra"];
    for (const char* key : {"errors", "cpu_pct", "clients", "duration_s", "queries_done"})
        CHECK(e.contains(key));
    CHECK(e["clients"] == clients);
    CHECK(e["duration_s"].get<double>() == duration);
    CHECK(e["errors"] == 0);
    CHECK(s["ids"].size() == 1000);
    CHECK(s["latency_ms"].size() == e["queries_done"].get<std::size_t>());
    CHECK(e["queries_done"].get<std::size_t>() >= 1000);
    CHECK(s["search_params"]["ef"] == 64);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_load <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        std::string dev = g_root + "/data/processed/dev";
        vro::Matrix v = vro::npy::read_f32(dev + "/vectors.npy", kRows);
        vro::Matrix q = vro::npy::read_f32(dev + "/queries.npy");
        auto t = truth(v, q);

        // (a) 8 clients against 1 client, 5 s each.
        auto one = bench("c1", "--clients 1 --duration 5");
        auto eight = bench("c8", "--clients 8 --duration 5");
        check_load_keys(one["searches"][0], 1, 5.0);
        check_load_keys(eight["searches"][0], 8, 5.0);
        double qps1 = one["searches"][0]["qps"], qps8 = eight["searches"][0]["qps"];
        double r8 = recall(eight["searches"][0]["ids"], t);
        std::cerr << "qps 1 client " << qps1 << ", 8 clients " << qps8 << ", cpu_pct 8 clients "
                  << eight["searches"][0]["extra"]["cpu_pct"] << ", first-pass recall " << r8 << "\n";
        CHECK(qps8 > qps1);
        CHECK(r8 >= 0.95);
        CHECK(eight["extra"]["clients"] == 8);

        // (b) inserts during searches against a static build on all rows.
        auto stat = bench("static", "");
        CHECK(stat["searches"].size() == 1);
        CHECK(!stat["searches"][0]["search_params"].contains("phase"));
        CHECK(!stat["searches"][0]["extra"].contains("clients"));
        double r_static = recall(stat["searches"][0]["ids"], t);
        auto ins = bench("insert", "--clients 4 --duration 10 --insert-rate 2000");
        CHECK(ins["n"] == kRows);
        CHECK(ins["searches"].size() == 2);
        check_load_keys(ins["searches"][0], 4, 10.0);
        const auto& after = ins["searches"][1];
        CHECK(after["search_params"]["phase"] == "after_inserts");
        CHECK(after["ids"].size() == 1000);
        CHECK(after["extra"]["inserted_rows"] == 2000);
        CHECK(after["extra"].contains("insert_p50_ms"));
        CHECK(ins["extra"]["inserted_rows"] == 2000);
        for (const char* key : {"inserted_during_loop", "insert_tail_s", "insert_p50_ms"})
            CHECK(ins["extra"].contains(key) && after["extra"].contains(key));
        CHECK(ins["searches"][0]["extra"]["inserted_rows"] == 2000);
        std::cerr << "inserted_during_loop " << ins["extra"]["inserted_during_loop"]
                  << ", insert_tail_s " << ins["extra"]["insert_tail_s"] << "\n";
        CHECK(ins["extra"]["build_rows"] == 18000);
        double r_after = recall(after["ids"], t);
        std::cerr << "static recall " << r_static << ", after-inserts recall " << r_after
                  << ", insert_p50_ms " << after["extra"]["insert_p50_ms"] << "\n";
        CHECK(std::abs(r_after - r_static) <= 0.01);

        // (c) only hnsw takes load flags: exit 2 for another index.
        std::string cmd = "\"" + g_bench + "\" --index flat --data \"" + dev +
                          "\" --out /tmp/vro-cpp-test-load-flat.json --limit 1000 --clients 2 2>/dev/null";
        int rc = std::system(cmd.c_str());
        CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 2);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "load: PASS\n";
    return g_failures ? 1 : 0;
}
