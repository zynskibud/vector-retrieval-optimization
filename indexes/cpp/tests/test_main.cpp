// Tests of CONTRACT 9, items 1, 2, 3, and 7, plus a k-means smoke test.
// Usage: tests <name> <repo_root> [bench_path]
#include <sys/wait.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "distance.hpp"
#include "flat.hpp"
#include "kmeans.hpp"
#include "npy.hpp"
#include "splitmix.hpp"

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

void test_npy() {
    vro::Matrix q = vro::npy::read_f32(dev("queries.npy"));
    CHECK(q.rows == 1000);
    CHECK(q.dim == 384);
    const double ref[3] = {-0.05522317439317703, -0.03818117082118988, 0.01416025310754776};
    for (int j = 0; j < 3; ++j) CHECK(static_cast<double>(q.data[j]) == ref[j]);

    vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
    CHECK(gt.rows == 1000);
    CHECK(gt.cols == 100);

    bool threw = false;
    try {
        vro::npy::read_f32(dev("ground_truth.npy"));  // wrong descr
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_splitmix() {
    vro::SplitMix64 a(42);
    CHECK(a.next_u64() == 13679457532755275413ULL);
    vro::SplitMix64 b(0);
    CHECK(b.next_u64() == 16294208416658607535ULL);
    vro::SplitMix64 c(7);
    for (int i = 0; i < 1000; ++i) {
        double f = c.next_f64();
        CHECK(f >= 0.0 && f < 1.0);
        CHECK(c.next_below(10) < 10);
    }
}

void test_flat() {
    vro::Matrix vectors = vro::npy::read_f32(dev("vectors.npy"));
    vro::Matrix queries = vro::npy::read_f32(dev("queries.npy"));
    vro::npy::Int64Array gt = vro::npy::read_i64(dev("ground_truth.npy"));
    vro::Params params;
    vro::flat::Index index = vro::flat::build(vectors, params, vro::BuildContext{});
    const std::size_t k = 10;
    std::size_t hits = 0, top1_equal = 0;
    for (std::size_t i = 0; i < queries.rows; ++i) {
        vro::SearchResult r = vro::flat::search(index, queries.row(i), k, params);
        CHECK(r.distance_computations == static_cast<std::int64_t>(vectors.rows));
        std::set<std::int64_t> truth(gt.data.begin() + i * gt.cols,
                                     gt.data.begin() + i * gt.cols + k);
        for (auto id : r.ids) hits += truth.count(id);
        if (r.ids[0] == gt.data[i * gt.cols]) ++top1_equal;
    }
    double recall = static_cast<double>(hits) / static_cast<double>(queries.rows * k);
    std::cerr << "flat recall@10 = " << recall << ", top-1 equal = " << top1_equal << "/"
              << queries.rows << "\n";
    CHECK(recall == 1.0);
    CHECK(top1_equal == queries.rows);
}

void test_kmeans() {
    vro::Matrix v = vro::npy::read_f32(dev("vectors.npy"), 5000);
    vro::KMeansOptions opt;
    opt.k = 16;
    opt.iters = 5;
    opt.threads = 4;
    std::vector<float> c1 = vro::kmeans(v.data.data(), v.rows, v.dim, opt);
    opt.threads = 1;
    std::vector<float> c2 = vro::kmeans(v.data.data(), v.rows, v.dim, opt);
    CHECK(c1 == c2);  // thread count must not change the result
    for (std::size_t c = 0; c < opt.k; ++c) {
        const float* row = c1.data() + c * v.dim;
        CHECK(std::fabs(vro::dot(row, row, v.dim) - 1.0f) < 1e-4f);
    }
}

void check_shape(const nlohmann::json& rows, std::size_t q, std::size_t k) {
    CHECK(rows.is_array() && rows.size() == q);
    for (const auto& row : rows) CHECK(row.is_array() && row.size() == k);
}

void test_json() {
    std::string out = "/tmp/vro-cpp-test-flat.json";
    std::string cmd = "\"" + g_bench + "\" --index flat --data \"" + g_root +
                      "/data/processed/dev\" --out " + out + " --limit 20000 --search ''";
    // An empty --search value is a usage error, so first check that exit code 2.
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 2);

    cmd = "\"" + g_bench + "\" --index flat --data \"" + g_root + "/data/processed/dev\" --out " +
          out + " --limit 20000";
    rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in);

    for (const char* key : {"contract_version", "language", "index", "data_dir", "n", "dim", "q",
                            "k", "threads", "seed", "build_params", "build", "searches",
                            "machine", "extra"})
        CHECK(doc.contains(key));
    CHECK(doc["contract_version"] == 1);
    CHECK(doc["language"] == "cpp");
    CHECK(doc["index"] == "flat");
    CHECK(doc["n"] == 20000);
    CHECK(doc["dim"] == 384);
    CHECK(doc["k"] == 10);
    CHECK(doc["build_params"].is_object());
    for (const char* key : {"train_s", "add_s", "total_s", "peak_rss_mb", "index_bytes"})
        CHECK(doc["build"].contains(key));
    for (const char* key : {"os", "arch", "cpu", "cores"}) CHECK(doc["machine"].contains(key));
    CHECK(doc["searches"].size() == 1);
    std::size_t q = doc["q"], k = doc["k"];
    for (const auto& s : doc["searches"]) {
        for (const char* key : {"search_params", "ids", "scores", "latency_ms", "total_s", "qps",
                                "distance_computations", "extra"})
            CHECK(s.contains(key));
        CHECK(s["extra"].is_object() && s["extra"].empty());
        check_shape(s["ids"], q, k);
        check_shape(s["scores"], q, k);
        CHECK(s["ids"][0][0].is_number_integer());
        CHECK(s["scores"][0][0].is_number_float());
        CHECK(s["latency_ms"].size() == q);
        CHECK(s["distance_computations"] == 20000.0);
    }

    // Unknown index and unknown parameter exit 2. No stub index remains.
    std::string base = "\"" + g_bench + "\" --data \"" + g_root + "/data/processed/dev\" --out " +
                       out + " --limit 2000 2>/dev/null";
    rc = std::system((base + " --index nope").c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 2);
    rc = std::system((base + " --index hnsw --build bogus=1").c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 2);
    rc = std::system((base + " --index nope").c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 2);
}

}  // namespace

int main(int argc, char** argv) {
    const std::map<std::string, std::function<void()>> tests = {
        {"npy", test_npy},     {"splitmix", test_splitmix}, {"flat", test_flat},
        {"kmeans", test_kmeans}, {"json", test_json},
    };
    if (argc < 3 || !tests.count(argv[1])) {
        std::cerr << "usage: tests <npy|splitmix|flat|kmeans|json> <repo_root> [bench_path]\n";
        return 2;
    }
    g_root = argv[2];
    if (argc > 3) g_bench = argv[3];
    try {
        tests.at(argv[1])();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << argv[1] << ": PASS\n";
    return g_failures ? 1 : 0;
}
