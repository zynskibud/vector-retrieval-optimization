// Tests for the diskann index (CONTRACT 6.7, 9).
// Usage: test_diskann <repo_root> <bench_path>
// All builds use the first 20000 dev rows, with brute-force truth computed here.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "diskann.hpp"
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

constexpr std::size_t kRows = 20000;
constexpr std::size_t kK = 10;

std::string g_root, g_bench;
std::string dev(const std::string& f) { return g_root + "/data/processed/dev/" + f; }
std::string tmp_out(const std::string& tag) {
    return "/tmp/vro-cpp-test-diskann-" + std::to_string(getpid()) + "-" + tag + ".json";
}

vro::Params build_params(const std::string& metric) {
    vro::Params p;
    p.values = {{"r", "64"}, {"l_build", "100"}, {"alpha", "1.2"}, {"pq_m", "48"}, {"metric", metric}};
    return p;
}

vro::Params search_params(int l, const std::string& io) {
    vro::Params p;
    p.values = {{"l", std::to_string(l)}, {"beam", "4"}, {"rerank", "100"}, {"io", io}};
    return p;
}

int hw_threads() { return static_cast<int>(std::max(2u, std::thread::hardware_concurrency())); }

struct Run {
    std::vector<std::vector<std::int64_t>> ids;
    std::vector<double> latency_ms;
    double recall = 0, p50 = 0, dc = 0, reads = 0;
};

Run run(const vro::diskann::Index& ix, const vro::Matrix& q, const std::vector<std::vector<std::int64_t>>& truth,
        int l, const std::string& io) {
    vro::Params p = search_params(l, io);
    Run out;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < q.rows; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        vro::SearchResult r = vro::diskann::search(ix, q.row(i), kK, p);
        out.latency_ms.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000.0);
        CHECK(r.ids.size() == kK && r.scores.size() == kK);
        for (std::size_t j = 1; j < kK; ++j) CHECK(r.scores[j - 1] >= r.scores[j]);
        out.dc += static_cast<double>(r.distance_computations);
        out.reads += r.counters["disk_reads"];
        std::set<std::int64_t> t(truth[i].begin(), truth[i].end());
        for (auto id : r.ids) hits += t.count(id);
        out.ids.push_back(r.ids);
    }
    double nq = static_cast<double>(q.rows);
    out.recall = static_cast<double>(hits) / (nq * kK);
    out.dc /= nq;
    out.reads /= nq;
    std::vector<double> lat = out.latency_ms;
    std::nth_element(lat.begin(), lat.begin() + static_cast<std::ptrdiff_t>(lat.size() / 2), lat.end());
    out.p50 = lat[lat.size() / 2];
    std::cerr << "  l=" << l << " io=" << io << ": recall@10 " << out.recall << ", p50 " << out.p50
              << " ms, distance_computations " << out.dc << ", disk_reads " << out.reads << "\n";
    return out;
}

// Reads the file: size N * record_bytes, every node has 1..r valid edges
// (then -1 padding), no self-loop, no repeat, and the stored vector matches.
void check_file(const vro::diskann::Index& ix, const vro::Matrix& orig) {
    struct stat st{};
    CHECK(stat(ix.path.c_str(), &st) == 0);
    CHECK(static_cast<std::size_t>(st.st_size) == kRows * 4096);
    CHECK(ix.record_bytes == 4096 && ix.disk_bytes == kRows * 4096);
    std::ifstream in(ix.path, std::ios::binary);
    std::vector<char> rec(ix.record_bytes);
    std::size_t bad = 0, min_deg = ix.r, max_deg = 0;
    for (std::size_t i = 0; i < ix.n; ++i) {
        in.read(rec.data(), static_cast<std::streamsize>(rec.size()));
        if (!in) {
            CHECK(false && "short file");
            return;
        }
        if (std::memcmp(rec.data(), orig.row(i), ix.dim * sizeof(float)) != 0) ++bad;
        std::vector<std::int32_t> e(ix.r);
        std::memcpy(e.data(), rec.data() + ix.dim * sizeof(float), ix.r * sizeof(std::int32_t));
        std::size_t deg = 0;
        while (deg < ix.r && e[deg] >= 0) ++deg;
        for (std::size_t j = deg; j < ix.r; ++j)
            if (e[j] != -1) ++bad;
        std::set<std::int32_t> uniq(e.begin(), e.begin() + static_cast<std::ptrdiff_t>(deg));
        if (uniq.size() != deg || uniq.count(static_cast<std::int32_t>(i))) ++bad;
        if (deg > 0 && static_cast<std::size_t>(*uniq.rbegin()) >= ix.n) ++bad;
        min_deg = std::min(min_deg, deg);
        max_deg = std::max(max_deg, deg);
    }
    std::cerr << "  file: out-degree min " << min_deg << ", max " << max_deg << ", mean "
              << ix.mean_out_degree << ", bad records " << bad << "\n";
    CHECK(bad == 0);
    CHECK(min_deg >= 1 && max_deg <= ix.r);
}

vro::diskann::Index build(vro::Matrix v, const std::string& metric, int threads, const std::string& tag) {
    vro::BuildContext ctx;
    ctx.threads = threads;
    ctx.out_path = tmp_out(tag);
    vro::diskann::Index ix = vro::diskann::build(v, build_params(metric), ctx);
    std::cerr << "build metric=" << metric << " threads=" << threads << ": train_s " << ix.times.train_s
              << ", add_s " << ix.times.add_s << ", entry " << ix.entry << "\n";
    CHECK(v.data.empty() && v.rows == 0);  // the corpus was released
    return ix;
}

void test_json() {
    std::string out = tmp_out("bench");
    std::string cmd = "\"" + g_bench + "\" --index diskann --data \"" + g_root +
                      "/data/processed/dev\" --out " + out +
                      " --limit 5000 --search l=50,io=mmap --search l=50,io=nocache 2>/dev/null";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in);
    CHECK(doc["index"] == "diskann" && doc["n"] == 5000);
    CHECK(doc["build_params"]["r"] == 64 && doc["build_params"]["metric"] == "ip");
    CHECK(doc["extra"]["disk_bytes"] == 5000.0 * 4096);
    CHECK(doc["searches"].size() == 2);
    for (const auto& s : doc["searches"]) {
        CHECK(s["ids"].size() == 1000);
        CHECK(s["distance_computations"].is_number());
        CHECK(s["extra"]["disk_reads"].get<double>() > 0);
        CHECK(s["extra"]["disk_bytes_read"].get<double>() > 0);
    }
    CHECK(doc["searches"][0]["ids"] == doc["searches"][1]["ids"]);
    std::remove(out.c_str());
    std::remove((out + ".diskann").c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_diskann <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    std::vector<std::string> files;
    try {
        vro::Matrix v = vro::npy::read_f32(dev("vectors.npy"), kRows);
        vro::Matrix q = vro::npy::read_f32(dev("queries.npy"));
        std::vector<std::vector<std::int64_t>> truth(q.rows);
        for (std::size_t i = 0; i < q.rows; ++i) {
            vro::TopK top(kK);
            for (std::size_t j = 0; j < v.rows; ++j)
                top.push(static_cast<std::int64_t>(j), vro::dot(q.row(i), v.row(j), v.dim));
            std::vector<float> sc;
            top.result(truth[i], sc);
        }

        // All cores, metric=ip: graph file, recall per l, both io modes.
        vro::diskann::Index ix = build(v, "ip", hw_threads(), "ip");
        files.push_back(ix.path);
        check_file(ix, v);
        Run r50 = run(ix, q, truth, 50, "mmap");
        Run r100 = run(ix, q, truth, 100, "mmap");
        Run r200 = run(ix, q, truth, 200, "mmap");
        CHECK(r100.recall >= 0.90);
        CHECK(r200.recall >= r100.recall && r100.recall >= r50.recall);
        Run n100 = run(ix, q, truth, 100, "nocache");
        CHECK(n100.ids == r100.ids);
        CHECK(n100.reads > 0);
        CHECK(n100.p50 > r100.p50);
        Run r100b = run(ix, q, truth, 100, "mmap");  // mmap again after nocache
        CHECK(r100b.ids == r100.ids);

        // One thread: deterministic row order; recall within 0.01 of all cores.
        vro::diskann::Index ix1 = build(v, "ip", 1, "ip1");
        files.push_back(ix1.path);
        check_file(ix1, v);
        Run s100 = run(ix1, q, truth, 100, "mmap");
        CHECK(std::abs(s100.recall - r100.recall) <= 0.01);

        // metric=l2 (CONTRACT 9, item 5).
        vro::diskann::Index ixl2 = build(v, "l2", hw_threads(), "l2");
        files.push_back(ixl2.path);
        Run l2 = run(ixl2, q, truth, 100, "mmap");
        CHECK(l2.recall >= 0.90);
        Run l2n = run(ixl2, q, truth, 100, "nocache");
        CHECK(l2n.ids == l2.ids);

        test_json();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        for (const auto& f : files) std::remove(f.c_str());
        return 1;
    }
    for (const auto& f : files) std::remove(f.c_str());
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "diskann: PASS\n";
    return g_failures ? 1 : 0;
}
