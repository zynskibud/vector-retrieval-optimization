// Tests for metadata filtering (CONTRACT 11.5) on flat, ivf, hnsw.
// Usage: test_filter <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
// Runs on the first 20000 dev rows. The filtered truth is computed here.
#include <sys/wait.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "distance.hpp"
#include "flat.hpp"
#include "hnsw.hpp"
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

constexpr std::size_t kRows = 20000, kK = 10;
std::string g_root, g_bench;
std::string dev_dir() { return g_root + "/data/processed/dev"; }

using SearchFn = std::function<vro::SearchResult(const float*, const vro::Params&)>;

// Exact top-k among passing rows (mask == nullptr: all rows).
std::vector<std::vector<std::int64_t>> truth(const vro::Matrix& v, const vro::Matrix& q,
                                             const std::vector<std::uint8_t>* mask) {
    std::vector<std::vector<std::int64_t>> out(q.rows);
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::TopK top(kK);
        for (std::size_t j = 0; j < v.rows; ++j)
            if (!mask || (*mask)[j]) top.push(static_cast<std::int64_t>(j), vro::dot(q.row(i), v.row(j), v.dim));
        std::vector<float> sc;
        top.result(out[i], sc);
    }
    return out;
}

// Runs every query; checks that each returned ID passes the mask or is -1.
// Returns recall@10 against t (pad entries -1 in t do not count).
double run(const std::string& name, const SearchFn& fn, const vro::Matrix& q,
           const std::string& filter, const std::vector<std::uint8_t>* mask,
           const std::vector<std::vector<std::int64_t>>* t, std::vector<std::vector<std::int64_t>>* ids_out) {
    vro::Params p;
    p.values["filter"] = filter;
    std::size_t hits = 0, denom = 0, bad = 0;
    double dc = 0, fr = 0;
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::SearchResult r = fn(q.row(i), p);
        CHECK(r.ids.size() == kK);
        for (auto id : r.ids)
            if (id != -1 && (id < 0 || static_cast<std::size_t>(id) >= kRows || (mask && !(*mask)[static_cast<std::size_t>(id)])))
                ++bad;
        dc += static_cast<double>(r.distance_computations);
        fr += r.counters.count("filter_rows") ? r.counters.at("filter_rows") : -1.0;
        if (ids_out) ids_out->push_back(r.ids);
        if (t) {
            std::set<std::int64_t> s;
            for (auto id : (*t)[i]) if (id != -1) s.insert(id);
            denom += s.size();
            for (auto id : r.ids) hits += s.count(id);
        }
    }
    CHECK(bad == 0);
    double rec = denom ? static_cast<double>(hits) / static_cast<double>(denom) : 1.0;
    std::cerr << name << " filter=" << filter << ": recall@10 " << rec << ", mean dc "
              << dc / static_cast<double>(q.rows) << ", filter_rows " << fr / static_cast<double>(q.rows)
              << ", bad ids " << bad << "\n";
    return rec;
}

void test_index(const std::string& name, const SearchFn& fn, const vro::Matrix& q,
                const std::vector<std::uint8_t>& m10, const std::vector<std::uint8_t>& m01,
                const std::vector<std::vector<std::int64_t>>& t10, const vro::Params& base_none,
                double floor) {
    // filter=none gives the same IDs as a search without the filter key.
    std::vector<std::vector<std::int64_t>> a, b;
    run(name, fn, q, "none", nullptr, nullptr, &a);
    for (std::size_t i = 0; i < q.rows; ++i) b.push_back(fn(q.row(i), base_none).ids);
    CHECK(a == b);
    double r10 = run(name, fn, q, "top10", &m10, &t10, nullptr);
    CHECK(r10 >= floor);
    run(name, fn, q, "top01", &m01, nullptr, nullptr);
    // A bad filter name is a ParamError.
    vro::Params bad;
    bad.values["filter"] = "top7";
    bool threw = false;
    try {
        fn(q.row(0), bad);
    } catch (const vro::ParamError&) {
        threw = true;
    }
    CHECK(threw);
}

void test_bench() {
    std::string out = "/tmp/vro-cpp-test-filter.json";
    std::string cmd = "\"" + g_bench + "\" --index hnsw --data \"" + dev_dir() + "\" --out " + out +
                      " --limit 20000 --search filter=none --search filter=top10 --search filter=top01";
    int rc = std::system(cmd.c_str());
    CHECK(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
    std::ifstream in(out);
    nlohmann::json doc = nlohmann::json::parse(in);
    CHECK(doc["searches"].size() == 3);
    const char* names[3] = {"none", "top10", "top01"};
    for (std::size_t i = 0; i < 3 && i < doc["searches"].size(); ++i) {
        const auto& s = doc["searches"][i];
        CHECK(s["search_params"]["filter"] == names[i]);
        CHECK(s["search_params"]["ef"] == 64);
        if (i > 0) CHECK(s["extra"]["filter_rows"].is_number());
        CHECK(s["extra"]["visited"].is_number());
    }
    CHECK(!doc["searches"][0]["extra"].contains("filter_rows"));
    // Unknown filter name and filter on pq: exit 2.
    for (const std::string& c : {std::string("--index hnsw --search filter=top7"),
                                 std::string("--index pq --search filter=top10")}) {
        std::string cmd2 = "\"" + g_bench + "\" " + c + " --data \"" + dev_dir() + "\" --out " + out +
                           " --limit 20000 2>/dev/null";
        int rc2 = std::system(cmd2.c_str());
        CHECK(WIFEXITED(rc2) && WEXITSTATUS(rc2) == 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_filter <repo_root> <bench_path>\n";
        return 2;
    }
    g_root = argv[1];
    g_bench = argv[2];
    try {
        vro::Matrix v = vro::npy::read_f32(dev_dir() + "/vectors.npy", kRows);
        vro::Matrix q = vro::npy::read_f32(dev_dir() + "/queries.npy");
        auto m10 = vro::npy::read_bool(dev_dir() + "/filter_top10.npy");
        auto m01 = vro::npy::read_bool(dev_dir() + "/filter_top01.npy");
        CHECK(m10.size() == 100000);
        m10.resize(kRows);
        m01.resize(kRows);
        auto t10 = truth(v, q, &m10);

        vro::BuildContext ctx;
        ctx.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        ctx.data_dir = dev_dir();

        vro::Params none;
        vro::flat::Index fi = vro::flat::build(v, none, ctx);
        test_index("flat", [&](const float* x, const vro::Params& p) { return vro::flat::search(fi, x, kK, p); },
                   q, m10, m01, t10, none, 1.0);

        vro::Params ib;
        ib.values["nlist"] = "256";
        ib.values["iters"] = "20";
        ib.values["train_size"] = std::to_string(vro::default_train_size(kRows, 256));
        vro::ivf::Index ii = vro::ivf::build(v, ib, ctx);
        vro::Params is;
        is.values["nprobe"] = "8";
        test_index("ivf", [&](const float* x, const vro::Params& p) {
            vro::Params pp = p;
            pp.values["nprobe"] = "8";
            return vro::ivf::search(ii, x, kK, pp);
        }, q, m10, m01, t10, is, 0.70);

        vro::Params hb;
        hb.values["m"] = "16";
        hb.values["ef_construct"] = "100";
        vro::hnsw::Index hi = vro::hnsw::build(v, hb, ctx);
        vro::Params hs;
        hs.values["ef"] = "64";
        test_index("hnsw", [&](const float* x, const vro::Params& p) {
            vro::Params pp = p;
            pp.values["ef"] = "64";
            return vro::hnsw::search(hi, x, kK, pp);
        }, q, m10, m01, t10, hs, 0.85);

        test_bench();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "filter: PASS\n";
    return g_failures ? 1 : 0;
}
