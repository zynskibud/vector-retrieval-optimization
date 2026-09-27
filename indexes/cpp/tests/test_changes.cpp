// Tests for updates, deletes, and compaction (CONTRACT 13.5) on flat, ivf, hnsw.
// Usage: test_changes <repo_root> <bench_path>. Exit 0 = pass, 1 = failure.
// Uses the first 20000 dev rows and the dev change sets; the truth is computed here.
#include <sys/wait.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "distance.hpp"
#include "flat.hpp"
#include "hnsw.hpp"
#include "ivf.hpp"
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
using Truth = std::vector<std::vector<std::int64_t>>;
using SearchFn = std::function<vro::SearchResult(const float*)>;

// Exact top-10 over the rows with dead[i] == 0 (dead empty = all rows).
Truth truth(const vro::Matrix& v, const vro::Matrix& q, const std::vector<std::uint8_t>& dead) {
    Truth out(q.rows);
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::TopK top(kK);
        for (std::size_t j = 0; j < v.rows; ++j)
            if (dead.empty() || !dead[j])
                top.push(static_cast<std::int64_t>(j), vro::dot(q.row(i), v.row(j), v.dim));
        std::vector<float> sc;
        top.result(out[i], sc);
    }
    return out;
}

// Recall@10 against t; counts returned IDs that are deleted.
double recall(const SearchFn& fn, const vro::Matrix& q, const Truth& t,
              const std::vector<std::uint8_t>& dead, std::size_t& dead_returned) {
    std::size_t hits = 0;
    dead_returned = 0;
    for (std::size_t i = 0; i < q.rows; ++i) {
        vro::SearchResult r = fn(q.row(i));
        std::set<std::int64_t> s(t[i].begin(), t[i].end());
        for (std::int64_t id : r.ids) {
            hits += s.count(id);
            if (id >= 0 && !dead.empty() && dead[static_cast<std::size_t>(id)]) ++dead_returned;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(q.rows * kK);
}

vro::Params params(std::initializer_list<std::pair<const std::string, std::string>> kv) {
    vro::Params p;
    p.values = kv;
    return p;
}

// One index type: build, delete, compact, update. Idx is flat/ivf/hnsw::Index.
template <typename Idx, typename Build, typename Search, typename Bytes, typename Del,
          typename Upd, typename Compact, typename USearch>
void run_index(const std::string& name, const vro::Matrix& base, const vro::Matrix& q,
               const std::vector<std::uint8_t>& dead, const Truth& t_all, const Truth& t_del,
               const std::vector<std::int64_t>& upd_ids, const vro::Matrix& upd_rows,
               const std::vector<std::size_t>& sample, Build build, Search search, Bytes bytes,
               Del del, Upd upd, Compact compact, bool check_bytes, USearch upd_search) {
    const std::vector<std::uint8_t> none;
    std::size_t dr = 0;

    // Delete del30, then compact (rebuild).
    vro::Matrix v = base;
    Idx ix = build(v);
    SearchFn fn = [&](const float* x) { return search(ix, x); };
    const double r0 = recall(fn, q, t_all, none, dr);
    del(ix, dead);
    const double r1 = recall(fn, q, t_del, dead, dr);
    std::cerr << name << ": recall undeleted " << r0 << ", after del30 " << r1
              << ", deleted returned " << dr << "\n";
    CHECK(dr == 0);
    CHECK(std::abs(r1 - r0) <= 0.03);
    if (name == "flat") CHECK(r1 == 1.0);
    const std::size_t before = bytes(ix);
    compact(ix, vro::CompactMode::kRebuild);
    const std::size_t after = bytes(ix);
    const double rc = recall(fn, q, t_del, dead, dr);
    CHECK(dr == 0);

    // Fresh build on the remaining rows (IDs mapped back).
    vro::Matrix live;
    live.dim = base.dim;
    std::vector<std::int64_t> map;
    for (std::size_t i = 0; i < base.rows; ++i)
        if (!dead[i]) {
            live.data.insert(live.data.end(), base.row(i), base.row(i) + base.dim);
            map.push_back(static_cast<std::int64_t>(i));
        }
    live.rows = map.size();
    Idx fresh = build(live);
    SearchFn ffn = [&](const float* x) {
        vro::SearchResult r = search(fresh, x);
        for (auto& id : r.ids)
            if (id >= 0) id = map[static_cast<std::size_t>(id)];
        return r;
    };
    const double rf = recall(ffn, q, t_del, dead, dr);
    std::cerr << name << ": compacted recall " << rc << ", fresh build " << rf
              << ", index_bytes " << before << " -> " << after << "\n";
    CHECK(std::abs(rc - rf) <= 0.01);
    if (check_bytes) CHECK(after < before);

    // Update upd10: a query equal to the new vector returns the row as top-1.
    vro::Matrix vu = base;
    Idx iu = build(vu);
    upd(iu, upd_ids, upd_rows);
    std::size_t top1 = 0;
    for (std::size_t s : sample) {
        vro::SearchResult r = upd_search(iu, upd_rows.row(s));
        if (r.ids[0] == upd_ids[s]) ++top1;
        else
            std::cerr << name << ": miss row " << upd_ids[s] << ", top-1 " << r.ids[0] << " score "
                      << r.scores[0] << ", visited " << r.counters["visited"] << "\n";
    }
    std::cerr << name << ": upd10 top-1 hits " << top1 << " of " << sample.size() << "\n";
    CHECK(top1 == sample.size());
}

std::string g_root, g_bench;

int run_bench(const std::string& args) {
    std::string cmd = "\"" + g_bench + "\" --data \"" + g_root + "/data/processed/dev\" --limit " +
                      std::to_string(kRows) + " " + args;
    int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_changes <repo_root> <bench_path>\n";
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
        std::size_t n_dead = 0;
        for (auto b : dead) n_dead += b;
        const Truth t_all = truth(v, q, {});
        const Truth t_del = truth(v, q, dead);

        // Updated rows among the first kRows; 100 sampled at an even stride.
        std::vector<std::int64_t> all_ids = vro::npy::read_i64_1d(dev + "/update_upd10_ids.npy");
        vro::Matrix all_rows = vro::npy::read_f32(dev + "/update_upd10_vectors.npy");
        std::vector<std::int64_t> upd_ids;
        vro::Matrix upd_rows;
        upd_rows.dim = v.dim;
        for (std::size_t j = 0; j < all_ids.size(); ++j)
            if (all_ids[j] < static_cast<std::int64_t>(kRows)) {
                upd_ids.push_back(all_ids[j]);
                upd_rows.data.insert(upd_rows.data.end(), all_rows.row(j), all_rows.row(j) + v.dim);
            }
        upd_rows.rows = upd_ids.size();
        std::vector<std::size_t> sample;
        for (std::size_t s = 0; s < 100; ++s) sample.push_back(s * upd_ids.size() / 100);
        std::cerr << "rows " << kRows << ", del30 rows " << n_dead << ", updated rows "
                  << upd_ids.size() << "\n";
        CHECK(upd_ids.size() >= 100);

        const vro::BuildContext ctx{6, 42, "", dev};
        const vro::Params none = params({{"filter", "none"}});

        run_index<vro::flat::Index>(
            "flat", v, q, dead, t_all, t_del, upd_ids, upd_rows, sample,
            [&](vro::Matrix& m) { return vro::flat::build(m, none, ctx); },
            [&](const vro::flat::Index& ix, const float* x) {
                return vro::flat::search(ix, x, kK, none);
            },
            [](const vro::flat::Index& ix) { return vro::flat::index_bytes(ix); },
            [](vro::flat::Index& ix, const std::vector<std::uint8_t>& m) { vro::flat::delete_rows(ix, m); },
            [](vro::flat::Index& ix, const std::vector<std::int64_t>& ids, const vro::Matrix& r) {
                vro::flat::update_rows(ix, ids, r);
            },
            [](vro::flat::Index& ix, vro::CompactMode mode) { vro::flat::compact(ix, mode); }, true,
            [&](const vro::flat::Index& ix, const float* x) {
                return vro::flat::search(ix, x, kK, none);
            });

        const vro::Params ib = params({{"nlist", "256"}, {"iters", "20"}, {"train_size", "20000"}});
        const vro::Params is = params({{"nprobe", "8"}, {"filter", "none"}});
        run_index<vro::ivf::Index>(
            "ivf", v, q, dead, t_all, t_del, upd_ids, upd_rows, sample,
            [&](vro::Matrix& m) { return vro::ivf::build(m, ib, ctx); },
            [&](const vro::ivf::Index& ix, const float* x) { return vro::ivf::search(ix, x, kK, is); },
            [](const vro::ivf::Index& ix) { return vro::ivf::index_bytes(ix); },
            [](vro::ivf::Index& ix, const std::vector<std::uint8_t>& m) { vro::ivf::delete_rows(ix, m); },
            [](vro::ivf::Index& ix, const std::vector<std::int64_t>& ids, const vro::Matrix& r) {
                vro::ivf::update_rows(ix, ids, r);
            },
            [](vro::ivf::Index& ix, vro::CompactMode mode) { vro::ivf::compact(ix, mode); }, true,
            [&](const vro::ivf::Index& ix, const float* x) { return vro::ivf::search(ix, x, kK, is); });

        const vro::Params hb = params({{"m", "16"}, {"ef_construct", "100"}});
        const vro::Params hs = params({{"ef", "64"}, {"filter", "none"}});
        const vro::Params hs_upd = params({{"ef", "128"}, {"filter", "none"}});
        auto hbuild = [&](vro::Matrix& m) { return vro::hnsw::build(m, hb, ctx); };
        auto hsearch = [&](const vro::hnsw::Index& ix, const float* x) {
            return vro::hnsw::search(ix, x, kK, hs);
        };
        run_index<vro::hnsw::Index>(
            "hnsw", v, q, dead, t_all, t_del, upd_ids, upd_rows, sample, hbuild, hsearch,
            [](const vro::hnsw::Index& ix) { return vro::hnsw::index_bytes(ix); },
            [](vro::hnsw::Index& ix, const std::vector<std::uint8_t>& m) { vro::hnsw::delete_rows(ix, m); },
            [](vro::hnsw::Index& ix, const std::vector<std::int64_t>& ids, const vro::Matrix& r) {
                vro::hnsw::update_rows(ix, ids, r);
            },
            [](vro::hnsw::Index& ix, vro::CompactMode mode) { vro::hnsw::compact(ix, mode); }, false,
            // The new vectors lie off the corpus distribution (0.6 old + 0.8
            // random): at ef=64, about 1 in 100 such queries ends in a local
            // maximum of the greedy search. The top-1 check uses ef=128.
            [&](const vro::hnsw::Index& ix, const float* x) {
                return vro::hnsw::search(ix, x, kK, hs_upd);
            });

        // hnsw compact, mode repair: no deleted ID, no edge to a deleted node,
        // every live node reachable, recall close to the tombstoned index.
        {
            vro::Matrix vr = v;
            vro::hnsw::Index ix = hbuild(vr);
            vro::hnsw::delete_rows(ix, dead);
            std::size_t dr = 0;
            SearchFn fn = [&](const float* x) { return hsearch(ix, x); };
            const double r1 = recall(fn, q, t_del, dead, dr);
            const std::size_t before = vro::hnsw::index_bytes(ix);
            vro::hnsw::compact(ix, vro::CompactMode::kRepair);
            const double rr = recall(fn, q, t_del, dead, dr);
            CHECK(dr == 0);
            std::size_t dead_edges = 0;
            for (std::size_t i = 0; i < kRows; ++i) {
                std::int32_t c;
                const auto* nb = ix.neighbors(0, static_cast<std::int32_t>(i), c);
                for (std::int32_t j = 0; j < c; ++j) dead_edges += dead[static_cast<std::size_t>(nb[j].load())];
            }
            CHECK(dead_edges == 0);
            CHECK(!dead[static_cast<std::size_t>(ix.entry.load())]);
            // Unreachable count includes the dead nodes, which have no in-edges.
            CHECK(vro::hnsw::unreachable_layer0(ix) == n_dead);
            std::cerr << "hnsw repair mode: recall " << r1 << " -> " << rr << ", index_bytes "
                      << before << " -> " << vro::hnsw::index_bytes(ix) << "\n";
            CHECK(rr >= r1 - 0.03);
        }

        // bench: --delete del30 on hnsw writes the change keys.
        const std::string out = "/tmp/vro-cpp-test-changes.json";
        CHECK(run_bench("--index hnsw --out " + out + " --delete del30 --search ef=64") == 0);
        std::ifstream in(out);
        nlohmann::json doc = nlohmann::json::parse(in);
        const auto& sp = doc["searches"][0]["search_params"];
        CHECK(sp["deleted"] == "del30");
        CHECK(sp["compacted"] == 0);
        CHECK(!sp.contains("updated"));
        CHECK(doc["extra"]["deleted_rows"] == n_dead);
        CHECK(doc["extra"].contains("delete_s"));
        CHECK(!doc["extra"].contains("compact_s"));
        std::size_t bench_dead = 0;
        for (const auto& row : doc["searches"][0]["ids"])
            for (const auto& id : row)
                if (id.get<std::int64_t>() >= 0) bench_dead += dead[id.get<std::size_t>()];
        CHECK(bench_dead == 0);
        // Only flat, ivf, hnsw take the change flags; bad combinations exit 2.
        CHECK(run_bench("--index pq --out /tmp/vro-cpp-test-changes-pq.json --delete del10 2>/dev/null") == 2);
        CHECK(run_bench("--index flat --out /tmp/x.json --delete del10 --update upd10 2>/dev/null") == 2);
        CHECK(run_bench("--index flat --out /tmp/x.json --compact 2>/dev/null") == 2);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " check(s) failed\n";
    else std::cerr << "changes: PASS\n";
    return g_failures ? 1 : 0;
}
