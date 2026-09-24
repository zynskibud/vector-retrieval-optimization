// bench: builds one index, runs the queries, writes one JSON file (CONTRACT 2-4).
#include <sys/resource.h>
#include <sys/utsname.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "diskann.hpp"
#include "flat.hpp"
#include "hnsw.hpp"
#include "ivf.hpp"
#include "ivf_pq.hpp"
#include "kmeans.hpp"
#include "npy.hpp"
#include "pq.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace vro {
namespace {

// ---------- Errors and exit codes ----------

struct UsageError : std::runtime_error {  // exit 2
    using std::runtime_error::runtime_error;
};

// ---------- Parameter specs (CONTRACT 6) ----------

enum class PType { kInt, kFloat, kString };

struct ParamSpec {
    std::string key;
    std::string def;  // "" = computed by bench (ivf train_size)
    PType type;
    std::vector<std::string> allowed;  // for strings; empty = any
};

struct IndexSpec {
    std::vector<ParamSpec> build;
    std::vector<ParamSpec> search;
};

const std::map<std::string, IndexSpec>& index_specs() {
    static const std::map<std::string, IndexSpec> specs = {
        {"flat", {{}, {}}},
        {"ivf",
         {{{"nlist", "1024", PType::kInt, {}},
           {"train_size", "", PType::kInt, {}},
           {"iters", "20", PType::kInt, {}}},
          {{"nprobe", "8", PType::kInt, {}}}}},
        {"pq",
         {{{"m", "48", PType::kInt, {}},
           {"nbits", "8", PType::kInt, {}},
           {"metric", "ip", PType::kString, {"ip", "l2"}},
           {"train_size", "100000", PType::kInt, {}},
           {"iters", "20", PType::kInt, {}}},
          {{"rerank", "0", PType::kInt, {}}}}},
        {"ivf_pq",
         {{{"nlist", "1024", PType::kInt, {}},
           {"iters", "20", PType::kInt, {}},
           {"m", "48", PType::kInt, {}},
           {"nbits", "8", PType::kInt, {}},
           {"metric", "ip", PType::kString, {"ip", "l2"}},
           {"train_size", "100000", PType::kInt, {}}},
          {{"nprobe", "8", PType::kInt, {}}, {"rerank", "0", PType::kInt, {}}}}},
        {"hnsw",
         {{{"m", "16", PType::kInt, {}}, {"ef_construct", "100", PType::kInt, {}}},
          {{"ef", "64", PType::kInt, {}}}}},
        {"diskann",
         {{{"r", "64", PType::kInt, {}},
           {"l_build", "100", PType::kInt, {}},
           {"alpha", "1.2", PType::kFloat, {}},
           {"pq_m", "48", PType::kInt, {}},
           {"metric", "ip", PType::kString, {"ip", "l2"}}},
          {{"l", "100", PType::kInt, {}},
           {"beam", "4", PType::kInt, {}},
           {"rerank", "100", PType::kInt, {}},
           {"io", "mmap", PType::kString, {"mmap", "nocache"}}}}},
    };
    return specs;
}

const ParamSpec* find_spec(const std::vector<ParamSpec>& specs, const std::string& key) {
    for (const auto& s : specs)
        if (s.key == key) return &s;
    return nullptr;
}

void check_value(const ParamSpec& spec, const std::string& value) {
    Params p;
    p.values[spec.key] = value;
    try {
        if (spec.type == PType::kInt) p.get_int(spec.key);
        if (spec.type == PType::kFloat) p.get_double(spec.key);
    } catch (const ParamError& e) {
        throw UsageError(e.what());
    }
    if (!spec.allowed.empty() &&
        std::find(spec.allowed.begin(), spec.allowed.end(), value) == spec.allowed.end())
        throw UsageError("parameter " + spec.key + " has invalid value: " + value);
}

// Parses "a=1,b=2" into params, checking each key against specs.
void parse_pairs(const std::string& text, const std::vector<ParamSpec>& specs,
                 const std::string& index, Params& out) {
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t comma = text.find(',', start);
        std::string pair = text.substr(start, comma == std::string::npos ? std::string::npos
                                                                         : comma - start);
        std::size_t eq = pair.find('=');
        if (eq == std::string::npos || eq == 0)
            throw UsageError("expected KEY=VALUE, got: " + pair);
        std::string key = pair.substr(0, eq);
        std::string value = pair.substr(eq + 1);
        const ParamSpec* spec = find_spec(specs, key);
        if (!spec) throw UsageError("unknown parameter for " + index + ": " + key);
        check_value(*spec, value);
        out.values[key] = value;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
}

void fill_defaults(const std::vector<ParamSpec>& specs, Params& p) {
    for (const auto& s : specs)
        if (!p.has(s.key) && !s.def.empty()) p.values[s.key] = s.def;
}

json params_to_json(const std::vector<ParamSpec>& specs, const Params& p) {
    json out = json::object();
    for (const auto& s : specs) {
        if (s.type == PType::kInt) out[s.key] = p.get_int(s.key);
        else if (s.type == PType::kFloat) out[s.key] = p.get_double(s.key);
        else out[s.key] = p.get_string(s.key);
    }
    return out;
}

// ---------- Command line (CONTRACT 2) ----------

struct Args {
    std::string index, data, out;
    std::size_t k = 10;
    int threads = 0;
    std::uint64_t seed = 42;
    std::size_t warmup = 100;
    std::size_t limit = 0;
    std::vector<std::string> build;
    std::vector<std::string> search;
};

long long parse_int_flag(const std::string& flag, const std::string& value, long long min) {
    char* end = nullptr;
    long long v = std::strtoll(value.c_str(), &end, 10);
    if (value.empty() || *end != '\0' || v < min)
        throw UsageError(flag + " needs an integer >= " + std::to_string(min) + ", got: " + value);
    return v;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) throw UsageError("missing value for " + flag);
        std::string v = argv[++i];
        if (flag == "--index") a.index = v;
        else if (flag == "--data") a.data = v;
        else if (flag == "--out") a.out = v;
        else if (flag == "--k") a.k = static_cast<std::size_t>(parse_int_flag(flag, v, 1));
        else if (flag == "--build") a.build.push_back(v);
        else if (flag == "--search") a.search.push_back(v);
        else if (flag == "--threads") a.threads = static_cast<int>(parse_int_flag(flag, v, 1));
        else if (flag == "--seed") a.seed = static_cast<std::uint64_t>(parse_int_flag(flag, v, 0));
        else if (flag == "--warmup") a.warmup = static_cast<std::size_t>(parse_int_flag(flag, v, 0));
        else if (flag == "--limit") a.limit = static_cast<std::size_t>(parse_int_flag(flag, v, 1));
        else throw UsageError("unknown option: " + flag);
    }
    if (a.index.empty()) throw UsageError("missing --index");
    if (a.data.empty()) throw UsageError("missing --data");
    if (a.out.empty()) throw UsageError("missing --out");
    if (!index_specs().count(a.index)) throw UsageError("unknown index: " + a.index);
    if (a.threads == 0) a.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return a;
}

// ---------- Index dispatch ----------

struct AnyIndex {
    virtual ~AnyIndex() = default;
    virtual BuildTimes times() const = 0;
    virtual SearchResult search(const float* q, std::size_t k, const Params& p) const = 0;
    virtual std::size_t bytes() const = 0;
    virtual std::map<std::string, double> extra() const = 0;
};

template <typename Idx, SearchResult (*Search)(const Idx&, const float*, std::size_t, const Params&),
          std::size_t (*Bytes)(const Idx&), std::map<std::string, double> (*Extra)(const Idx&)>
struct Wrapped : AnyIndex {
    explicit Wrapped(Idx i) : idx(std::move(i)) {}
    BuildTimes times() const override { return idx.times; }
    SearchResult search(const float* q, std::size_t k, const Params& p) const override {
        return Search(idx, q, k, p);
    }
    std::size_t bytes() const override { return Bytes(idx); }
    std::map<std::string, double> extra() const override { return Extra(idx); }
    Idx idx;
};

#define VRO_WRAP(ns)                                                                      \
    if (name == #ns)                                                                      \
        return std::make_unique<Wrapped<ns::Index, &ns::search, &ns::index_bytes, &ns::extra>>( \
            ns::build(vectors, params, ctx));

std::unique_ptr<AnyIndex> build_index(const std::string& name, Matrix& vectors,
                                      const Params& params, const BuildContext& ctx) {
    VRO_WRAP(flat)
    VRO_WRAP(ivf)
    VRO_WRAP(pq)
    VRO_WRAP(ivf_pq)
    VRO_WRAP(hnsw)
    VRO_WRAP(diskann)
    throw UsageError("unknown index: " + name);
}

#undef VRO_WRAP

// ---------- Machine info and memory (CONTRACT 4) ----------

double peak_rss_mb() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    double bytes = static_cast<double>(ru.ru_maxrss);  // bytes on macOS
#else
    double bytes = static_cast<double>(ru.ru_maxrss) * 1024.0;  // KB on Linux
#endif
    return bytes / (1024.0 * 1024.0);
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string cpu_name() {
#if defined(__APPLE__)
    char buf[256] = {};
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) return buf;
#else
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
#endif
    return "unknown";
}

json machine_info() {
    utsname u{};
    uname(&u);
    return json{{"os", lower(u.sysname)},
                {"arch", u.machine},
                {"cpu", cpu_name()},
                {"cores", std::thread::hardware_concurrency()}};
}

// ---------- Search runs ----------

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

json score_to_json(float s) {
    if (std::isinf(s) && s < 0) return nullptr;  // pad score -inf -> null
    return static_cast<double>(s);
}

void warm_up(const AnyIndex& index, const Matrix& queries, std::size_t k, std::size_t n,
             const Params& p) {
    for (std::size_t i = 0; i < std::min(n, queries.rows); ++i) index.search(queries.row(i), k, p);
}

// One timed pass over all queries, one at a time, on this thread.
json run_search(const AnyIndex& index, const Matrix& queries, std::size_t k, const Params& p,
                const std::vector<ParamSpec>& specs) {
    std::vector<SearchResult> results(queries.rows);
    std::vector<double> latency_ms(queries.rows);
    auto t_all = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < queries.rows; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        SearchResult r = index.search(queries.row(i), k, p);
        latency_ms[i] = seconds_since(t0) * 1000.0;
        results[i] = std::move(r);
    }
    double total_s = seconds_since(t_all);

    json ids = json::array(), scores = json::array();
    double dist_sum = 0.0;
    bool counted = true;
    std::map<std::string, double> counter_sums;
    for (const auto& r : results) {
        ids.push_back(r.ids);
        json row = json::array();
        for (float s : r.scores) row.push_back(score_to_json(s));
        scores.push_back(std::move(row));
        if (r.distance_computations < 0) counted = false;
        dist_sum += static_cast<double>(r.distance_computations);
        for (const auto& [key, v] : r.counters) counter_sums[key] += v;
    }
    double q = static_cast<double>(queries.rows);
    json extra = json::object();  // per-run counters, mean per query
    for (const auto& [key, sum] : counter_sums) extra[key] = sum / q;

    json out;
    out["search_params"] = params_to_json(specs, p);
    out["ids"] = std::move(ids);
    out["scores"] = std::move(scores);
    out["latency_ms"] = latency_ms;
    out["total_s"] = total_s;
    out["qps"] = q / total_s;
    out["distance_computations"] = counted ? json(dist_sum / q) : json(nullptr);
    out["extra"] = std::move(extra);
    return out;
}

// ---------- Main flow ----------

int run(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const IndexSpec& spec = index_specs().at(args.index);

    Params build_params;
    for (const auto& b : args.build) parse_pairs(b, spec.build, args.index, build_params);
    std::vector<Params> search_sets;
    for (const auto& s : args.search) {
        Params p;
        parse_pairs(s, spec.search, args.index, p);
        search_sets.push_back(std::move(p));
    }
    if (search_sets.empty()) search_sets.emplace_back();
    fill_defaults(spec.build, build_params);
    for (auto& p : search_sets) fill_defaults(spec.search, p);

    Matrix vectors = npy::read_f32(args.data + "/vectors.npy", args.limit);
    Matrix queries = npy::read_f32(args.data + "/queries.npy");
    if (queries.dim != vectors.dim) throw std::runtime_error("queries and vectors differ in dim");
    std::size_t n = vectors.rows, dim = vectors.dim;
    if (args.index == "ivf" && !build_params.has("train_size")) {
        auto nlist = static_cast<std::size_t>(build_params.get_int("nlist"));
        build_params.values["train_size"] = std::to_string(default_train_size(n, nlist));
    }

    BuildContext ctx{args.threads, args.seed, args.out};
    std::cerr << "building " << args.index << " on " << n << " rows\n";
    std::unique_ptr<AnyIndex> index = build_index(args.index, vectors, build_params, ctx);
    BuildTimes times = index->times();

    json doc;
    doc["contract_version"] = 1;
    doc["language"] = "cpp";
    doc["index"] = args.index;
    doc["data_dir"] = args.data;
    doc["n"] = n;
    doc["dim"] = dim;
    doc["q"] = queries.rows;
    doc["k"] = args.k;
    doc["threads"] = args.threads;
    doc["seed"] = args.seed;
    doc["build_params"] = params_to_json(spec.build, build_params);
    doc["build"] = {{"train_s", times.train_s},
                    {"add_s", times.add_s},
                    {"total_s", times.train_s + times.add_s},
                    {"peak_rss_mb", peak_rss_mb()},
                    {"index_bytes", index->bytes()}};

    warm_up(*index, queries, args.k, args.warmup, search_sets.front());
    json searches = json::array();
    for (const auto& p : search_sets) {
        std::cerr << "searching " << search_sets.size() << " set(s)\n";
        searches.push_back(run_search(*index, queries, args.k, p, spec.search));
    }
    doc["searches"] = std::move(searches);
    doc["machine"] = machine_info();

    json extra = json::object();
    for (const auto& [key, v] : index->extra()) extra[key] = v;
    doc["extra"] = std::move(extra);

    std::ofstream out(args.out);
    if (!out) throw std::runtime_error("cannot write " + args.out);
    out << doc.dump() << '\n';
    if (!out) throw std::runtime_error("write failed: " + args.out);
    return 0;
}

}  // namespace
}  // namespace vro

int main(int argc, char** argv) {
    try {
        return vro::run(argc, argv);
    } catch (const vro::UsageError& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    } catch (const vro::ParamError& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    } catch (const vro::NotImplemented& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
