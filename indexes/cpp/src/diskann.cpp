#include "diskann.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "distance.hpp"
#include "pq.hpp"
#include "splitmix.hpp"

namespace vro::diskann {

namespace {

constexpr std::size_t kStripes = 65536;  // mutex stripe over node IDs (parallel build)
constexpr std::size_t kChunk = 64;       // rows claimed per worker step
constexpr std::size_t kAlign = 4096;     // record alignment on disk
constexpr std::size_t kCentroids = pq::kCentroids;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::runtime_error sys_error(const std::string& what) {
    return std::runtime_error("diskann: " + what + ": " + std::strerror(errno));
}

struct Cand {
    float score;
    std::int32_t id;
};

// "a is better than b": higher score, ties to the lower ID.
inline bool better(const Cand& a, const Cand& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

// Visited set: node is visited when mark[node] == gen.
struct Visited {
    std::vector<std::uint32_t> mark;
    std::uint32_t gen = 0;
    void reset(std::size_t n) {
        if (mark.size() != n) {
            mark.assign(n, 0);
            gen = 0;
        }
        if (++gen == 0) {
            std::fill(mark.begin(), mark.end(), 0);
            gen = 1;
        }
    }
    bool test_and_set(std::int32_t id) {
        auto& v = mark[static_cast<std::size_t>(id)];
        if (v == gen) return true;
        v = gen;
        return false;
    }
};

// ---------- Build: Vamana graph in RAM ----------

struct Graph {
    const Matrix* v = nullptr;
    std::size_t r = 0;
    std::vector<std::int32_t> links;   // n * r slots
    std::vector<std::int32_t> counts;  // n
    std::mutex* locks = nullptr;       // nullptr (one thread) or kStripes mutexes

    std::mutex& lock_of(std::int32_t node) const {
        return locks[static_cast<std::size_t>(node) & (kStripes - 1)];
    }
    std::int32_t* list(std::int32_t node) { return links.data() + static_cast<std::size_t>(node) * r; }
    // Copies node's out-edges into buf (under the node's lock in a parallel build).
    void read(std::int32_t node, std::vector<std::int32_t>& buf) {
        std::unique_lock<std::mutex> g;
        if (locks) g = std::unique_lock<std::mutex>(lock_of(node));
        const std::int32_t* p = list(node);
        buf.assign(p, p + counts[static_cast<std::size_t>(node)]);
    }
    void write(std::int32_t node, const std::vector<std::int32_t>& ids) {  // caller holds the lock
        std::copy(ids.begin(), ids.end(), list(node));
        counts[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(ids.size());
    }
};

// Sorted list of at most cap candidates, best first, with an expanded flag.
struct CandList {
    struct Item {
        float score;
        std::int32_t id;
        bool expanded;
        std::int32_t slot;  // search: row in the vector scratch buffer
    };
    std::vector<Item> items;
    std::size_t cap = 0;

    void reset(std::size_t c) {
        items.clear();
        items.reserve(c + 1);
        cap = c;
    }
    // Returns the position of the new item, or cap if it was not inserted.
    std::size_t insert(const Cand& c) {
        if (items.size() >= cap && !better(c, {items.back().score, items.back().id})) return cap;
        auto it = std::lower_bound(items.begin(), items.end(), c, [](const Item& a, const Cand& b) {
            return better({a.score, a.id}, b);
        });
        std::size_t pos = static_cast<std::size_t>(it - items.begin());
        items.insert(it, Item{c.score, c.id, false, -1});
        if (items.size() > cap) items.pop_back();
        return pos;
    }
};

// Greedy search (Vamana, Algorithm 1) with full vectors, list size l.
// Fills visited_out with every expanded node and its score to q.
void greedy_build(Graph& g, const float* q, std::int32_t entry, std::size_t l, Visited& vis,
                  CandList& list, std::vector<std::int32_t>& buf, std::vector<Cand>& visited_out) {
    const Matrix& v = *g.v;
    vis.reset(v.rows);
    list.reset(l);
    visited_out.clear();
    vis.test_and_set(entry);
    list.insert({dot(q, v.row(static_cast<std::size_t>(entry)), v.dim), entry});
    std::size_t start = 0;  // no unexpanded item before this position
    while (true) {
        std::size_t p = start;
        while (p < list.items.size() && list.items[p].expanded) ++p;
        if (p >= list.items.size()) break;
        list.items[p].expanded = true;
        Cand cur{list.items[p].score, list.items[p].id};
        visited_out.push_back(cur);
        start = p + 1;
        g.read(cur.id, buf);
        for (std::int32_t e : buf) {
            if (vis.test_and_set(e)) continue;
            std::size_t pos = list.insert({dot(q, v.row(static_cast<std::size_t>(e)), v.dim), e});
            start = std::min(start, pos);  // the new item can land before start
        }
    }
}

// RobustPrune(p, cands, alpha, r). cands: IDs, any order, may hold p and repeats.
// Distance d(a, b) = 2 - 2 * dot(a, b). Returns at most r IDs, nearest first.
std::vector<std::int32_t> robust_prune(const Matrix& v, std::int32_t p, std::vector<std::int32_t> cands,
                                       double alpha, std::size_t r) {
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    const float* xp = v.row(static_cast<std::size_t>(p));
    std::vector<Cand> c;
    c.reserve(cands.size());
    for (std::int32_t id : cands)
        if (id != p) c.push_back({dot(xp, v.row(static_cast<std::size_t>(id)), v.dim), id});
    std::sort(c.begin(), c.end(), better);
    std::vector<char> alive(c.size(), 1);
    std::vector<std::int32_t> out;
    out.reserve(r);
    const float a = static_cast<float>(alpha);
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!alive[i]) continue;
        out.push_back(c[i].id);
        if (out.size() >= r) break;
        const float* xs = v.row(static_cast<std::size_t>(c[i].id));
        for (std::size_t j = i + 1; j < c.size(); ++j) {
            if (!alive[j]) continue;
            float d_star = 2.0f - 2.0f * dot(xs, v.row(static_cast<std::size_t>(c[j].id)), v.dim);
            float d_p = 2.0f - 2.0f * c[j].score;
            if (a * d_star <= d_p) alive[j] = 0;
        }
    }
    return out;
}

struct Worker {
    Visited vis;
    CandList list;
    std::vector<std::int32_t> buf;
    std::vector<Cand> visited;
};

// One Vamana step for row i.
void process(Graph& g, std::int32_t i, std::int32_t entry, std::size_t l_build, double alpha, Worker& w) {
    const Matrix& v = *g.v;
    greedy_build(g, v.row(static_cast<std::size_t>(i)), entry, l_build, w.vis, w.list, w.buf, w.visited);
    std::vector<std::int32_t> out;
    {
        std::unique_lock<std::mutex> lk;
        if (g.locks) lk = std::unique_lock<std::mutex>(g.lock_of(i));
        std::vector<std::int32_t> cands;
        cands.reserve(w.visited.size() + g.r);
        for (const Cand& c : w.visited) cands.push_back(c.id);
        const std::int32_t* cur = g.list(i);
        cands.insert(cands.end(), cur, cur + g.counts[static_cast<std::size_t>(i)]);
        out = robust_prune(v, i, std::move(cands), alpha, g.r);
        g.write(i, out);
    }
    for (std::int32_t j : out) {
        std::unique_lock<std::mutex> lk;
        if (g.locks) lk = std::unique_lock<std::mutex>(g.lock_of(j));
        std::int32_t* lj = g.list(j);
        std::int32_t& cj = g.counts[static_cast<std::size_t>(j)];
        if (std::find(lj, lj + cj, i) != lj + cj) continue;
        if (static_cast<std::size_t>(cj) < g.r) {
            lj[cj++] = i;
            continue;
        }
        std::vector<std::int32_t> cands(lj, lj + cj);
        cands.push_back(i);
        g.write(j, robust_prune(v, j, std::move(cands), alpha, g.r));
    }
}

std::int32_t medoid(const Matrix& v) {
    std::vector<double> sum(v.dim, 0.0);
    for (std::size_t i = 0; i < v.rows; ++i) {
        const float* x = v.row(i);
        for (std::size_t d = 0; d < v.dim; ++d) sum[d] += x[d];
    }
    std::vector<float> mean(v.dim);
    for (std::size_t d = 0; d < v.dim; ++d)
        mean[d] = static_cast<float>(sum[d] / static_cast<double>(v.rows));
    Cand best{-std::numeric_limits<float>::infinity(), -1};
    for (std::size_t i = 0; i < v.rows; ++i) {
        Cand c{dot(mean.data(), v.row(i), v.dim), static_cast<std::int32_t>(i)};
        if (best.id < 0 || better(c, best)) best = c;
    }
    return best.id;
}

void init_random(Graph& g, std::uint64_t seed) {
    const std::size_t n = g.v->rows;
    const std::size_t want = std::min(g.r, n - 1);
    SplitMix64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        std::int32_t* l = g.list(static_cast<std::int32_t>(i));
        std::int32_t& c = g.counts[i];
        while (static_cast<std::size_t>(c) < want) {
            auto j = static_cast<std::int32_t>(rng.next_below(n));
            if (static_cast<std::size_t>(j) == i || std::find(l, l + c, j) != l + c) continue;
            l[c++] = j;
        }
    }
}

// ---------- File ----------

void write_all(int fd, const char* p, std::size_t len, const std::string& path) {
    while (len > 0) {
        ssize_t w = ::write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw sys_error("write " + path);
        }
        p += w;
        len -= static_cast<std::size_t>(w);
    }
}

// Record i: vector (dim float32), then r int32 edges (-1 = empty), zero padding.
void write_file(const Index& ix, const Matrix& v, const Graph& g) {
    int fd = ::open(ix.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw sys_error("open " + ix.path);
#if defined(__APPLE__)
    fcntl(fd, F_NOCACHE, 1);  // keep the written pages out of the cache
#endif
    const std::size_t batch = 256;
    std::vector<char> buf(batch * ix.record_bytes);
    try {
        for (std::size_t lo = 0; lo < ix.n; lo += batch) {
            std::size_t hi = std::min(ix.n, lo + batch);
            std::fill(buf.begin(), buf.end(), 0);
            for (std::size_t i = lo; i < hi; ++i) {
                char* rec = buf.data() + (i - lo) * ix.record_bytes;
                std::memcpy(rec, v.row(i), ix.dim * sizeof(float));
                std::int32_t* e = reinterpret_cast<std::int32_t*>(rec + ix.dim * sizeof(float));
                std::fill(e, e + ix.r, -1);
                std::copy_n(g.links.data() + i * g.r, g.counts[i], e);
            }
            write_all(fd, buf.data(), (hi - lo) * ix.record_bytes, ix.path);
        }
#if defined(__linux__)
        fdatasync(fd);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    } catch (...) {
        ::close(fd);
        throw;
    }
    if (::close(fd) != 0) throw sys_error("close " + ix.path);
}

// Aligned buffer for one record (O_DIRECT needs 4096 alignment).
struct AlignedBuf {
    char* p = nullptr;
    std::size_t size = 0;
    ~AlignedBuf() { std::free(p); }
    char* get(std::size_t n) {
        if (size < n) {
            std::free(p);
            void* q = nullptr;
            if (posix_memalign(&q, kAlign, n) != 0) throw std::bad_alloc();
            p = static_cast<char*>(q);
            size = n;
        }
        return p;
    }
};

}  // namespace

// ---------- File access for search (CONTRACT 6.7.1) ----------

enum class IoMode { kNone, kMmap, kNoCache };

struct FileIo {
    std::string path;
    std::size_t len = 0;
    std::size_t record_bytes = 0;
    int fd = -1;                     // uncached descriptor
    const char* map = nullptr;
    std::atomic<IoMode> mode{IoMode::kNone};
    std::mutex mu;

    ~FileIo() {
        if (map) munmap(const_cast<char*>(map), len);
        if (fd >= 0) ::close(fd);
    }

    // Opens the uncached descriptor. No map has touched the file yet.
    void open_nocache() {
#if defined(__linux__)
        fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#else
        fd = ::open(path.c_str(), O_RDONLY);
#endif
        if (fd < 0) throw sys_error("open " + path);
#if defined(__APPLE__)
        if (fcntl(fd, F_NOCACHE, 1) != 0) throw sys_error("fcntl F_NOCACHE " + path);
#endif
    }

    // Switches mode. Leaving mmap invalidates the cached pages and unmaps the
    // file, so uncached reads start cold.
    void ensure(IoMode want) {
        if (mode.load(std::memory_order_acquire) == want) return;
        std::lock_guard<std::mutex> g(mu);
        if (mode.load(std::memory_order_relaxed) == want) return;
        if (want == IoMode::kMmap && !map) {
            int mfd = ::open(path.c_str(), O_RDONLY);
            if (mfd < 0) throw sys_error("open " + path);
            void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, mfd, 0);
            ::close(mfd);
            if (p == MAP_FAILED) throw sys_error("mmap " + path);
            map = static_cast<const char*>(p);
        } else if (want == IoMode::kNoCache && map) {
            msync(const_cast<char*>(map), len, MS_INVALIDATE);
#if defined(__linux__)
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            munmap(const_cast<char*>(map), len);
            map = nullptr;
        }
        mode.store(want, std::memory_order_release);
    }

    // Returns a pointer to record i: in the map, or pread into buf.
    const char* record(std::size_t i, AlignedBuf& buf) const {
        const std::size_t off = i * record_bytes;
        if (mode.load(std::memory_order_relaxed) == IoMode::kMmap) return map + off;
        char* dst = buf.get(record_bytes);
        std::size_t done = 0;
        while (done < record_bytes) {
            ssize_t got = ::pread(fd, dst + done, record_bytes - done, static_cast<off_t>(off + done));
            if (got < 0) {
                if (errno == EINTR) continue;
                throw sys_error("pread " + path);
            }
            if (got == 0) throw std::runtime_error("diskann: short read in " + path);
            done += static_cast<std::size_t>(got);
        }
        return dst;
    }
};

Index build(Matrix& vectors, const Params& params, const BuildContext& ctx) {
    const long long r = params.get_int("r");
    const long long l_build = params.get_int("l_build");
    const double alpha = params.get_double("alpha");
    const long long pq_m = params.get_int("pq_m");
    const std::string& metric = params.get_string("metric");
    if (r < 1) throw ParamError("diskann: r must be >= 1");
    if (l_build < 1) throw ParamError("diskann: l_build must be >= 1");
    if (alpha < 1.0) throw ParamError("diskann: alpha must be >= 1");
    if (pq_m < 1 || vectors.dim % static_cast<std::size_t>(pq_m) != 0)
        throw ParamError("diskann: pq_m must divide dim " + std::to_string(vectors.dim));
    if (metric != "ip" && metric != "l2") throw ParamError("diskann: metric must be ip or l2");
    if (vectors.rows < 2) throw std::runtime_error("diskann: needs at least 2 rows");
    if (vectors.rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("diskann: too many rows for int32 IDs");
    if (ctx.out_path.empty()) throw std::runtime_error("diskann: BuildContext.out_path is empty");

    Index ix;
    ix.n = vectors.rows;
    ix.dim = vectors.dim;
    ix.r = static_cast<std::size_t>(r);
    ix.record_bytes = (ix.dim * sizeof(float) + ix.r * sizeof(std::int32_t) + kAlign - 1) / kAlign * kAlign;
    ix.path = ctx.out_path + ".diskann";
    ix.threads = std::max(1, ctx.threads);
    const std::size_t n = ix.n;

    // Graph (add_s).
    auto t0 = std::chrono::steady_clock::now();
    Graph g;
    g.v = &vectors;
    g.r = ix.r;
    g.links.assign(n * ix.r, -1);
    g.counts.assign(n, 0);
    ix.entry = medoid(vectors);
    init_random(g, ctx.seed);
    const std::size_t lb = static_cast<std::size_t>(l_build);
    for (double a : {1.0, alpha}) {
        if (ix.threads == 1) {
            Worker w;
            for (std::size_t i = 0; i < n; ++i) process(g, static_cast<std::int32_t>(i), ix.entry, lb, a, w);
        } else {
            std::unique_ptr<std::mutex[]> locks(new std::mutex[kStripes]);
            g.locks = locks.get();
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> pool;
            for (int t = 0; t < ix.threads; ++t)
                pool.emplace_back([&] {
                    Worker w;
                    for (std::size_t c0 = next.fetch_add(kChunk); c0 < n; c0 = next.fetch_add(kChunk))
                        for (std::size_t i = c0; i < std::min(n, c0 + kChunk); ++i)
                            process(g, static_cast<std::int32_t>(i), ix.entry, lb, a, w);
                });
            for (auto& th : pool) th.join();
            g.locks = nullptr;
        }
    }
    double graph_s = seconds_since(t0);

    // PQ codes (6.4 with m = pq_m, train_size = N): train_s + encode time.
    Params pp;
    pp.values = {{"m", std::to_string(pq_m)}, {"nbits", "8"}, {"metric", metric},
                 {"train_size", std::to_string(n)}, {"iters", "20"}};
    pq::Index pqi = pq::build(vectors, pp, ctx);
    ix.metric = pqi.metric;
    ix.m = pqi.m;
    ix.dsub = pqi.dsub;
    ix.codebooks = std::move(pqi.codebooks);
    ix.codes = std::move(pqi.codes);
    ix.times.train_s = pqi.times.train_s;

    // File, then release the corpus.
    t0 = std::chrono::steady_clock::now();
    write_file(ix, vectors, g);
    ix.disk_bytes = n * ix.record_bytes;
    std::size_t edges = 0;
    for (std::int32_t c : g.counts) edges += static_cast<std::size_t>(c);
    ix.mean_out_degree = static_cast<double>(edges) / static_cast<double>(n);
    vectors.data = std::vector<float>();
    vectors.rows = 0;
    ix.times.add_s = graph_s + pqi.times.add_s + seconds_since(t0);

    ix.io = std::make_shared<FileIo>();
    ix.io->path = ix.path;
    ix.io->len = ix.disk_bytes;
    ix.io->record_bytes = ix.record_bytes;
    ix.io->open_nocache();
    return ix;
}

SearchResult search(const Index& ix, const float* query, std::size_t k, const Params& params) {
    const long long l_in = params.get_int("l");
    const long long beam_in = params.get_int("beam");
    const long long rerank_in = params.get_int("rerank");
    const std::string& io = params.get_string("io");
    if (l_in < 1) throw ParamError("diskann: l must be >= 1");
    if (beam_in < 1) throw ParamError("diskann: beam must be >= 1");
    if (rerank_in < 0) throw ParamError("diskann: rerank must be >= 0");
    if (io != "mmap" && io != "nocache") throw ParamError("diskann: io must be mmap or nocache");
    const std::size_t l = std::max(static_cast<std::size_t>(l_in), k);
    const std::size_t beam = static_cast<std::size_t>(beam_in);
    const std::size_t rerank = static_cast<std::size_t>(rerank_in);
    FileIo& f = *ix.io;
    f.ensure(io == "mmap" ? IoMode::kMmap : IoMode::kNoCache);

    SearchResult res;
    std::int64_t dc = 0, reads = 0;
    const std::size_t m = ix.m, dsub = ix.dsub, dim = ix.dim;

    // PQ table T (m, 256); higher is better in both metrics (6.4.1).
    thread_local std::vector<float> table;
    table.resize(m * kCentroids);
    const bool l2 = ix.metric == Metric::kL2;
    for (std::size_t j = 0; j < m; ++j) {
        const float* qj = query + j * dsub;
        const float* cb = ix.codebooks.data() + j * kCentroids * dsub;
        for (std::size_t c = 0; c < kCentroids; ++c)
            table[j * kCentroids + c] = l2 ? -l2_sq(qj, cb + c * dsub, dsub) : dot(qj, cb + c * dsub, dsub);
    }
    auto pq_score = [&](std::int32_t id) {
        const std::uint8_t* code = ix.codes.data() + static_cast<std::size_t>(id) * m;
        float s = 0.0f;
        for (std::size_t j = 0; j < m; ++j) s += table[j * kCentroids + code[j]];
        ++dc;
        return s;
    };

    thread_local Visited vis;
    thread_local CandList list;
    thread_local AlignedBuf buf;
    thread_local std::vector<float> vecs;  // full vectors of expanded nodes, one row per slot
    thread_local std::vector<std::size_t> pick;
    vis.reset(ix.n);
    list.reset(l);
    vecs.clear();
    vis.test_and_set(ix.entry);
    list.insert({pq_score(ix.entry), ix.entry});

    std::vector<std::int32_t> nbrs(ix.r);
    while (true) {
        pick.clear();
        for (std::size_t p = 0; p < list.items.size() && pick.size() < beam; ++p)
            if (!list.items[p].expanded) pick.push_back(p);
        if (pick.empty()) break;
        // Read the records first, then insert neighbors (inserts move items).
        std::vector<std::int32_t> all;
        all.reserve(pick.size() * ix.r);
        for (std::size_t p : pick) {
            auto& it = list.items[p];
            it.expanded = true;
            it.slot = static_cast<std::int32_t>(vecs.size() / dim);
            const char* rec = f.record(static_cast<std::size_t>(it.id), buf);
            ++reads;
            vecs.insert(vecs.end(), reinterpret_cast<const float*>(rec),
                        reinterpret_cast<const float*>(rec) + dim);
            std::memcpy(nbrs.data(), rec + dim * sizeof(float), ix.r * sizeof(std::int32_t));
            for (std::int32_t e : nbrs)
                if (e >= 0) all.push_back(e);
        }
        for (std::int32_t e : all) {
            if (vis.test_and_set(e)) continue;
            list.insert({pq_score(e), e});
        }
    }

    // Every node left in the list was expanded, so its vector is in vecs.
    if (rerank == 0) {
        TopK top(k);
        for (const auto& it : list.items) top.push(it.id, it.score);
        top.result(res.ids, res.scores);
    } else {
        TopK top(k);
        const std::size_t nr = std::min(rerank, list.items.size());
        for (std::size_t p = 0; p < nr; ++p) {
            const auto& it = list.items[p];
            top.push(it.id, dot(query, vecs.data() + static_cast<std::size_t>(it.slot) * dim, dim));
            ++dc;
        }
        top.result(res.ids, res.scores);
    }
    res.distance_computations = dc;
    res.counters["disk_reads"] = static_cast<double>(reads);
    res.counters["disk_bytes_read"] = static_cast<double>(reads) * static_cast<double>(ix.record_bytes);
    return res;
}

std::size_t index_bytes(const Index& ix) {
    return ix.codes.size() + ix.codebooks.size() * sizeof(float) + sizeof(ix.entry);
}

std::map<std::string, double> extra(const Index& ix) {
    return {{"disk_bytes", static_cast<double>(ix.disk_bytes)},
            {"record_bytes", static_cast<double>(ix.record_bytes)},
            {"entry_point", static_cast<double>(ix.entry)},
            {"mean_out_degree", ix.mean_out_degree},
            {"build_threads", static_cast<double>(ix.threads)}};
}

}  // namespace vro::diskann
