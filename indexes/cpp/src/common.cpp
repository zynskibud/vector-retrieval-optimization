#include "common.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>

#include "npy.hpp"

namespace vro {

const std::string& Params::get_string(const std::string& key) const {
    auto it = values.find(key);
    if (it == values.end()) throw ParamError("missing parameter: " + key);
    return it->second;
}

long long Params::get_int(const std::string& key) const {
    const std::string& s = get_string(key);
    char* end = nullptr;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0') throw ParamError("parameter " + key + " is not an integer: " + s);
    return v;
}

double Params::get_double(const std::string& key) const {
    const std::string& s = get_string(key);
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0') throw ParamError("parameter " + key + " is not a number: " + s);
    return v;
}

const FilterMask* get_filter(const std::string& data_dir, const Params& params, std::size_t n) {
    if (!params.has("filter")) return nullptr;
    const std::string& name = params.get_string("filter");
    if (name == "none") return nullptr;
    if (name != "top50" && name != "top10" && name != "top1" && name != "top01")
        throw ParamError("unknown filter: " + name);
    if (data_dir.empty()) throw ParamError("filter " + name + " needs BuildContext.data_dir");
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<FilterMask>> cache;
    const std::string key = data_dir + "\n" + name + "\n" + std::to_string(n);
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    auto m = std::make_unique<FilterMask>();
    m->pass = npy::read_bool(data_dir + "/filter_" + name + ".npy");
    if (m->pass.size() < n)
        throw ParamError("filter_" + name + ".npy has fewer rows than the corpus");
    m->pass.resize(n);
    for (std::uint8_t b : m->pass) m->rows += b;
    return cache.emplace(key, std::move(m)).first->second.get();
}

}  // namespace vro
