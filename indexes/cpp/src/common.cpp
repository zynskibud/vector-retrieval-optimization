#include "common.hpp"

#include <cstdlib>

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

}  // namespace vro
