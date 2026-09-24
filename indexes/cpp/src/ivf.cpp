#include "ivf.hpp"

namespace vro::ivf {

Index build(Matrix& /*vectors*/, const Params& /*params*/, const BuildContext& /*ctx*/) {
    throw NotImplemented("ivf");
}

SearchResult search(const Index& /*index*/, const float* /*query*/, std::size_t /*k*/,
                    const Params& /*params*/) {
    throw NotImplemented("ivf");
}

std::size_t index_bytes(const Index& /*index*/) { return 0; }

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::ivf
