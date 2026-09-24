#include "ivf_pq.hpp"

namespace vro::ivf_pq {

Index build(Matrix& /*vectors*/, const Params& /*params*/, const BuildContext& /*ctx*/) {
    throw NotImplemented("ivf_pq");
}

SearchResult search(const Index& /*index*/, const float* /*query*/, std::size_t /*k*/,
                    const Params& /*params*/) {
    throw NotImplemented("ivf_pq");
}

std::size_t index_bytes(const Index& /*index*/) { return 0; }

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::ivf_pq
