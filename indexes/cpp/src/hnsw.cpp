#include "hnsw.hpp"

namespace vro::hnsw {

Index build(Matrix& /*vectors*/, const Params& /*params*/, const BuildContext& /*ctx*/) {
    throw NotImplemented("hnsw");
}

SearchResult search(const Index& /*index*/, const float* /*query*/, std::size_t /*k*/,
                    const Params& /*params*/) {
    throw NotImplemented("hnsw");
}

std::size_t index_bytes(const Index& /*index*/) { return 0; }

std::map<std::string, double> extra(const Index& /*index*/) { return {}; }

}  // namespace vro::hnsw
