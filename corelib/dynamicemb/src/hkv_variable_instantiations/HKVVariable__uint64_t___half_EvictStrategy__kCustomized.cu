#include "hkv_variable.cuh"
namespace dyn_emb {
template class HKVVariable<uint64_t, __half, EvictStrategy::kCustomized>;
}
