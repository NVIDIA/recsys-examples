#include "hkv_variable.cuh"
namespace dyn_emb {
template class HKVVariable<int64_t, float, EvictStrategy::kLru>;
}
