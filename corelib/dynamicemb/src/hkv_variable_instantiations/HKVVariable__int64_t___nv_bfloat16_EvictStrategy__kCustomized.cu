#include "hkv_variable.cuh"
namespace dyn_emb {
template class HKVVariable<int64_t, __nv_bfloat16, EvictStrategy::kCustomized>;
}
