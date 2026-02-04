#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype());
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous(), "self_attention: all tensors must be contiguous.");

    size_t query_len = q->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t kv_len = k->shape()[0];
    size_t num_kv_heads  = k->shape()[1];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    }
}
} // namespace llaisys::ops
