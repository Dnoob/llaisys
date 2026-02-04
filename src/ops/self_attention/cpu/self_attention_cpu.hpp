#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    llaisysDataType_t type, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, float scale);
}