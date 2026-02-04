#include "self_attention_cpu.hpp" 
#include "../../../utils.hpp"
#include <cmath>                                                                                                     
#include <vector>                                                                                                    
#include <limits> 

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, 
                     size_t query_len, size_t kv_len, size_t num_heads, 
                     size_t num_kv_heads, size_t head_dim, float scale ) {
    size_t group = num_heads / num_kv_heads;
    int offset = static_cast<int>(kv_len) - static_cast<int>(query_len);

    std::vector<float> score(kv_len);

    for (size_t h = 0; h < num_heads; h++) {
        size_t kv_h = h / group;

        for (size_t i = 0; i < query_len; i++) {
            const T *q_ptr = q + (i * num_heads + h) * head_dim;

            // score + causal mask
            for (size_t j = 0; j < kv_len; j++) {
                if (static_cast<int>(j) > static_cast<int>(i) + offset) {
                    score[j] = -std::numeric_limits<float>::infinity();
                } else {
                    const T *k_ptr = k + (j * num_kv_heads + kv_h) * head_dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        dot += llaisys::utils::cast<float>(q_ptr[d]) * llaisys::utils::cast<float>(k_ptr[d]);
                    }
                    score[j] = dot * scale;
                }
            }

            // softmax
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kv_len; j++) {
                if (score[j] > max_val)
                    max_val = score[j];
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < kv_len; j++) {
                score[j] = std::exp(score[j] - max_val);
                sum_exp += score[j];
            }
            for (size_t j = 0; j < kv_len; j++) {
                score[j] /= sum_exp;
            }

            // weight x V
            T *out_ptr = attn_val + (i * num_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (size_t j = 0; j < kv_len; j++) {
                    const T *v_ptr = v + (j * num_kv_heads + kv_h) * head_dim;
                    val += score[j] * llaisys::utils::cast<float>(v_ptr[d]);
                }
                out_ptr[d] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<fp16_t *>(attn_val),
                               reinterpret_cast<const fp16_t *>(q),
                               reinterpret_cast<const fp16_t *>(k),
                               reinterpret_cast<const fp16_t *>(v),
                               query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<bf16_t *>(attn_val),
                               reinterpret_cast<const bf16_t *>(q),
                               reinterpret_cast<const bf16_t *>(k),
                               reinterpret_cast<const bf16_t *>(v),
                               query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} 