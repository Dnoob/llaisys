#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < n_heads; h++) {
            const T *in_ptr = in + (s * n_heads + h) * head_dim;
            T *out_ptr = out + (s * n_heads + h) * head_dim;

            for (size_t i = 0; i < half; i++) {
                float angle = pos / std::pow(theta, 2.0f * i / static_cast<float>(head_dim));
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                float a = llaisys::utils::cast<float>(in_ptr[i]);
                float b = llaisys::utils::cast<float>(in_ptr[i + half]);

                out_ptr[i] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out_ptr[i + half] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), pos, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), pos, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}