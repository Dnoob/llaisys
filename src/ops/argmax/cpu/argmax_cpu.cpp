#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cstdint>
#include <limits>


template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    float best_val = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;
    for (size_t i = 0; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<int64_t>(i);
        }
    }

    *max_idx = best_idx;
    *max_val = llaisys::utils::cast<T>(best_val);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<fp16_t *>(max_val), reinterpret_cast<const fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<bf16_t *>(max_val), reinterpret_cast<const bf16_t *>(vals), numel);
    default:                                                                                                                                   
          EXCEPTION_UNSUPPORTED_DATATYPE(type); 
    }
}
}