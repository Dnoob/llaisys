#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t r = 0; r < rows; r++) {
        const T *in_row = in + r * cols;
        T *out_row = out + r * cols;

        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float v = llaisys::utils::cast<float>(in_row[c]);
            sum_sq += v * v;
        }
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);

        for (size_t c = 0; c < cols; c++) {
            float v = llaisys::utils::cast<float>(in_row[c]);
            float w = llaisys::utils::cast<float>(weight[c]);
            out_row[c] = llaisys::utils::cast<T>(v * rms * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, std::byte *weight, llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const fp16_t *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const bf16_t *>(weight), rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}