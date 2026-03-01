#include "rope_cuda.cuh"
#include "../../../device/nvidia/cuda_cast.cuh"
#include <cmath>

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half = head_dim / 2;
    size_t total = seq_len * n_heads * half;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t i = idx % half;
    size_t h = (idx / half) % n_heads;
    size_t s = idx / (half * n_heads);

    float pos = static_cast<float>(pos_ids[s]);
    float angle = pos / powf(theta, 2.0f * i / static_cast<float>(head_dim));
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    size_t base = (s * n_heads + h) * head_dim;
    float a = cuda_cast<float>(in[base + i]);
    float b = cuda_cast<float>(in[base + i + half]);

    out[base + i]        = cuda_cast<T>(a * cos_val - b * sin_val);
    out[base + i + half] = cuda_cast<T>(b * cos_val + a * sin_val);
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta, llaisysStream_t stream) {
    size_t half = head_dim / 2;
    size_t total = seq_len * n_heads * half;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    cudaStream_t s = (cudaStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta);
        return;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta);
        return;
    default:
        break;
    }
}
}
