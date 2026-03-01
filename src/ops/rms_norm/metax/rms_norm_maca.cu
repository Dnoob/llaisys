#include "rms_norm_maca.cuh"
#include "../../../device/metax/maca_cast.cuh"
#include <mcr/mc_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t cols, float eps) {
    size_t row = blockIdx.x;
    const T *in_row = in + row * cols;
    T *out_row = out + row * cols;

    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = maca_cast<float>(in_row[i]);
        local_sum += v * v;
    }

    __shared__ float s_sum[BLOCK_SIZE];
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }

    float rms = rsqrtf(s_sum[0] / static_cast<float>(cols) + eps);

    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        out_row[c] = maca_cast<T>(maca_cast<float>(in_row[c]) * rms * maca_cast<float>(weight[c]));
    }
}

namespace llaisys::ops::metax {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t cols, float eps,
              llaisysStream_t stream) {
    int grid_size = static_cast<int>(rows);
    mcStream_t s = (mcStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<<<grid_size, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), cols, eps);
        return;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<<<grid_size, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<__maca_bfloat16 *>(out),
            reinterpret_cast<const __maca_bfloat16 *>(in),
            reinterpret_cast<const __maca_bfloat16 *>(weight), cols, eps);
        return;
    default:
        break;
    }
}
}
