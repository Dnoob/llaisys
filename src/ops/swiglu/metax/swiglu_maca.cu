#include "swiglu_maca.cuh"
#include "../../../device/metax/maca_cast.cuh"

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float g = maca_cast<float>(gate[i]);
        float u = maca_cast<float>(up[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[i] = maca_cast<T>(u * g * sigmoid_g);
    }
}

namespace llaisys::ops::metax {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    mcStream_t s = (mcStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<__maca_bfloat16 *>(out),
            reinterpret_cast<const __maca_bfloat16 *>(gate),
            reinterpret_cast<const __maca_bfloat16 *>(up), numel);
        return;
    default:
        break;
    }
}
}
