#include "add_maca.cuh"
#include "../../../device/metax/maca_cast.cuh"

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float va = maca_cast<float>(a[i]);
        float vb = maca_cast<float>(b[i]);
        c[i] = maca_cast<T>(va + vb);
    }
}

namespace llaisys::ops::metax {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    mcStream_t s = (mcStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<__maca_bfloat16 *>(c),
            reinterpret_cast<const __maca_bfloat16 *>(a),
            reinterpret_cast<const __maca_bfloat16 *>(b), numel);
        return;
    default:
        break;
    }
}
}
