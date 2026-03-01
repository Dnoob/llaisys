#include "add_cuda.cuh"
#include "../../../device/nvidia/cuda_cast.cuh"

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float va = cuda_cast<float>(a[i]);
        float vb = cuda_cast<float>(b[i]);
        c[i] = cuda_cast<T>(va + vb);
    }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    cudaStream_t s = (cudaStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        return;
    default:
        break;
    }
}
}
