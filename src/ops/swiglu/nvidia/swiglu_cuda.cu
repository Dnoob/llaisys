#include "swiglu_cuda.cuh"
#include "../../../device/nvidia/cuda_cast.cuh"

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float g = cuda_cast<float>(gate[i]);
        float u = cuda_cast<float>(up[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[i] = cuda_cast<T>(u * g * sigmoid_g);
    }
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    cudaStream_t s = (cudaStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<grid_size, block_size, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        return;
    default:
        break;
    }
}
}
