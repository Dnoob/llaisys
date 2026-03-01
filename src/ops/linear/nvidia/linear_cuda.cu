#include "linear_cuda.cuh"
#include "../../../device/nvidia/cuda_cast.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();
    return handle;
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    for (size_t i = idx; i < total; i += gridDim.x * blockDim.x) {
        float val = cuda_cast<float>(out[i]) + cuda_cast<float>(bias[i % N]);
        out[i] = cuda_cast<T>(val);
    }
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,
            llaisysStream_t stream) {
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t s = (cudaStream_t)stream;
    cublasSetStream(handle, s);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha,
                    reinterpret_cast<const float *>(weight), K,
                    reinterpret_cast<const float *>(in), K,
                    &beta,
                    reinterpret_cast<float *>(out), N);
        if (bias) {
            int block = 256, grid = (M * N + block - 1) / block;
            add_bias_kernel<<<grid, block, 0, s>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(bias), M, N);
        }
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha,
                     reinterpret_cast<const __nv_bfloat16 *>(weight), CUDA_R_16BF, K,
                     reinterpret_cast<const __nv_bfloat16 *>(in), CUDA_R_16BF, K,
                     &beta,
                     reinterpret_cast<__nv_bfloat16 *>(out), CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (bias) {
            int block = 256, grid = (M * N + block - 1) / block;
            add_bias_kernel<<<grid, block, 0, s>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(bias), M, N);
        }
        return;
    }
    default:
        break;
    }
}
}
