#include "linear_maca.cuh"
#include "../../../device/metax/maca_cast.cuh"
#include <mcr/mc_runtime.h>
#include <mcblas/mcblas.h>

static mcblasHandle_t get_mcblas_handle() {
    static mcblasHandle_t handle = []() {
        mcblasHandle_t h;
        mcblasCreate(&h);
        return h;
    }();
    return handle;
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    for (size_t i = idx; i < total; i += gridDim.x * blockDim.x) {
        float val = maca_cast<float>(out[i]) + maca_cast<float>(bias[i % N]);
        out[i] = maca_cast<T>(val);
    }
}

namespace llaisys::ops::metax {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,
            llaisysStream_t stream) {
    mcblasHandle_t handle = get_mcblas_handle();
    mcStream_t s = (mcStream_t)stream;
    mcblasSetStream(handle, s);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float alpha = 1.0f, beta = 0.0f;
        mcblasSgemm(handle, MCBLAS_OP_T, MCBLAS_OP_N, N, M, K,
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
        mcblasGemmEx(handle, MCBLAS_OP_T, MCBLAS_OP_N, N, M, K,
                     &alpha,
                     reinterpret_cast<const __maca_bfloat16 *>(weight), MACA_R_16BF, K,
                     reinterpret_cast<const __maca_bfloat16 *>(in), MACA_R_16BF, K,
                     &beta,
                     reinterpret_cast<__maca_bfloat16 *>(out), MACA_R_16BF, N,
                     MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
        if (bias) {
            int block = 256, grid = (M * N + block - 1) / block;
            add_bias_kernel<<<grid, block, 0, s>>>(
                reinterpret_cast<__maca_bfloat16 *>(out),
                reinterpret_cast<const __maca_bfloat16 *>(bias), M, N);
        }
        return;
    }
    default:
        break;
    }
}
}
